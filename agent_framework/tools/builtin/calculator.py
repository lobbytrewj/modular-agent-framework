from __future__ import annotations

import ast
import math
import operator
from typing import Any, Union

from agent_framework.tools.base import Tool, ToolPermission

# Arithmetic an agent can be trusted with.
#
# The obvious implementation of this tool is `eval(expression)`, and it is a
# remote code execution hole: a model that can be talked into emitting
# `__import__("os").system(...)` as its "math" now runs it. So the expression
# is parsed to an AST and walked, and anything that isn't arithmetic is
# rejected before evaluation - a whitelist, because a blacklist of dangerous
# syntax is a list you will always be one item behind on.

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Named functions callable from an expression. All pure, all numeric.
FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *values: sum(values),
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "hypot": math.hypot,
    "factorial": math.factorial,
    "degrees": math.degrees,
    "radians": math.radians,
}

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Nothing here can escape the process, but it can still burn it: `9**9**9`
# parses as pure arithmetic and then tries to allocate a number with hundreds
# of millions of digits. These bounds keep a hostile expression from turning
# into a denial of service on the agent that ran it.
MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT = 1000
MAX_FACTORIAL_INPUT = 1000

Number = Union[int, float]


class CalculatorError(ValueError):
    """Raised for anything the calculator refuses to evaluate."""


def _describe(node: ast.AST) -> str:
    return type(node).__name__


def _evaluate(node: ast.AST) -> Any:
    """Recursively evaluate one whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"only numeric literals are allowed, got {node.value!r}")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalculatorError(
            f"unknown name '{node.id}'; known constants: {sorted(CONSTANTS)}"
        )

    if isinstance(node, ast.BinOp):
        op = BINARY_OPERATORS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator {_describe(node.op)} is not allowed")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _guard_power(left, right)
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = UNARY_OPERATORS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator {_describe(node.op)} is not allowed")
        return op(_evaluate(node.operand))

    if isinstance(node, ast.Call):
        return _evaluate_call(node)

    # Everything else - attribute access, subscripts, comprehensions, lambdas,
    # walrus assignments - lands here and is refused. This branch is the whole
    # security boundary, so it stays a hard failure with no fallback.
    raise CalculatorError(f"expression element {_describe(node)} is not allowed")


def _evaluate_call(node: ast.Call) -> Any:
    if not isinstance(node.func, ast.Name):
        # Blocks `(...).__class__(...)` and every other attribute-based route
        # out of the arithmetic sandbox.
        raise CalculatorError("only direct calls to named functions are allowed")
    if node.keywords:
        raise CalculatorError("keyword arguments are not supported")

    function = FUNCTIONS.get(node.func.id)
    if function is None:
        raise CalculatorError(
            f"unknown function '{node.func.id}'; available: {sorted(FUNCTIONS)}"
        )

    arguments = [_evaluate(argument) for argument in node.args]
    if node.func.id == "factorial":
        _guard_factorial(arguments)
    return function(*arguments)


def _guard_power(base: Number, exponent: Number) -> None:
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(
            f"exponent {exponent} exceeds the maximum of {MAX_EXPONENT}"
        )
    if abs(base) > 1 and abs(base) ** min(abs(exponent), 64) == math.inf:
        raise CalculatorError("result would overflow")


def _guard_factorial(arguments: list) -> None:
    if arguments and isinstance(arguments[0], (int, float)):
        if arguments[0] > MAX_FACTORIAL_INPUT:
            raise CalculatorError(
                f"factorial argument exceeds the maximum of {MAX_FACTORIAL_INPUT}"
            )


def calculate(expression: str) -> Number:
    """Evaluate an arithmetic expression, or raise CalculatorError.

    Returns the number itself rather than a formatted string: the Tool wrapper
    keeps structured output, and a caller that wants text can render it.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError(
            f"expression exceeds {MAX_EXPRESSION_CHARS} characters"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse expression: {exc.msg}") from exc

    result = _evaluate(tree)

    if isinstance(result, float) and (math.isinf(result) or math.isnan(result)):
        raise CalculatorError(f"result is not a finite number ({result})")
    return result


CALCULATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": (
                "An arithmetic expression, e.g. '2 + 2 * 10' or 'sqrt(144) / 3'. "
                "Supports + - * / // % **, parentheses, the constants "
                "pi/e/tau, and functions like abs, round, min, max, sqrt, "
                "log, exp and the trigonometric ones."
            ),
        }
    },
    "required": ["expression"],
}


def build_calculator_tool() -> Tool:
    """The calculator as a registrable Tool. Reads nothing, writes nothing."""
    return Tool(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression exactly. Use this instead of "
            "doing arithmetic yourself whenever a number matters."
        ),
        func=calculate,
        parameters_schema=CALCULATOR_SCHEMA,
        required_permission=ToolPermission.READ_ONLY,
    )
