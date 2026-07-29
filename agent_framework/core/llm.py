from __future__ import annotations

from typing import List, Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Local pipelines keyed by (model, device, dtype). Model weights are the
# expensive part of local inference, so they're loaded once and shared by every
# agent using the same checkpoint.
_PIPELINE_CACHE: dict = {}


class LLMError(Exception):
    """Raised when a generation call fails for any reason (load, runtime, empty output)."""


class LLMClient:
    """Runs a local Hugging Face model through transformers.

    Everything happens on this machine: no API keys, no network calls once the
    weights are cached.

        client = LLMClient()
        reply = client.complete(
            system_prompt="You are a helpful assistant.",
            user_message="Say hello.",
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise LLMError(
                "The 'transformers' package (and torch) is required to run local "
                "models. Install them with: pip install transformers torch"
            ) from exc

        self.device, self.dtype = self._resolve_device_and_dtype(device, dtype)

        try:
            self._client = self._build_pipeline(pipeline)
        except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
            raise LLMError(
                f"Couldn't load model '{self.model}' on device '{self.device}' "
                f"with dtype '{self.dtype}': {exc}"
            ) from exc

    # --- Backend setup ----------------------------------------------------

    @staticmethod
    def _resolve_device_and_dtype(
        device: Optional[str], dtype: Optional[str]
    ) -> tuple:
        """Pick the accelerator and weight precision for this machine.

        Checkpoints are commonly published in bfloat16, which Apple's MPS
        backend rejects on older torch builds, so precision is pinned per
        device rather than left to the checkpoint: float16 on GPU/MPS, float32
        on CPU. Both can be overridden from config.
        """
        try:
            import torch
        except ImportError as exc:
            raise LLMError(
                "The 'torch' package is required to run local models. "
                "Install it with: pip install torch"
            ) from exc

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype = dtype or ("float32" if device == "cpu" else "float16")
        return device, dtype

    def _build_pipeline(self, pipeline):
        """Load the text-generation pipeline, reusing an already-loaded one.

        Weights are read-only and generation settings are passed per call, so
        agents that differ only in temperature share a single copy in memory.
        """
        cache_key = (self.model, self.device, self.dtype)
        if cache_key not in _PIPELINE_CACHE:
            _PIPELINE_CACHE[cache_key] = self._load_pipeline(pipeline)
        return _PIPELINE_CACHE[cache_key]

    def _load_pipeline(self, pipeline):
        # The precision argument was renamed from `torch_dtype` to `dtype`, so
        # try the current name first and fall back for older installs.
        try:
            return pipeline(
                "text-generation",
                model=self.model,
                device=self.device,
                dtype=self.dtype,
            )
        except TypeError:
            return pipeline(
                "text-generation",
                model=self.model,
                device=self.device,
                torch_dtype=self.dtype,
            )

    # --- Public API -------------------------------------------------------

    def complete(self, system_prompt: str, user_message: str, history: Optional[List[dict]] = None) -> str:
        """Sends a system prompt and user message to the model and returns response text."""
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        prompt = self._to_prompt(messages)
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "return_full_text": False,
            # Greedy decoding at temperature 0, sampling above it.
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        tokenizer = getattr(self._client, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            # Silences the pipeline's "setting pad_token_id" warning on every call.
            generation_kwargs["pad_token_id"] = eos_token_id

        try:
            outputs = self._client(prompt, **generation_kwargs)
        except Exception as exc:  # noqa: BLE001 - one error type for the agent to catch
            raise LLMError(f"Local generation failed: {exc}") from exc

        try:
            content = (outputs[0]["generated_text"] or "").strip()
        except (IndexError, KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected output from pipeline: {outputs!r}") from exc

        if not content:
            raise LLMError("Model returned an empty response.")
        return content

    def _to_prompt(self, messages: List[dict]) -> str:
        """Flatten chat messages into a single prompt string.

        Instruct models ship a chat template; when the tokenizer has one we use
        it so the model sees the format it was tuned on, and fall back to a
        plain transcript otherwise.
        """
        tokenizer = getattr(self._client, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:  # noqa: BLE001 - fall back to the plain transcript
                pass

        lines = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
        lines.append("Assistant:")
        return "\n\n".join(lines)
