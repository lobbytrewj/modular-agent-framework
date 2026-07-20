from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

class LLMError(Exception):
    """Raised when an LLM call fails for any reason (auth, network, API error)."""


class LLMClient:
    """Small wrapper around the OpenAI api

    Example of how to use:
        client = LLMClient()
        reply = client.complete(
            system_prompt="You are a helpful assistant.",
            user_message="Say hello.",
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise LLMError(
                "No OpenAI API key found. Set OPENAI_API_KEY in your .env file "
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    def complete(self, system_prompt: str, user_message: str, history: Optional[List[dict]] = None) -> str:
        """Sends a system prompt and user message to the model and returns response text."""
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except APIConnectionError as exc:
            raise LLMError(f"Couldn't reach the OpenAI API: {exc}") from exc
        except APIStatusError as exc:
            raise LLMError(
                f"OpenAI API returned an error (status {exc.status_code}): {exc.message}"
            ) from exc
        except OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise LLMError("Model returned an empty response.")
        return content
