"""Small OpenAI-compatible client used by submitted inference policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from inference_opt.budget import Budget


@dataclass
class VLLMClient:
    """Client for one OpenAI-compatible vLLM chat-completions endpoint."""

    base_url: str
    model: str = "student"
    timeout: float = 120.0
    budget: Budget | None = None

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.budget is not None:
            self.budget.reserve_student_call()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            self.base_url.rstrip("/") + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return str(payload["choices"][0]["message"]["content"])


class PolicyModelClient:
    """Restricted view of a model client exposed to submitted policies."""

    def __init__(self, client: VLLMClient):
        self._client = client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text without exposing endpoint or model identity."""
        return self._client.generate(prompt, **kwargs)
