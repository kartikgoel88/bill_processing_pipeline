"""Ollama (local) OpenAI-compatible API provider."""

from __future__ import annotations

import logging
from typing import Any

import requests

from core.models import LLMResponse
from providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)
DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"


class OllamaProvider(BaseLLMProvider):
    """Ollama local server; same HTTP contract as OpenAI chat/completions."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str = "",
        model: str = "llama3.2",
        timeout_sec: int = 120,
    ) -> None:
        self._base_url = (base_url or DEFAULT_OLLAMA_BASE).rstrip("/")
        self._api_key = api_key or ""
        self._model = model
        self._timeout = timeout_sec

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        model = kwargs.get("model") or self._model
        messages = [{"role": "user", "content": prompt}]
        text = self.chat(messages, model=model, **kwargs)
        return LLMResponse(text=text, model=model)

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": kwargs.get("model") or self._model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": False,
        }
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("response_format") is not None:
            payload["response_format"] = kwargs["response_format"]
        resp = requests.post(
            url, json=payload, headers=self._headers(), timeout=self._timeout
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        return (choice.get("message") or {}).get("content", "").strip()
