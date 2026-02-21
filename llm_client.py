"""
OpenAI-compatible LLM client with retry and dependency injection.
Supports chat and vision (image URL or base64); works with Ollama, OpenAI, Grok.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Callable, Protocol

import requests

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for injectable LLM client."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
    ) -> str: ...

    def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
    ) -> str: ...


class OpenAICompatibleClient:
    """HTTP client for OpenAI-compatible API (Ollama, OpenAI, Grok)."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        default_model: str = "llama3.2",
        timeout_sec: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout_sec

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
    ) -> str:
        """Single request; no retry. Optional temperature, top_p for deterministic output."""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        return (choice.get("message") or {}).get("content", "").strip()

    def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
    ) -> str:
        """Chat with exponential backoff retry. Forwards temperature, top_p, stream to chat()."""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self.chat(
                    messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream,
                )
            except requests.RequestException as e:
                last_err = e
                logger.warning("LLM request attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_sec * (attempt + 1))
        if last_err:
            raise last_err
        raise RuntimeError("LLM chat failed after retries")

    def chat_vision(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Same as chat but messages may include image_url with base64 or URL.
        Expects message content as list of content parts, e.g.:
        [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
        """
        return self.chat(messages, model=model, max_tokens=max_tokens)

    def chat_vision_with_retry(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ) -> str:
        """Vision chat with retry."""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self.chat_vision(messages, model=model, max_tokens=max_tokens)
            except requests.RequestException as e:
                last_err = e
                logger.warning("LLM vision attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_sec * (attempt + 1))
        if last_err:
            raise last_err
        raise RuntimeError("LLM vision failed after retries")


def image_to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """Encode image bytes as data URL for vision API."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{b64}"
