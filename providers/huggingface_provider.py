"""Hugging Face Inference API provider (optional)."""

from __future__ import annotations

import logging
from typing import Any

from core.models import LLMResponse
from providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Inference API for text generation. Vision can use HF VL models elsewhere."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        api_key: str = "",
        timeout_sec: int = 120,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key or ""
        self._timeout = timeout_sec

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        model = kwargs.get("model") or self._model_id
        text = self.chat([{"role": "user", "content": prompt}], model=model, **kwargs)
        return LLMResponse(text=text, model=model)

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests required for HuggingFaceProvider")
        import requests as req
        url = "https://api-inference.huggingface.co/models/" + (kwargs.get("model") or self._model_id)
        # HF inference often expects a single string input for text generation
        last_content = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                last_content = m["content"]
                break
        payload: dict[str, Any] = {"inputs": last_content}
        resp = req.post(url, json=payload, headers=self._headers(), timeout=self._timeout)
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return (out[0].get("generated_text") or "").strip()
        if isinstance(out, dict) and out.get("generated_text"):
            return str(out["generated_text"]).strip()
        return str(out).strip()
