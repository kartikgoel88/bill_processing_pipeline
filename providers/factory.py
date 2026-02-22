"""Factory for creating LLM providers from config. No hardcoded model names."""

from __future__ import annotations

from typing import Any

from core.interfaces import ILLMProvider
from providers.base import BaseLLMProvider
from providers.openai_provider import OpenAIProvider
from providers.ollama_provider import OllamaProvider
from providers.huggingface_provider import HuggingFaceProvider


def create_provider(
    provider: str,
    *,
    base_url: str | None = None,
    api_key: str = "",
    model: str = "llama3.2",
    timeout_sec: int = 120,
    **kwargs: Any,
) -> ILLMProvider:
    """
    Create an LLM provider by name. All settings from config; easy to add new providers.
    """
    name = (provider or "ollama").strip().lower()
    if name == "openai":
        return OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            model=model or "gpt-4o-mini",
            timeout_sec=timeout_sec,
        )
    if name == "ollama":
        return OllamaProvider(
            base_url=base_url or "http://localhost:11434/v1",
            api_key=api_key,
            model=model or "llama3.2",
            timeout_sec=timeout_sec,
        )
    if name in ("hf", "huggingface"):
        return HuggingFaceProvider(
            model_id=kwargs.get("model_id") or model,
            api_key=api_key,
            timeout_sec=timeout_sec,
        )
    raise ValueError(f"Unknown LLM provider: {provider}. Use openai, ollama, or huggingface.")
