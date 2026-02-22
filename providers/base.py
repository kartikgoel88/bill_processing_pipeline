"""
Abstract base for all LLM providers.
Pipeline depends only on this interface; no concrete provider imports in services.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.interfaces import ILLMProvider
from core.models import LLMResponse


class BaseLLMProvider(ILLMProvider, ABC):
    """Abstract LLM provider. Implement generate() and chat(); chat_vision has default."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate text from prompt. kwargs: model, max_tokens, temperature, etc."""
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Chat completion. Returns content string."""
        ...

    def chat_vision(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Vision-capable chat. Default: delegate to chat (OpenAI-compatible APIs support it)."""
        return self.chat(messages, **kwargs)
