"""One factory function to create the chat model — swap provider by config."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def create_llm() -> BaseChatModel:
    """Return a chat model based on LLM_PROVIDER env var.

    Supported values:
      - "fake"      → scripted FakeChatModel, no API key needed (default)
      - "anthropic"  → Claude via langchain-anthropic
    """
    provider = os.getenv("LLM_PROVIDER", "fake").lower()

    if provider == "fake":
        from src.fake_llm import FakeChatModel

        return FakeChatModel()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            temperature=0,
            max_tokens=1024,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'fake' or 'anthropic'.")
