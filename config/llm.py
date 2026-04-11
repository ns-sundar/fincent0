from __future__ import annotations

from langchain_openai import ChatOpenAI


def make_chat_model(
    api_key: str,
    *,
    model: str,
    temperature: float,
) -> ChatOpenAI:
    """Construct a configured OpenAI chat model (used by hub and spoke agents)."""
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
