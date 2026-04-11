from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.hub.prompts import ROUTER_SYSTEM_PROMPT


class IntentDecision(BaseModel):
    """Structured routing decision produced by the hub planner LLM."""

    route: Literal["qa", "decline"] = Field(
        description="Which spoke to invoke, or decline for out-of-scope requests."
    )


def classify_intent(
    router_llm: BaseChatModel,
    conversation: Sequence[BaseMessage],
) -> IntentDecision:
    """
    Ask the hub model to choose a route using multi-turn context.

    The router sees the same user/assistant transcript the spokes see (no RAG).
    """
    structured = router_llm.with_structured_output(IntentDecision)
    messages: list[BaseMessage] = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        *conversation,
    ]
    decision = structured.invoke(messages)
    if not isinstance(decision, IntentDecision):
        raise TypeError(f"Expected IntentDecision, got {type(decision)}")
    return decision
