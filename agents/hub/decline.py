from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agents.hub.prompts import decline_user_message
from state.schema import FinnieState


def decline_node(state: FinnieState) -> dict[str, Any]:
    """Hub-authored refusal when the request is outside the allowed Q&A spoke."""
    _ = state  # Decline message is intentionally not personalized.
    return {"messages": [AIMessage(content=decline_user_message())]}
