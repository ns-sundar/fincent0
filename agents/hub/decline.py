from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from agents.hub.prompts import decline_user_message
from config.runtime_logging import fincent_log
from state.schema import FincentState

logger = logging.getLogger(__name__)


def decline_node(state: FincentState) -> dict[str, Any]:
    """Hub-authored refusal when the request is outside the allowed Q&A spoke."""
    _ = state  # Decline message is intentionally not personalized.
    fincent_log(
        logger,
        logging.INFO,
        "agent.hub_decline.response",
        component="agent.hub_decline",
        rag=False,
    )
    return {"messages": [AIMessage(content=decline_user_message())]}
