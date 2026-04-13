from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agents.hub.planner import classify_intent
from config.runtime_logging import fincent_log
from state.schema import FincentState, RouteName

logger = logging.getLogger(__name__)


def make_route_intent_node(router_llm: BaseChatModel):
    """Factory: LangGraph node that sets `route` based on planner output."""

    def route_intent_node(state: FincentState) -> dict[str, Any]:
        n_msg = len(state["messages"])
        fincent_log(
            logger,
            logging.INFO,
            "agent.hub_route.start",
            component="agent.hub",
            lc_message_count=n_msg,
        )
        decision = classify_intent(router_llm, state["messages"])
        route: RouteName = decision.route
        fincent_log(
            logger,
            logging.INFO,
            "agent.hub_route.decision",
            component="agent.hub",
            route=route,
        )
        return {"route": route}

    return route_intent_node


def route_from_state(state: FincentState) -> RouteName:
    """Edge function: read planner output for conditional routing."""
    route = state["route"]
    fincent_log(
        logger,
        logging.INFO,
        "graph.edge.from_hub_route",
        component="graph",
        target=route,
    )
    return route
