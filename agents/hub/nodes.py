from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agents.hub.planner import classify_intent
from state.schema import FinnieState, RouteName


def make_route_intent_node(router_llm: BaseChatModel):
    """Factory: LangGraph node that sets `route` based on planner output."""

    def route_intent_node(state: FinnieState) -> dict[str, Any]:
        decision = classify_intent(router_llm, state["messages"])
        route: RouteName = decision.route
        return {"route": route}

    return route_intent_node


def route_from_state(state: FinnieState) -> RouteName:
    """Edge function: read planner output for conditional routing."""
    return state["route"]
