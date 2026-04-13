from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.hub.decline import decline_node
from agents.hub.nodes import make_route_intent_node, route_from_state
from agents.qa.nodes import make_qa_financial_docs_node
from config.llm import make_chat_model
from config.runtime_logging import fincent_log
from config.settings import AppSettings
from state.schema import FincentState

logger = logging.getLogger(__name__)


def build_compiled_graph(
    api_key: str,
    settings: AppSettings,
    *,
    model_name: str,
) -> CompiledStateGraph[FincentState, None, FincentState, FincentState]:
    """
    Hub-and-spoke graph:

    START → hub_route → (qa_financial_docs | hub_decline) → END
    """
    fincent_log(
        logger,
        logging.INFO,
        "workflow.build.start",
        component="workflow",
        chat_model=model_name,
        router_temperature=settings.router_temperature,
        qa_temperature=settings.qa_temperature,
    )
    router_llm = make_chat_model(
        api_key,
        model=model_name,
        temperature=settings.router_temperature,
    )
    qa_llm = make_chat_model(
        api_key,
        model=model_name,
        temperature=settings.qa_temperature,
    )

    workflow = StateGraph(FincentState)
    workflow.add_node("hub_route", make_route_intent_node(router_llm))
    workflow.add_node(
        "qa_financial_docs",
        make_qa_financial_docs_node(qa_llm, api_key=api_key, settings=settings),
    )
    workflow.add_node("hub_decline", decline_node)

    workflow.add_edge(START, "hub_route")
    workflow.add_conditional_edges(
        "hub_route",
        route_from_state,
        {
            "qa": "qa_financial_docs",
            "decline": "hub_decline",
        },
    )
    workflow.add_edge("qa_financial_docs", END)
    workflow.add_edge("hub_decline", END)

    compiled = workflow.compile()
    fincent_log(
        logger,
        logging.INFO,
        "workflow.build.done",
        component="workflow",
        edges="hub_route -> qa_financial_docs | hub_decline -> END",
    )
    return compiled
