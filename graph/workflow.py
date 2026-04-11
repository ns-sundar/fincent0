from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.hub.decline import decline_node
from agents.hub.nodes import make_route_intent_node, route_from_state
from agents.qa.nodes import make_qa_financial_docs_node
from config.llm import make_chat_model
from config.settings import AppSettings
from state.schema import FinnieState


def build_compiled_graph(
    api_key: str,
    settings: AppSettings,
    *,
    model_name: str,
) -> CompiledStateGraph[FinnieState, None, FinnieState, FinnieState]:
    """
    Hub-and-spoke graph:

    START → hub_route → (qa_financial_docs | hub_decline) → END
    """
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

    workflow = StateGraph(FinnieState)
    workflow.add_node("hub_route", make_route_intent_node(router_llm))
    workflow.add_node("qa_financial_docs", make_qa_financial_docs_node(qa_llm))
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

    return workflow.compile()
