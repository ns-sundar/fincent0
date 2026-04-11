from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from langgraph.graph import MessagesState

RouteName = Literal["qa", "decline"]


class FinnieState(MessagesState):
    """
    Hub-and-spoke orchestration state.

    `messages` carries multi-turn conversational memory (user + assistant turns).
    `route` is written by the hub planner and consumed by conditional routing.
    """

    route: NotRequired[RouteName]


class UiChatRow(TypedDict):
    """Serializable chat row for Streamlit `st.session_state` storage."""

    role: Literal["user", "assistant"]
    content: str
