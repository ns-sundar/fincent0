from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from config.settings import AppSettings, load_app_settings
from graph.workflow import build_compiled_graph
from state.adapters import chat_rows_to_messages, messages_to_chat_rows
from state.schema import UiChatRow

SESSION_CHAT_ROWS_KEY = "fincent_ui_chat_rows"


def configure_page() -> None:
    st.set_page_config(page_title="Fincent", page_icon="🏦", layout="centered")
    st.title("Fincent - Your Friendly Neighborhood Financial Assistant")
    st.caption("Self-help Finance for Fun but Not Profit.")


def render_sidebar(settings: AppSettings) -> tuple[str, str]:
    """Collect secrets and model override. Returns `(api_key, model_name)`."""
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Not stored on disk; kept in this Streamlit session.",
        ).strip()

        model_name = st.text_input(
            "Model",
            value=settings.default_chat_model,
            help="Default is gpt-4o-mini (overridable).",
        ).strip() or settings.default_chat_model

        if st.button("Clear conversation"):
            st.session_state.pop(SESSION_CHAT_ROWS_KEY, None)
            st.rerun()

    return api_key, model_name


def ensure_chat_rows() -> list[UiChatRow]:
    if SESSION_CHAT_ROWS_KEY not in st.session_state:
        st.session_state[SESSION_CHAT_ROWS_KEY] = []
    return cast(list[UiChatRow], st.session_state[SESSION_CHAT_ROWS_KEY])


@st.cache_resource(show_spinner=False)
def get_compiled_graph(
    api_key: str,
    model_name: str,
    settings: AppSettings,
) -> CompiledStateGraph:
    """Compile once per (key, model, settings) tuple; safe for Streamlit reruns."""
    return build_compiled_graph(api_key, settings, model_name=model_name)


def render_chat_history(rows: Sequence[UiChatRow]) -> None:
    for row in rows:
        with st.chat_message(row["role"]):
            st.markdown(row["content"])


def run_graph_turn(graph: CompiledStateGraph, rows: list[UiChatRow]) -> None:
    """Invoke LangGraph and replace the UI transcript from the returned message list."""
    lc_messages = chat_rows_to_messages(rows)
    result = graph.invoke({"messages": lc_messages})
    updated = messages_to_chat_rows(result["messages"])
    rows.clear()
    rows.extend(updated)


def handle_new_prompt(
    prompt: str,
    *,
    api_key: str,
    model_name: str,
    settings: AppSettings,
    rows: list[UiChatRow],
) -> None:
    """Append the user turn, run the hub-and-spoke graph, and render this turn inline."""
    rows.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        assistant_text = (
            "Please set your OpenAI API key in the sidebar (or export OPENAI_API_KEY)."
        )
        rows.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.warning(assistant_text)
        return

    graph = get_compiled_graph(
        api_key,
        model_name,
        settings,
    )

    with st.chat_message("assistant"):
        try:
            run_graph_turn(graph, rows)
        except Exception as e:
            st.exception(e)
            return

        if rows and rows[-1]["role"] == "assistant":
            st.markdown(rows[-1]["content"])


def main() -> None:
    configure_page()
    settings = load_app_settings()
    api_key, model_name = render_sidebar(settings)
    rows = ensure_chat_rows()

    render_chat_history(rows)

    prompt = st.chat_input("Ask a finance question…")
    if prompt:
        handle_new_prompt(
            prompt,
            api_key=api_key,
            model_name=model_name,
            settings=settings,
            rows=rows,
        )


if __name__ == "__main__":
    main()
