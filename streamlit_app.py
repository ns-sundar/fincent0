from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import cast

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from config.runtime_logging import (
    configure_fincent_logging,
    fincent_log,
    fincent_log_exception,
)
from config.settings import AppSettings, load_app_settings
from graph.workflow import build_compiled_graph
from state.adapters import chat_rows_to_messages, messages_to_chat_rows
from state.schema import UiChatRow

logger = logging.getLogger(__name__)

SESSION_CHAT_ROWS_KEY = "fincent_ui_chat_rows"

# OpenAI Chat Completions model ids (see https://platform.openai.com/docs/models)
CHAT_MODEL_OPTIONS: tuple[str, ...] = (
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-4o-mini",
)


def configure_page() -> None:
    st.set_page_config(page_title="Fincent", page_icon="🏦", layout="centered")
    st.title("Fincent")
    st.caption("Self-help Finance for Fun")


def render_sidebar(settings: AppSettings) -> str:
    """Model choice for this session. API key comes from OPENAI_API_KEY (e.g. HF Secrets)."""
    with st.sidebar:
        st.header("Settings")
        default_model = settings.default_chat_model
        if default_model in CHAT_MODEL_OPTIONS:
            default_index = CHAT_MODEL_OPTIONS.index(default_model)
        else:
            default_index = 0
        model_name = st.selectbox(
            "Model",
            options=list(CHAT_MODEL_OPTIONS),
            index=default_index,
            help="OpenAI chat model id for routing and Q&A.",
        )

        if st.button("Clear conversation"):
            st.session_state.pop(SESSION_CHAT_ROWS_KEY, None)
            st.rerun()

    return model_name


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
    fincent_log(
        logger,
        logging.INFO,
        "streamlit.graph_compile",
        component="streamlit",
        chat_model=model_name,
        embedding_model=settings.qa_embedding_model,
        note="cache miss",
    )
    return build_compiled_graph(api_key, settings, model_name=model_name)


def render_chat_history(rows: Sequence[UiChatRow]) -> None:
    for row in rows:
        with st.chat_message(row["role"]):
            st.markdown(row["content"])


def run_graph_turn(graph: CompiledStateGraph, rows: list[UiChatRow]) -> None:
    """Invoke LangGraph and replace the UI transcript from the returned message list."""
    lc_messages = chat_rows_to_messages(rows)
    fincent_log(
        logger,
        logging.INFO,
        "graph.invoke.start",
        component="graph",
        lc_message_count=len(lc_messages),
    )
    result = graph.invoke({"messages": lc_messages})
    updated = messages_to_chat_rows(result["messages"])
    fincent_log(
        logger,
        logging.INFO,
        "graph.invoke.done",
        component="graph",
        ui_row_count=len(updated),
        final_route=result.get("route"),
    )
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
    preview = prompt if len(prompt) <= 120 else f"{prompt[:117]}..."
    fincent_log(
        logger,
        logging.INFO,
        "ingress.chat_input",
        component="ingress",
        chat_model=model_name,
        prior_turns=len(rows),
        prompt_length=len(prompt),
        prompt_preview=preview,
    )
    rows.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        assistant_text = (
            "OpenAI API key is missing. Set the OPENAI_API_KEY secret (e.g. Hugging Face "
            "Space **Settings → Variables and secrets**) or export it in your environment."
        )
        rows.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.warning(assistant_text)
        fincent_log(
            logger,
            logging.WARNING,
            "ingress.missing_openai_key",
            component="ingress",
            skipped="graph",
        )
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
            fincent_log_exception(logger, "graph.invoke.failed", component="graph")
            st.exception(e)
            return

        if rows and rows[-1]["role"] == "assistant":
            st.markdown(rows[-1]["content"])


def main() -> None:
    configure_fincent_logging()
    configure_page()
    settings = load_app_settings()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = render_sidebar(settings)
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
