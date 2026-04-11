from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from agents.qa.prompts import QA_SYSTEM_PROMPT
from agents.qa.rag import (
    build_or_load_vectorstore,
    format_retrieved_context,
    format_source_list,
    last_user_query,
)
from config.settings import AppSettings
from state.schema import FincentState


def make_qa_financial_docs_node(
    qa_llm: BaseChatModel,
    *,
    api_key: str,
    settings: AppSettings,
):
    """Factory: RAG-backed Q&A node that appends one assistant message."""
    vectorstore_cache: dict[str, Any] = {"value": None}

    def get_vectorstore():
        if vectorstore_cache["value"] is None:
            vectorstore_cache["value"] = build_or_load_vectorstore(api_key, settings)
        return vectorstore_cache["value"]

    def qa_financial_docs_node(state: FincentState) -> dict[str, Any]:
        query = last_user_query(state["messages"])
        vectorstore = get_vectorstore()
        retrieved_docs = vectorstore.similarity_search(query=query, k=settings.qa_top_k)
        context = format_retrieved_context(retrieved_docs)
        source_footer = format_source_list(retrieved_docs)

        messages: list[BaseMessage] = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            SystemMessage(
                content=(
                    "Retrieved context snippets from indexed docs:\n\n"
                    f"{context if context else 'No snippets retrieved.'}"
                )
            ),
            *state["messages"],
        ]
        reply = qa_llm.invoke(messages)
        reply_text = str(getattr(reply, "content", ""))
        if "Sources:" not in reply_text:
            reply_text = f"{reply_text.rstrip()}\n\n{source_footer}"
        return {"messages": [AIMessage(content=reply_text)]}

    return qa_financial_docs_node
