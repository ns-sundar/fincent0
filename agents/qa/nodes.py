from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from agents.qa.prompts import QA_SYSTEM_PROMPT
from state.schema import FinnieState


def make_qa_financial_docs_node(qa_llm: BaseChatModel):
    """Factory: LangGraph node that appends a single assistant message."""

    def qa_financial_docs_node(state: FinnieState) -> dict[str, Any]:
        messages: list[BaseMessage] = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            *state["messages"],
        ]
        reply = qa_llm.invoke(messages)
        return {"messages": [reply]}

    return qa_financial_docs_node
