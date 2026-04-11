from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from state.schema import UiChatRow


def chat_rows_to_messages(rows: Sequence[UiChatRow]) -> list[BaseMessage]:
    """Convert UI-friendly rows into LangChain messages for graph input."""
    out: list[BaseMessage] = []
    for row in rows:
        if row["role"] == "user":
            out.append(HumanMessage(content=row["content"]))
        elif row["role"] == "assistant":
            out.append(AIMessage(content=row["content"]))
        else:
            raise ValueError(f"Unsupported UI role: {row['role']!r}")
    return out


def messages_to_chat_rows(messages: Sequence[BaseMessage]) -> list[UiChatRow]:
    """Keep only user/assistant turns for the Streamlit transcript."""
    rows: list[UiChatRow] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            rows.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            if msg.content:
                rows.append({"role": "assistant", "content": str(msg.content)})
    return rows
