from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.runtime_logging import fincent_log
from config.settings import AppSettings

logger = logging.getLogger(__name__)


def _split_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Parse optional YAML frontmatter from a markdown file."""
    if not raw_text.startswith("---\n"):
        return {}, raw_text

    parts = raw_text.split("\n---\n", maxsplit=1)
    if len(parts) != 2:
        return {}, raw_text

    frontmatter_block = parts[0][4:]
    body = parts[1]
    try:
        parsed = yaml.safe_load(frontmatter_block) or {}
    except yaml.YAMLError:
        return {}, raw_text

    if not isinstance(parsed, dict):
        return {}, raw_text
    return parsed, body


def load_markdown_documents(data_dir: Path) -> list[Document]:
    """
    Load markdown files under the configured data directory.

    Files inside `vector_store` and README markdowns are skipped.
    """
    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*.md")):
        if "vector_store" in path.parts:
            continue
        if path.name.lower() == "readme.md":
            continue

        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        title = str(frontmatter.get("title") or path.stem)
        doc_id = str(frontmatter.get("doc_id") or path.stem)
        source_urls = frontmatter.get("source_urls")
        if not isinstance(source_urls, list):
            source_urls = []

        relative_path = str(path.relative_to(data_dir.parent).as_posix())
        documents.append(
            Document(
                page_content=body.strip(),
                metadata={
                    "doc_id": doc_id,
                    "title": title,
                    "path": relative_path,
                    "source_urls": source_urls,
                },
            )
        )
    return documents


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Chunk docs with large segments to keep related context together."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks


def _index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def build_or_load_vectorstore(api_key: str, settings: AppSettings) -> FAISS:
    """
    Build the FAISS index from data markdown files once, then load from disk.
    """
    index_dir = settings.qa_index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=settings.qa_embedding_model,
    )

    if _index_exists(index_dir):
        fincent_log(
            logger,
            logging.INFO,
            "rag.faiss.load",
            component="rag",
            index_dir=str(index_dir),
        )
        return FAISS.load_local(
            str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    fincent_log(
        logger,
        logging.INFO,
        "rag.faiss.build.start",
        component="rag",
        data_dir=str(settings.qa_data_dir),
        index_dir=str(index_dir),
    )
    docs = load_markdown_documents(settings.qa_data_dir)
    if not docs:
        raise ValueError(f"No markdown docs found in {settings.qa_data_dir}")

    chunks = chunk_documents(
        docs,
        chunk_size=settings.qa_chunk_size,
        chunk_overlap=settings.qa_chunk_overlap,
    )
    fincent_log(
        logger,
        logging.INFO,
        "rag.faiss.build.chunked",
        component="rag",
        markdown_doc_count=len(docs),
        chunk_count=len(chunks),
        chunk_size=settings.qa_chunk_size,
        chunk_overlap=settings.qa_chunk_overlap,
    )
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(str(index_dir))
    fincent_log(
        logger,
        logging.INFO,
        "rag.faiss.build.saved",
        component="rag",
        index_dir=str(index_dir),
    )
    return vectorstore


def last_user_query(messages: Sequence[BaseMessage]) -> str:
    """Extract latest user message text from conversation history."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def format_retrieved_context(documents: Sequence[Document]) -> str:
    """Render retrieved chunks into a prompt-friendly context block."""
    context_parts: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        title = str(doc.metadata.get("title", "Untitled"))
        doc_id = str(doc.metadata.get("doc_id", "unknown"))
        path = str(doc.metadata.get("path", ""))
        context_parts.append(
            (
                f"[Snippet {idx}] doc_id={doc_id} title={title} path={path}\n"
                f"{doc.page_content}"
            )
        )
    return "\n\n".join(context_parts)


def format_source_list(documents: Sequence[Document]) -> str:
    """Create a deduplicated, human-readable sources footer."""
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for doc in documents:
        doc_id = str(doc.metadata.get("doc_id", "unknown"))
        title = str(doc.metadata.get("title", "Untitled"))
        path = str(doc.metadata.get("path", ""))
        key = (doc_id, path)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {doc_id}: {title} (`{path}`)")

    if not lines:
        return "Sources:\n- No indexed document matched this question."
    return "Sources:\n" + "\n".join(lines)
