from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Static defaults and environment-derived configuration."""

    default_chat_model: str
    router_temperature: float
    qa_temperature: float
    qa_data_dir: Path
    qa_index_dir: Path
    qa_chunk_size: int
    qa_chunk_overlap: int
    qa_top_k: int
    qa_embedding_model: str


def load_app_settings() -> AppSettings:
    """Read non-secret defaults from the environment."""
    project_root = Path(__file__).resolve().parent.parent
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    router_temp = float(os.getenv("FINCENT_ROUTER_TEMPERATURE", "0"))
    qa_temp = float(os.getenv("FINCENT_QA_TEMPERATURE", "0.2"))
    qa_data_dir = Path(os.getenv("FINCENT_QA_DATA_DIR", str(project_root / "data")))
    qa_index_dir = Path(
        os.getenv(
            "FINCENT_QA_INDEX_DIR",
            str(project_root / "data" / "vector_store" / "qa_faiss"),
        )
    )
    qa_chunk_size = int(os.getenv("FINCENT_QA_CHUNK_SIZE", "1800"))
    qa_chunk_overlap = int(os.getenv("FINCENT_QA_CHUNK_OVERLAP", "250"))
    qa_top_k = int(os.getenv("FINCENT_QA_TOP_K", "4"))
    qa_embedding_model = os.getenv("FINCENT_QA_EMBEDDING_MODEL", "text-embedding-3-small")
    return AppSettings(
        default_chat_model=model,
        router_temperature=router_temp,
        qa_temperature=qa_temp,
        qa_data_dir=qa_data_dir,
        qa_index_dir=qa_index_dir,
        qa_chunk_size=qa_chunk_size,
        qa_chunk_overlap=qa_chunk_overlap,
        qa_top_k=qa_top_k,
        qa_embedding_model=qa_embedding_model,
    )
