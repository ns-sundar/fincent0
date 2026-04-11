from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Static defaults and environment-derived configuration."""

    default_chat_model: str
    router_temperature: float
    qa_temperature: float


def load_app_settings() -> AppSettings:
    """Read non-secret defaults from the environment."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    router_temp = float(os.getenv("FINCENT_ROUTER_TEMPERATURE", "0"))
    qa_temp = float(os.getenv("FINCENT_QA_TEMPERATURE", "0.2"))
    return AppSettings(
        default_chat_model=model,
        router_temperature=router_temp,
        qa_temperature=qa_temp,
    )
