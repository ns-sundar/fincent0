from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

_configured = False


class FincentJsonFormatter(logging.Formatter):
    """One JSON object per line (stdout/stderr), for HF Spaces and aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        out: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
        }
        fincent = getattr(record, "fincent", None)
        if isinstance(fincent, dict):
            out.update(fincent)
        else:
            out["message"] = record.getMessage()
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info).strip()
        return json.dumps(out, default=str, ensure_ascii=False)


def configure_fincent_logging() -> None:
    """Idempotent: root logger emits structured JSON lines to stderr."""
    global _configured
    if _configured:
        return
    level_name = os.getenv("FINCENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(FincentJsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    _configured = True


def fincent_log(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a single structured record; `event` is the stable name for tracing."""
    data: dict[str, Any] = {"event": event, **fields}
    logger.log(level, event, extra={"fincent": data})


def fincent_log_exception(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Like fincent_log at ERROR with exception chain attached (for JSON `exception` field)."""
    data: dict[str, Any] = {"event": event, **fields}
    logger.error(event, exc_info=True, extra={"fincent": data})
