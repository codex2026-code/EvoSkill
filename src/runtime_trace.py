"""Structured runtime trace logging for external interactions."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _is_enabled() -> bool:
    return (os.getenv("EVOSKILL_RUNTIME_TRACE_ENABLED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def append_runtime_trace(event_type: str, payload: dict[str, Any]) -> None:
    """Append one JSONL runtime trace event when tracing is enabled."""
    if not _is_enabled():
        return

    trace_file_raw = (os.getenv("EVOSKILL_RUNTIME_TRACE_FILE") or "").strip()
    if not trace_file_raw:
        return

    trace_path = Path(trace_file_raw)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "pid": os.getpid(),
        **payload,
    }

    line = json.dumps(event, ensure_ascii=False, default=str)
    with _LOCK:
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
