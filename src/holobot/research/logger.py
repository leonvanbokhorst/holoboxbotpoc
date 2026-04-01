"""Research event logger — structured JSONL for interaction analysis.

Logs every event (user utterance, agent response, spontaneous impulse,
state transitions) for Pieter and Maaike's believability research.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)


class ResearchLogger:
    """Append-only JSONL logger for research events."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._log_dir = Path(self.cfg.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = f"session_{int(time.time())}"
        self._file = self._log_dir / f"{self._session_id}.jsonl"
        self._start_time = time.time()

        self.log_event("session_start", {
            "spontaneity_level": self.cfg.spontaneity_level.name,
            "persona_path": str(self.cfg.persona_path),
        })
        log.info("research log: %s", self._file)

    @property
    def session_id(self) -> str:
        return self._session_id

    def log_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        entry = {
            "ts": time.time(),
            "elapsed_s": round(time.time() - self._start_time, 3),
            "session": self._session_id,
            "event": event,
            **(data or {}),
        }
        line = json.dumps(entry, ensure_ascii=False)
        with open(self._file, "a") as f:
            f.write(line + "\n")

    def log_user_utterance(self, text: str, duration_ms: int = 0) -> None:
        self.log_event("user_utterance", {"text": text, "duration_ms": duration_ms})

    def log_agent_response(
        self, text: str, triggered_by: str = "user", latency_ms: int = 0
    ) -> None:
        self.log_event("agent_response", {
            "text": text,
            "triggered_by": triggered_by,
            "latency_ms": latency_ms,
        })

    def log_spontaneous_impulse(
        self, impulse_type: str, text: str, silence_before_s: float
    ) -> None:
        self.log_event("spontaneous_impulse", {
            "type": impulse_type,
            "text": text,
            "silence_before_s": round(silence_before_s, 2),
            "level": self.cfg.spontaneity_level.name,
        })

    def log_state_change(self, state: str, detail: str = "") -> None:
        self.log_event("state_change", {"state": state, "detail": detail})

    def close(self) -> None:
        self.log_event("session_end")
