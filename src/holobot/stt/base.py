"""STT provider interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class STTProvider(Protocol):
    """Transcribe a speech segment to text."""

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str: ...
