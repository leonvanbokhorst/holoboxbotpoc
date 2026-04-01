"""TTS provider interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TTSProvider(Protocol):
    """Synthesize text to audio."""

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (audio_array, sample_rate)."""
        ...
