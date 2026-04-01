"""Audio playback with interruption support.

Plays audio chunks through speakers. Can be interrupted mid-playback
(e.g. when a kid starts speaking while the agent is talking).
"""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np
import sounddevice as sd

from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)


class AudioPlayback:
    """Async audio playback with cancel support."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._playing = False
        self._cancel = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def play(self, audio: np.ndarray, sample_rate: int | None = None) -> bool:
        """Play audio array. Returns True if completed, False if interrupted."""
        sr = sample_rate or self.cfg.sample_rate
        self._cancel.clear()
        self._playing = True

        loop = asyncio.get_event_loop()
        completed = await loop.run_in_executor(None, self._play_blocking, audio, sr)
        self._playing = False
        return completed

    def _play_blocking(self, audio: np.ndarray, sr: int) -> bool:
        """Blocking playback in a thread — checks cancel event periodically."""
        try:
            chunk_samples = int(sr * 0.1)  # 100ms chunks for responsive interruption
            offset = 0
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            with stream:
                while offset < len(audio):
                    if self._cancel.is_set():
                        log.debug("playback interrupted at %.2fs", offset / sr)
                        return False

                    end = min(offset + chunk_samples, len(audio))
                    stream.write(audio[offset:end].reshape(-1, 1))
                    offset = end

            return True
        except Exception:
            log.exception("playback error")
            return False

    def interrupt(self) -> None:
        """Cancel current playback (e.g. kid started speaking)."""
        if self._playing:
            self._cancel.set()
