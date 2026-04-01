"""Microphone capture with Voice Activity Detection (silero-vad).

Streams audio from the mic, runs VAD per frame, and yields complete
speech segments (as numpy arrays) when the user stops talking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import AsyncIterator

import numpy as np
import sounddevice as sd
import torch

from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)

_vad_model: torch.jit.ScriptModule | None = None


def _get_vad_model() -> torch.jit.ScriptModule:
    global _vad_model
    if _vad_model is None:
        _vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
    return _vad_model


class AudioCapture:
    """Async mic capture with VAD-based speech segmentation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._vad = _get_vad_model()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._running = False
        self._last_speech_time: float = 0.0
        self._speech_started: bool = False

    @property
    def last_speech_time(self) -> float:
        return self._last_speech_time

    @property
    def silence_duration(self) -> float:
        if self._last_speech_time == 0.0:
            return 0.0
        return time.monotonic() - self._last_speech_time

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        if status:
            log.warning("audio input status: %s", status)
        self._queue.put_nowait(indata[:, 0].copy())

    async def stream_speech_segments(self) -> AsyncIterator[np.ndarray]:
        """Yield numpy arrays of complete speech segments (16kHz mono float32)."""

        vad_threshold = 0.5
        pre_speech_buffer_frames = 10  # ~300ms lookback so we don't clip onsets
        post_speech_frames = int(self.cfg.silence_threshold_s / (self.cfg.block_duration_ms / 1000))

        ring: deque[np.ndarray] = deque(maxlen=pre_speech_buffer_frames)
        speech_frames: list[np.ndarray] = []
        silent_count = 0
        self._speech_started = False
        self._running = True

        stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="float32",
            blocksize=self.cfg.block_size,
            callback=self._audio_callback,
        )

        with stream:
            log.info("mic capture started (rate=%d, block=%d)", self.cfg.sample_rate, self.cfg.block_size)
            while self._running:
                try:
                    frame = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                tensor = torch.from_numpy(frame)
                speech_prob = self._vad(tensor, self.cfg.sample_rate).item()

                if speech_prob >= vad_threshold:
                    self._last_speech_time = time.monotonic()
                    silent_count = 0

                    if not self._speech_started:
                        self._speech_started = True
                        speech_frames.extend(ring)
                        log.debug("speech start detected")

                    speech_frames.append(frame)

                else:
                    ring.append(frame)

                    if self._speech_started:
                        silent_count += 1
                        speech_frames.append(frame)

                        if silent_count >= post_speech_frames:
                            segment = np.concatenate(speech_frames)
                            speech_frames.clear()
                            self._speech_started = False
                            silent_count = 0
                            log.debug("speech segment: %.2fs", len(segment) / self.cfg.sample_rate)
                            yield segment

    def stop(self) -> None:
        self._running = False
