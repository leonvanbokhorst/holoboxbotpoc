"""OpenAI Whisper API speech-to-text provider."""

from __future__ import annotations

import io
import logging
import wave

import numpy as np
from openai import AsyncOpenAI

from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)


class WhisperSTT:
    """Transcribe speech segments using the OpenAI Whisper API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._client = AsyncOpenAI(api_key=self.cfg.openai_api_key)

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        wav_bytes = self._to_wav(audio, sample_rate)

        response = await self._client.audio.transcriptions.create(
            model="whisper-1",
            file=("speech.wav", wav_bytes, "audio/wav"),
            language="nl",
            prompt="Dit is een gesprek met een kind in de bibliotheek.",
        )

        text = response.text.strip()
        log.debug("whisper transcription: %s", text)
        return text

    @staticmethod
    def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        pcm = (audio * 32767).astype(np.int16)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return buf.read()
