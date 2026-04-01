"""ElevenLabs TTS provider with streaming support."""

from __future__ import annotations

import io
import logging
import wave

import numpy as np
from elevenlabs import AsyncElevenLabs

from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)


class ElevenLabsTTS:
    """Synthesize speech using the ElevenLabs API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._client = AsyncElevenLabs(api_key=self.cfg.elevenlabs_api_key)

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        voice_id = self.cfg.elevenlabs_voice_id
        if not voice_id:
            log.warning("no ElevenLabs voice_id configured, using default")
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel — fallback

        audio_iter = await self._client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_16000",
        )

        chunks: list[bytes] = []
        async for chunk in audio_iter:
            chunks.append(chunk)

        raw = b"".join(chunks)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        log.debug("tts synthesized %.2fs of audio", len(audio) / 16_000)
        return audio, 16_000


class OpenAITTS:
    """Fallback TTS using OpenAI's tts-1 model (no ElevenLabs key needed)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.cfg.openai_api_key)

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        response = await self._client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="wav",
        )

        wav_bytes = response.content
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

        log.debug("openai-tts synthesized %.2fs of audio", len(audio) / sr)
        return audio, sr
