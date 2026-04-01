"""Holobot main orchestrator — ties all components into a running agent.

Supports two modes:
  - Voice mode (default): mic -> VAD -> STT -> Brain -> TTS -> speaker
  - Text mode (--text):   stdin -> Brain -> stdout  (no audio hardware needed)

And two conversation styles:
  - Free chat (default): open-ended, spontaneity-only agency
  - Scenario (--scenario): phase-guided interaction from YAML (Meike's designs)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

from holobot.brain.conversation import ConversationEngine
from holobot.brain.persona import load_persona
from holobot.brain.spontaneity import Impulse, SpontaneityEngine
from holobot.config import SpontaneityLevel, get_settings
from holobot.integration.websocket import HoloboxEventServer
from holobot.research.logger import ResearchLogger
from holobot.scenario.child_profile import ChildProfile
from holobot.scenario.runner import ScenarioRunner, load_scenario

log = logging.getLogger(__name__)

SCENARIOS_DIR = Path("scenarios")


class Holobot:
    """Main agent orchestrator."""

    def __init__(
        self,
        text_mode: bool = False,
        scenario_name: str | None = None,
    ) -> None:
        self.cfg = get_settings()
        self.text_mode = text_mode

        self._validate_config()

        self.persona = load_persona(self.cfg.persona_path)
        self.brain = ConversationEngine(self.persona, self.cfg)
        self.logger = ResearchLogger(self.cfg)
        self.events = HoloboxEventServer(self.cfg)
        self.spontaneity = SpontaneityEngine(
            state=self.brain.state,
            on_impulse=self._handle_impulse,
            settings=self.cfg,
        )

        self.profile = ChildProfile()
        self.scenario_runner: ScenarioRunner | None = None

        if scenario_name:
            self._init_scenario(scenario_name)

        self._tts = None
        self._playback = None
        self._capture = None

    def _validate_config(self) -> None:
        if not self.cfg.openai_api_key:
            log.error("OPENAI_API_KEY not set -- copy .env.example to .env and fill in keys")
            sys.exit(1)
        if not self.text_mode and not self.cfg.elevenlabs_api_key:
            log.warning("ELEVENLABS_API_KEY not set -- will fall back to OpenAI TTS")

    def _init_scenario(self, name: str) -> None:
        path = SCENARIOS_DIR / f"{name}.yaml"
        if not path.exists():
            available = [p.stem for p in SCENARIOS_DIR.glob("*.yaml")]
            log.error("scenario '%s' not found. available: %s", name, available)
            sys.exit(1)

        scenario_def = load_scenario(path)
        self.scenario_runner = ScenarioRunner(
            scenario=scenario_def,
            brain=self.brain,
            profile=self.profile,
            settings=self.cfg,
            on_action=self._handle_action,
            on_phase_change=self._handle_phase_change,
            on_profile_update=self._handle_profile_update,
        )
        log.info("loaded scenario: %s (%s)", scenario_def.name, scenario_def.id)

    def _get_tts(self):
        if self._tts is None:
            if self.cfg.elevenlabs_api_key:
                from holobot.tts.elevenlabs import ElevenLabsTTS
                self._tts = ElevenLabsTTS(self.cfg)
            else:
                from holobot.tts.elevenlabs import OpenAITTS
                self._tts = OpenAITTS(self.cfg)
                log.info("no ElevenLabs key, using OpenAI TTS fallback")
        return self._tts

    async def run(self) -> None:
        mode_str = "text" if self.text_mode else "voice"
        scenario_str = self.scenario_runner.scenario_id if self.scenario_runner else "free_chat"
        log.info("starting holobot: mode=%s, scenario=%s", mode_str, scenario_str)
        log.info("persona: %s (%s)", self.persona.name, self.persona.presentation)
        log.info("spontaneity level: %s", self.cfg.spontaneity_level.name)

        self.logger.log_event("session_config", {
            "scenario": scenario_str,
            "text_mode": self.text_mode,
        })

        await self.events.start()
        self.spontaneity.start()

        try:
            if self.text_mode:
                await self._text_loop()
            else:
                await self._voice_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.spontaneity.stop()
            await self.events.stop()
            self.logger.close()
            log.info("holobot stopped")

    # ── Text mode ────────────────────────────────────────────────────

    async def _text_loop(self) -> None:
        print(f"\n  {self.persona.name} is hier! (type 'quit' om te stoppen)\n")

        greeting = await self._get_opening()
        print(f"  {self.persona.name}: {greeting}\n")
        self.logger.log_agent_response(greeting, triggered_by="greeting")
        await self.events.emit_agent_speaking(greeting)

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("  Jij: ")
                )
            except EOFError:
                break

            if user_input.strip().lower() in ("quit", "exit", "stop", "doei"):
                farewell = await self.brain.respond("Het kind zegt gedag en loopt weg.")
                print(f"  {self.persona.name}: {farewell}\n")
                self.logger.log_agent_response(farewell, triggered_by="farewell")
                break

            if not user_input.strip():
                continue

            self.logger.log_user_utterance(user_input)
            await self.events.emit_user_speaking()

            t0 = time.time()
            await self.events.emit_agent_thinking()
            reply = await self._process_input(user_input)
            latency = int((time.time() - t0) * 1000)

            print(f"  {self.persona.name}: {reply}\n")
            self.logger.log_agent_response(reply, latency_ms=latency)
            await self.events.emit_agent_speaking(reply)

            if self.scenario_runner and self.scenario_runner.is_finished:
                print(f"\n  [scenario '{self.scenario_runner.scenario_id}' voltooid]\n")
                break

    # ── Voice mode ───────────────────────────────────────────────────

    async def _voice_loop(self) -> None:
        from holobot.audio.capture import AudioCapture
        from holobot.audio.playback import AudioPlayback
        from holobot.stt.whisper import WhisperSTT

        stt = WhisperSTT(self.cfg)
        self._capture = AudioCapture(self.cfg)
        self._playback = AudioPlayback(self.cfg)

        print(f"\n  {self.persona.name} luistert... (Ctrl+C om te stoppen)\n")

        greeting = await self._get_opening()
        self.logger.log_agent_response(greeting, triggered_by="greeting")
        await self._speak(greeting)

        await self.events.emit_agent_listening()
        self.logger.log_state_change("listening")

        async for segment in self._capture.stream_speech_segments():
            self.brain.state.is_agent_speaking = False

            if self._playback.is_playing:
                self._playback.interrupt()
                self.logger.log_state_change("interrupted")

            await self.events.emit_user_speaking()

            duration_ms = int(len(segment) / self.cfg.sample_rate * 1000)
            t0 = time.time()

            try:
                text = await stt.transcribe(segment, self.cfg.sample_rate)
            except Exception:
                log.exception("STT error -- skipping segment")
                await self.events.emit_agent_listening()
                continue

            if not text.strip():
                await self.events.emit_agent_listening()
                continue

            log.info("user: %s", text)
            self.logger.log_user_utterance(text, duration_ms=duration_ms)

            await self.events.emit_agent_thinking()
            self.logger.log_state_change("thinking")

            try:
                reply = await self._process_input(text)
            except Exception:
                log.exception("LLM error -- using fallback")
                reply = "Oeps, ik was even kwijt wat ik wilde zeggen... Kun je dat nog een keer zeggen?"
                self.logger.log_state_change("llm_error")

            latency = int((time.time() - t0) * 1000)

            log.info("agent: %s", reply)
            self.logger.log_agent_response(reply, latency_ms=latency)

            await self._speak(reply)
            await self.events.emit_agent_listening()
            self.logger.log_state_change("listening")

            if self.scenario_runner and self.scenario_runner.is_finished:
                log.info("scenario '%s' completed", self.scenario_runner.scenario_id)
                break

    # ── Shared helpers ───────────────────────────────────────────────

    async def _get_opening(self) -> str:
        if self.scenario_runner:
            return await self.scenario_runner.start()
        return await self.brain.respond(
            "Een kind komt net de bibliotheek binnen en kijkt naar je. Begroet het kind."
        )

    async def _process_input(self, user_text: str) -> str:
        if self.scenario_runner and not self.scenario_runner.is_finished:
            return await self.scenario_runner.process_user_input(user_text)
        return await self.brain.respond(user_text)

    async def _speak(self, text: str) -> None:
        self.brain.state.is_agent_speaking = True
        await self.events.emit_agent_speaking(text)

        try:
            tts = self._get_tts()
            audio, sr = await tts.synthesize(text)
            if self._playback:
                await self._playback.play(audio, sr)
        except Exception:
            log.exception("TTS/playback error -- continuing without audio")
        finally:
            self.brain.state.is_agent_speaking = False

    # ── Spontaneity callback ─────────────────────────────────────────

    async def _handle_impulse(self, impulse: Impulse) -> None:
        log.info("spontaneous impulse: %s", impulse.impulse_type.value)

        silence_before = (
            time.time() - self.brain.state.last_agent_time
            if self.brain.state.last_agent_time else 0
        )

        text = await self.brain.generate_spontaneous(impulse.instruction)

        self.logger.log_spontaneous_impulse(
            impulse_type=impulse.impulse_type.value,
            text=text,
            silence_before_s=silence_before,
        )

        await self.events.emit_agent_spontaneous(impulse.impulse_type.value, text)

        if self.text_mode:
            print(f"  {self.persona.name}: *{impulse.impulse_type.value}* {text}\n")
        else:
            await self._speak(text)

    # ── Scenario callbacks ───────────────────────────────────────────

    async def _handle_action(self, action: str, scenario_id: str, phase_id: str) -> None:
        log.info("action trigger: %s (scenario=%s, phase=%s)", action, scenario_id, phase_id)
        self.logger.log_action_trigger(action, scenario_id, phase_id)
        await self.events.emit_action(action, scenario_id, phase_id)

    async def _handle_phase_change(
        self, scenario_id: str, from_phase: str, to_phase: str
    ) -> None:
        log.info("phase transition: %s -> %s (scenario=%s)", from_phase, to_phase, scenario_id)
        self.logger.log_phase_transition(scenario_id, from_phase, to_phase)
        await self.events.emit_phase_transition(scenario_id, from_phase, to_phase)

    async def _handle_profile_update(
        self, field: str, value: Any, phase: str
    ) -> None:
        log.info("child profile: %s = %s (phase=%s)", field, value, phase)
        self.logger.log_child_data_extracted(field, value, phase)


# ── CLI ──────────────────────────────────────────────────────────────


def _list_scenarios() -> list[str]:
    if SCENARIOS_DIR.exists():
        return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))
    return []


def cli() -> None:
    scenarios = _list_scenarios()
    scenario_help = f"Run a scenario ({', '.join(scenarios)})" if scenarios else "Run a scenario"

    parser = argparse.ArgumentParser(description="Holobot -- conversational AI agent")
    parser.add_argument("--text", action="store_true", help="Run in text mode (no audio hardware)")
    parser.add_argument("--scenario", type=str, metavar="NAME", help=scenario_help)
    parser.add_argument(
        "--level", type=int, choices=[0, 1, 2, 3],
        help="Override spontaneity level (0=off, 1=idle, 2=moderate, 3=frequent)",
    )
    parser.add_argument("--persona", type=str, help="Path to persona YAML file")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_scenarios:
        if scenarios:
            print("Available scenarios:")
            for s in scenarios:
                print(f"  - {s}")
        else:
            print("No scenarios found in scenarios/")
        sys.exit(0)

    cfg = get_settings()
    if args.level is not None:
        cfg.spontaneity_level = SpontaneityLevel(args.level)
    if args.persona:
        cfg.persona_path = Path(args.persona)

    bot = Holobot(text_mode=args.text, scenario_name=args.scenario)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nDoei!")
        sys.exit(0)
