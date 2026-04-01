"""Conversation engine — manages LLM interaction, message history, and persona."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from holobot.brain.persona import Persona
from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)

MAX_HISTORY = 40  # keep last N messages to stay within context window


@dataclass
class ConversationState:
    """Observable state for the spontaneity engine and research logger."""

    turn_count: int = 0
    last_user_text: str = ""
    last_agent_text: str = ""
    last_user_time: float = 0.0
    last_agent_time: float = 0.0
    topics_mentioned: list[str] = field(default_factory=list)
    is_agent_speaking: bool = False


class ConversationEngine:
    """Stateful conversation manager wrapping an LLM with persona."""

    def __init__(
        self,
        persona: Persona,
        settings: Settings | None = None,
    ) -> None:
        self.cfg = settings or get_settings()
        self.persona = persona
        self._client = AsyncOpenAI(api_key=self.cfg.openai_api_key)
        self._base_system_prompt = persona.build_system_prompt()
        self._phase_instruction: str | None = None
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": self._base_system_prompt}
        ]
        self.state = ConversationState()

    def set_phase_instruction(self, instruction: str | None) -> None:
        """Overlay a scenario phase instruction onto the system prompt.

        This updates the system message in-place so the LLM sees both
        the persona and the current phase goal.
        """
        self._phase_instruction = instruction
        if instruction:
            combined = f"{self._base_system_prompt}\n\n{instruction}"
        else:
            combined = self._base_system_prompt
        self._messages[0] = {"role": "system", "content": combined}

    async def respond(self, user_text: str) -> str:
        """Generate a response to user input."""
        self._messages.append({"role": "user", "content": user_text})
        self._trim_history()

        self.state.last_user_text = user_text
        self.state.last_user_time = time.time()
        self.state.turn_count += 1

        response = await self._client.chat.completions.create(
            model=self.cfg.llm_model,
            messages=self._messages,
            max_tokens=150,
            temperature=0.9,
        )

        reply = response.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": reply})

        self.state.last_agent_text = reply
        self.state.last_agent_time = time.time()

        log.debug("brain response: %s", reply[:80])
        return reply

    async def generate_spontaneous(self, impulse_instruction: str) -> str:
        """Generate a spontaneous utterance given an impulse instruction.

        This injects a system-level nudge without adding a fake user message,
        preserving natural conversation flow.
        """
        messages = self._messages.copy()
        messages.append({
            "role": "system",
            "content": (
                f"[IMPULS — INTERNE GEDACHTE] {impulse_instruction}\n"
                "Reageer als jezelf. Kort. Alsof je hardop denkt of iets spontaan wilt zeggen."
            ),
        })

        response = await self._client.chat.completions.create(
            model=self.cfg.llm_model,
            messages=messages,
            max_tokens=80,
            temperature=1.0,
        )

        reply = response.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": reply})

        self.state.last_agent_text = reply
        self.state.last_agent_time = time.time()

        log.debug("spontaneous: %s", reply[:80])
        return reply

    def _trim_history(self) -> None:
        system = self._messages[:1]
        rest = self._messages[1:]
        if len(rest) > MAX_HISTORY:
            rest = rest[-MAX_HISTORY:]
        self._messages = system + rest
