"""Spontaneity Engine — the core research component.

Background async loop that monitors conversation state and fires "impulses":
spontaneous utterances the agent generates without user input.

Configurable via SpontaneityLevel for A/B research on virtual human believability.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Awaitable

from holobot.brain.conversation import ConversationState
from holobot.config import Settings, SpontaneityLevel, get_settings

log = logging.getLogger(__name__)


class ImpulseType(str, Enum):
    IDLE_WONDER = "idle_wonder"
    CURIOUS_QUESTION = "curious_question"
    SELF_CORRECTION = "self_correction"
    TOPIC_TANGENT = "topic_tangent"
    THINKING_ALOUD = "thinking_aloud"
    PLAYFUL_TEASE = "playful_tease"


IMPULSE_INSTRUCTIONS: dict[ImpulseType, str] = {
    ImpulseType.IDLE_WONDER: (
        "Je was even stil en er schiet je ineens iets te binnen. "
        "Deel een gedachte of herinnering die vaag verband houdt met het gesprek."
    ),
    ImpulseType.CURIOUS_QUESTION: (
        "Je bent ineens heel nieuwsgierig naar iets over het kind. "
        "Stel een vraag die niks te maken heeft met wat jullie net bespraken."
    ),
    ImpulseType.SELF_CORRECTION: (
        "Je bedenkt dat je eerder iets zei dat niet helemaal klopte of raar was. "
        "Corrigeer jezelf of nuanceer wat je eerder zei."
    ),
    ImpulseType.TOPIC_TANGENT: (
        "Je gedachten dwalen af naar een gerelateerd onderwerp. "
        "Begin erover alsof het de normaalste zaak van de wereld is."
    ),
    ImpulseType.THINKING_ALOUD: (
        "Je denkt hardop na. Niet per se over iets belangrijks. "
        "Laat het kind even meekijken in je hoofd."
    ),
    ImpulseType.PLAYFUL_TEASE: (
        "Je wilt het kind op een speelse manier plagen of uitdagen. "
        "Houd het lief en grappig, nooit gemeen."
    ),
}

LEVEL_ALLOWED_TYPES: dict[SpontaneityLevel, list[ImpulseType]] = {
    SpontaneityLevel.OFF: [],
    SpontaneityLevel.IDLE_ONLY: [
        ImpulseType.IDLE_WONDER,
        ImpulseType.THINKING_ALOUD,
    ],
    SpontaneityLevel.MODERATE: [
        ImpulseType.IDLE_WONDER,
        ImpulseType.CURIOUS_QUESTION,
        ImpulseType.THINKING_ALOUD,
        ImpulseType.TOPIC_TANGENT,
    ],
    SpontaneityLevel.FREQUENT: list(ImpulseType),
}


@dataclass
class Impulse:
    """A spontaneous impulse ready for the conversation engine."""

    impulse_type: ImpulseType
    instruction: str
    timestamp: float


class SpontaneityEngine:
    """Background engine that monitors silence and conversation state,
    then fires impulses through a callback."""

    def __init__(
        self,
        state: ConversationState,
        on_impulse: Callable[[Impulse], Awaitable[None]],
        settings: Settings | None = None,
    ) -> None:
        self.cfg = settings or get_settings()
        self._state = state
        self._on_impulse = on_impulse
        self._running = False
        self._impulse_timestamps: list[float] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def allowed_types(self) -> list[ImpulseType]:
        return LEVEL_ALLOWED_TYPES.get(self.cfg.spontaneity_level, [])

    def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info(
            "spontaneity engine started (level=%s, check_interval=%.1fs)",
            self.cfg.spontaneity_level.name,
            self.cfg.impulse_check_interval_s,
        )

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.cfg.impulse_check_interval_s)
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("spontaneity tick error")

    async def _tick(self) -> None:
        if self.cfg.spontaneity_level == SpontaneityLevel.OFF:
            return

        if self._state.is_agent_speaking:
            return

        now = time.time()

        if self._is_rate_limited(now):
            return

        silence = now - self._state.last_agent_time if self._state.last_agent_time else 0

        # Idle impulses: fire after prolonged silence
        if silence >= self.cfg.min_silence_before_impulse:
            if random.random() < self.cfg.impulse_probability:
                idle_types = [
                    t for t in self.allowed_types
                    if t in (ImpulseType.IDLE_WONDER, ImpulseType.THINKING_ALOUD, ImpulseType.CURIOUS_QUESTION)
                ]
                if idle_types:
                    await self._fire(random.choice(idle_types), now)
                    return

        # Mid-conversation impulses (level >= MODERATE): occasional during active chat
        if self.cfg.spontaneity_level >= SpontaneityLevel.MODERATE:
            if self._state.turn_count >= 3 and silence >= 3.0:
                p = self.cfg.impulse_probability * 0.3  # lower probability for mid-conversation
                if random.random() < p:
                    mid_types = [
                        t for t in self.allowed_types
                        if t not in (ImpulseType.IDLE_WONDER, ImpulseType.THINKING_ALOUD)
                    ]
                    if mid_types:
                        await self._fire(random.choice(mid_types), now)

    async def _fire(self, impulse_type: ImpulseType, now: float) -> None:
        impulse = Impulse(
            impulse_type=impulse_type,
            instruction=IMPULSE_INSTRUCTIONS[impulse_type],
            timestamp=now,
        )
        self._impulse_timestamps.append(now)
        log.info("firing impulse: %s", impulse_type.value)
        await self._on_impulse(impulse)

    def _is_rate_limited(self, now: float) -> bool:
        cutoff = now - 60.0
        self._impulse_timestamps = [t for t in self._impulse_timestamps if t > cutoff]
        return len(self._impulse_timestamps) >= self.cfg.max_impulses_per_minute
