"""Scenario runner — phase-based state machine that guides conversation flow.

Sits between user input and the conversation brain. Injects phase-specific
instructions into the LLM prompt, tracks phase goals, handles transitions,
and fires Holobox action triggers at the right moments.

Spontaneity still fires within phases — the runner doesn't suppress it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Awaitable

import yaml

from holobot.brain.conversation import ConversationEngine
from holobot.config import Settings, get_settings
from holobot.scenario.child_profile import ChildProfile, ProfileExtractor
from holobot.scenario.phase import MemoryPlant, Phase, ScenarioDefinition

log = logging.getLogger(__name__)


def load_scenario(path: Path) -> ScenarioDefinition:
    with open(path) as f:
        data = yaml.safe_load(f)

    phases = [Phase.from_dict(p) for p in data.get("phases", [])]

    return ScenarioDefinition(
        id=data.get("id", path.stem),
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        phases=phases,
        metadata=data.get("metadata", {}),
    )


class ScenarioRunner:
    """Drives a conversation through scenario phases.

    Callbacks:
      on_action -- called when a phase transition triggers a Holobox animation
      on_phase_change -- called on every phase transition (for logging)
      on_profile_update -- called when child profile data is extracted
    """

    def __init__(
        self,
        scenario: ScenarioDefinition,
        brain: ConversationEngine,
        profile: ChildProfile,
        settings: Settings | None = None,
        on_action: Callable[[str, str, str], Awaitable[None]] | None = None,
        on_phase_change: Callable[[str, str, str], Awaitable[None]] | None = None,
        on_profile_update: Callable[[str, Any, str], Awaitable[None]] | None = None,
    ) -> None:
        self.cfg = settings or get_settings()
        self.scenario = scenario
        self.brain = brain
        self.profile = profile
        self._extractor = ProfileExtractor(self.cfg)

        self._on_action = on_action
        self._on_phase_change = on_phase_change
        self._on_profile_update = on_profile_update

        self._current_phase: Phase | None = None
        self._phase_turn_count: int = 0
        self._planted_memories: dict[str, str] = {}
        self._finished = False

    @property
    def current_phase(self) -> Phase | None:
        return self._current_phase

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def scenario_id(self) -> str:
        return self.scenario.id

    async def start(self) -> str:
        """Start the scenario. Returns the opening agent utterance."""
        first = self.scenario.first_phase
        if not first:
            self._finished = True
            return ""

        await self._enter_phase(first)

        opening_instruction = (
            "Een kind komt net de bibliotheek binnen en kijkt naar je. "
            f"[FASE: {first.id}] {first.instruction}"
        )
        reply = await self.brain.respond(opening_instruction)
        self._phase_turn_count += 1
        return reply

    async def process_user_input(self, user_text: str) -> str:
        """Process user input within the current scenario phase.

        Returns the agent's response. Handles extraction, phase goal checking,
        and transitions automatically.
        """
        if self._finished or not self._current_phase:
            return await self.brain.respond(user_text)

        phase = self._current_phase

        # Extract profile data if this phase needs it
        if phase.extract:
            fields_needed = [f for f in phase.extract if not self.profile.has(f)]
            if fields_needed:
                extracted = await self._extractor.extract(user_text, fields_needed)
                changed = self.profile.update(extracted)
                for field_name in changed:
                    val = getattr(self.profile, field_name)
                    log.info("extracted %s = %s in phase %s", field_name, val, phase.id)
                    if self._on_profile_update:
                        await self._on_profile_update(field_name, val, phase.id)

        # Handle memory tests in this phase
        for memory_key in phase.memory_tests:
            if memory_key in self._planted_memories:
                pass  # The test happens naturally via the phase instruction

        # Build phase-aware prompt
        phase_context = self._build_phase_context(phase, user_text)
        reply = await self.brain.respond(phase_context)
        self._phase_turn_count += 1

        # Check if phase goal is met or turn limit reached
        if self._should_advance(phase):
            await self._advance()

        return reply

    def _build_phase_context(self, phase: Phase, user_text: str) -> str:
        """Wrap user text with phase-specific context for the LLM."""
        parts = [user_text]

        # Inject phase instruction as a subtle system nudge
        hint = f"\n[INTERNE NOTITIE — niet hardop zeggen: fase={phase.id}, doel={phase.goal}]"

        # If there are memory plants for this phase, remind the VH to plant them
        for plant in phase.memory_plants:
            if plant.key not in self._planted_memories:
                hint += f"\n[PLANT: {plant.description}]"

        # If there are memory tests, add the test prompt
        for memory_key in phase.memory_tests:
            if memory_key in self._planted_memories:
                plant_desc = self._planted_memories[memory_key]
                hint += f"\n[TEST HERINNERING: vraag of het kind zich herinnert: {plant_desc}]"

        # Add profile context so the VH can use the kid's name
        if self.profile.child_name:
            hint += f"\n[Het kind heet {self.profile.child_name}]"

        return parts[0] + hint

    def _should_advance(self, phase: Phase) -> bool:
        """Determine if we should move to the next phase."""
        # All required fields extracted
        if phase.extract:
            all_extracted = all(self.profile.has(f) for f in phase.extract)
            if all_extracted:
                log.info("phase %s: all fields extracted, advancing", phase.id)
                return True

        # Turn limit reached
        if self._phase_turn_count >= phase.max_turns:
            log.info("phase %s: turn limit reached (%d), advancing", phase.id, phase.max_turns)
            return True

        # Memory plants all planted (mark them as planted after agent speaks)
        for plant in phase.memory_plants:
            if plant.key not in self._planted_memories:
                self._planted_memories[plant.key] = plant.description

        return False

    async def _advance(self) -> None:
        """Transition to the next phase."""
        phase = self._current_phase
        if not phase:
            return

        if phase.on_complete_action and self._on_action:
            await self._on_action(
                phase.on_complete_action,
                self.scenario.id,
                phase.id,
            )

        next_id = phase.next_phase
        if not next_id:
            self._finished = True
            log.info("scenario %s finished after phase %s", self.scenario.id, phase.id)
            return

        next_phase = self.scenario.get_phase(next_id)
        if not next_phase:
            log.error("phase %s references unknown next phase %s", phase.id, next_id)
            self._finished = True
            return

        old_id = phase.id
        await self._enter_phase(next_phase)

        if self._on_phase_change:
            await self._on_phase_change(self.scenario.id, old_id, next_phase.id)

    async def _enter_phase(self, phase: Phase) -> None:
        """Enter a new phase — update system prompt overlay."""
        self._current_phase = phase
        self._phase_turn_count = 0

        self.brain.set_phase_instruction(
            f"[SCENARIO FASE: {phase.id}]\n"
            f"Doel: {phase.goal}\n"
            f"Instructie: {phase.instruction}"
        )

        log.info("entered phase: %s (goal: %s)", phase.id, phase.goal)
