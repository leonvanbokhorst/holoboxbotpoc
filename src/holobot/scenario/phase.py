"""Phase definitions — the building blocks of a scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryPlant:
    """Information the VH deliberately introduces for later recall testing."""

    key: str  # e.g. "vh_name", "pen_action"
    description: str  # what to plant, e.g. "Noem je naam duidelijk"
    test_prompt: str  # how to test recall, e.g. "Weet je nog hoe ik heet?"


@dataclass
class Phase:
    """A single phase in a scenario's conversation flow."""

    id: str
    goal: str
    instruction: str  # injected into LLM system prompt during this phase
    extract: list[str] = field(default_factory=list)  # profile fields to extract
    max_turns: int = 5
    next_phase: str | None = None
    on_complete_action: str | None = None  # Holobox animation trigger
    memory_plants: list[MemoryPlant] = field(default_factory=list)
    memory_tests: list[str] = field(default_factory=list)  # keys from earlier plants to test

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Phase:
        plants = [
            MemoryPlant(**p) for p in data.get("memory_plants", [])
        ]
        return Phase(
            id=data["id"],
            goal=data.get("goal", ""),
            instruction=data.get("instruction", ""),
            extract=data.get("extract", []),
            max_turns=data.get("max_turns", 5),
            next_phase=data.get("on_complete", {}).get("next") if isinstance(data.get("on_complete"), dict) else data.get("next"),
            on_complete_action=data.get("on_complete", {}).get("action") if isinstance(data.get("on_complete"), dict) else data.get("action"),
            memory_plants=plants,
            memory_tests=data.get("memory_tests", []),
        )


@dataclass
class ScenarioDefinition:
    """A complete scenario loaded from YAML."""

    id: str
    name: str
    description: str
    phases: list[Phase]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def first_phase(self) -> Phase | None:
        return self.phases[0] if self.phases else None

    def get_phase(self, phase_id: str) -> Phase | None:
        for p in self.phases:
            if p.id == phase_id:
                return p
        return None
