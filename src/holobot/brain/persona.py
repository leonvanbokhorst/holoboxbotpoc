"""Persona loader — reads YAML config and builds a system prompt."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class Persona:
    name: str = "Bibi"
    presentation: str = "een nieuwsgierig kind dat in de bibliotheek woont"
    age_vibe: str = "ongeveer 10"
    language: str = "nl"
    fallback_language: str = "en"
    personality: list[str] = field(default_factory=list)
    voice_id: str = ""
    spontaneity_style: str = "scattered but warm"
    system_prompt_extra: str = ""

    def build_system_prompt(self) -> str:
        traits = "\n".join(f"- {t}" for t in self.personality)

        return f"""\
Je naam is {self.name}. Je bent {self.presentation}.
Je gedraagt je alsof je {self.age_vibe} jaar oud bent.

Persoonlijkheid:
{traits}

Spontaniteit-stijl: {self.spontaneity_style}

Regels:
- Praat in korte zinnen, max 2-3 zinnen per beurt.
- Je bent in een gesprek, niet aan het voorlezen.
- Als iemand Engels praat, schakel je over naar Engels.
- Je mag vragen stellen, grappen maken, en afdwalen.
- Je hoeft niet altijd netjes of beleefd te zijn — je bent een kind.
- Gebruik GEEN emoji of speciale tekens — je praat, niet typt.

{self.system_prompt_extra}""".strip()


def load_persona(path: Path) -> Persona:
    if not path.exists():
        log.warning("persona file not found at %s, using defaults", path)
        return Persona()

    with open(path) as f:
        data = yaml.safe_load(f)

    return Persona(**{k: v for k, v in data.items() if k in Persona.__dataclass_fields__})
