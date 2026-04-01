"""Child profile — structured data extracted from conversation.

Tracks name, age, interests, and scenario-specific fields (e.g. guess).
Extraction uses the LLM with JSON mode after each user turn in relevant phases.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

from openai import AsyncOpenAI

from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Je analyseert wat een kind zei in een gesprek. Extraheer de volgende velden als je ze kunt vinden.
Geef ALLEEN velden terug die je met zekerheid kunt afleiden uit wat het kind zei.
Laat onbekende velden op null.

Velden:
- child_name: De naam van het kind (string of null)
- child_age: De leeftijd van het kind (integer of null)
- child_interests: Hobby's, activiteiten, favoriete boeken of onderwerpen (lijst van strings, of lege lijst)
- child_guess: Als het kind een keuze maakte (links/rechts), wat koos het? (string of null)

Reageer in JSON. Niets anders.
"""


@dataclass
class ChildProfile:
    """Per-session child data, progressively filled during conversation."""

    child_name: str | None = None
    child_age: int | None = None
    child_interests: list[str] = field(default_factory=list)
    child_guess: str | None = None

    def update(self, extracted: dict[str, Any]) -> list[str]:
        """Merge extracted fields. Returns list of newly set field names."""
        changed: list[str] = []

        name = extracted.get("child_name")
        if name and not self.child_name:
            self.child_name = name
            changed.append("child_name")

        age = extracted.get("child_age")
        if age is not None and self.child_age is None:
            self.child_age = int(age)
            changed.append("child_age")

        interests = extracted.get("child_interests") or []
        for interest in interests:
            if interest and interest not in self.child_interests:
                self.child_interests.append(interest)
                if "child_interests" not in changed:
                    changed.append("child_interests")

        guess = extracted.get("child_guess")
        if guess and not self.child_guess:
            self.child_guess = guess
            changed.append("child_guess")

        return changed

    def has(self, field_name: str) -> bool:
        val = getattr(self, field_name, None)
        if val is None:
            return False
        if isinstance(val, list):
            return len(val) > 0
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProfileExtractor:
    """Uses LLM JSON mode to extract structured child data from utterances."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._client = AsyncOpenAI(api_key=self.cfg.openai_api_key)

    async def extract(self, user_text: str, fields_needed: list[str]) -> dict[str, Any]:
        """Extract profile fields from a user utterance.

        Only runs if there are fields still needed — avoids wasting API calls.
        """
        if not fields_needed:
            return {}

        fields_hint = ", ".join(fields_needed)

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Focus op deze velden: {fields_hint}\n\n"
                            f"Het kind zei: \"{user_text}\""
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0.0,
            )

            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            log.debug("extracted profile data: %s", data)
            return data

        except Exception:
            log.exception("profile extraction failed")
            return {}
