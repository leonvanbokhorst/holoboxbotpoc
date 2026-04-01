# Holobot -- Conversational AI Agent for the Holobox PoC

Voice-in/voice-out conversational agent for kids in the Tilburg public library.
Built as a standalone agent that integrates with the Holobox virtual human later.

## Two layers of agency

**Spontaneity Engine** -- the agent initiates without being asked (questions,
tangents, self-corrections, thinking aloud). Configurable levels for A/B research
on virtual human believability.

**Scenario System** -- structured interaction sequences from Meike's research
designs. Phase-guided conversation (greeting, game, memory test, farewell) while
spontaneity still fires within phases.

## Quick start

```bash
# Install
uv sync

# Copy and fill in API keys
cp .env.example .env

# Free chat (text mode, no audio hardware needed)
uv run holobot --text

# Free chat (voice mode, needs mic + speakers)
uv run holobot

# Run a scenario
uv run holobot --text --scenario surprise_game
uv run holobot --text --scenario memory_test
uv run holobot --text --scenario avatar_interests

# Override spontaneity level
uv run holobot --text --scenario surprise_game --level 3

# List available scenarios
uv run holobot --list-scenarios
```

## Scenarios (Meike's interaction designs)

| Scenario | Description | Phases |
|----------|-------------|--------|
| `surprise_game` | Guess-the-hand game with avatar reward | greeting, tease, game, reveal, farewell |
| `avatar_interests` | Interest-driven personalized avatar | greeting, age, interests, creation, reveal, farewell |
| `memory_test` | Conversational memory and attention test | greeting, conversation + plants, name test, action test, farewell |
| `free_chat` | Open-ended conversation (control condition) | single open phase |

Scenarios are YAML files in `scenarios/`. Each defines phases with goals,
LLM instructions, data extraction targets, memory plants/tests, and
Holobox animation triggers.

## Spontaneity levels

| Level | Behaviour |
|-------|-----------|
| 0 | Control -- only responds when spoken to |
| 1 | Idle only -- speaks after prolonged silence |
| 2 | Moderate -- occasional mid-conversation impulses (default) |
| 3 | Frequent -- human-like messy, chaotic but warm |

## Architecture

```
Mic -> VAD (silero) -> STT (Whisper) -> Scenario Runner -> Brain (GPT-4o) -> TTS (ElevenLabs) -> Speaker
                                             |                  ^
                                        Child Profile     Spontaneity Engine
                                             |                  |
                                        Action Triggers    Background impulses
                                             |
                                        WebSocket -> Holobox / UE MetaHuman
```

## Holobox integration (WebSocket events)

State events: `agent_listening`, `agent_thinking`, `agent_speaking`, `agent_spontaneous`,
`user_speaking`, `user_silent`

Action triggers: `present_hands`, `reveal_surprise`, `take_pen`, `put_pen_back`,
`show_avatar_card`, `wave_goodbye`

Phase transitions: `phase_transition` with scenario_id, from_phase, to_phase

Connect to `ws://localhost:8765` to receive all events as JSON.

## Research logging

All events logged as structured JSONL in `logs/`:
- User utterances, agent responses, spontaneous impulses
- Phase transitions, child data extraction, memory plant/test results
- Action triggers with scenario and phase context
