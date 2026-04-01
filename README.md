# Holobot — Conversational AI Agent for the Holobox PoC

Voice-in/voice-out conversational agent for kids in the Tilburg public library.
Built as a standalone agent that integrates with the Holobox virtual human later.

## Key feature: Spontaneity Engine

The agent doesn't just respond — it *initiates*. A background engine fires
spontaneous utterances (questions, tangents, self-corrections, thinking aloud)
to study what this does to perceived believability in a virtual human context.

Spontaneity levels are configurable for A/B research:

| Level | Behaviour |
|-------|-----------|
| 0     | Control — only responds when spoken to |
| 1     | Idle only — speaks after prolonged silence |
| 2     | Moderate — occasional mid-conversation impulses |
| 3     | Frequent — human-like messy, chaotic but warm |

## Quick start

```bash
# Install
uv sync

# Copy and fill in API keys
cp .env.example .env

# Run (voice mode — needs mic + speakers)
uv run holobot

# Run (text mode — terminal chat, no audio hardware needed)
uv run holobot --text
```

## Architecture

```
Mic → VAD (silero) → STT (Whisper) → Brain (GPT-4o + persona) → TTS (ElevenLabs) → Speaker
                                         ↑
                                  Spontaneity Engine (background impulses)
                                         ↓
                                  WebSocket → Holobox (future)
```

## Persona

Edit `personas/default.yaml` to change the character. The default is **Bibi** —
a curious, slightly chaotic 10-year-old who lives in the library.

## Research logging

All events (user utterances, agent responses, spontaneous impulses) are logged
as structured JSONL in `logs/` for analysis.
