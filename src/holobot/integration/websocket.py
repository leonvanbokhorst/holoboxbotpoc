"""WebSocket event server for Holobox integration.

Broadcasts agent state events so the Unreal Engine MetaHuman can
drive animations (lip sync, idle, thinking, etc.) in real time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from holobot.config import Settings, get_settings

log = logging.getLogger(__name__)


class HoloboxEventServer:
    """Lightweight WebSocket server that broadcasts state events."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._clients: set[ServerConnection] = set()
        self._server: Any = None

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self.cfg.ws_host,
            self.cfg.ws_port,
        )
        log.info("holobox event server listening on ws://%s:%d", self.cfg.ws_host, self.cfg.ws_port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        remote = ws.remote_address
        log.info("holobox client connected: %s", remote)
        try:
            async for msg in ws:
                log.debug("received from holobox client: %s", msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.info("holobox client disconnected: %s", remote)

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        if not self._clients:
            return

        payload = json.dumps({
            "event": event,
            "ts": time.time(),
            **(data or {}),
        })

        dead: list[ServerConnection] = []
        for ws in self._clients:
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                dead.append(ws)

        for ws in dead:
            self._clients.discard(ws)

    async def emit_agent_listening(self) -> None:
        await self.emit("agent_listening")

    async def emit_agent_thinking(self) -> None:
        await self.emit("agent_thinking")

    async def emit_agent_speaking(self, text: str) -> None:
        await self.emit("agent_speaking", {"text": text})

    async def emit_agent_spontaneous(self, impulse_type: str, text: str) -> None:
        await self.emit("agent_spontaneous", {"impulse_type": impulse_type, "text": text})

    async def emit_user_speaking(self) -> None:
        await self.emit("user_speaking")

    async def emit_user_silent(self) -> None:
        await self.emit("user_silent")

    # ── Action triggers (scenario-driven Holobox animations) ─────────

    async def emit_action(self, action: str, scenario_id: str, phase_id: str) -> None:
        await self.emit("action_trigger", {
            "action": action,
            "scenario_id": scenario_id,
            "phase_id": phase_id,
        })

    async def emit_phase_transition(
        self, scenario_id: str, from_phase: str, to_phase: str
    ) -> None:
        await self.emit("phase_transition", {
            "scenario_id": scenario_id,
            "from_phase": from_phase,
            "to_phase": to_phase,
        })
