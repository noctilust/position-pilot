"""Async websocket transports for DXLink and Tastytrade account events."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from websockets.asyncio.client import connect

from .account import AccountStreamerProtocol
from .dxlink import DxLinkProtocol

EventCallback = Callable[[Any], None | Awaitable[None]]
HANDSHAKE_TIMEOUT_SECONDS = 15
KEEPALIVE_INTERVAL_SECONDS = 25
STALE_TIMEOUT_SECONDS = 90


class StreamStaleError(ConnectionError):
    """No inbound protocol message arrived inside the liveness window."""


async def _dispatch(callback: EventCallback, event: Any) -> None:
    result = callback(event)
    if inspect.isawaitable(result):
        await result


async def _periodic_send(
    websocket: Any,
    message_factory: Callable[[], dict],
    stop: asyncio.Event,
    *,
    interval: float = KEEPALIVE_INTERVAL_SECONDS,
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            await websocket.send(json.dumps(message_factory()))


async def _receive(websocket: Any, *, timeout: float, stale: bool = False) -> Any:
    try:
        return await asyncio.wait_for(websocket.recv(), timeout=timeout)
    except TimeoutError as error:
        exception = (
            StreamStaleError("stream stopped sending liveness messages")
            if stale
            else ConnectionError("stream handshake timed out")
        )
        raise exception from error


async def _dxlink_handshake(
    websocket: Any,
    *,
    token: str,
    symbols: list[str],
    timeout: float = HANDSHAKE_TIMEOUT_SECONDS,
    on_message: Callable[[], None] | None = None,
) -> None:
    try:
        async with asyncio.timeout(timeout):
            await websocket.send(json.dumps(DxLinkProtocol.setup()))
            while True:
                message = json.loads(await websocket.recv())
                if on_message:
                    on_message()
                if message.get("type") == "AUTH_STATE" and message.get("state") == "UNAUTHORIZED":
                    await websocket.send(json.dumps(DxLinkProtocol.authorize(token)))
                elif message.get("type") == "AUTH_STATE" and message.get("state") == "AUTHORIZED":
                    await websocket.send(json.dumps(DxLinkProtocol.channel_request()))
                elif message.get("type") == "CHANNEL_OPENED":
                    await websocket.send(json.dumps(DxLinkProtocol.feed_setup()))
                elif message.get("type") == "FEED_CONFIG":
                    await websocket.send(json.dumps(DxLinkProtocol.subscription(symbols)))
                    return
    except TimeoutError as error:
        raise ConnectionError("stream handshake timed out") from error


class DxLinkClient:
    """One authenticated DXLink connection with explicit protocol ordering."""

    async def run(
        self,
        *,
        url: str,
        token: str,
        symbols: list[str],
        on_event: EventCallback,
        stop: asyncio.Event,
        on_connected: Callable[[], None] | None = None,
        on_message: Callable[[], None] | None = None,
        subscription_updates: asyncio.Queue[list[str]] | None = None,
    ) -> None:
        async with connect(url, ping_interval=None) as websocket:
            await _dxlink_handshake(
                websocket,
                token=token,
                symbols=symbols,
                on_message=on_message,
            )
            if on_connected:
                on_connected()

            keepalive = asyncio.create_task(
                _periodic_send(websocket, DxLinkProtocol.keepalive, stop)
            )
            receive_task = asyncio.create_task(
                _receive(websocket, timeout=STALE_TIMEOUT_SECONDS, stale=True)
            )
            update_task: asyncio.Task[list[str]] | None = None
            try:
                while not stop.is_set():
                    wait_for = {receive_task}
                    if subscription_updates is not None:
                        update_task = asyncio.create_task(subscription_updates.get())
                        wait_for.add(update_task)
                    completed, _ = await asyncio.wait(
                        wait_for,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if update_task is not None and update_task in completed:
                        await websocket.send(
                            json.dumps(DxLinkProtocol.subscription(update_task.result()))
                        )
                        update_task = None
                        if receive_task not in completed:
                            continue
                    elif update_task is not None:
                        update_task.cancel()
                        await asyncio.gather(update_task, return_exceptions=True)
                        update_task = None

                    raw = receive_task.result()
                    if on_message:
                        on_message()
                    message = json.loads(raw)
                    for event in DxLinkProtocol.decode(message):
                        await _dispatch(on_event, event)
                    receive_task = asyncio.create_task(
                        _receive(websocket, timeout=STALE_TIMEOUT_SECONDS, stale=True)
                    )
            finally:
                keepalive.cancel()
                receive_task.cancel()
                if update_task is not None:
                    update_task.cancel()
                await asyncio.gather(
                    keepalive,
                    receive_task,
                    *(tuple([update_task]) if update_task is not None else ()),
                    return_exceptions=True,
                )


class AccountStreamerClient:
    """Production account-streamer transport for full-object notifications."""

    async def run(
        self,
        *,
        url: str,
        access_token: str,
        account_numbers: list[str],
        on_event: EventCallback,
        stop: asyncio.Event,
        on_connected: Callable[[], None] | None = None,
        on_message: Callable[[], None] | None = None,
    ) -> None:
        async with connect(url, ping_interval=None) as websocket:
            await websocket.send(
                json.dumps(AccountStreamerProtocol.connect(account_numbers, access_token))
            )
            response = json.loads(await _receive(websocket, timeout=HANDSHAKE_TIMEOUT_SECONDS))
            if on_message:
                on_message()
            AccountStreamerProtocol.validate_connect_response(response)
            if on_connected:
                on_connected()
            heartbeat = asyncio.create_task(
                _periodic_send(
                    websocket,
                    lambda: AccountStreamerProtocol.heartbeat(access_token),
                    stop,
                )
            )
            try:
                while not stop.is_set():
                    raw = await _receive(
                        websocket,
                        timeout=STALE_TIMEOUT_SECONDS,
                        stale=True,
                    )
                    if on_message:
                        on_message()
                    event = AccountStreamerProtocol.decode(json.loads(raw))
                    if event is not None:
                        await _dispatch(on_event, event)
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)


class StreamingSupervisor:
    """Reconnect streams with bounded backoff and observable lifecycle hooks."""

    def __init__(
        self,
        *,
        name: str,
        status: dict[str, dict[str, str | None]],
        on_reconnect: Callable[[], None],
    ) -> None:
        self.name = name
        self.status = status
        self.on_reconnect = on_reconnect
        self._ever_connected = False
        self._delay = 1.0

    def connected(self) -> None:
        """Mark a completed handshake and reconcile only true reconnections."""

        if self._ever_connected:
            self.on_reconnect()
        self._ever_connected = True
        self._delay = 1.0
        self.status[self.name] = {
            "state": "live",
            "error": None,
            "last_message_at": datetime.now(UTC).isoformat(),
        }

    def activity(self) -> None:
        current = self.status.get(self.name, {})
        self.status[self.name] = {
            **current,
            "last_message_at": datetime.now(UTC).isoformat(),
        }

    async def run_forever(
        self,
        runner: Callable[[], Awaitable[None]],
        stop: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                self.status[self.name] = {"state": "connecting", "error": None}
                await runner()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.status[self.name] = {
                    "state": "stale" if isinstance(error, StreamStaleError) else "degraded",
                    "error": type(error).__name__,
                    "last_message_at": self.status.get(self.name, {}).get("last_message_at"),
                }
                await asyncio.sleep(self._delay)
                self._delay = min(self._delay * 2, 30.0)
