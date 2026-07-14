"""Tests for the local dashboard process launcher."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from position_pilot.web.launcher import (
    DEFAULT_DASHBOARD_PORT,
    GRACEFUL_SHUTDOWN_SECONDS,
    LOOPBACK_HOST,
    _DashboardServer,
    run_web_dashboard,
)


def test_run_web_dashboard_defaults_to_fixed_loopback_port(monkeypatch) -> None:
    recorded: dict[str, object] = {}
    bind_calls: list[tuple[str, int]] = []

    class RecordingServer:
        def __init__(self, config: object) -> None:
            recorded["config"] = config

        def run(self, sockets: list | None = None) -> None:
            recorded["sockets"] = sockets

    fake_socket = MagicMock()
    fake_socket.getsockname.return_value = (LOOPBACK_HOST, DEFAULT_DASHBOARD_PORT)

    def fake_loopback(host: str, port: int) -> MagicMock:
        bind_calls.append((host, port))
        return fake_socket

    monkeypatch.setattr("position_pilot.web.launcher._DashboardServer", RecordingServer)
    monkeypatch.setattr("position_pilot.web.launcher.create_app", lambda settings: object())
    monkeypatch.setattr("position_pilot.web.launcher._loopback_socket", fake_loopback)

    run_web_dashboard(open_browser=False)

    assert bind_calls == [(LOOPBACK_HOST, DEFAULT_DASHBOARD_PORT)]
    config = recorded["config"]
    assert getattr(config, "host") == LOOPBACK_HOST
    assert getattr(config, "port") == DEFAULT_DASHBOARD_PORT
    assert getattr(config, "timeout_graceful_shutdown") == GRACEFUL_SHUTDOWN_SECONDS
    assert recorded["sockets"] == [fake_socket]


def test_run_web_dashboard_uses_custom_port_in_bind_url_and_config(monkeypatch, capsys) -> None:
    recorded: dict[str, object] = {}
    bind_calls: list[tuple[str, int]] = []
    custom_port = 9123

    class RecordingServer:
        def __init__(self, config: object) -> None:
            recorded["config"] = config

        def run(self, sockets: list | None = None) -> None:
            recorded["sockets"] = sockets

    fake_socket = MagicMock()
    fake_socket.getsockname.return_value = (LOOPBACK_HOST, custom_port)

    def fake_loopback(host: str, port: int) -> MagicMock:
        bind_calls.append((host, port))
        return fake_socket

    monkeypatch.setattr("position_pilot.web.launcher._DashboardServer", RecordingServer)
    monkeypatch.setattr("position_pilot.web.launcher.create_app", lambda settings: object())
    monkeypatch.setattr("position_pilot.web.launcher._loopback_socket", fake_loopback)

    run_web_dashboard(open_browser=False, port=custom_port)

    assert bind_calls == [(LOOPBACK_HOST, custom_port)]
    config = recorded["config"]
    assert getattr(config, "host") == LOOPBACK_HOST
    assert getattr(config, "port") == custom_port
    printed = capsys.readouterr().out
    assert f"http://{LOOPBACK_HOST}:{custom_port}/?" in printed
    assert "launch_token=" in printed


def test_run_web_dashboard_fails_clearly_when_port_unavailable(monkeypatch) -> None:
    def raise_address_in_use(host: str, port: int) -> socket.socket:
        raise OSError(48, f"Address already in use ({host}:{port})")

    monkeypatch.setattr("position_pilot.web.launcher.create_app", lambda settings: object())
    monkeypatch.setattr("position_pilot.web.launcher._loopback_socket", raise_address_in_use)

    with pytest.raises(RuntimeError, match=r"Could not bind dashboard to 127\.0\.0\.1:8765") as exc:
        run_web_dashboard(open_browser=False, port=DEFAULT_DASHBOARD_PORT)

    assert "port" in str(exc.value).lower()
    assert isinstance(exc.value.__cause__, OSError)


def test_dashboard_server_closes_open_transports_before_parent_shutdown(monkeypatch) -> None:
    closed: list[bool] = []
    parent_called = {"value": False}

    class FakeTransport:
        def close(self) -> None:
            closed.append(True)

    async def fake_super_shutdown(self, sockets=None) -> None:  # noqa: ANN001
        parent_called["value"] = True

    monkeypatch.setattr(uvicorn.Server, "shutdown", fake_super_shutdown)

    async def scenario() -> None:
        server = _DashboardServer.__new__(_DashboardServer)
        server.server_state = SimpleNamespace(
            connections=[SimpleNamespace(transport=FakeTransport())]
        )
        await server.shutdown(sockets=None)

    asyncio.run(scenario())

    assert closed == [True]
    assert parent_called["value"] is True


def test_dashboard_server_exits_promptly_with_open_sse_stream() -> None:
    """Ctrl+C must not hang forever on infinite EventSource responses."""

    app = FastAPI()

    @app.get("/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream():
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(1.0)
                yield ": keepalive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(64)
    host, port = server_socket.getsockname()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )
    server = _DashboardServer(config)
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [server_socket]},
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 5
    while time.time() < deadline and not server.started:
        time.sleep(0.05)
    assert server.started, "server failed to start"

    sse_error: list[str] = []

    def hold_sse() -> None:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/events", timeout=30) as response:
                while response.read(64):
                    pass
        except Exception as error:  # noqa: BLE001 — connection drop is expected
            sse_error.append(type(error).__name__)

    client = threading.Thread(target=hold_sse, daemon=True)
    client.start()
    time.sleep(0.4)

    t0 = time.time()
    server.should_exit = True
    thread.join(timeout=GRACEFUL_SHUTDOWN_SECONDS + 3)
    elapsed = time.time() - t0

    assert not thread.is_alive(), f"server thread still running after {elapsed:.2f}s"
    assert elapsed < GRACEFUL_SHUTDOWN_SECONDS + 2
    assert sse_error  # stream ended because the transport was closed
