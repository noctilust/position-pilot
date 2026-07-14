"""Local process launcher for the Position Pilot web dashboard."""

from __future__ import annotations

import socket
import threading
import webbrowser
from urllib.parse import urlencode

import uvicorn

from .app import WebSettings, create_app

# Bound graceful exit so Ctrl+C cannot hang forever on long-lived SSE streams.
GRACEFUL_SHUTDOWN_SECONDS = 5

# Fixed default loopback port; override via CLI --port. Never fall back to random.
DEFAULT_DASHBOARD_PORT = 8765
LOOPBACK_HOST = "127.0.0.1"


class _DashboardServer(uvicorn.Server):
    """Uvicorn server that can exit cleanly on Ctrl+C.

    Browser EventSource connections never complete a response. Default uvicorn
    graceful shutdown only clears keep-alive and then waits in
    ``asyncio.Server.wait_closed()`` until every connection drops — which never
    happens for infinite SSE. Closing transports drops those streams so the
    wait can finish; ``timeout_graceful_shutdown`` is a safety net.
    """

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        for connection in list(self.server_state.connections):
            transport = getattr(connection, "transport", None)
            if transport is not None:
                transport.close()
        await super().shutdown(sockets=sockets)


def _loopback_socket(host: str, port: int) -> socket.socket:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(2048)
    return server_socket


def run_web_dashboard(
    *,
    open_browser: bool = True,
    port: int = DEFAULT_DASHBOARD_PORT,
) -> None:
    """Run the local dashboard until the user stops the foreground process.

    Binds only to IPv4 loopback. Uses a fixed port (default 8765); if that port
    is unavailable the launch fails clearly rather than choosing a random port.
    """

    host = LOOPBACK_HOST
    settings = WebSettings()
    app = create_app(settings)
    try:
        server_socket = _loopback_socket(host, port)
    except OSError as exc:
        raise RuntimeError(
            f"Could not bind dashboard to {host}:{port}. "
            "The port may already be in use; free it or pass a different --port. "
            f"({exc})"
        ) from exc
    bound_port = server_socket.getsockname()[1]
    query = urlencode({"launch_token": settings.launch_token})
    dashboard_url = f"http://{host}:{bound_port}/?{query}"

    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(dashboard_url,)).start()
    else:
        print(f"Position Pilot dashboard: {dashboard_url}")

    config = uvicorn.Config(
        app,
        host=host,
        port=bound_port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )
    _DashboardServer(config).run(sockets=[server_socket])
