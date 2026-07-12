"""Local process launcher for the Position Pilot web dashboard."""

from __future__ import annotations

import socket
import threading
import webbrowser
from urllib.parse import urlencode

import uvicorn

from .app import WebSettings, create_app


def _loopback_socket(host: str, port: int) -> socket.socket:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(2048)
    return server_socket


def run_web_dashboard(*, open_browser: bool = True) -> None:
    """Run the local dashboard until the user stops the foreground process."""

    host = "127.0.0.1"
    settings = WebSettings()
    app = create_app(settings)
    server_socket = _loopback_socket(host, 0)
    port = server_socket.getsockname()[1]
    query = urlencode({"launch_token": settings.launch_token})
    dashboard_url = f"http://{host}:{port}/?{query}"

    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(dashboard_url,)).start()
    else:
        print(f"Position Pilot dashboard: {dashboard_url}")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    uvicorn.Server(config).run(sockets=[server_socket])
