"""FastAPI application factory for the local Position Pilot dashboard."""

from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__

STATIC_DIR = Path(__file__).with_name("static")


@dataclass(frozen=True, slots=True)
class WebSettings:
    """Process-local security and hosting settings for the web dashboard."""

    launch_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    session_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    enforce_loopback: bool = True


class SessionExchangeRequest(BaseModel):
    """One-time launch credential supplied by the locally opened browser."""

    launch_token: str


def _configured(*environment_keys: str) -> str:
    return (
        "configured"
        if all(bool(os.getenv(environment_key)) for environment_key in environment_keys)
        else "not_configured"
    )


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Create an isolated local dashboard application."""

    load_dotenv(Path.cwd() / ".env", override=False)
    active_settings = settings or WebSettings()
    app = FastAPI(
        title="Position Pilot",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.web_settings = active_settings
    app.state.launch_token_available = True
    app.state.launch_token_lock = asyncio.Lock()
    assets_directory = STATIC_DIR / "assets"
    if assets_directory.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_directory), name="assets")

    @app.middleware("http")
    async def require_loopback_client(request, call_next):
        client_host = request.client.host if request.client else ""
        request_host = request.url.hostname or ""
        is_loopback = False
        try:
            is_loopback = ip_address(client_host).is_loopback
        except ValueError:
            is_loopback = client_host == "localhost"

        if active_settings.enforce_loopback and not is_loopback:
            return JSONResponse(
                {"detail": "Position Pilot accepts local connections only."},
                status_code=status.HTTP_403_FORBIDDEN,
                headers={"Cache-Control": "no-store"},
            )
        if active_settings.enforce_loopback and request_host not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            return JSONResponse(
                {"detail": "Invalid local dashboard host."},
                status_code=status.HTTP_400_BAD_REQUEST,
                headers={"Cache-Control": "no-store"},
            )
        return await call_next(request)

    @app.middleware("http")
    async def add_private_cache_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/api/v1/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "position-pilot",
                "version": __version__,
            }
        )

    @app.post("/api/v1/session/exchange", status_code=status.HTTP_204_NO_CONTENT)
    async def exchange_session(payload: SessionExchangeRequest) -> Response:
        async with app.state.launch_token_lock:
            if not app.state.launch_token_available or not secrets.compare_digest(
                payload.launch_token,
                active_settings.launch_token,
            ):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
            app.state.launch_token_available = False

        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.set_cookie(
            "position_pilot_session",
            active_settings.session_token,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/",
        )
        return response

    @app.get("/api/v1/bootstrap")
    async def bootstrap(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> JSONResponse:
        if session is None or not secrets.compare_digest(
            session,
            active_settings.session_token,
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return JSONResponse(
            {
                "application": {
                    "name": "Position Pilot",
                    "version": __version__,
                    "phase": "web-foundation",
                },
                "providers": {
                    "tastytrade": _configured(
                        "TASTYTRADE_CLIENT_SECRET",
                        "TASTYTRADE_REFRESH_TOKEN",
                    ),
                    "codex": "not_checked",
                    "massive": _configured("MASSIVE_API_KEY"),
                    "benzinga": _configured("BENZINGA_API_KEY"),
                },
                "monitoring": {
                    "market_timezone": "America/New_York",
                    "window_start": "07:30",
                    "window_end": "18:00",
                    "evaluation_minutes": 30,
                    "risk_refresh_seconds": 60,
                },
                "navigation": [
                    "Overview",
                    "Positions",
                    "Roll analytics",
                    "Markets",
                    "Alerts",
                    "Settings",
                ],
                "data_state": "awaiting_portfolio_snapshot",
                "server_time": datetime.now(UTC).isoformat(),
            }
        )

    @app.get("/{path:path}", include_in_schema=False)
    async def dashboard_shell(path: str) -> Response:
        if path.startswith("api/"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        index_file = STATIC_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return HTMLResponse(
            "<h1>Position Pilot frontend is not built</h1>"
            "<p>Run <code>pnpm run build</code> in the frontend directory.</p>",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return app
