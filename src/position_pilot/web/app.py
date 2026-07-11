"""FastAPI application factory for the local Position Pilot dashboard."""

from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Protocol

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..domain.factory import get_market_service, get_portfolio_service, get_roll_service
from ..domain.market import MarketSnapshot
from ..domain.portfolio import PortfolioService
from ..domain.rolls import RollChainSnapshot
from ..domain.snapshots import PortfolioSnapshot, PositionHorizon, StrategySnapshot

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


class PrimaryAccountRequest(BaseModel):
    account_id: str


class StrategyHorizonRequest(BaseModel):
    horizon: PositionHorizon


class PortfolioReader(Protocol):
    """Web-facing portfolio service contract."""

    def latest(self, account_id: str = "all") -> PortfolioSnapshot | None: ...

    def refresh(self) -> PortfolioSnapshot: ...

    def primary_account_id(self) -> str: ...

    def set_primary_account(self, account_id: str) -> None: ...

    def set_strategy_horizon(
        self,
        strategy_id: str,
        horizon: PositionHorizon,
    ) -> StrategySnapshot: ...


def _configured(*environment_keys: str) -> str:
    return (
        "configured"
        if all(bool(os.getenv(environment_key)) for environment_key in environment_keys)
        else "not_configured"
    )


def _default_portfolio_service() -> PortfolioService:
    return get_portfolio_service()


def create_app(
    settings: WebSettings | None = None,
    *,
    portfolio_service: PortfolioReader | None = None,
) -> FastAPI:
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
    app.state.portfolio_service = portfolio_service
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
                "primary_account_id": (
                    app.state.portfolio_service.primary_account_id()
                    if app.state.portfolio_service
                    and hasattr(app.state.portfolio_service, "primary_account_id")
                    else "all"
                ),
                "data_state": "awaiting_portfolio_snapshot",
                "server_time": datetime.now(UTC).isoformat(),
            }
        )

    @app.get("/api/v1/portfolio", response_model=PortfolioSnapshot)
    async def portfolio(
        refresh: bool = False,
        account_id: str = "all",
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> PortfolioSnapshot:
        if session is None or not secrets.compare_digest(
            session,
            active_settings.session_token,
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        service = app.state.portfolio_service
        if service is None:
            service = _default_portfolio_service()
            app.state.portfolio_service = service
        try:
            if refresh:
                snapshot = await run_in_threadpool(service.refresh)
                return snapshot.for_account(account_id)
            snapshot = await run_in_threadpool(service.latest, account_id)
            if snapshot is None:
                snapshot = await run_in_threadpool(service.refresh)
                return snapshot.for_account(account_id)
            return snapshot
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Portfolio data is currently unavailable.",
            ) from error

    @app.put("/api/v1/settings/primary-account", status_code=status.HTTP_204_NO_CONTENT)
    async def save_primary_account(
        payload: PrimaryAccountRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        if session is None or not secrets.compare_digest(session, active_settings.session_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        service = app.state.portfolio_service or _default_portfolio_service()
        app.state.portfolio_service = service
        try:
            await run_in_threadpool(service.set_primary_account, payload.account_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.patch("/api/v1/strategies/{strategy_id}/horizon", response_model=StrategySnapshot)
    async def save_strategy_horizon(
        strategy_id: str,
        payload: StrategyHorizonRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> StrategySnapshot:
        if session is None or not secrets.compare_digest(session, active_settings.session_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        service = app.state.portfolio_service or _default_portfolio_service()
        app.state.portfolio_service = service
        try:
            return await run_in_threadpool(
                service.set_strategy_horizon,
                strategy_id,
                payload.horizon,
            )
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error

    @app.get("/api/v1/markets/{symbol}", response_model=MarketSnapshot)
    async def market_snapshot(
        symbol: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> MarketSnapshot:
        if session is None or not secrets.compare_digest(session, active_settings.session_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        snapshot = await run_in_threadpool(get_market_service().snapshot, symbol)
        if snapshot is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return snapshot

    @app.get("/api/v1/accounts/{account_id}/rolls", response_model=list[RollChainSnapshot])
    async def roll_chains(
        account_id: str,
        symbol: str | None = None,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[RollChainSnapshot]:
        if session is None or not secrets.compare_digest(session, active_settings.session_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return await run_in_threadpool(
            get_roll_service().chains,
            account_id,
            symbol=symbol,
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
