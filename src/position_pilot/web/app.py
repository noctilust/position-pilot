"""FastAPI application factory for the local Position Pilot dashboard."""

from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Protocol

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..client import get_client
from ..domain.factory import (
    get_database,
    get_field_router,
    get_market_service,
    get_portfolio_service,
    get_roll_service,
)
from ..domain.market import MarketSnapshot
from ..domain.portfolio import PortfolioService
from ..domain.rolls import RollChainSnapshot
from ..domain.snapshots import PortfolioSnapshot, PositionHorizon, StrategySnapshot
from ..streaming.account import AccountStreamEvent
from ..streaming.hub import LiveEvent, LiveStateHub
from ..streaming.reconciliation import ReconciliationCoordinator, ReconciliationWorkQueue
from ..streaming.service import TastytradeStreamingService

STATIC_DIR = Path(__file__).with_name("static")


@dataclass(frozen=True, slots=True)
class WebSettings:
    """Process-local security and hosting settings for the web dashboard."""

    launch_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    session_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    enforce_loopback: bool = True
    enable_streaming: bool = True


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


def _stream_symbols(snapshot: PortfolioSnapshot | None) -> list[str]:
    return sorted(
        {
            symbol
            for account in (snapshot.accounts if snapshot is not None else [])
            for position in account.positions
            for symbol in (position.symbol, position.underlying_symbol)
        }
    )


async def _start_streaming_runtime(app: FastAPI, settings: WebSettings) -> None:
    app.state.live_hub = LiveStateHub(get_database())
    tastytrade_configured = (
        _configured("TASTYTRADE_CLIENT_SECRET", "TASTYTRADE_REFRESH_TOKEN") == "configured"
    )
    if not settings.enable_streaming or not tastytrade_configured:
        return
    client = get_client()
    service = app.state.portfolio_service or _default_portfolio_service()
    app.state.portfolio_service = service
    try:
        snapshot = await run_in_threadpool(service.latest)
        if snapshot is None:
            snapshot = await run_in_threadpool(service.refresh)
    except Exception:
        snapshot = None
    try:
        accounts = await run_in_threadpool(client.get_accounts)
    except Exception:
        accounts = []
    if snapshot is None and not accounts:
        app.state.streaming_startup_error = "BrokerStateUnavailable"
        return
    account_numbers = [account.account_number for account in accounts]
    symbols = _stream_symbols(snapshot)

    async def refresh_and_publish(reason: str) -> None:
        try:
            refreshed = await run_in_threadpool(service.refresh)
            streaming.update_symbols(_stream_symbols(refreshed))
            await app.state.live_hub.publish(
                LiveEvent(
                    event_type="portfolio.reconciled",
                    payload={
                        "snapshot_id": refreshed.snapshot_id,
                        "reason": reason,
                        "captured_at": refreshed.captured_at.isoformat(),
                    },
                    received_at=datetime.now(UTC),
                )
            )
        except Exception as error:
            await app.state.live_hub.publish(
                LiveEvent(
                    event_type="portfolio.reconciliation_failed",
                    payload={"reason": reason, "error": type(error).__name__},
                    received_at=datetime.now(UTC),
                )
            )

    reconciliation_queue = ReconciliationWorkQueue(refresh_and_publish)
    app.state.reconciliation_queue = reconciliation_queue

    coordinator = ReconciliationCoordinator(reconcile=reconciliation_queue.submit)
    streaming = TastytradeStreamingService(client=client, reconciliation=coordinator)
    app.state.streaming_service = streaming

    async def run_streams() -> None:
        async def handle_account_event(event: AccountStreamEvent) -> None:
            await app.state.live_hub.publish_account(event)
            coordinator.on_account_event(event.event_type)

        try:
            await streaming.run(
                account_numbers=account_numbers,
                symbols=symbols,
                on_market_event=app.state.live_hub.publish_market,
                on_account_event=handle_account_event,
            )
        except Exception as error:
            await app.state.live_hub.publish(
                LiveEvent(
                    event_type="provider.tastytrade_streaming",
                    payload={"state": "unavailable", "error": type(error).__name__},
                    received_at=datetime.now(UTC),
                )
            )

    async def reconciliation_clock() -> None:
        while not streaming.stop.is_set():
            await asyncio.sleep(30)
            coordinator.run_if_due()

    app.state.streaming_tasks = [
        asyncio.create_task(run_streams()),
        asyncio.create_task(reconciliation_clock()),
    ]


async def _stop_streaming_runtime(app: FastAPI) -> None:
    streaming = app.state.streaming_service
    if streaming is not None:
        streaming.close()
    for task in app.state.streaming_tasks:
        task.cancel()
    if app.state.streaming_tasks:
        await asyncio.gather(*app.state.streaming_tasks, return_exceptions=True)
    if app.state.reconciliation_queue is not None:
        await app.state.reconciliation_queue.close()


def create_app(
    settings: WebSettings | None = None,
    *,
    portfolio_service: PortfolioReader | None = None,
) -> FastAPI:
    """Create an isolated local dashboard application."""

    load_dotenv(Path.cwd() / ".env", override=False)
    active_settings = settings or WebSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await _start_streaming_runtime(application, active_settings)
        try:
            yield
        finally:
            await _stop_streaming_runtime(application)

    app = FastAPI(
        title="Position Pilot",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.web_settings = active_settings
    app.state.portfolio_service = portfolio_service
    app.state.live_hub = None
    app.state.streaming_tasks = []
    app.state.streaming_service = None
    app.state.streaming_startup_error = None
    app.state.reconciliation_queue = None
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

    @app.get("/api/v1/providers/health")
    async def provider_health(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict:
        if session is None or not secrets.compare_digest(session, active_settings.session_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        health = get_field_router().health()
        for provider, state in health.items():
            get_database().save_provider_health(provider, state.model_dump(mode="json"))
        return {provider: state.model_dump(mode="json") for provider, state in health.items()}

    @app.get("/api/v1/streaming/status")
    async def streaming_status(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict:
        if session is None or not secrets.compare_digest(session, active_settings.session_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        streaming = app.state.streaming_service
        if streaming is None:
            if app.state.streaming_startup_error is not None:
                return {
                    "market": {
                        "state": "degraded",
                        "error": app.state.streaming_startup_error,
                    },
                    "account": {
                        "state": "degraded",
                        "error": app.state.streaming_startup_error,
                    },
                }
            return {
                "market": {"state": "disabled", "error": None},
                "account": {"state": "disabled", "error": None},
            }
        return streaming.status

    @app.get("/api/v1/events")
    async def live_events(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> StreamingResponse:
        if session is None or not secrets.compare_digest(session, active_settings.session_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        queue = app.state.live_hub.subscribe()

        async def stream():
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=20)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                app.state.live_hub.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
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
