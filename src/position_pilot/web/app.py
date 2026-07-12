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
from typing import Annotated, Any, Protocol

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..client import get_client
from ..domain.alerts import AlertCategory
from ..domain.catalysts import (
    CatalystFeedbackEvent,
    CatalystFeedbackKind,
    CatalystScanSnapshot,
    SymbolCatalystResult,
)
from ..domain.factory import (
    get_alert_service,
    get_catalyst_service,
    get_database,
    get_field_router,
    get_market_service,
    get_monitoring_service,
    get_operations_service,
    get_order_service,
    get_plans_service,
    get_portfolio_service,
    get_recommendation_service,
    get_risk_service,
    get_roll_service,
    get_watchlist_service,
    set_live_market_hub,
)
from ..domain.market import ChartSnapshot, MarketOverview, MarketSnapshot
from ..domain.operations import package_diagnostic_zip
from ..domain.orders import OrderSnapshot
from ..domain.plans import AuditEvent, Thesis, TradePlan
from ..domain.portfolio import PortfolioService
from ..domain.recommendations import SubjectType, TraderDecisionKind
from ..domain.risk import PortfolioRisk, StrategyRisk
from ..domain.rolls import RollChainSnapshot, RollHeatmapSnapshot, RollPatternsSnapshot
from ..domain.snapshots import PortfolioSnapshot, PositionHorizon, StrategySnapshot
from ..domain.watchlist import WatchlistSnapshot
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
    # Production default is one-time launch tokens. Browser suites may reuse a fixed token.
    single_use_launch_token: bool = True


class SessionExchangeRequest(BaseModel):
    """One-time launch credential supplied by the locally opened browser."""

    launch_token: str


class PrimaryAccountRequest(BaseModel):
    account_id: str


class StrategyHorizonRequest(BaseModel):
    horizon: PositionHorizon


class ThesisRequest(BaseModel):
    purpose: str = Field(default="", max_length=4_000)
    expected_duration: str = Field(default="", max_length=1_000)
    target_range: str = Field(default="", max_length=1_000)
    invalidation: str = Field(default="", max_length=4_000)
    income_or_hedge_intent: str = Field(default="", max_length=4_000)
    events_to_watch: list[str] = Field(default_factory=list, max_length=100)


class TradePlanRequest(BaseModel):
    entry_thesis: str = Field(default="", max_length=4_000)
    intended_duration: str = Field(default="", max_length=1_000)
    profit_target: str = Field(default="", max_length=1_000)
    max_loss: str = Field(default="", max_length=1_000)
    roll_criteria: str = Field(default="", max_length=4_000)
    event_exposure: str = Field(default="", max_length=4_000)
    exit_deadline: str = Field(default="", max_length=1_000)


class WatchlistRequest(BaseModel):
    symbols: list[str] = Field(max_length=100)


class CatalystFeedbackRequest(BaseModel):
    kind: CatalystFeedbackKind
    catalyst_id: str | None = None
    symbol: str | None = None
    note: str = Field(default="", max_length=2_000)


class CatalystSettingsRequest(BaseModel):
    stock_move_threshold_pct: float | None = Field(default=None, ge=0.1, le=50)
    etf_move_threshold_pct: float | None = Field(default=None, ge=0.1, le=50)
    news_cadence_seconds: int | None = Field(default=None, ge=60, le=3600)
    benzinga_enabled: bool | None = None
    public_web_enabled: bool | None = None
    public_web_sources: list[dict[str, Any]] | None = None
    store_full_text_providers: list[str] | None = None
    full_text_consent: dict[str, dict[str, Any]] | None = None
    clear_full_text_provider: str | None = Field(default=None, max_length=64)


class MonitoringConsentRequest(BaseModel):
    enabled: bool


class RecommendationSettingsRequest(BaseModel):
    rich_notification_preview: bool | None = None


class TraderDecisionRequest(BaseModel):
    decision: TraderDecisionKind
    note: str = Field(default="", max_length=2_000)


class AlertSnoozeRequest(BaseModel):
    minutes: int = Field(default=60, ge=1, le=60 * 24 * 7)


class AlertMuteRequest(BaseModel):
    category: AlertCategory | None = None
    alert_type: str | None = Field(default=None, max_length=100)
    symbol: str | None = Field(default=None, max_length=32)
    strategy_type: str | None = Field(default=None, max_length=100)


class EvaluateRequest(BaseModel):
    force: bool = False


class RetentionSettingsRequest(BaseModel):
    portfolio_snapshots_days: int | None = Field(default=None, ge=30, le=3650)
    catalyst_events_days: int | None = Field(default=None, ge=30, le=3650)
    article_metadata_days: int | None = Field(default=None, ge=7, le=365)
    recommendation_history_days: int | None = Field(default=None, ge=0, le=3650)
    transaction_history: str | None = Field(default=None, max_length=32)


class ConfirmRequest(BaseModel):
    confirm: bool = False


class BackupCreateRequest(BaseModel):
    reason: str = Field(default="manual", max_length=40)


class RestoreBackupRequest(BaseModel):
    confirm: bool = False


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


def _require_session(session: str | None, expected: str) -> None:
    if session is None or not secrets.compare_digest(session, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def _stream_symbols(snapshot: PortfolioSnapshot | None) -> list[str]:
    return sorted(
        {
            symbol
            for account in (snapshot.accounts if snapshot is not None else [])
            for position in account.positions
            for symbol in (position.symbol, position.underlying_symbol)
        }
    )


def _find_strategy(snapshot: PortfolioSnapshot, strategy_id: str) -> StrategySnapshot:
    strategy = next(
        (item for item in snapshot.strategies if item.strategy_id == strategy_id),
        None,
    )
    if strategy is None:
        raise KeyError(strategy_id)
    return strategy


async def _portfolio_or_503(
    service: PortfolioReader,
    *,
    refresh: bool = False,
    account_id: str = "all",
) -> PortfolioSnapshot:
    try:
        if refresh:
            snapshot = await run_in_threadpool(service.refresh)
            return snapshot.for_account(account_id)
        snapshot = await run_in_threadpool(service.latest, account_id)
        if snapshot is None:
            snapshot = await run_in_threadpool(service.refresh)
            return snapshot.for_account(account_id)
        return snapshot
    except KeyError:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Portfolio data is currently unavailable.",
        ) from error


async def _start_streaming_runtime(app: FastAPI, settings: WebSettings) -> None:
    app.state.live_hub = LiveStateHub(get_database())
    # Prefer redacted DXLink values for tactical monitoring when available.
    set_live_market_hub(app.state.live_hub)
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
    set_live_market_hub(None)


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
        # Safe retention pass on startup (never blocks portfolio panels on failure).
        try:
            get_catalyst_service().apply_retention()
        except Exception:
            pass
        await _start_streaming_runtime(application, active_settings)
        # Codex monitoring is independent — failures must not block core APIs.
        try:
            monitoring = get_monitoring_service()
            application.state.monitoring_service = monitoring
            await monitoring.start()
        except Exception:
            application.state.monitoring_service = None
        try:
            yield
        finally:
            monitoring = getattr(application.state, "monitoring_service", None)
            if monitoring is not None:
                try:
                    await monitoring.stop()
                except Exception:
                    pass
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
    app.state.monitoring_service = None
    app.state.launch_token_available = True
    app.state.launch_token_lock = asyncio.Lock()
    assets_directory = STATIC_DIR / "assets"
    if assets_directory.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_directory), name="assets")

    async def resolve_strategy(strategy_id: str) -> StrategySnapshot:
        service = app.state.portfolio_service or _default_portfolio_service()
        app.state.portfolio_service = service
        try:
            snapshot = await _portfolio_or_503(service)
            return _find_strategy(snapshot, strategy_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error

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
            if active_settings.single_use_launch_token:
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
        _require_session(session, active_settings.session_token)

        def _bootstrap_side_state() -> tuple[dict[str, Any], str, dict[str, Any]]:
            """Blocking probes off the event loop; auth status is TTL-cached."""

            try:
                monitoring_payload = get_monitoring_service().public_bootstrap()
            except Exception:
                monitoring_payload = {
                    "market_timezone": "America/New_York",
                    "window_start": "07:30",
                    "window_end": "18:00",
                    "evaluation_minutes": 30,
                    "risk_refresh_seconds": 60,
                    "enabled": False,
                    "consented": False,
                    "inside_window": False,
                    "is_trading_day": False,
                    "is_holiday": False,
                    "is_early_close": False,
                    "provider_status": "unavailable",
                    "running": False,
                    "notice": "Monitoring status unavailable.",
                    "last_evaluation_at": None,
                }
            # Prefer monitoring's cached provider_status; fall back to one probe.
            codex_status = str(monitoring_payload.get("provider_status") or "not_checked")
            if codex_status in {"not_checked", ""}:
                try:
                    codex_status = get_recommendation_service().provider_public_status()
                except Exception:
                    codex_status = "unavailable"
            try:
                recommendation_settings = get_recommendation_service().settings()
            except Exception:
                recommendation_settings = {
                    "selected_provider": "codex-cli",
                    "api_key_fallback_available": False,
                    "api_key_fallback_enabled": False,
                    "rich_notification_preview": False,
                }
            return monitoring_payload, codex_status, recommendation_settings

        monitoring_payload, codex_status, recommendation_settings = await run_in_threadpool(
            _bootstrap_side_state
        )
        return JSONResponse(
            {
                "application": {
                    "name": "Position Pilot",
                    "version": __version__,
                    "phase": "hardening-retirement",
                },
                "providers": {
                    "tastytrade": _configured(
                        "TASTYTRADE_CLIENT_SECRET",
                        "TASTYTRADE_REFRESH_TOKEN",
                    ),
                    "codex": codex_status,
                    "massive": _configured("MASSIVE_API_KEY"),
                    "benzinga": _configured("BENZINGA_API_KEY"),
                },
                "monitoring": monitoring_payload,
                "recommendations": recommendation_settings,
                "catalysts": get_catalyst_service().public_settings(),
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
        _require_session(session, active_settings.session_token)
        service = app.state.portfolio_service
        if service is None:
            service = _default_portfolio_service()
            app.state.portfolio_service = service
        try:
            return await _portfolio_or_503(service, refresh=refresh, account_id=account_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error

    @app.get("/api/v1/portfolio/risk", response_model=PortfolioRisk)
    async def portfolio_risk(
        account_id: str = "all",
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> PortfolioRisk:
        _require_session(session, active_settings.session_token)
        service = app.state.portfolio_service or _default_portfolio_service()
        app.state.portfolio_service = service
        try:
            snapshot = await _portfolio_or_503(service, account_id=account_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return await run_in_threadpool(get_risk_service().portfolio_risk, snapshot)

    @app.put("/api/v1/settings/primary-account", status_code=status.HTTP_204_NO_CONTENT)
    async def save_primary_account(
        payload: PrimaryAccountRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        _require_session(session, active_settings.session_token)
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
        _require_session(session, active_settings.session_token)
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

    @app.get("/api/v1/strategies/{strategy_id}")
    async def strategy_detail(
        strategy_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        strategy = await resolve_strategy(strategy_id)

        market = await run_in_threadpool(get_market_service().snapshot, strategy.underlying)
        underlying_price = market.price if market else None
        risk = await run_in_threadpool(
            lambda: get_risk_service().strategy_risk(
                strategy,
                underlying_price=underlying_price,
            )
        )
        catalyst = await run_in_threadpool(
            get_catalyst_service().analyze_symbol,
            strategy.underlying,
        )
        markers = await run_in_threadpool(
            get_catalyst_service().event_markers,
            strategy.underlying,
        )
        chart = await run_in_threadpool(
            lambda: get_market_service().chart(
                strategy.underlying,
                prior_close=catalyst.prior_close,
                event_markers=markers,
                include_extended_hours=True,
                window_start=catalyst.lookback_start,
                window_end=catalyst.lookback_end,
            )
        )
        thesis = await run_in_threadpool(get_plans_service().get_thesis, strategy_id)
        trade_plan = await run_in_threadpool(get_plans_service().get_trade_plan, strategy_id)
        audit = await run_in_threadpool(get_plans_service().audit_history, strategy_id)
        try:
            recommendation = await run_in_threadpool(
                get_recommendation_service().get,
                SubjectType.STRATEGY,
                strategy_id,
            )
        except Exception:
            recommendation = None
        try:
            recommendation_history = await run_in_threadpool(
                lambda: get_recommendation_service().history(
                    SubjectType.STRATEGY,
                    strategy_id,
                    limit=20,
                )
            )
        except Exception:
            recommendation_history = []
        try:
            decisions = await run_in_threadpool(
                lambda: get_recommendation_service().list_decisions(
                    SubjectType.STRATEGY,
                    strategy_id,
                    limit=20,
                )
            )
        except Exception:
            decisions = []
        all_rolls = await run_in_threadpool(
            lambda: get_roll_service().chains(
                strategy.account_id,
                symbol=strategy.underlying,
            )
        )
        rolls = [chain for chain in all_rolls if chain.strategy_type == strategy.strategy_type]
        return {
            "strategy": strategy.model_dump(mode="json"),
            "risk": risk.model_dump(mode="json"),
            "market": market.model_dump(mode="json") if market else None,
            "chart": chart.model_dump(mode="json"),
            "catalyst": catalyst.model_dump(mode="json"),
            "thesis": thesis.model_dump(mode="json") if thesis else None,
            "trade_plan": trade_plan.model_dump(mode="json") if trade_plan else None,
            "recommendation": recommendation.model_dump(mode="json") if recommendation else None,
            "recommendation_history": [
                entry.model_dump(mode="json") for entry in recommendation_history
            ],
            "decisions": [item.model_dump(mode="json") for item in decisions],
            "audit": [event.model_dump(mode="json") for event in audit],
            "rolls": [chain.model_dump(mode="json") for chain in rolls],
            "events": [
                {
                    "kind": "catalyst",
                    "timestamp": event.event_at.isoformat(),
                    "summary": event.headline,
                    "action": event.confidence.value,
                }
                for event in catalyst.catalysts
            ]
            + [
                {
                    "kind": "audit",
                    "timestamp": event.recorded_at.isoformat(),
                    "summary": event.summary,
                    "action": event.action,
                }
                for event in audit
            ]
            + [
                {
                    "kind": "roll",
                    "timestamp": roll.timestamp.isoformat(),
                    "summary": (
                        f"Rolled {roll.old_strike} → {roll.new_strike} "
                        f"({roll.old_dte}→{roll.new_dte} DTE)"
                    ),
                    "action": "roll",
                }
                for chain in rolls
                for roll in chain.rolls
            ],
        }

    @app.get("/api/v1/strategies/{strategy_id}/risk", response_model=StrategyRisk)
    async def strategy_risk_endpoint(
        strategy_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> StrategyRisk:
        _require_session(session, active_settings.session_token)
        strategy = await resolve_strategy(strategy_id)
        market = await run_in_threadpool(get_market_service().snapshot, strategy.underlying)
        price = market.price if market else None
        return await run_in_threadpool(
            lambda: get_risk_service().strategy_risk(strategy, underlying_price=price)
        )

    @app.get("/api/v1/strategies/{strategy_id}/thesis", response_model=Thesis | None)
    async def get_thesis(
        strategy_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Thesis | None:
        _require_session(session, active_settings.session_token)
        await resolve_strategy(strategy_id)
        return await run_in_threadpool(get_plans_service().get_thesis, strategy_id)

    @app.put("/api/v1/strategies/{strategy_id}/thesis", response_model=Thesis)
    async def put_thesis(
        strategy_id: str,
        payload: ThesisRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Thesis:
        _require_session(session, active_settings.session_token)
        await resolve_strategy(strategy_id)
        return await run_in_threadpool(
            get_plans_service().save_thesis,
            strategy_id,
            payload.model_dump(),
        )

    @app.get("/api/v1/strategies/{strategy_id}/trade-plan", response_model=TradePlan | None)
    async def get_trade_plan(
        strategy_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> TradePlan | None:
        _require_session(session, active_settings.session_token)
        await resolve_strategy(strategy_id)
        return await run_in_threadpool(get_plans_service().get_trade_plan, strategy_id)

    @app.put("/api/v1/strategies/{strategy_id}/trade-plan", response_model=TradePlan)
    async def put_trade_plan(
        strategy_id: str,
        payload: TradePlanRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> TradePlan:
        _require_session(session, active_settings.session_token)
        await resolve_strategy(strategy_id)
        return await run_in_threadpool(
            get_plans_service().save_trade_plan,
            strategy_id,
            payload.model_dump(),
        )

    @app.get("/api/v1/strategies/{strategy_id}/audit", response_model=list[AuditEvent])
    async def strategy_audit(
        strategy_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[AuditEvent]:
        _require_session(session, active_settings.session_token)
        await resolve_strategy(strategy_id)
        return await run_in_threadpool(get_plans_service().audit_history, strategy_id)

    @app.get("/api/v1/markets", response_model=MarketOverview)
    async def markets_overview(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> MarketOverview:
        _require_session(session, active_settings.session_token)
        return await run_in_threadpool(get_market_service().overview)

    @app.get("/api/v1/markets/{symbol}", response_model=MarketSnapshot)
    async def market_snapshot(
        symbol: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> MarketSnapshot:
        _require_session(session, active_settings.session_token)
        snapshot = await run_in_threadpool(get_market_service().snapshot, symbol)
        if snapshot is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return snapshot

    @app.get("/api/v1/markets/{symbol}/chart", response_model=ChartSnapshot)
    async def market_chart(
        symbol: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> ChartSnapshot:
        _require_session(session, active_settings.session_token)
        catalyst = await run_in_threadpool(get_catalyst_service().analyze_symbol, symbol)
        markers = await run_in_threadpool(get_catalyst_service().event_markers, symbol)
        return await run_in_threadpool(
            lambda: get_market_service().chart(
                symbol,
                prior_close=catalyst.prior_close,
                event_markers=markers,
                include_extended_hours=True,
                window_start=catalyst.lookback_start,
                window_end=catalyst.lookback_end,
            )
        )

    @app.get("/api/v1/catalysts", response_model=CatalystScanSnapshot)
    async def catalysts_scan(
        account_id: str = "all",
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> CatalystScanSnapshot:
        _require_session(session, active_settings.session_token)
        service = app.state.portfolio_service or _default_portfolio_service()
        app.state.portfolio_service = service
        try:
            snapshot = await _portfolio_or_503(service, account_id=account_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        symbols = sorted({strategy.underlying for strategy in snapshot.strategies})
        return await run_in_threadpool(get_catalyst_service().scan_held, symbols)

    @app.get("/api/v1/catalysts/{symbol}", response_model=SymbolCatalystResult)
    async def catalyst_for_symbol(
        symbol: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> SymbolCatalystResult:
        _require_session(session, active_settings.session_token)
        return await run_in_threadpool(get_catalyst_service().analyze_symbol, symbol)

    @app.post("/api/v1/catalysts/feedback", response_model=CatalystFeedbackEvent)
    async def catalyst_feedback(
        payload: CatalystFeedbackRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> CatalystFeedbackEvent:
        _require_session(session, active_settings.session_token)
        if payload.kind is CatalystFeedbackKind.MISSING_CATALYST and not payload.symbol:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing catalyst feedback requires a symbol.",
            )
        if payload.kind is not CatalystFeedbackKind.MISSING_CATALYST and not payload.catalyst_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feedback requires a catalyst_id.",
            )
        try:
            return await run_in_threadpool(
                lambda: get_catalyst_service().submit_feedback(
                    payload.catalyst_id,
                    payload.kind,
                    symbol=payload.symbol,
                    note=payload.note,
                )
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    @app.get("/api/v1/settings/catalysts")
    async def get_catalyst_settings(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        return get_catalyst_service().public_settings()

    @app.put("/api/v1/settings/catalysts")
    async def put_catalyst_settings(
        payload: CatalystSettingsRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        updates = {key: value for key, value in payload.model_dump().items() if value is not None}
        return await run_in_threadpool(get_catalyst_service().update_settings, updates)

    @app.get("/api/v1/recommendations")
    async def list_recommendations(
        account_id: str = "all",
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[dict[str, Any]]:
        _require_session(session, active_settings.session_token)
        try:
            records = await run_in_threadpool(
                get_recommendation_service().list_for_account,
                account_id,
            )
            return [record.model_dump(mode="json") for record in records]
        except Exception:
            # Never block the dashboard if Codex/recommendation store fails.
            return []

    @app.get("/api/v1/recommendations/{subject_type}/{subject_id}")
    async def get_recommendation(
        subject_type: str,
        subject_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            SubjectType(subject_type)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST) from error
        record = await run_in_threadpool(
            get_recommendation_service().get,
            subject_type,
            subject_id,
        )
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return record.model_dump(mode="json")

    @app.get("/api/v1/recommendations/{subject_type}/{subject_id}/history")
    async def recommendation_history(
        subject_type: str,
        subject_id: str,
        limit: int = 50,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[dict[str, Any]]:
        _require_session(session, active_settings.session_token)
        try:
            SubjectType(subject_type)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST) from error
        entries = await run_in_threadpool(
            lambda: get_recommendation_service().history(
                subject_type,
                subject_id,
                limit=min(max(limit, 1), 200),
            )
        )
        return [entry.model_dump(mode="json") for entry in entries]

    @app.post("/api/v1/strategies/{strategy_id}/recommend")
    async def evaluate_strategy_recommendation(
        strategy_id: str,
        payload: EvaluateRequest | None = None,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        force = bool(payload.force) if payload else False
        strategy = await resolve_strategy(strategy_id)

        def _run():
            catalysts = []
            try:
                catalyst = get_catalyst_service().analyze_symbol(strategy.underlying)
                catalysts = [event.model_dump(mode="json") for event in catalyst.catalysts]
            except Exception:
                catalysts = []
            thesis = get_plans_service().get_thesis(strategy_id)
            plan = get_plans_service().get_trade_plan(strategy_id)
            return get_recommendation_service().evaluate_strategy(
                strategy,
                catalysts=catalysts,
                thesis=thesis.model_dump(mode="json") if thesis else None,
                trade_plan=plan.model_dump(mode="json") if plan else None,
                force=force,
            )

        try:
            record = await run_in_threadpool(_run)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Recommendation evaluation is currently unavailable.",
            ) from error
        return record.model_dump(mode="json")

    @app.post("/api/v1/recommendations/{recommendation_id}/decisions")
    async def record_trader_decision(
        recommendation_id: str,
        payload: TraderDecisionRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            decision = await run_in_threadpool(
                lambda: get_recommendation_service().record_decision(
                    recommendation_id=recommendation_id,
                    decision=payload.decision,
                    note=payload.note,
                )
            )
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return decision.model_dump(mode="json")

    @app.get("/api/v1/decisions")
    async def list_decisions(
        subject_type: str | None = None,
        subject_id: str | None = None,
        limit: int = 50,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[dict[str, Any]]:
        _require_session(session, active_settings.session_token)
        decisions = await run_in_threadpool(
            lambda: get_recommendation_service().list_decisions(
                subject_type,
                subject_id,
                limit=min(max(limit, 1), 200),
            )
        )
        return [item.model_dump(mode="json") for item in decisions]

    @app.get("/api/v1/alerts")
    async def list_alerts(
        account_id: str = "all",
        include_resolved: bool = False,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[dict[str, Any]]:
        _require_session(session, active_settings.session_token)
        try:
            alerts = await run_in_threadpool(
                lambda: get_alert_service().list_alerts(
                    account_id=account_id,
                    include_resolved=include_resolved,
                )
            )
            return [alert.model_dump(mode="json") for alert in alerts]
        except Exception:
            return []

    @app.post("/api/v1/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(
        alert_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            alert = await run_in_threadpool(get_alert_service().acknowledge, alert_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return alert.model_dump(mode="json")

    @app.post("/api/v1/alerts/{alert_id}/snooze")
    async def snooze_alert(
        alert_id: str,
        payload: AlertSnoozeRequest | None = None,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        minutes = payload.minutes if payload else 60
        try:
            alert = await run_in_threadpool(
                lambda: get_alert_service().snooze(alert_id, minutes=minutes)
            )
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return alert.model_dump(mode="json")

    @app.post("/api/v1/alerts/{alert_id}/resolve")
    async def resolve_alert(
        alert_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            alert = await run_in_threadpool(get_alert_service().resolve, alert_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        return alert.model_dump(mode="json")

    @app.post("/api/v1/alerts/mute")
    async def mute_alerts(
        payload: AlertMuteRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        rule = await run_in_threadpool(
            lambda: get_alert_service().mute_by_rule(
                category=payload.category,
                alert_type=payload.alert_type,
                symbol=payload.symbol,
                strategy_type=payload.strategy_type,
            )
        )
        return rule.model_dump(mode="json")

    @app.get("/api/v1/monitoring")
    async def monitoring_status(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            return await run_in_threadpool(
                lambda: get_monitoring_service().status().model_dump(mode="json")
            )
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Monitoring status is currently unavailable.",
            ) from error

    @app.put("/api/v1/monitoring/consent")
    async def monitoring_consent(
        payload: MonitoringConsentRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)

        def _set() -> dict[str, Any]:
            consent = get_monitoring_service().set_consent(enabled=payload.enabled)
            return {
                "consent": consent.model_dump(mode="json"),
                "status": get_monitoring_service().status().model_dump(mode="json"),
            }

        return await run_in_threadpool(_set)

    @app.post("/api/v1/monitoring/evaluate")
    async def monitoring_evaluate(
        payload: EvaluateRequest | None = None,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        force = bool(payload.force) if payload else False
        try:
            return await run_in_threadpool(
                lambda: get_monitoring_service().run_once(reason="on_demand", force=force)
            )
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="On-demand evaluation failed while core portfolio data remains available.",
            ) from error

    @app.get("/api/v1/settings/recommendations")
    async def get_recommendation_settings(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        return await run_in_threadpool(get_recommendation_service().settings)

    @app.put("/api/v1/settings/recommendations")
    async def put_recommendation_settings(
        payload: RecommendationSettingsRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        updates = {key: value for key, value in payload.model_dump().items() if value is not None}
        try:
            return await run_in_threadpool(get_recommendation_service().update_settings, updates)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    @app.get("/api/v1/watchlist", response_model=WatchlistSnapshot)
    async def watchlist(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> WatchlistSnapshot:
        _require_session(session, active_settings.session_token)
        return await run_in_threadpool(get_watchlist_service().snapshot)

    @app.put("/api/v1/watchlist", response_model=WatchlistSnapshot)
    async def replace_watchlist(
        payload: WatchlistRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> WatchlistSnapshot:
        _require_session(session, active_settings.session_token)
        try:
            await run_in_threadpool(get_watchlist_service().set_symbols, payload.symbols)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        return await run_in_threadpool(get_watchlist_service().snapshot)

    @app.get("/api/v1/accounts/{account_id}/orders", response_model=list[OrderSnapshot])
    async def account_orders(
        account_id: str,
        limit: int = 100,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[OrderSnapshot]:
        _require_session(session, active_settings.session_token)
        capped = min(max(limit, 1), 250)
        try:
            return await run_in_threadpool(
                lambda: get_order_service().list_orders(account_id, limit=capped)
            )
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Order history is currently unavailable.",
            ) from error

    @app.get("/api/v1/accounts/{account_id}/rolls", response_model=list[RollChainSnapshot])
    async def roll_chains(
        account_id: str,
        symbol: str | None = None,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[RollChainSnapshot]:
        _require_session(session, active_settings.session_token)
        return await run_in_threadpool(lambda: get_roll_service().chains(account_id, symbol=symbol))

    @app.get(
        "/api/v1/accounts/{account_id}/rolls/patterns",
        response_model=RollPatternsSnapshot,
    )
    async def roll_patterns(
        account_id: str,
        symbol: str | None = None,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> RollPatternsSnapshot:
        _require_session(session, active_settings.session_token)
        return await run_in_threadpool(
            lambda: get_roll_service().patterns(account_id, symbol=symbol)
        )

    @app.get(
        "/api/v1/accounts/{account_id}/rolls/heatmap",
        response_model=RollHeatmapSnapshot,
    )
    async def roll_heatmap(
        account_id: str,
        symbol: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> RollHeatmapSnapshot:
        _require_session(session, active_settings.session_token)
        return await run_in_threadpool(
            lambda: get_roll_service().heatmap(account_id, symbol=symbol)
        )

    @app.get("/api/v1/providers/health")
    async def provider_health(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict:
        _require_session(session, active_settings.session_token)
        health_map = get_field_router().health()
        for provider, state in health_map.items():
            get_database().save_provider_health(provider, state.model_dump(mode="json"))
        return {provider: state.model_dump(mode="json") for provider, state in health_map.items()}

    # ── Phase 7 operations: export, diagnostics, backup, retention, update ──

    @app.get("/api/v1/exports/portfolio.csv")
    async def export_portfolio_csv(
        account_id: str = "all",
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        _require_session(session, active_settings.session_token)
        try:
            filename, body = await run_in_threadpool(
                lambda: get_operations_service().export_portfolio_csv(account_id)
            )
        except LookupError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return Response(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/v1/exports/history.csv")
    async def export_history_csv(
        limit: int = 365,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        _require_session(session, active_settings.session_token)
        filename, body = await run_in_threadpool(
            lambda: get_operations_service().export_history_csv(limit=limit)
        )
        return Response(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/v1/exports/snapshot.html")
    async def export_snapshot_html(
        account_id: str = "all",
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        _require_session(session, active_settings.session_token)
        try:
            filename, body = await run_in_threadpool(
                lambda: get_operations_service().printable_html(account_id)
            )
        except LookupError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return Response(
            content=body,
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/v1/exports/snapshot.pdf")
    async def export_snapshot_pdf(
        account_id: str = "all",
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        _require_session(session, active_settings.session_token)
        try:
            filename, body = await run_in_threadpool(
                lambda: get_operations_service().printable_pdf(account_id)
            )
        except LookupError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return Response(
            content=body,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/v1/diagnostics/bundle")
    async def diagnostics_bundle(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        bundle = await run_in_threadpool(get_operations_service().diagnostic_bundle)
        return bundle.model_dump(mode="json")

    @app.get("/api/v1/diagnostics/bundle.zip")
    async def diagnostics_bundle_zip(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        _require_session(session, active_settings.session_token)
        bundle = await run_in_threadpool(get_operations_service().diagnostic_bundle)
        payload = package_diagnostic_zip(bundle)
        return Response(
            content=payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="position-pilot-diagnostics.zip"'
            },
        )

    @app.get("/api/v1/diagnostics/env")
    async def diagnostics_env(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        report = await run_in_threadpool(get_operations_service().env_diagnostics)
        return report.model_dump(mode="json")

    @app.get("/api/v1/settings/retention")
    async def get_retention_settings(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        settings = await run_in_threadpool(get_operations_service().retention_settings)
        return settings.model_dump(mode="json")

    @app.put("/api/v1/settings/retention")
    async def put_retention_settings(
        payload: RetentionSettingsRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        updates = {key: value for key, value in payload.model_dump().items() if value is not None}
        try:
            settings = await run_in_threadpool(
                lambda: get_operations_service().update_retention_settings(updates)
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        return settings.model_dump(mode="json")

    @app.get("/api/v1/settings/retention/preview")
    async def retention_preview(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        preview = await run_in_threadpool(get_operations_service().retention_preview)
        return preview.model_dump(mode="json")

    @app.post("/api/v1/settings/retention/apply")
    async def retention_apply(
        payload: ConfirmRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            result = await run_in_threadpool(
                lambda: get_operations_service().apply_retention(confirm=payload.confirm)
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        return result

    @app.get("/api/v1/backups")
    async def list_backups(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> list[dict[str, Any]]:
        _require_session(session, active_settings.session_token)
        items = await run_in_threadpool(get_operations_service().list_backups)
        return [item.model_dump(mode="json") for item in items]

    @app.post("/api/v1/backups")
    async def create_backup(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            item = await run_in_threadpool(
                lambda: get_operations_service().create_backup(reason="manual")
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        return item.model_dump(mode="json")

    @app.get("/api/v1/backups/{backup_id}")
    async def download_backup(
        backup_id: str,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> Response:
        """Browser download: sanitized portable archive (not the faithful restore file)."""

        _require_session(session, active_settings.session_token)
        try:
            filename, payload = await run_in_threadpool(
                lambda: get_operations_service().portable_backup_archive(backup_id)
            )
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/v1/backups/{backup_id}/restore")
    async def restore_backup(
        backup_id: str,
        payload: RestoreBackupRequest,
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        try:
            result = await run_in_threadpool(
                lambda: get_operations_service().restore_backup(backup_id, confirm=payload.confirm)
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        return result.model_dump(mode="json")

    @app.get("/api/v1/update/status")
    async def update_status(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict[str, Any]:
        _require_session(session, active_settings.session_token)
        readiness = await run_in_threadpool(get_operations_service().update_readiness)
        return readiness.model_dump(mode="json")

    @app.get("/api/v1/streaming/status")
    async def streaming_status(
        session: Annotated[
            str | None,
            Cookie(alias="position_pilot_session"),
        ] = None,
    ) -> dict:
        _require_session(session, active_settings.session_token)
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
        _require_session(session, active_settings.session_token)
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
