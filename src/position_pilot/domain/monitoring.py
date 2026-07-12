"""Trading-day monitoring window, consent, and non-blocking evaluation scheduler."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Any, Callable
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from ..persistence.sqlite import PositionPilotDatabase
from ..providers.codex import CodexProviderStatus
from .alerts import AlertCategory, AlertService, AlertSeverity
from .notifications import NotificationService
from .recommendations import (
    RecommendationService,
    SubjectType,
    equity_subject_id,
    fingerprint_inputs,
    is_stock_strategy,
    sanitize_market_context,
    strategic_due,
    strategy_context,
    tactical_due,
)
from .snapshots import PortfolioSnapshot, PositionHorizon, PositionType

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
WINDOW_START = time(7, 30)
WINDOW_END = time(18, 0)
# Equity early close is 13:00 ET; monitoring preserves a two-hour post-market window → 15:00.
EARLY_CLOSE_MARKET = time(13, 0)
EARLY_CLOSE_MONITOR_END = time(15, 0)
RISK_INTERVAL_SECONDS = 60
REEVALUATION_SECONDS = 30 * 60
WAKE_GAP_SECONDS = 120  # > 2 risk intervals implies sleep/wake gap

# Live-market materiality thresholds for the 60-second tactical pulse.
PRICE_MATERIAL_PCT = 0.5  # percent move vs last baseline
IV_RANK_MATERIAL = 2.0  # absolute IV-rank points
IV_MATERIAL = 0.02  # absolute IV (e.g. 0.02 = 2 vol points)
SPREAD_MATERIAL_PCT = 0.5  # percentage-point spread change


class RiskLevelState(StrEnum):
    """Deterministic portfolio risk bands for alert materiality."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_RANK = {
    RiskLevelState.NORMAL: 0,
    RiskLevelState.ELEVATED: 1,
    RiskLevelState.HIGH: 2,
    RiskLevelState.CRITICAL: 3,
}


def classify_risk_state(
    *,
    total_delta: float = 0.0,
    total_gamma: float = 0.0,
    concentration_top_share: float = 0.0,
    unrealized_pnl: float = 0.0,
) -> RiskLevelState:
    """Map aggregate risk inputs to a discrete state (no pennies-as-alerts)."""

    delta = abs(float(total_delta or 0.0))
    gamma = abs(float(total_gamma or 0.0))
    conc = float(concentration_top_share or 0.0)
    pnl = float(unrealized_pnl or 0.0)

    if pnl <= -5_000 or delta >= 200 or conc >= 0.50:
        return RiskLevelState.CRITICAL
    if pnl <= -2_000 or delta >= 100 or conc >= 0.35 or gamma >= 0.50:
        return RiskLevelState.HIGH
    if pnl <= -500 or delta >= 50 or conc >= 0.25 or gamma >= 0.20:
        return RiskLevelState.ELEVATED
    return RiskLevelState.NORMAL


def should_emit_risk_alert(
    previous: RiskLevelState | None,
    current: RiskLevelState,
) -> bool:
    """Alert only on entering a worse state — not ordinary normal P/L noise."""

    if previous is None:
        # First observation: alert only if already elevated or worse.
        return current != RiskLevelState.NORMAL
    return _RISK_RANK[current] > _RISK_RANK[previous]


def is_material_market_change(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
    *,
    price_pct: float = PRICE_MATERIAL_PCT,
    iv_rank_points: float = IV_RANK_MATERIAL,
    iv_abs: float = IV_MATERIAL,
    spread_pct: float = SPREAD_MATERIAL_PCT,
) -> bool:
    """True when live market inputs crossed tactical materiality thresholds."""

    if not current:
        return False
    if not previous:
        return False  # establish baseline without forcing Codex
    prev_price = previous.get("price")
    curr_price = current.get("price")
    if isinstance(prev_price, (int, float)) and isinstance(curr_price, (int, float)):
        if prev_price != 0 and abs(curr_price - prev_price) / abs(prev_price) * 100 >= price_pct:
            return True
    for key, threshold in (("iv_rank", iv_rank_points), ("spread_percent", spread_pct)):
        prev = previous.get(key)
        curr = current.get(key)
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            if abs(curr - prev) >= threshold:
                return True
    prev_iv = previous.get("iv")
    curr_iv = current.get("iv")
    if isinstance(prev_iv, (int, float)) and isinstance(curr_iv, (int, float)):
        if abs(curr_iv - prev_iv) >= iv_abs:
            return True
    return False


class MonitoringConsent(BaseModel):
    enabled: bool = False
    consented_at: datetime | None = None
    consent_version: str = "monitoring-consent.v1"


class MonitoringStatus(BaseModel):
    enabled: bool
    consented: bool
    inside_window: bool
    market_timezone: str = "America/New_York"
    window_start: str = "07:30"
    window_end: str = "18:00"
    evaluation_minutes: int = 30
    risk_refresh_seconds: int = 60
    is_trading_day: bool
    is_holiday: bool
    is_early_close: bool
    last_risk_tick_at: datetime | None = None
    last_evaluation_at: datetime | None = None
    last_recovery_evaluation_at: datetime | None = None
    provider_status: str = "not_checked"
    running: bool = False
    notice: str | None = None


@dataclass(slots=True)
class MarketSession:
    trading_day: bool
    holiday: bool
    early_close: bool
    window_start: datetime
    window_end: datetime


def local_now(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(MARKET_TZ)


def is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th weekday (Mon=0) of month."""

    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (n - 1) * 7
    return date(year, month, day)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    cursor = next_month - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def _observed(day: date) -> date:
    """NYSE observed holiday: Sat→Fri, Sun→Mon."""

    if day.weekday() == 5:  # Saturday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm for Western Easter Sunday."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def good_friday(year: int) -> date:
    return easter_sunday(year) - timedelta(days=2)


def us_market_holidays_for_year(year: int) -> set[date]:
    """Deterministic NYSE full-day holidays for any year (stdlib algorithm).

    Covers fixed, observed, and floating holidays including Good Friday and
    Juneteenth. Does not model ad-hoc emergency closures (weather, national
    mourning) — those remain an honest exceptional-closure limitation.
    """

    holidays = {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # MLK Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        good_friday(year),
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 6, 19)),  # Juneteenth
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }
    return holidays


def us_early_close_days_for_year(year: int) -> set[date]:
    """Deterministic NYSE 13:00 ET early-close sessions for a year."""

    days: set[date] = set()
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    day_after = thanksgiving + timedelta(days=1)
    if day_after.weekday() < 5:
        days.add(day_after)

    # Day before Independence Day when July 3 is a weekday and July 4 is the holiday.
    july3 = date(year, 7, 3)
    if july3.weekday() < 5 and july3 not in us_market_holidays_for_year(year):
        days.add(july3)

    # Christmas Eve when a weekday and not itself a full holiday.
    eve = date(year, 12, 24)
    if eve.weekday() < 5 and eve not in us_market_holidays_for_year(year):
        days.add(eve)

    return days


def is_market_holiday(day: date) -> bool:
    return day in us_market_holidays_for_year(day.year)


def is_early_close_day(day: date) -> bool:
    return day in us_early_close_days_for_year(day.year)


def session_for(now: datetime | None = None) -> MarketSession:
    local = local_now(now)
    day = local.date()
    holiday = is_market_holiday(day)
    weekend = is_weekend(day)
    trading_day = not weekend and not holiday
    early_close = trading_day and is_early_close_day(day)
    end_time = EARLY_CLOSE_MONITOR_END if early_close else WINDOW_END
    start = datetime.combine(day, WINDOW_START, tzinfo=MARKET_TZ)
    end = datetime.combine(day, end_time, tzinfo=MARKET_TZ)
    return MarketSession(
        trading_day=trading_day,
        holiday=holiday,
        early_close=early_close,
        window_start=start,
        window_end=end,
    )


def inside_monitoring_window(now: datetime | None = None) -> bool:
    session = session_for(now)
    if not session.trading_day:
        return False
    local = local_now(now)
    return session.window_start <= local <= session.window_end


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class MonitoringService:
    """Consent-gated background evaluator that never blocks core APIs."""

    def __init__(
        self,
        database: PositionPilotDatabase,
        recommendations: RecommendationService,
        alerts: AlertService,
        notifications: NotificationService,
        *,
        portfolio_loader: Callable[[], PortfolioSnapshot | None] | None = None,
        catalyst_loader: Callable[[str], list[dict[str, Any]]] | None = None,
        thesis_loader: Callable[[str], dict[str, Any] | None] | None = None,
        plan_loader: Callable[[str], dict[str, Any] | None] | None = None,
        risk_snapshot_loader: Callable[[PortfolioSnapshot], dict[str, Any] | None] | None = None,
        market_context_loader: Callable[[str], dict[str, Any] | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        risk_interval_seconds: int = RISK_INTERVAL_SECONDS,
        reevaluation_seconds: int = REEVALUATION_SECONDS,
    ) -> None:
        self.database = database
        self.recommendations = recommendations
        self.alerts = alerts
        self.notifications = notifications
        self.portfolio_loader = portfolio_loader
        self.catalyst_loader = catalyst_loader or (lambda _symbol: [])
        self.thesis_loader = thesis_loader or (lambda _sid: None)
        self.plan_loader = plan_loader or (lambda _sid: None)
        self.risk_snapshot_loader = risk_snapshot_loader
        self.market_context_loader = market_context_loader
        self._clock = clock or (lambda: datetime.now(UTC))
        self.risk_interval_seconds = risk_interval_seconds
        self.reevaluation_seconds = reevaluation_seconds
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._cycle_lock = threading.Lock()
        self._last_risk_tick_at: datetime | None = None
        self._last_evaluation_at: datetime | None = None
        self._last_recovery_evaluation_at: datetime | None = None
        self._last_loop_tick_at: datetime | None = None
        self._next_eval_at: datetime | None = None
        self._network_available = True
        self._portfolio_unavailable = False
        self._last_risk_fingerprint: str | None = None
        self._last_risk_state: RiskLevelState | None = None
        self._market_baselines: dict[str, dict[str, Any]] = {}
        self._seen_catalyst_keys: set[str] = set()
        self._restore_runtime()

    def _restore_runtime(self) -> None:
        runtime = self.database.get_setting("monitoring.runtime", {}) or {}
        self._last_evaluation_at = _parse_iso(runtime.get("last_evaluation_at"))
        self._last_recovery_evaluation_at = _parse_iso(runtime.get("last_recovery_evaluation_at"))
        self._last_risk_tick_at = _parse_iso(
            self.database.get_setting("monitoring.last_risk_tick_at")
        )
        self._last_risk_fingerprint = runtime.get("last_risk_fingerprint")
        state_name = runtime.get("last_risk_state")
        if state_name in {item.value for item in RiskLevelState}:
            self._last_risk_state = RiskLevelState(state_name)
        baselines = runtime.get("market_baselines") or {}
        if isinstance(baselines, dict):
            self._market_baselines = {
                str(symbol).upper(): dict(payload)
                for symbol, payload in baselines.items()
                if isinstance(payload, dict)
            }
        seen = runtime.get("seen_catalyst_keys") or []
        if isinstance(seen, list):
            self._seen_catalyst_keys = {str(item) for item in seen}

    def get_consent(self) -> MonitoringConsent:
        stored = self.database.get_setting("monitoring.consent", {}) or {}
        return MonitoringConsent.model_validate(
            {
                "enabled": bool(stored.get("enabled", False)),
                "consented_at": stored.get("consented_at"),
                "consent_version": stored.get("consent_version", "monitoring-consent.v1"),
            }
        )

    def set_consent(self, *, enabled: bool) -> MonitoringConsent:
        now = self._clock()
        consent = MonitoringConsent(
            enabled=enabled,
            consented_at=now if enabled else None,
            consent_version="monitoring-consent.v1",
        )
        self.database.set_setting("monitoring.consent", consent.model_dump(mode="json"))
        if enabled:
            try:
                self.database.ensure_daily_backup(now=now)
            except Exception:
                logger.exception("monitoring backup failed")
        return consent

    def status(self) -> MonitoringStatus:
        consent = self.get_consent()
        session = session_for(self._clock())
        inside = inside_monitoring_window(self._clock())
        end_label = (
            EARLY_CLOSE_MONITOR_END.strftime("%H:%M") if session.early_close else "18:00"
        )
        notice = None
        if not consent.enabled:
            notice = "Monitoring is disabled until you grant onboarding consent."
        elif not session.trading_day:
            notice = "Outside trading days (weekend or US exchange holiday)."
        elif not inside:
            notice = "Outside the America/New_York monitoring window."
        return MonitoringStatus(
            enabled=consent.enabled,
            consented=consent.enabled,
            inside_window=inside,
            window_end=end_label,
            is_trading_day=session.trading_day,
            is_holiday=session.holiday,
            is_early_close=session.early_close,
            last_risk_tick_at=self._last_risk_tick_at,
            last_evaluation_at=self._last_evaluation_at,
            last_recovery_evaluation_at=self._last_recovery_evaluation_at,
            provider_status=self.recommendations.provider_public_status(),
            running=self._running,
            notice=notice,
        )

    def public_bootstrap(self) -> dict[str, Any]:
        status = self.status()
        return {
            "market_timezone": status.market_timezone,
            "window_start": status.window_start,
            "window_end": status.window_end,
            "evaluation_minutes": status.evaluation_minutes,
            "risk_refresh_seconds": status.risk_refresh_seconds,
            "enabled": status.enabled,
            "consented": status.consented,
            "inside_window": status.inside_window,
            "is_trading_day": status.is_trading_day,
            "is_holiday": status.is_holiday,
            "is_early_close": status.is_early_close,
            "provider_status": status.provider_status,
            "running": status.running,
            "notice": status.notice,
            "last_evaluation_at": status.last_evaluation_at.isoformat()
            if status.last_evaluation_at
            else None,
            "last_risk_tick_at": status.last_risk_tick_at.isoformat()
            if status.last_risk_tick_at
            else None,
        }

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="position-pilot-monitoring")

    async def stop(self) -> None:
        self._stop.set()
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def on_network_recovery(self) -> dict[str, Any]:
        """One recovery evaluation if inside the window (no missed-interval replay)."""

        self._network_available = True
        self._portfolio_unavailable = False
        if not self.get_consent().enabled:
            return {"skipped": True, "reason": "consent_required"}
        if not inside_monitoring_window(self._clock()):
            return {"skipped": True, "reason": "outside_window"}
        result = self.run_once(reason="network_recovery")
        self._last_recovery_evaluation_at = self._clock()
        self._persist_runtime()
        return result

    def on_wake(self) -> dict[str, Any]:
        if not self.get_consent().enabled:
            return {"skipped": True, "reason": "consent_required"}
        if not inside_monitoring_window(self._clock()):
            return {"skipped": True, "reason": "outside_window"}
        result = self.run_once(reason="wake")
        self._last_recovery_evaluation_at = self._clock()
        self._persist_runtime()
        return result

    def run_once(self, *, reason: str = "manual", force: bool = False) -> dict[str, Any]:
        """Evaluate due subjects once. Single-flight — concurrent cycles skip."""

        if not self._cycle_lock.acquire(blocking=False):
            return {"skipped": True, "reason": "already_running"}
        try:
            return self._run_once_locked(reason=reason, force=force)
        finally:
            self._cycle_lock.release()

    def _run_once_locked(self, *, reason: str, force: bool) -> dict[str, Any]:
        consent = self.get_consent()
        if not consent.enabled and reason not in {"manual", "on_demand"}:
            return {"skipped": True, "reason": "consent_required"}
        now = self._clock()
        if reason not in {"manual", "on_demand"} and not inside_monitoring_window(now):
            return {"skipped": True, "reason": "outside_window"}

        snapshot = self._load_portfolio()
        if snapshot is None:
            return {"skipped": True, "reason": "no_portfolio"}

        evaluated = 0
        material_alerts = 0
        failures = 0
        stock_strategy_keys = {
            (s.account_id, s.underlying.upper())
            for s in snapshot.strategies
            if is_stock_strategy(s)
        }

        # Portfolio-level strategic assessment.
        try:
            portfolio_fp = fingerprint_inputs(
                __import__(
                    "position_pilot.domain.recommendations",
                    fromlist=["portfolio_context"],
                ).portfolio_context(snapshot)
            )
            portfolio_rec = self.recommendations.get(
                SubjectType.PORTFOLIO,
                f"portfolio:{snapshot.selected_account_id}",
            )
            if force or strategic_due(
                portfolio_rec, now=now, material_event=False, fingerprint=portfolio_fp
            ):
                previous = portfolio_rec
                result = self.recommendations.evaluate_portfolio(snapshot, force=force)
                evaluated += 1
                if result.provider_status not in {
                    CodexProviderStatus.OK,
                    CodexProviderStatus.SKIPPED_UNCHANGED,
                }:
                    failures += 1
                    self.alerts.raise_provider_health(
                        provider=result.provider,
                        status=result.provider_status.value,
                        detail=result.error or result.provider_status.value,
                    )
                material_alerts += self._maybe_notify(result, previous=previous)
        except Exception:
            failures += 1
            logger.exception("portfolio evaluation failed")
            self.alerts.raise_provider_health(
                provider="recommendation-service",
                status="unavailable",
                detail="Portfolio recommendation evaluation failed",
            )

        for strategy in snapshot.strategies:
            try:
                catalysts = self.catalyst_loader(strategy.underlying)
                self._raise_catalyst_alerts(strategy.underlying, catalysts, strategy.strategy_type)
                material_event = any(bool(item.get("high_impact")) for item in catalysts)
                thesis = self.thesis_loader(strategy.strategy_id)
                plan = self.plan_loader(strategy.strategy_id)
                market = self._market_for(strategy.underlying)
                # Scheduled/on-demand reevaluation always uses current market context.
                self._remember_market_baseline(strategy.underlying, market, force=True)
                context = strategy_context(
                    strategy,
                    catalysts=catalysts,
                    thesis=thesis,
                    trade_plan=plan,
                    market=market,
                )
                fingerprint = fingerprint_inputs(context)
                previous = self.recommendations.get(SubjectType.STRATEGY, strategy.strategy_id)
                horizon = strategy.horizon
                if horizon == PositionHorizon.UNCLASSIFIED:
                    from .recommendations import default_horizon_for_strategy

                    horizon = default_horizon_for_strategy(strategy)

                if force:
                    due = True
                elif horizon == PositionHorizon.STRATEGIC:
                    due = strategic_due(
                        previous,
                        now=now,
                        material_event=material_event,
                        fingerprint=fingerprint,
                    )
                else:
                    due = tactical_due(
                        previous,
                        now=now,
                        fingerprint=fingerprint,
                        reevaluation_seconds=self.reevaluation_seconds,
                        material_event=material_event,
                    )
                if not due:
                    continue
                result = self.recommendations.evaluate_strategy(
                    strategy,
                    catalysts=catalysts,
                    thesis=thesis,
                    trade_plan=plan,
                    market=market,
                    force=force,
                )
                evaluated += 1
                if result.provider_status not in {
                    CodexProviderStatus.OK,
                    CodexProviderStatus.SKIPPED_UNCHANGED,
                }:
                    failures += 1
                    self.alerts.raise_provider_health(
                        provider=result.provider,
                        status=result.provider_status.value,
                        detail=result.error or result.provider_status.value,
                    )
                material_alerts += self._maybe_notify(result, previous=previous)
            except Exception:
                failures += 1
                logger.exception("strategy evaluation failed for %s", strategy.strategy_id)
                self.alerts.raise_provider_health(
                    provider="recommendation-service",
                    status="unavailable",
                    detail=f"Strategy evaluation failed for {strategy.underlying}",
                )

        # Standalone equities — skip when already covered as Long/Short Stock strategy.
        for account in snapshot.accounts:
            for position in account.positions:
                if position.position_type != PositionType.EQUITY:
                    continue
                key = (account.account_id, position.underlying_symbol.upper())
                if key in stock_strategy_keys:
                    continue
                subject_id = equity_subject_id(account.account_id, position.underlying_symbol)
                try:
                    catalysts = self.catalyst_loader(position.underlying_symbol)
                    self._raise_catalyst_alerts(
                        position.underlying_symbol.upper(),
                        catalysts,
                        "Long Stock",
                    )
                    material_event = any(bool(item.get("high_impact")) for item in catalysts)
                    from .recommendations import equity_context

                    market = self._market_for(position.underlying_symbol)
                    self._remember_market_baseline(
                        position.underlying_symbol, market, force=True
                    )
                    context = equity_context(
                        symbol=position.underlying_symbol.upper(),
                        quantity=position.quantity,
                        mark_price=position.mark_price,
                        unrealized_pnl=position.unrealized_pnl,
                        unrealized_pnl_percent=position.unrealized_pnl_percent,
                        catalysts=catalysts,
                        market=market,
                    )
                    fingerprint = fingerprint_inputs(context)
                    previous = self.recommendations.get(SubjectType.EQUITY, subject_id)
                    if not force and not strategic_due(
                        previous,
                        now=now,
                        material_event=material_event,
                        fingerprint=fingerprint,
                    ):
                        continue
                    result = self.recommendations.evaluate_equity(
                        subject_id=subject_id,
                        account_id=account.account_id,
                        symbol=position.underlying_symbol.upper(),
                        quantity=position.quantity,
                        mark_price=position.mark_price,
                        unrealized_pnl=position.unrealized_pnl,
                        unrealized_pnl_percent=position.unrealized_pnl_percent,
                        catalysts=catalysts,
                        market=market,
                        force=force,
                    )
                    evaluated += 1
                    if result.provider_status not in {
                        CodexProviderStatus.OK,
                        CodexProviderStatus.SKIPPED_UNCHANGED,
                    }:
                        failures += 1
                        self.alerts.raise_provider_health(
                            provider=result.provider,
                            status=result.provider_status.value,
                            detail=result.error or result.provider_status.value,
                        )
                    material_alerts += self._maybe_notify(result, previous=previous)
                except Exception:
                    failures += 1
                    logger.exception("equity evaluation failed for %s", subject_id)
                    self.alerts.raise_provider_health(
                        provider="recommendation-service",
                        status="unavailable",
                        detail=f"Equity evaluation failed for {position.underlying_symbol}",
                    )

        self._raise_risk_alerts(snapshot)
        self._last_evaluation_at = now
        if reason in {"wake", "network_recovery"}:
            self._last_recovery_evaluation_at = now
        self._persist_runtime(reason=reason, evaluated=evaluated)
        return {
            "skipped": False,
            "reason": reason,
            "evaluated": evaluated,
            "material_alerts": material_alerts,
            "failures": failures,
        }

    def risk_tick(self) -> dict[str, Any]:
        """60-second tactical pulse: live market inputs + material-change Codex only."""

        now = self._clock()
        if not self.get_consent().enabled or not inside_monitoring_window(now):
            return {"skipped": True}
        self._last_risk_tick_at = now
        self.database.set_setting("monitoring.last_risk_tick_at", now.isoformat())

        snapshot = self._load_portfolio()
        if snapshot is None:
            return {"skipped": False, "at": now.isoformat(), "evaluated": 0, "no_portfolio": True}

        evaluated = 0
        material_market_hits = 0

        for strategy in snapshot.strategies:
            try:
                horizon = strategy.horizon
                if horizon == PositionHorizon.UNCLASSIFIED:
                    from .recommendations import default_horizon_for_strategy

                    horizon = default_horizon_for_strategy(strategy)
                if horizon != PositionHorizon.TACTICAL:
                    continue
                catalysts = self.catalyst_loader(strategy.underlying)
                self._raise_catalyst_alerts(strategy.underlying, catalysts, strategy.strategy_type)
                material_event = any(bool(item.get("high_impact")) for item in catalysts)
                market = self._market_for(strategy.underlying)
                baseline = self._market_baselines.get(strategy.underlying.upper())
                material_market = is_material_market_change(baseline, market)
                # Pin the baseline until a material move occurs so cumulative drift
                # cannot evade the threshold through many small observations.
                self._remember_market_baseline(
                    strategy.underlying,
                    market,
                    force=material_market,
                )
                if not material_market and not material_event:
                    continue
                if material_market:
                    material_market_hits += 1

                thesis = self.thesis_loader(strategy.strategy_id)
                plan = self.plan_loader(strategy.strategy_id)
                context = strategy_context(
                    strategy,
                    catalysts=catalysts,
                    thesis=thesis,
                    trade_plan=plan,
                    market=market,
                )
                fingerprint = fingerprint_inputs(context)
                previous = self.recommendations.get(SubjectType.STRATEGY, strategy.strategy_id)
                # Pulse never uses pure 30m cadence — only material live/event input changes.
                if previous is not None and previous.input_fingerprint == fingerprint:
                    continue
                if not tactical_due(
                    previous,
                    now=now,
                    fingerprint=fingerprint,
                    reevaluation_seconds=self.reevaluation_seconds,
                    material_event=material_event or material_market,
                ):
                    continue
                result = self.recommendations.evaluate_strategy(
                    strategy,
                    catalysts=catalysts,
                    thesis=thesis,
                    trade_plan=plan,
                    market=market,
                    force=False,
                )
                evaluated += 1
                self._maybe_notify(result, previous=previous)
                if result.provider_status not in {
                    CodexProviderStatus.OK,
                    CodexProviderStatus.SKIPPED_UNCHANGED,
                }:
                    self.alerts.raise_provider_health(
                        provider=result.provider,
                        status=result.provider_status.value,
                        detail=result.error or result.provider_status.value,
                    )
            except Exception:
                logger.exception("tactical risk tick failed for %s", strategy.strategy_id)

        self._raise_risk_alerts(snapshot)
        self._persist_runtime()
        return {
            "skipped": False,
            "at": now.isoformat(),
            "evaluated": evaluated,
            "material_market_hits": material_market_hits,
        }

    def _market_for(self, symbol: str) -> dict[str, Any] | None:
        if self.market_context_loader is None:
            return None
        try:
            raw = self.market_context_loader(symbol)
        except Exception:
            logger.exception("market_context_loader failed for %s", symbol)
            return None
        return sanitize_market_context(raw) if raw else None

    def _remember_market_baseline(
        self,
        symbol: str,
        market: dict[str, Any] | None,
        *,
        force: bool,
    ) -> None:
        if not market:
            return
        key = symbol.upper()
        cleaned = sanitize_market_context(market)
        if force or key not in self._market_baselines:
            self._market_baselines[key] = cleaned

    def _load_portfolio(self) -> PortfolioSnapshot | None:
        if self.portfolio_loader is None:
            self._portfolio_unavailable = True
            self._network_available = False
            return None
        try:
            snapshot = self.portfolio_loader()
        except Exception:
            logger.exception("portfolio loader failed")
            was_available = not self._portfolio_unavailable
            self._portfolio_unavailable = True
            self._network_available = False
            if was_available:
                self.alerts.raise_provider_health(
                    provider="portfolio",
                    status="unavailable",
                    detail="Portfolio data became unavailable",
                )
            return None
        if snapshot is None:
            was_available = not self._portfolio_unavailable
            self._portfolio_unavailable = True
            self._network_available = False
            if was_available:
                self.alerts.raise_provider_health(
                    provider="portfolio",
                    status="unavailable",
                    detail="Portfolio snapshot missing",
                )
            return None
        recovered = self._portfolio_unavailable or not self._network_available
        self._portfolio_unavailable = False
        self._network_available = True
        if recovered:
            # Caller may still invoke on_network_recovery; flag only.
            pass
        return snapshot

    def _raise_catalyst_alerts(
        self,
        symbol: str,
        catalysts: list[dict[str, Any]],
        strategy_type: str | None,
    ) -> None:
        for item in catalysts:
            if not isinstance(item, dict) or not item.get("high_impact"):
                continue
            cat_id = str(
                item.get("catalyst_id") or item.get("id") or item.get("headline") or "unknown"
            )
            key = f"{symbol.upper()}:{cat_id}"
            if key in self._seen_catalyst_keys:
                continue
            self._seen_catalyst_keys.add(key)
            self.alerts.raise_alert(
                category=AlertCategory.CATALYST,
                severity=AlertSeverity.HIGH,
                alert_type="high_impact_catalyst",
                title=f"{symbol.upper()} high-impact catalyst",
                summary=str(item.get("headline") or item.get("summary") or "High-impact event")[
                    :500
                ],
                source="catalyst-monitor",
                symbol=symbol.upper(),
                strategy_type=strategy_type,
                payload={"catalyst_id": cat_id},
                dedupe_key=f"catalyst:{key}",
            )

    def _raise_risk_alerts(self, snapshot: PortfolioSnapshot) -> None:
        risk_payload: dict[str, Any] = {
            "total_delta": sum(s.total_delta for s in snapshot.strategies),
            "total_gamma": 0.0,
            "total_theta": sum(s.total_theta for s in snapshot.strategies),
            "unrealized_pnl": snapshot.totals.unrealized_pnl,
            "concentration_top_share": 0.0,
        }
        # Concentration from market values when available.
        totals_mv = 0.0
        by_symbol: dict[str, float] = {}
        for account in snapshot.accounts:
            for position in account.positions:
                symbol = position.underlying_symbol.upper()
                mv = abs(float(position.market_value or 0.0))
                by_symbol[symbol] = by_symbol.get(symbol, 0.0) + mv
                totals_mv += mv
        if totals_mv > 0 and by_symbol:
            risk_payload["concentration_top_share"] = max(by_symbol.values()) / totals_mv

        if self.risk_snapshot_loader is not None:
            try:
                extra = self.risk_snapshot_loader(snapshot) or {}
                for key in (
                    "total_delta",
                    "total_gamma",
                    "total_theta",
                    "total_vega",
                    "unrealized_pnl",
                    "concentration_top_share",
                ):
                    if key in extra and extra[key] is not None:
                        risk_payload[key] = extra[key]
            except Exception:
                logger.exception("risk snapshot loader failed")

        current_state = classify_risk_state(
            total_delta=float(risk_payload.get("total_delta") or 0),
            total_gamma=float(risk_payload.get("total_gamma") or 0),
            concentration_top_share=float(risk_payload.get("concentration_top_share") or 0),
            unrealized_pnl=float(risk_payload.get("unrealized_pnl") or 0),
        )
        previous_state = self._last_risk_state
        fingerprint = fingerprint_inputs(
            {**risk_payload, "state": current_state.value}
        )
        # Update tracked state even when we do not alert.
        emit = should_emit_risk_alert(previous_state, current_state)
        self._last_risk_state = current_state
        self._last_risk_fingerprint = fingerprint
        if not emit:
            return

        severity = {
            RiskLevelState.ELEVATED: AlertSeverity.WARNING,
            RiskLevelState.HIGH: AlertSeverity.HIGH,
            RiskLevelState.CRITICAL: AlertSeverity.CRITICAL,
            RiskLevelState.NORMAL: AlertSeverity.INFO,
        }[current_state]
        self.alerts.raise_alert(
            category=AlertCategory.RISK,
            severity=severity,
            alert_type=f"portfolio_risk_{current_state.value}",
            title=f"Portfolio risk {current_state.value}",
            summary=(
                f"state={current_state.value} · Δ {risk_payload.get('total_delta')} · "
                f"γ {risk_payload.get('total_gamma')} · "
                f"conc {float(risk_payload.get('concentration_top_share') or 0):.0%}"
            )[:500],
            source="risk-monitor",
            payload={
                "state": current_state.value,
                "previous_state": previous_state.value if previous_state else None,
            },
            dedupe_key=f"risk-state:{current_state.value}",
        )

    def _maybe_notify(self, result, *, previous) -> int:
        from .recommendations import is_notification_material

        if result.action is None or result.urgency is None or result.risk is None:
            return 0
        if result.provider_status not in {CodexProviderStatus.OK}:
            return 0
        if previous is not None and not is_notification_material(
            previous, result.action, result.urgency, result.risk
        ):
            return 0
        alert = self.alerts.raise_recommendation_change(
            action=result.action,
            urgency=result.urgency,
            risk=result.risk,
            account_id=result.account_id,
            symbol=result.symbol,
            strategy_type=result.strategy_type,
            subject_type=str(result.subject_type),
            subject_id=result.subject_id,
            recommendation_id=result.recommendation_id,
        )
        if alert is None:
            return 0
        rich = bool(self.recommendations.settings().get("rich_notification_preview"))
        self.notifications.notify_alert(alert, rich_preview=rich)
        return 1

    def _persist_runtime(self, *, reason: str | None = None, evaluated: int | None = None) -> None:
        state = {
            "last_evaluation_at": self._last_evaluation_at.isoformat()
            if self._last_evaluation_at
            else None,
            "last_recovery_evaluation_at": self._last_recovery_evaluation_at.isoformat()
            if self._last_recovery_evaluation_at
            else None,
            "last_risk_fingerprint": self._last_risk_fingerprint,
            "last_risk_state": self._last_risk_state.value if self._last_risk_state else None,
            "market_baselines": {
                symbol: payload for symbol, payload in list(self._market_baselines.items())[:200]
            },
            "seen_catalyst_keys": sorted(self._seen_catalyst_keys)[-200:],
            "last_reason": reason,
            "evaluated": evaluated,
        }
        self.database.set_setting("monitoring.runtime", state)

    async def tick_once(self) -> dict[str, Any]:
        """One scheduler iteration — testable wake/recovery/risk/eval seam.

        Preserves single-flight via run_once and never replays missed intervals:
        at most one wake recovery, one network recovery, one risk pulse, and one
        scheduled evaluation per tick.
        """

        consent = self.get_consent()
        now = self._clock()
        outcome: dict[str, Any] = {
            "wake": False,
            "recovery": False,
            "risk": None,
            "scheduled": None,
            "inside_window": inside_monitoring_window(now),
            "enabled": consent.enabled,
        }

        if (
            self._last_loop_tick_at is not None
            and consent.enabled
            and inside_monitoring_window(now)
        ):
            gap = (now - self._last_loop_tick_at).total_seconds()
            if gap >= WAKE_GAP_SECONDS:
                outcome["wake"] = True
                outcome["wake_result"] = await asyncio.to_thread(self.on_wake)
                self._next_eval_at = now + timedelta(seconds=self.reevaluation_seconds)
        self._last_loop_tick_at = now

        if consent.enabled and inside_monitoring_window(now):
            before_unavailable = self._portfolio_unavailable
            outcome["risk"] = await asyncio.to_thread(self.risk_tick)
            if before_unavailable and not self._portfolio_unavailable:
                outcome["recovery"] = True
                outcome["recovery_result"] = await asyncio.to_thread(self.on_network_recovery)
                self._next_eval_at = now + timedelta(seconds=self.reevaluation_seconds)

            due = self._next_eval_at is None or now >= self._next_eval_at
            if due:
                outcome["scheduled"] = await asyncio.to_thread(
                    self.run_once, reason="scheduled"
                )
                self._next_eval_at = now + timedelta(seconds=self.reevaluation_seconds)
            try:
                await asyncio.to_thread(self.database.ensure_daily_backup, now=now)
            except Exception:
                logger.exception("daily monitoring backup failed")
        else:
            self._next_eval_at = None
        return outcome

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick_once()
            except Exception:
                logger.exception("monitoring loop iteration failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.risk_interval_seconds)
            except TimeoutError:
                continue
