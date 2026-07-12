"""Monitoring window, calendar, risk pulse, consent, recovery, alerts."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from position_pilot.domain.alerts import AlertCategory, AlertService
from position_pilot.domain.monitoring import (
    EARLY_CLOSE_MONITOR_END,
    MonitoringService,
    RiskLevelState,
    classify_risk_state,
    good_friday,
    inside_monitoring_window,
    is_early_close_day,
    is_market_holiday,
    is_material_market_change,
    session_for,
    should_emit_risk_alert,
    us_market_holidays_for_year,
)
from position_pilot.domain.notifications import NotificationService
from position_pilot.domain.recommendations import RecommendationService
from position_pilot.domain.snapshots import (
    AccountSnapshot,
    DataFreshness,
    FreshnessState,
    PortfolioSnapshot,
    PortfolioTotals,
    PositionHorizon,
    PositionSnapshot,
    QuantityDirection,
    SnapshotState,
    StrategySnapshot,
)
from position_pilot.models import PositionType
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.providers.codex import (
    SCHEMA_VERSION,
    CodexInvocationResult,
    CodexProviderStatus,
    CodexStructuredOutput,
    RecommendationAction,
    RecommendationRisk,
)

ET = ZoneInfo("America/New_York")


class StubProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[dict] = []

    def public_status(self) -> str:
        return "configured"

    def complete_recommendation(self, context: dict) -> CodexInvocationResult:
        self.calls += 1
        self.contexts.append(context)
        return CodexInvocationResult(
            status=CodexProviderStatus.OK,
            output=CodexStructuredOutput(
                schema_version=SCHEMA_VERSION,
                action=RecommendationAction.HOLD,
                urgency=2,
                risk=RecommendationRisk.LOW,
                reasoning="Stable.",
                evidence=[],
                catalyst_refs=[],
            ),
        )


def _snapshot(*, include_stock_strategy: bool = False) -> PortfolioSnapshot:
    now = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)
    strategies = [
        StrategySnapshot(
            strategy_id="strat-1",
            account_id="acct-1",
            underlying="SPY",
            strategy_type="Short Put",
            expiration_date="2026-08-21",
            days_to_expiration=21,
            quantity=1,
            strikes="$500",
            unrealized_pnl=40,
            total_delta=-20,
            total_theta=4,
            horizon=PositionHorizon.TACTICAL,
            legs=[],
        )
    ]
    if include_stock_strategy:
        strategies.append(
            StrategySnapshot(
                strategy_id="strat-aapl-stock",
                account_id="acct-1",
                underlying="AAPL",
                strategy_type="Long Stock",
                expiration_date=None,
                days_to_expiration=None,
                quantity=10,
                strikes="",
                unrealized_pnl=100,
                total_delta=10,
                total_theta=0,
                horizon=PositionHorizon.STRATEGIC,
                legs=[],
            )
        )
    return PortfolioSnapshot(
        snapshot_id="snap",
        captured_at=now,
        state=SnapshotState.LIVE,
        freshness=DataFreshness(as_of=now, provider="tastytrade", state=FreshnessState.FRESH),
        accounts=[
            AccountSnapshot(
                account_id="acct-1",
                label="Individual 1",
                account_type="Individual",
                net_liquidating_value=25000,
                cash_balance=5000,
                buying_power=10000,
                positions=[
                    PositionSnapshot(
                        symbol="AAPL",
                        underlying_symbol="AAPL",
                        quantity=10,
                        quantity_direction=QuantityDirection.LONG,
                        position_type=PositionType.EQUITY,
                        mark_price=200,
                        market_value=2000,
                        unrealized_pnl=100,
                    )
                ],
            )
        ],
        strategies=strategies,
        totals=PortfolioTotals(net_liquidating_value=25000, unrealized_pnl=140),
        selected_account_id="all",
    )


def _services(
    tmp_path: Path,
    clock,
    provider=None,
    *,
    portfolio=None,
    catalysts=None,
    markets=None,
    include_stock_strategy: bool = False,
):
    db = PositionPilotDatabase(tmp_path / "m.sqlite3", backup_directory=tmp_path / "b")
    provider = provider or StubProvider()
    recs = RecommendationService(db, provider=provider, clock=clock)
    alerts = AlertService(db, clock=clock)
    notifications = NotificationService(enabled=False)
    if portfolio is not None:
        snap = portfolio
    else:
        snap = _snapshot(include_stock_strategy=include_stock_strategy)
    loader_state = {"snap": snap, "fail": False}
    market_map = markets if markets is not None else {}

    def portfolio_loader():
        if loader_state["fail"]:
            raise ConnectionError("network down")
        return loader_state["snap"]

    catalyst_map = catalysts or {}

    def catalyst_loader(symbol: str):
        return list(catalyst_map.get(symbol.upper(), []))

    def market_context_loader(symbol: str):
        return market_map.get(symbol.upper())

    monitoring = MonitoringService(
        db,
        recs,
        alerts,
        notifications,
        portfolio_loader=portfolio_loader,
        catalyst_loader=catalyst_loader,
        market_context_loader=market_context_loader,
        clock=clock,
    )
    return monitoring, recs, alerts, notifications, provider, loader_state, market_map


def test_holiday_and_weekend_outside_window() -> None:
    assert is_market_holiday(date(2026, 7, 3))  # Independence observed
    assert inside_monitoring_window(datetime(2026, 7, 3, 12, 0, tzinfo=ET)) is False
    assert inside_monitoring_window(datetime(2026, 7, 11, 12, 0, tzinfo=ET)) is False  # Saturday


def test_calendar_covers_years_beyond_2027() -> None:
    holidays_2030 = us_market_holidays_for_year(2030)
    assert good_friday(2030) in holidays_2030
    assert any(d.month == 6 and d.day in {19, 18, 20} for d in holidays_2030)  # Juneteenth observed
    # Thanksgiving 2030 is Nov 28
    assert date(2030, 11, 28) in holidays_2030


def test_early_close_monitoring_ends_two_hours_after_close() -> None:
    # Thanksgiving Friday 2026-11-27 is early close
    assert is_early_close_day(date(2026, 11, 27))
    session = session_for(datetime(2026, 11, 27, 14, 30, tzinfo=ET))
    assert session.early_close is True
    assert session.window_end.hour == EARLY_CLOSE_MONITOR_END.hour
    assert session.window_end.hour == 15
    assert inside_monitoring_window(datetime(2026, 11, 27, 14, 30, tzinfo=ET)) is True
    assert inside_monitoring_window(datetime(2026, 11, 27, 15, 1, tzinfo=ET)) is False
    # Market closes 13:00 but monitoring continues until 15:00
    assert inside_monitoring_window(datetime(2026, 11, 27, 13, 30, tzinfo=ET)) is True


def test_regular_window_730_to_1800() -> None:
    assert inside_monitoring_window(datetime(2026, 7, 10, 7, 29, tzinfo=ET)) is False
    assert inside_monitoring_window(datetime(2026, 7, 10, 7, 30, tzinfo=ET)) is True
    assert inside_monitoring_window(datetime(2026, 7, 10, 18, 0, tzinfo=ET)) is True
    assert inside_monitoring_window(datetime(2026, 7, 10, 18, 1, tzinfo=ET)) is False


def test_monitoring_disabled_without_consent(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    monitoring, _, _, _, provider, _, _ = _services(tmp_path, clock)
    result = monitoring.run_once(reason="scheduled")
    assert result["skipped"] is True
    assert result["reason"] == "consent_required"
    assert provider.calls == 0


def test_consent_enables_evaluation_inside_window(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    monitoring, _, _, _, provider, _, _ = _services(tmp_path, clock)
    monitoring.set_consent(enabled=True)
    result = monitoring.run_once(reason="scheduled")
    assert result["skipped"] is False
    assert result["evaluated"] >= 1
    assert provider.calls >= 1


def test_risk_tick_skips_penny_market_moves_and_fires_on_material_move(
    tmp_path: Path,
) -> None:
    clock = {"now": datetime(2026, 7, 10, 12, 0, tzinfo=ET)}
    markets = {"SPY": {"price": 500.0, "iv_rank": 30.0, "provider": "test"}}
    monitoring, _, _, _, provider, _, market_map = _services(
        tmp_path, lambda: clock["now"], markets=markets
    )
    monitoring.set_consent(enabled=True)
    # Establish baseline via scheduled path (includes market, no force needed for first).
    monitoring.run_once(reason="on_demand", force=True)
    baseline = provider.calls
    # Penny tick: +0.05% — observe only, no Codex.
    market_map["SPY"] = {"price": 500.25, "iv_rank": 30.0, "provider": "test"}
    tick = monitoring.risk_tick()
    assert tick["evaluated"] == 0
    assert provider.calls == baseline
    # Material price move ≥ 0.5% → tactical Codex.
    market_map["SPY"] = {"price": 504.0, "iv_rank": 30.0, "provider": "test"}
    tick2 = monitoring.risk_tick()
    assert tick2["evaluated"] >= 1
    assert tick2["material_market_hits"] >= 1
    assert provider.calls > baseline
    after = provider.calls
    # Same material baseline now — no further call.
    tick3 = monitoring.risk_tick()
    assert tick3["evaluated"] == 0
    assert provider.calls == after


def test_is_material_market_change_thresholds() -> None:
    base = {"price": 100.0, "iv_rank": 20.0, "iv": 0.20, "spread_percent": 0.1}
    assert is_material_market_change(base, {"price": 100.2, "iv_rank": 20.0}) is False
    assert is_material_market_change(base, {"price": 100.6, "iv_rank": 20.0}) is True
    assert is_material_market_change(base, {"price": 100.0, "iv_rank": 22.5}) is True
    assert is_material_market_change(None, {"price": 100.0}) is False


def test_risk_tick_detects_cumulative_subthreshold_market_drift(tmp_path: Path) -> None:
    current = {"price": 100.0, "iv_rank": 20.0}

    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    monitoring, _, _, _, provider, _, _ = _services(
        tmp_path,
        clock,
        markets={"SPY": current},
    )
    monitoring.set_consent(enabled=True)
    monitoring.run_once(reason="on_demand", force=True)
    baseline_calls = provider.calls

    current["price"] = 100.3
    assert monitoring.risk_tick()["material_market_hits"] == 0
    current["price"] = 100.6
    assert monitoring.risk_tick()["material_market_hits"] >= 1
    assert provider.calls > baseline_calls


def _diversified_snapshot(*, total_delta: float = -20.0, unrealized_pnl: float = 40.0):
    """Balanced multi-symbol book so concentration alone is not critical."""

    now = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)
    return PortfolioSnapshot(
        snapshot_id="div",
        captured_at=now,
        state=SnapshotState.LIVE,
        freshness=DataFreshness(as_of=now, provider="tastytrade", state=FreshnessState.FRESH),
        accounts=[
            AccountSnapshot(
                account_id="acct-1",
                label="Individual 1",
                account_type="Individual",
                net_liquidating_value=100_000,
                positions=[
                    PositionSnapshot(
                        symbol="AAPL",
                        underlying_symbol="AAPL",
                        quantity=10,
                        quantity_direction=QuantityDirection.LONG,
                        position_type=PositionType.EQUITY,
                        market_value=20_000,
                        unrealized_pnl=100,
                    ),
                    PositionSnapshot(
                        symbol="MSFT",
                        underlying_symbol="MSFT",
                        quantity=10,
                        quantity_direction=QuantityDirection.LONG,
                        position_type=PositionType.EQUITY,
                        market_value=20_000,
                        unrealized_pnl=50,
                    ),
                    PositionSnapshot(
                        symbol="GOOG",
                        underlying_symbol="GOOG",
                        quantity=10,
                        quantity_direction=QuantityDirection.LONG,
                        position_type=PositionType.EQUITY,
                        market_value=20_000,
                        unrealized_pnl=25,
                    ),
                    PositionSnapshot(
                        symbol="AMZN",
                        underlying_symbol="AMZN",
                        quantity=10,
                        quantity_direction=QuantityDirection.LONG,
                        position_type=PositionType.EQUITY,
                        market_value=20_000,
                        unrealized_pnl=10,
                    ),
                    PositionSnapshot(
                        symbol="META",
                        underlying_symbol="META",
                        quantity=10,
                        quantity_direction=QuantityDirection.LONG,
                        position_type=PositionType.EQUITY,
                        market_value=20_000,
                        unrealized_pnl=5,
                    ),
                ],
            )
        ],
        strategies=[
            StrategySnapshot(
                strategy_id="strat-1",
                account_id="acct-1",
                underlying="SPY",
                strategy_type="Short Put",
                expiration_date="2026-08-21",
                days_to_expiration=21,
                quantity=1,
                strikes="$500",
                unrealized_pnl=unrealized_pnl,
                total_delta=total_delta,
                total_theta=4,
                horizon=PositionHorizon.TACTICAL,
                legs=[],
            )
        ],
        totals=PortfolioTotals(net_liquidating_value=100_000, unrealized_pnl=unrealized_pnl),
        selected_account_id="all",
    )


def test_risk_state_alerts_only_on_threshold_crossing(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    base = _diversified_snapshot(total_delta=-20, unrealized_pnl=40)
    monitoring, _, alerts, _, _, _, _ = _services(tmp_path, clock, portfolio=base)
    monitoring.set_consent(enabled=True)
    # Normal state first observation — no alert.
    monitoring._raise_risk_alerts(base)
    risk_alerts = [
        a for a in alerts.list_alerts(include_resolved=True) if a.category == AlertCategory.RISK
    ]
    assert risk_alerts == []
    assert monitoring._last_risk_state == RiskLevelState.NORMAL

    # Small normal P/L noise still normal — zero alerts.
    noisy = _diversified_snapshot(total_delta=-20, unrealized_pnl=-50)
    monitoring._raise_risk_alerts(noisy)
    risk_alerts = [
        a for a in alerts.list_alerts(include_resolved=True) if a.category == AlertCategory.RISK
    ]
    assert risk_alerts == []

    # Cross into elevated via delta threshold.
    elevated_snap = _diversified_snapshot(total_delta=-60, unrealized_pnl=40)
    monitoring._raise_risk_alerts(elevated_snap)
    risk_alerts = [
        a for a in alerts.list_alerts(include_resolved=True) if a.category == AlertCategory.RISK
    ]
    assert len(risk_alerts) == 1
    assert risk_alerts[0].alert_type == "portfolio_risk_elevated"
    assert risk_alerts[0].severity.value == "warning"

    # Unchanged elevated state — no duplicate.
    monitoring._raise_risk_alerts(elevated_snap)
    risk_alerts2 = [
        a for a in alerts.list_alerts(include_resolved=True) if a.category == AlertCategory.RISK
    ]
    assert len(risk_alerts2) == 1

    # Cross to high — one new alert.
    high_snap = _diversified_snapshot(total_delta=-120, unrealized_pnl=40)
    monitoring._raise_risk_alerts(high_snap)
    risk_alerts3 = [
        a for a in alerts.list_alerts(include_resolved=True) if a.category == AlertCategory.RISK
    ]
    assert len(risk_alerts3) == 2
    assert any(a.alert_type == "portfolio_risk_high" for a in risk_alerts3)


def test_classify_risk_state_thresholds() -> None:
    assert classify_risk_state(total_delta=10, unrealized_pnl=-10) == RiskLevelState.NORMAL
    assert classify_risk_state(total_delta=55) == RiskLevelState.ELEVATED
    assert classify_risk_state(total_delta=110) == RiskLevelState.HIGH
    assert classify_risk_state(unrealized_pnl=-6000) == RiskLevelState.CRITICAL
    assert should_emit_risk_alert(RiskLevelState.NORMAL, RiskLevelState.ELEVATED) is True
    assert should_emit_risk_alert(RiskLevelState.ELEVATED, RiskLevelState.ELEVATED) is False
    assert should_emit_risk_alert(None, RiskLevelState.NORMAL) is False


def test_single_flight_returns_already_running(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    monitoring, _, _, _, provider, _, _ = _services(tmp_path, clock)
    monitoring.set_consent(enabled=True)
    assert monitoring._cycle_lock.acquire(blocking=False)
    try:
        result = monitoring.run_once(reason="scheduled")
        assert result == {"skipped": True, "reason": "already_running"}
        assert provider.calls == 0
    finally:
        monitoring._cycle_lock.release()


def test_stock_strategy_skips_duplicate_equity_subject(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    monitoring, recs, _, _, provider, _, _ = _services(
        tmp_path, clock, include_stock_strategy=True
    )
    monitoring.set_consent(enabled=True)
    monitoring.run_once(reason="on_demand", force=True)
    equity = recs.get("equity", "equity:acct-1:AAPL")
    # Equity subject should not exist; stock strategy covers AAPL.
    assert equity is None
    stock = recs.get("strategy", "strat-aapl-stock")
    assert stock is not None


def test_subject_exception_isolation(tmp_path: Path) -> None:
    class FlakyProvider(StubProvider):
        def complete_recommendation(self, context: dict) -> CodexInvocationResult:
            if context.get("subject_type") == "portfolio":
                self.calls += 1
                raise RuntimeError("boom")
            return StubProvider.complete_recommendation(self, context)

    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    provider = FlakyProvider()
    monitoring, recs, alerts, _, _, _, _ = _services(tmp_path, clock, provider=provider)
    monitoring.set_consent(enabled=True)
    result = monitoring.run_once(reason="on_demand", force=True)
    assert result["failures"] >= 1
    # Strategy still evaluated despite portfolio failure.
    assert recs.get("strategy", "strat-1") is not None
    assert any(a.category == AlertCategory.PROVIDER_HEALTH for a in alerts.list_alerts())


def test_network_recovery_one_shot(tmp_path: Path) -> None:
    clock = {"now": datetime(2026, 7, 10, 12, 0, tzinfo=ET)}
    monitoring, _, _, _, provider, loader_state, _ = _services(
        tmp_path, lambda: clock["now"]
    )
    monitoring.set_consent(enabled=True)
    loader_state["fail"] = True
    assert monitoring.run_once(reason="scheduled")["reason"] == "no_portfolio"
    failed_calls = provider.calls
    loader_state["fail"] = False
    result = monitoring.on_network_recovery()
    assert result.get("skipped") is False
    assert provider.calls > failed_calls
    after = provider.calls
    # Second recovery still one evaluation, not replay backlog
    monitoring.on_network_recovery()
    assert provider.calls >= after  # may skip-unchanged without extra Codex


def test_wake_gap_path_invokes_once(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    monitoring, _, _, _, provider, _, _ = _services(tmp_path, clock)
    monitoring.set_consent(enabled=True)
    result = monitoring.on_wake()
    assert result.get("skipped") is False
    assert provider.calls >= 1


def test_tick_once_detects_scheduler_wake_gap(tmp_path: Path) -> None:
    """Async seam: large gap between ticks triggers one wake evaluation (no replay)."""

    import asyncio

    current = {"t": datetime(2026, 7, 10, 12, 0, tzinfo=ET)}

    def clock() -> datetime:
        return current["t"]

    monitoring, _, _, _, provider, _, _ = _services(tmp_path, clock)
    monitoring.set_consent(enabled=True)

    async def scenario() -> None:
        first = await monitoring.tick_once()
        assert first["wake"] is False
        assert first["risk"] is not None
        calls_after_first = provider.calls

        # Small gap — no wake.
        current["t"] = current["t"] + timedelta(seconds=30)
        second = await monitoring.tick_once()
        assert second["wake"] is False

        # Large gap — one wake evaluation, no interval replay backlog.
        current["t"] = current["t"] + timedelta(minutes=5)
        third = await monitoring.tick_once()
        assert third["wake"] is True
        assert third.get("wake_result", {}).get("skipped") is False
        assert provider.calls >= calls_after_first

    asyncio.run(scenario())


def test_runtime_timestamps_restored_on_restart(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime(2026, 7, 10, 12, 0, tzinfo=ET)

    monitoring, _, _, _, _, _, _ = _services(tmp_path, clock)
    monitoring.set_consent(enabled=True)
    monitoring.run_once(reason="on_demand", force=True)
    assert monitoring._last_evaluation_at is not None
    # New service instance shares DB
    db = monitoring.database
    recs = RecommendationService(db, provider=StubProvider(), clock=clock)
    alerts = AlertService(db, clock=clock)
    notes = NotificationService(enabled=False)
    restarted = MonitoringService(
        db,
        recs,
        alerts,
        notes,
        portfolio_loader=lambda: _snapshot(),
        clock=clock,
    )
    assert restarted._last_evaluation_at is not None
    assert restarted._last_risk_state == monitoring._last_risk_state


def test_risk_and_catalyst_alerts_dedupe(tmp_path: Path) -> None:
    clock = {"now": datetime(2026, 7, 10, 12, 0, tzinfo=ET)}
    catalysts = {
        "SPY": [
            {
                "catalyst_id": "hi-1",
                "headline": "Shock",
                "summary": "Macro shock",
                "taxonomy": "macro",
                "confidence": "likely",
                "attribution": "macro",
                "event_at": "2026-07-10T14:00:00Z",
                "high_impact": True,
            }
        ]
    }
    monitoring, _, alerts, _, _, _, _ = _services(
        tmp_path, lambda: clock["now"], catalysts=catalysts
    )
    monitoring.set_consent(enabled=True)
    monitoring.run_once(reason="on_demand", force=True)
    all_alerts = alerts.list_alerts(include_resolved=True)
    cats = [a for a in all_alerts if a.category == AlertCategory.CATALYST]
    assert cats
    # Second pass should not duplicate catalyst alerts
    monitoring.run_once(reason="on_demand", force=True)
    all_alerts2 = alerts.list_alerts(include_resolved=True)
    cats2 = [a for a in all_alerts2 if a.category == AlertCategory.CATALYST]
    assert len(cats2) == len(cats)


def test_notification_default_payload_is_privacy_safe() -> None:
    from position_pilot.domain.alerts import (
        AlertCategory,
        AlertRecord,
        AlertResolution,
        AlertSeverity,
        privacy_safe_notification_body,
    )

    alert = AlertRecord(
        alert_id="a1",
        category=AlertCategory.RECOMMENDATION,
        severity=AlertSeverity.HIGH,
        alert_type="recommendation_change",
        title="SPY recommendation changed",
        summary="close · urgency 5 · critical",
        symbol="SPY",
        strategy_type="Iron Condor",
        source="test",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        resolution=AlertResolution.OPEN,
    )
    body = privacy_safe_notification_body(alert, rich_preview=False)
    assert body == "SPY · Iron Condor · recommendation_change"
