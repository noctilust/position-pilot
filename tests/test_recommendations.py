"""Recommendation fingerprints, prompt minimization, history, decisions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from position_pilot.domain.recommendations import (
    HistoryEntryKind,
    RecommendationService,
    SubjectType,
    TraderDecisionKind,
    equity_context,
    fingerprint_inputs,
    is_notification_material,
    strategic_due,
    strategy_context,
    tactical_due,
)
from position_pilot.domain.snapshots import (
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
    ExplicitApiKeyFallbackProvider,
    RecommendationAction,
    RecommendationRisk,
)

FIXED = datetime(2026, 7, 11, 15, 0, tzinfo=UTC)


class StubProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.status = CodexProviderStatus.OK
        self.action = RecommendationAction.HOLD
        self.urgency = 2
        self.risk = RecommendationRisk.LOW
        self.reasoning = "Stable premium decay."
        self.evidence = ["theta"]
        self.catalyst_refs: list[str] = []

    def public_status(self) -> str:
        return "configured"

    def complete_recommendation(self, context: dict) -> CodexInvocationResult:
        self.calls += 1
        if self.status != CodexProviderStatus.OK:
            return CodexInvocationResult(status=self.status, error="forced failure")
        return CodexInvocationResult(
            status=CodexProviderStatus.OK,
            output=CodexStructuredOutput(
                schema_version=SCHEMA_VERSION,
                action=self.action,
                urgency=self.urgency,
                risk=self.risk,
                reasoning=self.reasoning,
                evidence=list(self.evidence),
                catalyst_refs=list(self.catalyst_refs),
            ),
        )


def _db(tmp_path: Path) -> PositionPilotDatabase:
    return PositionPilotDatabase(tmp_path / "test.sqlite3", backup_directory=tmp_path / "backups")


def _strategy(**overrides) -> StrategySnapshot:
    base = dict(
        strategy_id="strat-1",
        account_id="acct-1",
        underlying="SPY",
        strategy_type="Iron Condor",
        expiration_date="2026-08-21",
        days_to_expiration=21,
        quantity=1,
        strikes="500/505/480/475",
        unrealized_pnl=80.0,
        unrealized_pnl_percent=12.0,
        total_delta=-5.0,
        total_theta=12.0,
        horizon=PositionHorizon.TACTICAL,
        legs=[
            PositionSnapshot(
                symbol="SPY260821P00480000",
                underlying_symbol="SPY",
                quantity=-1,
                quantity_direction=QuantityDirection.SHORT,
                position_type=PositionType.EQUITY_OPTION,
                strike_price=480,
                option_type="P",
                expiration_date="2026-08-21",
                days_to_expiration=21,
                mark_price=1.2,
                market_value=120,
                unrealized_pnl=40,
                delta=-0.12,
                theta=0.05,
            )
        ],
    )
    base.update(overrides)
    return StrategySnapshot(**base)


def test_strategy_context_allowlists_and_redacts_sensitive_fields() -> None:
    strategy = _strategy(account_id="5WT00001", strikes="500/505/480/475")
    context = strategy_context(
        strategy,
        catalysts=[
            {
                "catalyst_id": "c1",
                "headline": "Earnings beat",
                "summary": "Beat estimates",
                "taxonomy": "earnings",
                "confidence": "likely",
                "attribution": "company",
                "event_at": "2026-07-11T12:00:00Z",
                "full_text": "should not appear",
                "account_number": "5WT00001",
                "provider_payload": {"raw": True},
                "high_impact": True,
            }
        ],
        thesis={
            "strategy_id": "strat-1",
            "account_id": "acct-deadbeef",
            "purpose": "Income",
            "updated_at": "2026-07-01T00:00:00Z",
            "invalidation": "Break below support",
        },
        trade_plan={
            "strategy_id": "strat-1",
            "profit_target": "50%",
            "max_loss": "2x credit",
            "entry_thesis": "Sell premium",
        },
        portfolio_exposure={
            "net_delta": -10,
            "account_number": "5WT00001",
            "token": "secret",
            "concentration": [{"symbol": "SPY", "market_value": 1000}],
        },
        market={
            "price": 500.25,
            "iv_rank": 42.0,
            "provider_payload": {"raw": True},
            "token": "sk-not-this",
        },
    )
    blob = str(context)
    # Exact strike chain and normal analytical text must survive.
    assert context["strikes"] == "500/505/480/475"
    assert context["dte"] == 21
    assert context["legs"][0]["strike"] == 480
    assert context["market"]["price"] == 500.25
    assert context["market"]["iv_rank"] == 42.0
    assert "provider_payload" not in context["market"]
    # Realistic broker account identifiers / secrets redacted or dropped.
    assert "5WT00001" not in blob
    assert "account_number" not in blob
    assert "account_id" not in blob
    assert "strategy_id" not in blob
    assert "full_text" not in blob
    assert "provider_payload" not in blob
    assert "updated_at" not in blob
    assert "token" not in blob
    assert "sk-not-this" not in blob
    assert context["catalysts"][0]["headline"] == "Earnings beat"
    assert context["thesis"]["purpose"] == "Income"
    assert context["trade_plan"]["profit_target"] == "50%"
    assert context["portfolio_exposure"]["net_delta"] == -10


def test_equity_context_redacts_forbidden_keys() -> None:
    context = equity_context(
        symbol="AAPL",
        quantity=10,
        mark_price=200,
        unrealized_pnl=50,
        unrealized_pnl_percent=1.0,
        catalysts=[{"id": "x", "headline": "H", "password": "nope", "summary": "S"}],
        portfolio_exposure={"net_theta": 1, "refresh_token": "abc"},
    )
    assert "password" not in str(context)
    assert "refresh_token" not in str(context)
    assert context["catalysts"][0]["headline"] == "H"


def test_fingerprint_is_deterministic() -> None:
    payload = {"symbol": "SPY", "quantity": 1, "delta": -0.1}
    assert fingerprint_inputs(payload) == fingerprint_inputs(dict(reversed(list(payload.items()))))


def test_unchanged_fingerprint_skips_codex_call(tmp_path: Path) -> None:
    provider = StubProvider()
    service = RecommendationService(_db(tmp_path), provider=provider, clock=lambda: FIXED)
    strategy = _strategy()
    first = service.evaluate_strategy(strategy)
    assert provider.calls == 1
    second = service.evaluate_strategy(strategy)
    assert provider.calls == 1
    assert second.provider_status == CodexProviderStatus.SKIPPED_UNCHANGED
    assert second.last_evaluated_at == FIXED
    assert second.recommendation_updated_at == first.recommendation_updated_at


def test_material_event_does_not_bypass_fingerprint_skip(tmp_path: Path) -> None:
    provider = StubProvider()
    service = RecommendationService(_db(tmp_path), provider=provider, clock=lambda: FIXED)
    strategy = _strategy()
    service.evaluate_strategy(strategy)
    # Sticky high-impact flag must not force Codex when inputs identical.
    service.evaluate_strategy(strategy, material_event=True, force=False)
    assert provider.calls == 1


def test_reasoning_change_creates_audit_history_without_notification_material(
    tmp_path: Path,
) -> None:
    provider = StubProvider()
    clock = {"now": FIXED}
    service = RecommendationService(_db(tmp_path), provider=provider, clock=lambda: clock["now"])
    strategy = _strategy()
    first = service.evaluate_strategy(strategy)
    provider.reasoning = "Updated wording only; still hold."
    clock["now"] = FIXED + timedelta(hours=1)
    second = service.evaluate_strategy(strategy, force=True)
    assert provider.calls == 2
    assert second.action == first.action
    assert second.urgency == first.urgency
    assert second.risk == first.risk
    # recommendation_updated_at stays when only wording changes
    assert second.recommendation_updated_at == first.recommendation_updated_at
    history = service.history(SubjectType.STRATEGY, strategy.strategy_id)
    audit = [h for h in history if h.kind == HistoryEntryKind.AUDIT_CHANGE]
    assert audit
    assert "reasoning" in audit[0].diff
    assert is_notification_material(first, second.action, second.urgency, second.risk) is False


def test_material_action_change_updates_timestamp(tmp_path: Path) -> None:
    provider = StubProvider()
    clock = {"now": FIXED}
    service = RecommendationService(_db(tmp_path), provider=provider, clock=lambda: clock["now"])
    strategy = _strategy()
    first = service.evaluate_strategy(strategy)
    provider.action = RecommendationAction.ROLL
    provider.urgency = 4
    provider.risk = RecommendationRisk.HIGH
    provider.reasoning = "Roll for credit before gamma week."
    clock["now"] = FIXED + timedelta(hours=1)
    second = service.evaluate_strategy(strategy, force=True)
    assert second.recommendation_updated_at == clock["now"]
    assert second.recommendation_updated_at != first.recommendation_updated_at
    history = service.history(SubjectType.STRATEGY, strategy.strategy_id)
    assert any(entry.kind == HistoryEntryKind.MATERIAL_CHANGE for entry in history)


def test_unchanged_evaluations_collapse_to_daily_summary(tmp_path: Path) -> None:
    provider = StubProvider()
    service = RecommendationService(_db(tmp_path), provider=provider, clock=lambda: FIXED)
    strategy = _strategy()
    service.evaluate_strategy(strategy)
    service.evaluate_strategy(strategy)
    service.evaluate_strategy(strategy)
    history = service.history(SubjectType.STRATEGY, strategy.strategy_id)
    summaries = [entry for entry in history if entry.kind == HistoryEntryKind.DAILY_SUMMARY]
    assert len(summaries) == 1
    assert summaries[0].evaluation_count >= 2


def test_trader_decisions_are_immutable_and_separate(tmp_path: Path) -> None:
    provider = StubProvider()
    service = RecommendationService(_db(tmp_path), provider=provider, clock=lambda: FIXED)
    record = service.evaluate_strategy(_strategy())
    decision = service.record_decision(
        recommendation_id=record.recommendation_id,
        decision=TraderDecisionKind.ACCEPTED,
        note="Will manage in Tastytrade tomorrow",
    )
    assert decision.decision == TraderDecisionKind.ACCEPTED
    listed = service.list_decisions(SubjectType.STRATEGY, "strat-1")
    assert len(listed) == 1
    current = service.get(SubjectType.STRATEGY, "strat-1")
    assert current is not None
    assert current.action == RecommendationAction.HOLD


def test_provider_failure_is_explicit_never_silent_fallback(tmp_path: Path) -> None:
    provider = StubProvider()
    provider.status = CodexProviderStatus.SIGNED_OUT
    service = RecommendationService(
        _db(tmp_path),
        provider=provider,
        fallback_provider=ExplicitApiKeyFallbackProvider(),
        clock=lambda: FIXED,
    )
    record = service.evaluate_strategy(_strategy())
    assert record.provider_status == CodexProviderStatus.SIGNED_OUT
    assert record.action is None


def test_settings_never_enable_api_key_fallback(tmp_path: Path) -> None:
    service = RecommendationService(_db(tmp_path), provider=StubProvider(), clock=lambda: FIXED)
    updated = service.update_settings(
        {
            "api_key_fallback_enabled": True,
            "selected_provider": "api-key-fallback",
            "rich_notification_preview": True,
        }
    )
    assert updated["selected_provider"] == "codex-cli"
    assert updated["api_key_fallback_enabled"] is False
    assert updated["api_key_fallback_available"] is False
    assert updated["rich_notification_preview"] is True


def test_horizon_cadence_helpers() -> None:
    assert strategic_due(None, now=FIXED, material_event=False) is True
    rec = type(
        "R",
        (),
        {
            "last_evaluated_at": FIXED,
            "input_fingerprint": "abc",
            "provider_status": CodexProviderStatus.OK,
        },
    )()
    later = FIXED + timedelta(hours=2)
    assert strategic_due(rec, now=later, material_event=True, fingerprint="abc") is False
    assert strategic_due(rec, now=later, material_event=True, fingerprint="new") is True
    ten_min = FIXED + timedelta(minutes=10)
    assert tactical_due(rec, now=ten_min, fingerprint="abc", material_event=True) is False
    assert tactical_due(rec, now=FIXED + timedelta(minutes=31), fingerprint="abc") is True
    assert tactical_due(rec, now=FIXED + timedelta(minutes=1), fingerprint="changed") is True


def test_schema_version_is_six(tmp_path: Path) -> None:
    assert _db(tmp_path).schema_version == 6


def test_portfolio_evaluation(tmp_path: Path) -> None:
    provider = StubProvider()
    service = RecommendationService(_db(tmp_path), provider=provider, clock=lambda: FIXED)
    snapshot = PortfolioSnapshot(
        snapshot_id="snap-1",
        captured_at=FIXED,
        state=SnapshotState.LIVE,
        freshness=DataFreshness(as_of=FIXED, provider="tastytrade", state=FreshnessState.FRESH),
        accounts=[],
        strategies=[_strategy()],
        totals=PortfolioTotals(net_liquidating_value=10000, unrealized_pnl=80),
        selected_account_id="all",
    )
    record = service.evaluate_portfolio(snapshot)
    assert record.subject_type == SubjectType.PORTFOLIO
    assert provider.calls == 1
