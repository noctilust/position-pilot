"""Offline Tasty mechanics: playbook, facts, rules, candidates, settings."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from position_pilot.domain.mechanics import (
    PLAYBOOK_ID_V1,
    CandidateKind,
    MechanicsService,
    MechanicsSettings,
    PnlHistoryQuality,
    RiskClass,
    RuleStatus,
    compact_mechanics_context,
    evaluate_mechanics,
    get_playbook,
    parse_plan_profit_target_pct,
)
from position_pilot.domain.recommendations import (
    sanitize_mechanics_context,
    strategy_context,
)
from position_pilot.domain.risk import RiskService
from position_pilot.domain.snapshots import (
    PositionHorizon,
    PositionSnapshot,
    QuantityDirection,
    StrategySnapshot,
)
from position_pilot.models import PositionType
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.providers.codex import build_recommendation_prompt

FIXED = datetime(2026, 7, 18, 15, 0, tzinfo=UTC)


def _leg(
    *,
    symbol: str,
    underlying: str = "SPY",
    quantity: int = 1,
    direction: QuantityDirection = QuantityDirection.SHORT,
    strike: float | None = 500,
    option_type: str | None = "P",
    mark: float | None = 1.5,
    market_value: float = 150,
    cost_basis: float = 300,
    unrealized_pnl: float = 150,
    delta: float | None = -0.18,
    dte: int | None = 30,
    roll_history_status: str = "none",
    lifetime_net_credit: float | None = None,
    pnl_open: float | None = None,
    multiplier: int = 100,
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        underlying_symbol=underlying,
        quantity=quantity,
        quantity_direction=direction,
        position_type=PositionType.EQUITY_OPTION if strike is not None else PositionType.EQUITY,
        strike_price=strike,
        option_type=option_type,
        expiration_date="2026-08-21" if strike is not None else None,
        days_to_expiration=dte,
        mark_price=mark,
        market_value=market_value,
        cost_basis=cost_basis,
        unrealized_pnl=unrealized_pnl,
        pnl_open=pnl_open if pnl_open is not None else unrealized_pnl,
        delta=delta,
        multiplier=multiplier,
        roll_history_status=roll_history_status,
        lifetime_net_credit=lifetime_net_credit,
        horizon=PositionHorizon.TACTICAL,
    )


def _bull_put(
    *,
    dte: int = 30,
    short_mark: float = 1.5,
    long_mark: float = 0.5,
    short_cost: float = 300,
    long_cost: float = 100,
    short_pnl: float = 150,
    long_pnl: float = 50,
    short_delta: float = -0.18,
    strategy_id: str = "strat-bps-1",
    roll_history_status: str = "none",
    lifetime_short: float | None = None,
    lifetime_long: float | None = None,
) -> StrategySnapshot:
    short = _leg(
        symbol="SPY  260821P00500000",
        strike=500,
        option_type="P",
        direction=QuantityDirection.SHORT,
        mark=short_mark,
        market_value=short_mark * 100,
        cost_basis=short_cost,
        unrealized_pnl=short_pnl,
        delta=short_delta,
        dte=dte,
        roll_history_status=roll_history_status,
        lifetime_net_credit=lifetime_short,
    )
    long = _leg(
        symbol="SPY  260821P00490000",
        strike=490,
        option_type="P",
        direction=QuantityDirection.LONG,
        mark=long_mark,
        market_value=long_mark * 100,
        cost_basis=long_cost,
        unrealized_pnl=long_pnl,
        delta=-0.08,
        dte=dte,
        roll_history_status=roll_history_status,
        lifetime_net_credit=lifetime_long,
    )
    pnl = short_pnl + long_pnl
    return StrategySnapshot(
        strategy_id=strategy_id,
        account_id="acct-public-1",
        underlying="SPY",
        strategy_type="Bull Put Spread",
        expiration_date="2026-08-21",
        days_to_expiration=dte,
        quantity=1,
        strikes="$490/$500",
        unrealized_pnl=pnl,
        pnl_open=pnl,
        pnl_open_basis=abs(short_cost) - abs(long_cost) if short_cost else None,
        total_delta=-0.1,
        total_theta=0.05,
        horizon=PositionHorizon.TACTICAL,
        legs=[short, long],
    )


def _short_strangle(*, dte: int = 21, short_put_delta: float = -0.35) -> StrategySnapshot:
    put = _leg(
        symbol="SPY  260821P00480000",
        strike=480,
        option_type="P",
        direction=QuantityDirection.SHORT,
        mark=2.0,
        cost_basis=400,
        unrealized_pnl=100,
        delta=short_put_delta,
        dte=dte,
    )
    call = _leg(
        symbol="SPY  260821C00520000",
        strike=520,
        option_type="C",
        direction=QuantityDirection.SHORT,
        mark=1.8,
        cost_basis=350,
        unrealized_pnl=80,
        delta=0.12,
        dte=dte,
    )
    return StrategySnapshot(
        strategy_id="strat-strangle-1",
        account_id="acct-public-1",
        underlying="SPY",
        strategy_type="Short Strangle",
        expiration_date="2026-08-21",
        days_to_expiration=dte,
        quantity=1,
        strikes="$480/$520",
        unrealized_pnl=180,
        pnl_open=180,
        total_delta=-0.2,
        total_theta=0.1,
        horizon=PositionHorizon.TACTICAL,
        legs=[put, call],
    )


def test_playbook_v1_sources_and_rules() -> None:
    playbook = get_playbook(PLAYBOOK_ID_V1)
    assert playbook.playbook_id == PLAYBOOK_ID_V1
    assert playbook.version == "v1"
    assert len(playbook.sources) >= 8
    assert all(source.url.startswith("https://") for source in playbook.sources)
    # No article body fields
    for source in playbook.sources:
        dumped = source.model_dump()
        assert "body" not in dumped
        assert "text" not in dumped
    rule_ids = {rule.rule_id for rule in playbook.rules}
    assert "profit.manage_winner" in rule_ids
    assert "time.manage_at_dte" in rule_ids
    assert "size.small_position" in rule_ids
    assert "tested.side_review" in rule_ids
    assert "roll.credit_only" in rule_ids


def test_eligible_short_premium_winner_at_profit_target() -> None:
    strategy = _bull_put(short_pnl=120, long_pnl=40, short_cost=300, long_cost=100)
    # Net credit proxy ~200; pnl 160 → 80% capture
    evaluation = evaluate_mechanics(
        strategy,
        market={"price": 510.0, "spread_percent": 2.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        settings=MechanicsSettings(profit_target_pct=0.50),
        now=FIXED,
    )
    profit = next(rule for rule in evaluation.rules if rule.rule_id == "profit.manage_winner")
    assert profit.status == RuleStatus.DUE
    assert evaluation.facts.profit_capture_ratio is not None
    assert evaluation.facts.profit_capture_ratio >= 0.50
    # Option/complex liquidity always unknown today → close is blocked/manual-review.
    assert not any(
        c.kind == CandidateKind.CLOSE and not c.blocking_reasons for c in evaluation.candidates
    )
    assert any(
        c.kind == CandidateKind.MANUAL_REVIEW and "profit" in c.candidate_id
        for c in evaluation.candidates
    )
    assert all(c.kind.value != "add" for c in evaluation.candidates)


def test_21_dte_time_management() -> None:
    strategy = _bull_put(dte=21, short_pnl=20, long_pnl=10, short_cost=300, long_cost=100)
    evaluation = evaluate_mechanics(
        strategy,
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        settings=MechanicsSettings(manage_at_dte=21),
        now=FIXED,
    )
    time_rule = next(rule for rule in evaluation.rules if rule.rule_id == "time.manage_at_dte")
    assert time_rule.status == RuleStatus.DUE
    assert time_rule.reason_code == "dte_threshold_crossed"


def test_undefined_vs_defined_risk() -> None:
    defined = evaluate_mechanics(
        _bull_put(dte=21),
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    undefined = evaluate_mechanics(
        _short_strangle(dte=21),
        market={"price": 500.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert defined.facts.risk_class == RiskClass.DEFINED
    assert undefined.facts.risk_class == RiskClass.UNDEFINED
    undef_time = next(r for r in undefined.rules if r.rule_id == "time.manage_at_dte")
    assert "undefined" in undef_time.explanation.lower() or undef_time.status == RuleStatus.DUE
    # Undefined DTE management prefers manual-review over blind close
    undef_dte_cands = [c for c in undefined.candidates if "dte" in c.candidate_id]
    assert undef_dte_cands
    assert any(c.kind == CandidateKind.MANUAL_REVIEW for c in undef_dte_cands)


def test_tested_vs_untested_side() -> None:
    tested = evaluate_mechanics(
        _short_strangle(dte=30, short_put_delta=-0.40),
        market={"price": 485.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        settings=MechanicsSettings(tested_delta_threshold=0.30),
        now=FIXED,
    )
    assert tested.facts.tested_side == "put"
    tested_rule = next(r for r in tested.rules if r.rule_id == "tested.side_review")
    assert tested_rule.status == RuleStatus.DUE
    assert any(c.kind == CandidateKind.ROLL_REVIEW for c in tested.candidates)
    roll = next(c for c in tested.candidates if c.kind == CandidateKind.ROLL_REVIEW)
    assert "roll_economics_unknown" in roll.blocking_reasons
    assert roll.after_risk is not None
    assert "unknown" in (roll.after_risk.note or "").lower()

    untested = evaluate_mechanics(
        _bull_put(dte=30, short_delta=-0.12),
        market={"price": 510.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert untested.facts.tested_side == "untested"


def test_incomplete_roll_history() -> None:
    strategy = _bull_put(
        roll_history_status="partial",
        lifetime_short=None,
        lifetime_long=None,
        short_pnl=100,
        long_pnl=40,
    )
    evaluation = evaluate_mechanics(
        strategy,
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert evaluation.facts.pnl_history_quality == PnlHistoryQuality.PARTIAL
    assert "partial_roll_history" in evaluation.facts.data_quality_flags
    assert evaluation.facts.original_credit is None
    profit = next(r for r in evaluation.rules if r.rule_id == "profit.manage_winner")
    assert profit.status == RuleStatus.BLOCKED


def test_missing_opening_credit_and_quote() -> None:
    strategy = _bull_put(short_cost=0, long_cost=0, short_pnl=10, long_pnl=5)
    # Zero cost basis → no credit
    for leg in strategy.legs:
        leg.cost_basis = 0
    evaluation = evaluate_mechanics(
        strategy,
        market=None,
        account_nlv=None,
        now=FIXED,
    )
    assert evaluation.facts.original_credit is None
    assert "missing_opening_credit" in evaluation.facts.data_quality_flags
    flags = evaluation.facts.data_quality_flags
    assert "missing_underlying_quote" in flags or "missing_spot" in flags
    assert "missing_nlv" in evaluation.facts.data_quality_flags
    assert "missing_option_liquidity" in flags
    assert evaluation.facts.option_liquidity_known is False
    profit = next(r for r in evaluation.rules if r.rule_id == "profit.manage_winner")
    assert profit.status == RuleStatus.BLOCKED
    assert any(
        "missing" in c.candidate_id or c.kind == CandidateKind.MANUAL_REVIEW
        for c in evaluation.candidates
    )


def test_stale_and_wide_market() -> None:
    strategy = _bull_put()
    stale_as_of = datetime(2026, 7, 18, 10, 0, tzinfo=UTC).isoformat()
    evaluation = evaluate_mechanics(
        strategy,
        market={
            "price": 505.0,
            "spread_percent": 25.0,
            "as_of": stale_as_of,
            "freshness": "stale",
        },
        account_nlv=100_000,
        now=FIXED,
    )
    assert "stale_market" in evaluation.facts.data_quality_flags
    assert "wide_underlying_spread" in evaluation.facts.data_quality_flags
    assert evaluation.facts.underlying_spread_pct == 25.0
    assert evaluation.facts.option_liquidity_known is False
    data = next(r for r in evaluation.rules if r.rule_id == "gate.data_quality")
    assert data.status == RuleStatus.BLOCKED


def test_trade_plan_override_conflict() -> None:
    strategy = _bull_put(short_pnl=120, long_pnl=40, short_cost=300, long_cost=100)
    evaluation = evaluate_mechanics(
        strategy,
        market={"price": 510.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        trade_plan={
            "profit_target": "hold through expiration",
            "roll_criteria": "accept debit if needed",
        },
        settings=MechanicsSettings(profit_target_pct=0.50),
        now=FIXED,
    )
    assert any("holding longer" in note for note in evaluation.facts.plan_override_notes)
    profit = next(r for r in evaluation.rules if r.rule_id == "profit.manage_winner")
    assert profit.status in {RuleStatus.WATCH, RuleStatus.DUE}
    assert any(
        c.kind == CandidateKind.MANUAL_REVIEW or "plan" in c.candidate_id
        for c in evaluation.candidates
    )


def test_unsupported_strategy_fail_closed() -> None:
    strategy = StrategySnapshot(
        strategy_id="strat-stock",
        account_id="acct-public-1",
        underlying="AAPL",
        strategy_type="Long Stock",
        expiration_date=None,
        days_to_expiration=None,
        quantity=100,
        strikes="",
        unrealized_pnl=50,
        total_delta=100,
        total_theta=0,
        horizon=PositionHorizon.STRATEGIC,
        legs=[
            PositionSnapshot(
                symbol="AAPL",
                underlying_symbol="AAPL",
                quantity=100,
                quantity_direction=QuantityDirection.LONG,
                position_type=PositionType.EQUITY,
                mark_price=200,
                market_value=20000,
                unrealized_pnl=50,
                horizon=PositionHorizon.STRATEGIC,
            )
        ],
    )
    evaluation = evaluate_mechanics(strategy, now=FIXED)
    assert not evaluation.facts.supported
    supported = next(r for r in evaluation.rules if r.rule_id == "gate.strategy_supported")
    assert supported.status == RuleStatus.BLOCKED
    assert all(c.kind == CandidateKind.MANUAL_REVIEW for c in evaluation.candidates)
    assert not any(c.kind == CandidateKind.CLOSE for c in evaluation.candidates)


def test_stable_ids_and_deterministic_fingerprint() -> None:
    strategy = _bull_put(dte=21, short_pnl=120, long_pnl=40)
    market = {"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()}
    a = evaluate_mechanics(strategy, market=market, account_nlv=100_000, catalysts=[], now=FIXED)
    b = evaluate_mechanics(strategy, market=market, account_nlv=100_000, catalysts=[], now=FIXED)
    assert a.fingerprint == b.fingerprint
    assert [c.candidate_id for c in a.candidates] == [c.candidate_id for c in b.candidates]
    assert [r.rule_id for r in a.rules] == [r.rule_id for r in b.rules]
    # Compact context is stable JSON-serializable
    compact = compact_mechanics_context(a)
    assert compact["playbook_id"] == PLAYBOOK_ID_V1
    assert "candidates" in compact


def test_never_fabricates_roll_credit_or_add() -> None:
    evaluation = evaluate_mechanics(
        _short_strangle(short_put_delta=-0.42),
        market={"price": 480.0, "spread_percent": 2.0, "as_of": FIXED.isoformat()},
        account_nlv=50_000,
        catalysts=[],
        now=FIXED,
    )
    for candidate in evaluation.candidates:
        assert candidate.kind.value != "add"
        if candidate.kind == CandidateKind.ROLL_REVIEW:
            assert (
                "candidate_roll_credit" in candidate.missing_inputs
                or "option_chain_quote" in candidate.missing_inputs
            )
            assert candidate.after_risk is None or candidate.after_risk.max_loss is None


def test_offline_replay_seam(tmp_path: Path) -> None:
    db = PositionPilotDatabase(tmp_path / "m.sqlite3", backup_directory=tmp_path / "b")
    service = MechanicsService(db, clock=lambda: FIXED)
    strategy = _bull_put(dte=21)
    result = service.replay(
        strategy,
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
    )
    assert result.schema_version == "mechanics.v1"
    assert result.playbook_id == PLAYBOOK_ID_V1
    assert result.rules


def test_settings_validation_fail_closed(tmp_path: Path) -> None:
    db = PositionPilotDatabase(tmp_path / "m.sqlite3", backup_directory=tmp_path / "b")
    service = MechanicsService(db)
    public = service.public_settings()
    assert public["shadow_mode"] is True
    assert public["playbook_id"] == PLAYBOOK_ID_V1

    updated = service.update_settings({"profit_target_pct": 0.40, "manage_at_dte": 25})
    assert updated["profit_target_pct"] == 0.40
    assert updated["manage_at_dte"] == 25

    with pytest.raises(ValueError):
        service.update_settings({"profit_target_pct": 1.5})

    # Malformed stored settings fail closed: disabled, not re-enabled
    db.set_setting("mechanics", {"profit_target_pct": "nope", "shadow_mode": False})
    recovered = service.get_settings()
    assert recovered.enabled is False
    assert recovered.shadow_mode is True
    assert recovered.advisory_only is True
    # Do not overwrite invalid payload
    assert db.get_setting("mechanics")["profit_target_pct"] == "nope"
    disabled_eval = service.evaluate_strategy(_bull_put(), settings=recovered)
    assert disabled_eval.enabled is False
    assert disabled_eval.candidates == []


def test_parse_plan_profit_target() -> None:
    assert parse_plan_profit_target_pct("50%") == 0.50
    assert parse_plan_profit_target_pct("take half the credit") == 0.50
    assert parse_plan_profit_target_pct("") is None


def test_mechanics_in_recommendation_context_and_prompt() -> None:
    strategy = _bull_put(dte=21, short_pnl=120, long_pnl=40)
    evaluation = evaluate_mechanics(
        strategy,
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    mechanics = compact_mechanics_context(evaluation)
    ctx = strategy_context(strategy, mechanics=mechanics)
    assert "mechanics" in ctx
    assert ctx["mechanics"]["playbook_id"] == PLAYBOOK_ID_V1
    assert "account_id" not in str(ctx["mechanics"]).lower() or "acct-public" not in str(ctx)
    # Sanitizer strips unknown keys
    dirty = {**mechanics, "account_number": "5WT00001", "raw_payload": {"x": 1}}
    clean = sanitize_mechanics_context(dirty)
    assert "account_number" not in clean
    assert "raw_payload" not in clean

    # Default playbook settings use shadow_mode=true → observational prompt only.
    prompt = build_recommendation_prompt(ctx)
    assert "MECHANICS_OBSERVATION" in prompt
    assert "must NOT change or constrain" in prompt
    assert "MECHANICS_CONSTRAINTS" not in prompt

    non_shadow = {**mechanics, "shadow_mode": False}
    hard = build_recommendation_prompt({**ctx, "mechanics": non_shadow})
    assert "MECHANICS_CONSTRAINTS" in hard
    assert "Do not invent strikes" in hard
    assert "ONLY those supplied advisory candidates" in hard


def test_local_risk_on_profit_candidate_when_liquidity_blocks_close() -> None:
    strategy = _bull_put(short_pnl=120, long_pnl=40, short_cost=300, long_cost=100)
    risk = RiskService().strategy_risk(strategy, underlying_price=510.0)
    evaluation = evaluate_mechanics(
        strategy,
        risk=risk,
        market={"price": 510.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    # Missing option/complex quotes block unblocked close.
    assert "missing_option_liquidity" in evaluation.facts.data_quality_flags
    data = next(r for r in evaluation.rules if r.rule_id == "gate.data_quality")
    assert data.status == RuleStatus.BLOCKED
    close = next((c for c in evaluation.candidates if c.kind == CandidateKind.CLOSE), None)
    assert close is None or close.blocking_reasons
    review = next(
        (c for c in evaluation.candidates if "profit" in c.candidate_id),
        None,
    )
    assert review is not None
    assert review.before_risk is not None


def test_undefined_missing_bpr_never_size_above_cap_or_reduce() -> None:
    """Huge market value must not be compared to undefined BPR cap without strategy BPR."""
    strategy = _short_strangle(dte=30)
    for leg in strategy.legs:
        leg.market_value = 50_000  # large notional, still not BPR
    evaluation = evaluate_mechanics(
        strategy,
        market={"price": 500.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        settings=MechanicsSettings(undefined_bpr_cap_pct=0.05),
        now=FIXED,
    )
    assert evaluation.facts.size_ratio is None
    assert evaluation.facts.size_basis is None
    assert evaluation.facts.market_value_nlv_ratio is not None
    assert evaluation.facts.market_value_nlv_ratio > 0.05
    assert "missing_bpr" in evaluation.facts.data_quality_flags
    size = next(r for r in evaluation.rules if r.rule_id == "size.small_position")
    assert size.status == RuleStatus.WATCH
    assert size.reason_code == "size_unknown"
    assert not any(c.kind == CandidateKind.REDUCE for c in evaluation.candidates)
    assert not any(c.candidate_id.endswith(":size-cap") for c in evaluation.candidates)


def test_catalysts_none_is_unknown_not_no_event() -> None:
    evaluation = evaluate_mechanics(
        _bull_put(dte=30),
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=None,
        now=FIXED,
    )
    assert evaluation.facts.catalyst_availability.value == "unknown"
    assert evaluation.facts.high_impact_catalyst is None
    assert "catalysts_unknown" in evaluation.facts.data_quality_flags
    event = next(r for r in evaluation.rules if r.rule_id == "gate.event_exposure")
    assert event.status == RuleStatus.WATCH
    assert event.reason_code == "catalysts_unknown"
    # Profit winner with unknown catalysts is downgraded away from clean close.
    rich = evaluate_mechanics(
        _bull_put(short_pnl=120, long_pnl=40, short_cost=300, long_cost=100),
        market={"price": 510.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=None,
        now=FIXED,
    )
    assert not any(
        c.kind == CandidateKind.CLOSE and not c.blocking_reasons for c in rich.candidates
    )


def test_catalysts_empty_list_is_known_absence() -> None:
    evaluation = evaluate_mechanics(
        _bull_put(dte=30),
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert evaluation.facts.catalyst_availability.value == "known"
    assert evaluation.facts.high_impact_catalyst is False
    event = next(r for r in evaluation.rules if r.rule_id == "gate.event_exposure")
    assert event.status == RuleStatus.PASS
    assert event.reason_code == "no_high_impact"


def test_malformed_and_naive_market_timestamps_fail_closed() -> None:
    strategy = _bull_put(dte=30)
    bad = evaluate_mechanics(
        strategy,
        market={"price": 505.0, "spread_percent": 1.0, "as_of": "not-a-timestamp"},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert "invalid_market_timestamp" in bad.facts.data_quality_flags
    data = next(r for r in bad.rules if r.rule_id == "gate.data_quality")
    assert data.status == RuleStatus.BLOCKED

    naive = evaluate_mechanics(
        strategy,
        market={"price": 505.0, "spread_percent": 1.0, "as_of": "2026-07-18T14:00:00"},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert "invalid_market_timestamp" in naive.facts.data_quality_flags
    data_naive = next(r for r in naive.rules if r.rule_id == "gate.data_quality")
    assert data_naive.status == RuleStatus.BLOCKED


def test_butterfly_unsupported_fail_closed() -> None:
    wings = [
        _leg(
            symbol="SPY  260821C00490000",
            strike=490,
            option_type="C",
            direction=QuantityDirection.LONG,
            mark=1.0,
            cost_basis=100,
            unrealized_pnl=10,
            delta=0.2,
        ),
        _leg(
            symbol="SPY  260821C00500000",
            strike=500,
            option_type="C",
            direction=QuantityDirection.SHORT,
            mark=2.0,
            cost_basis=200,
            unrealized_pnl=20,
            delta=0.4,
            quantity=2,
        ),
        _leg(
            symbol="SPY  260821C00510000",
            strike=510,
            option_type="C",
            direction=QuantityDirection.LONG,
            mark=0.8,
            cost_basis=80,
            unrealized_pnl=5,
            delta=0.15,
        ),
    ]
    strategy = StrategySnapshot(
        strategy_id="strat-bfly",
        account_id="acct-public-1",
        underlying="SPY",
        strategy_type="Butterfly",
        expiration_date="2026-08-21",
        days_to_expiration=30,
        quantity=1,
        strikes="$490/$500/$510",
        unrealized_pnl=35,
        total_delta=0,
        total_theta=0,
        horizon=PositionHorizon.TACTICAL,
        legs=wings,
    )
    evaluation = evaluate_mechanics(strategy, catalysts=[], now=FIXED)
    assert not evaluation.facts.supported
    assert (
        next(r for r in evaluation.rules if r.rule_id == "gate.strategy_supported").status
        == RuleStatus.BLOCKED
    )


def test_underlying_spread_not_option_liquidity() -> None:
    evaluation = evaluate_mechanics(
        _bull_put(dte=30),
        market={"price": 505.0, "spread_percent": 2.0, "as_of": FIXED.isoformat()},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert evaluation.facts.underlying_spread_pct == 2.0
    assert evaluation.facts.option_liquidity_known is False
    assert "missing_option_liquidity" in evaluation.facts.data_quality_flags


def test_no_aggregate_portfolio_nlv_fallback_for_mismatched_account(tmp_path: Path) -> None:
    """Strategy account missing from portfolio must leave NLV unknown (no totals fallback)."""
    from position_pilot.domain.snapshots import (
        AccountSnapshot,
        DataFreshness,
        FreshnessState,
        PortfolioSnapshot,
        PortfolioTotals,
        SnapshotState,
    )

    strategy = _bull_put(dte=30)
    strategy = strategy.model_copy(update={"account_id": "acct-strategy"})
    portfolio = PortfolioSnapshot(
        snapshot_id="snap-multi",
        captured_at=FIXED,
        state=SnapshotState.LIVE,
        freshness=DataFreshness(as_of=FIXED, provider="test", state=FreshnessState.FRESH),
        accounts=[
            AccountSnapshot(
                account_id="acct-other",
                label="Other",
                account_type="Individual",
                net_liquidating_value=1_000_000,
                buying_power=500_000,
            )
        ],
        strategies=[strategy],
        totals=PortfolioTotals(
            net_liquidating_value=1_000_000,
            buying_power=500_000,
            unrealized_pnl=0,
        ),
    )
    db = PositionPilotDatabase(tmp_path / "nlv.sqlite3", backup_directory=tmp_path / "b")
    service = MechanicsService(db, clock=lambda: FIXED)
    evaluation = service.evaluate_strategy(
        strategy,
        portfolio=portfolio,
        market={"price": 505.0, "spread_percent": 1.0, "as_of": FIXED.isoformat()},
        catalysts=[],
    )
    assert evaluation.facts.account_nlv is None
    assert evaluation.facts.account_buying_power is None
    assert "missing_nlv" in evaluation.facts.data_quality_flags


@pytest.mark.parametrize("bad_value", [[], "", 0, False, "not-a-dict"])
def test_falsy_malformed_settings_disable_without_overwrite(tmp_path: Path, bad_value) -> None:
    db = PositionPilotDatabase(tmp_path / "set.sqlite3", backup_directory=tmp_path / "b")
    db.set_setting("mechanics", bad_value)
    service = MechanicsService(db)
    recovered = service.get_settings()
    assert recovered.enabled is False
    assert recovered.shadow_mode is True
    # Storage unchanged
    assert db.get_setting("mechanics") == bad_value
    assert db.has_setting("mechanics") is True


def test_price_only_and_missing_freshness_block_actionable_candidates() -> None:
    strategy = _bull_put(short_pnl=120, long_pnl=40, short_cost=300, long_cost=100)
    price_only = evaluate_mechanics(
        strategy,
        market={"price": 510.0},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    flags = price_only.facts.data_quality_flags
    assert "incomplete_underlying_quote" in flags
    assert "missing_market_freshness" in flags
    data = next(r for r in price_only.rules if r.rule_id == "gate.data_quality")
    assert data.status == RuleStatus.BLOCKED
    assert not any(
        c.kind == CandidateKind.CLOSE and not c.blocking_reasons for c in price_only.candidates
    )

    no_ts = evaluate_mechanics(
        strategy,
        market={"price": 510.0, "bid": 509.9, "ask": 510.1, "spread_percent": 0.04},
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert "missing_market_freshness" in no_ts.facts.data_quality_flags
    assert (
        next(r for r in no_ts.rules if r.rule_id == "gate.data_quality").status
        == RuleStatus.BLOCKED
    )


def test_missing_option_liquidity_blocks_close_with_complete_underlying() -> None:
    """Complete underlying quote/freshness still blocks actionable close without option quotes."""
    strategy = _bull_put(short_pnl=120, long_pnl=40, short_cost=300, long_cost=100)
    evaluation = evaluate_mechanics(
        strategy,
        market={
            "price": 510.0,
            "bid": 509.9,
            "ask": 510.1,
            "spread_percent": 0.04,
            "as_of": FIXED.isoformat(),
            "freshness": "fresh",
        },
        account_nlv=100_000,
        catalysts=[],
        now=FIXED,
    )
    assert evaluation.facts.option_liquidity_known is False
    assert evaluation.facts.underlying_spread_pct == 0.04
    assert "missing_option_liquidity" in evaluation.facts.data_quality_flags
    assert "incomplete_underlying_quote" not in evaluation.facts.data_quality_flags
    assert "missing_market_freshness" not in evaluation.facts.data_quality_flags
    data = next(r for r in evaluation.rules if r.rule_id == "gate.data_quality")
    assert data.status == RuleStatus.BLOCKED
    assert "missing_option_liquidity" in (
        data.data_quality_notes or data.observed.get("flags") or []
    )
    assert not any(
        c.kind in {CandidateKind.CLOSE, CandidateKind.REDUCE} and not c.blocking_reasons
        for c in evaluation.candidates
    )


def test_factory_market_loader_includes_freshness_state() -> None:
    """MarketService fallback used by monitoring must preserve freshness state."""
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from position_pilot.domain import factory as factory_mod
    from position_pilot.domain.market import MarketSnapshot
    from position_pilot.domain.snapshots import DataFreshness, FreshnessState

    snap = MarketSnapshot(
        symbol="SPY",
        price=500.0,
        bid=499.9,
        ask=500.1,
        spread_percent=0.04,
        freshness=DataFreshness(
            as_of=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
            provider="test",
            state=FreshnessState.UNAVAILABLE,
        ),
    )
    mock_ms = MagicMock()
    mock_ms.snapshot.return_value = snap
    factory_mod.get_market_service.cache_clear()
    # Patch the getter used inside the loader closure after rebuild.
    original = factory_mod.get_market_service
    factory_mod.get_market_service = lambda: mock_ms  # type: ignore[assignment]
    try:
        # Rebuild loader with patched market service.
        factory_mod._live_market_hub = None
        loader = factory_mod.build_market_context_loader()
        payload = loader("SPY")
    finally:
        factory_mod.get_market_service = original  # type: ignore[assignment]
        factory_mod.get_market_service.cache_clear()
    assert payload is not None
    assert payload["as_of"] is not None
    assert payload["freshness"] == "unavailable"
