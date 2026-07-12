"""Deterministic calculation regression suite (no AI, no network)."""

from __future__ import annotations

from datetime import UTC, datetime

from position_pilot.domain.risk import RiskService
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


def _now() -> datetime:
    return datetime(2026, 7, 11, 15, 0, tzinfo=UTC)


def _leg(
    *,
    symbol: str,
    underlying: str = "SPY",
    quantity: int = -1,
    strike: float = 500,
    option_type: str = "P",
    delta: float = -0.25,
    gamma: float = 0.02,
    theta: float = -0.04,
    vega: float = 0.1,
    market_value: float = 200,
    pnl: float = 50,
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        underlying_symbol=underlying,
        quantity=quantity,
        quantity_direction=QuantityDirection.SHORT if quantity < 0 else QuantityDirection.LONG,
        position_type=PositionType.EQUITY_OPTION,
        strike_price=strike,
        option_type=option_type,
        expiration_date="2026-08-21",
        days_to_expiration=40,
        market_value=market_value,
        unrealized_pnl=pnl,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        implied_volatility=0.2,
        multiplier=100,
        horizon=PositionHorizon.TACTICAL,
    )


def test_iron_condor_greeks_and_bounds_are_stable() -> None:
    legs = [
        _leg(symbol="SPY_P_480", strike=480, option_type="P", quantity=1, delta=0.08, pnl=20),
        _leg(symbol="SPY_P_500", strike=500, option_type="P", quantity=-1, delta=-0.25, pnl=40),
        _leg(symbol="SPY_C_530", strike=530, option_type="C", quantity=-1, delta=-0.22, pnl=35),
        _leg(symbol="SPY_C_550", strike=550, option_type="C", quantity=1, delta=0.09, pnl=15),
    ]
    strategy = StrategySnapshot(
        strategy_id="acct:SPY:ic",
        account_id="acct",
        underlying="SPY",
        strategy_type="Iron Condor",
        quantity=1,
        strikes="480/500/530/550",
        unrealized_pnl=sum(leg.unrealized_pnl for leg in legs),
        total_delta=sum((leg.delta or 0) for leg in legs),
        total_theta=sum((leg.theta or 0) for leg in legs),
        horizon=PositionHorizon.TACTICAL,
        legs=legs,
        days_to_expiration=40,
    )
    risk = RiskService().strategy_risk(strategy, underlying_price=515)
    combined = risk.combined
    assert combined.delta is not None
    # Share-equivalent option delta: sign(direction) * |qty| * multiplier * raw_delta
    expected_delta = (
        (1 * 1 * 100 * 0.08)
        + (-1 * 1 * 100 * -0.25)
        + (-1 * 1 * 100 * -0.22)
        + (1 * 1 * 100 * 0.09)
    )
    assert abs(combined.delta - expected_delta) < 1e-9
    assert risk.current_pnl == sum(leg.unrealized_pnl for leg in legs)
    assert risk.combined.nearest_dte == 40
    names = {item.name for item in risk.stress}
    assert "price_down_10" in names
    assert "iv_up_25" in names
    assert "expiration" in names
    # Stress estimates are finite numbers (no NaN).
    assert all(isinstance(item.estimated_pnl_change, (int, float)) for item in risk.stress)


def test_portfolio_aggregation_scales_to_many_strategies() -> None:
    strategies: list[StrategySnapshot] = []
    accounts: list[AccountSnapshot] = []
    for account_idx in range(5):
        account_id = f"acct-{account_idx}"
        legs: list[PositionSnapshot] = []
        account_strategies: list[StrategySnapshot] = []
        for strategy_idx in range(40):
            underlying = f"U{strategy_idx % 20}"
            leg = _leg(
                symbol=f"{underlying}_{account_idx}_{strategy_idx}",
                underlying=underlying,
                quantity=-1,
                delta=-0.1,
                market_value=100,
                pnl=10,
            )
            legs.append(leg)
            account_strategies.append(
                StrategySnapshot(
                    strategy_id=f"{account_id}:{underlying}:{strategy_idx}",
                    account_id=account_id,
                    underlying=underlying,
                    strategy_type="Short Put",
                    quantity=1,
                    strikes="100P",
                    unrealized_pnl=10,
                    total_delta=-0.1,
                    total_theta=-0.04,
                    horizon=PositionHorizon.TACTICAL,
                    legs=[leg],
                )
            )
        strategies.extend(account_strategies)
        accounts.append(
            AccountSnapshot(
                account_id=account_id,
                label=f"Account {account_idx}",
                account_type="Margin",
                net_liquidating_value=50_000,
                cash_balance=20_000,
                buying_power=40_000,
                positions=legs,
            )
        )

    snapshot = PortfolioSnapshot(
        snapshot_id="scale-snap",
        captured_at=_now(),
        state=SnapshotState.LIVE,
        freshness=DataFreshness(as_of=_now(), provider="test", state=FreshnessState.FRESH),
        accounts=accounts,
        strategies=strategies,
        totals=PortfolioTotals(
            net_liquidating_value=250_000,
            cash_balance=100_000,
            buying_power=200_000,
            unrealized_pnl=sum(s.unrealized_pnl for s in strategies),
        ),
    )
    risk = RiskService().portfolio_risk(snapshot)
    assert risk.account_count == 5
    assert risk.strategy_count == 200
    assert risk.position_count == 200
    # Each short option: sign=-1, qty=1, mult=100, raw delta=-0.1 → +10 share-eq delta
    assert abs(risk.total_delta - (10.0 * 200)) < 1e-6
    assert risk.unrealized_pnl == 10 * 200
    assert risk.concentration  # top underlyings present
