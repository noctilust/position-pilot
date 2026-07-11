"""Deterministic portfolio and strategy risk calculations."""

from datetime import UTC, datetime

from position_pilot.domain.risk import RiskService
from position_pilot.domain.snapshots import (
    AccountSnapshot,
    DataFreshness,
    PortfolioSnapshot,
    PortfolioTotals,
    PositionHorizon,
    PositionSnapshot,
    QuantityDirection,
    SnapshotState,
    StrategySnapshot,
)
from position_pilot.models import PositionType


def _leg(
    *,
    symbol: str,
    underlying: str = "SPY",
    quantity: int = 1,
    direction: QuantityDirection = QuantityDirection.SHORT,
    strike: float | None = 500,
    option_type: str | None = "C",
    mark: float | None = 2.5,
    market_value: float = 250,
    delta: float | None = 0.2,
    gamma: float | None = 0.01,
    theta: float | None = -0.05,
    vega: float | None = 0.08,
    iv: float | None = 0.18,
    dte: int | None = 21,
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
        unrealized_pnl=50,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        implied_volatility=iv,
        multiplier=multiplier,
        horizon=PositionHorizon.TACTICAL,
    )


def _strategy(*legs: PositionSnapshot) -> StrategySnapshot:
    return StrategySnapshot(
        strategy_id="strat-1",
        account_id="acct-1",
        underlying="SPY",
        strategy_type="Bull Put Spread",
        expiration_date="2026-08-21",
        days_to_expiration=21,
        quantity=1,
        strikes="$490/$500",
        unrealized_pnl=sum(leg.unrealized_pnl for leg in legs),
        total_delta=0,
        total_theta=0,
        horizon=PositionHorizon.TACTICAL,
        legs=list(legs),
    )


def test_strategy_risk_computes_greeks_max_loss_and_breakevens() -> None:
    short_put = _leg(
        symbol="SPY  260821P00500000",
        strike=500,
        option_type="P",
        direction=QuantityDirection.SHORT,
        mark=3.0,
        market_value=300,
        delta=-0.25,
        theta=-0.04,
    )
    long_put = _leg(
        symbol="SPY  260821P00490000",
        strike=490,
        option_type="P",
        direction=QuantityDirection.LONG,
        mark=1.5,
        market_value=150,
        delta=-0.12,
        theta=-0.02,
    )
    risk = RiskService().strategy_risk(_strategy(short_put, long_put), underlying_price=505)

    assert risk.strategy_id == "strat-1"
    assert risk.combined.delta is not None
    assert risk.combined.theta is not None
    assert risk.combined.delta == 13
    assert risk.combined.theta == 2
    assert risk.max_profit == 150
    assert risk.max_loss == 850
    assert risk.breakevens == [498.5]
    assert risk.distance_to_nearest_strike is not None
    assert risk.valuation_basis == "current_mark"


def test_structure_bounds_support_multi_leg_and_nonstandard_multipliers() -> None:
    legs = [
        _leg(
            symbol="SPY P490", strike=490, option_type="P", direction=QuantityDirection.LONG, mark=1
        ),
        _leg(
            symbol="SPY P495",
            strike=495,
            option_type="P",
            direction=QuantityDirection.SHORT,
            mark=2,
        ),
        _leg(
            symbol="SPY C505",
            strike=505,
            option_type="C",
            direction=QuantityDirection.SHORT,
            mark=2,
        ),
        _leg(
            symbol="SPY C510", strike=510, option_type="C", direction=QuantityDirection.LONG, mark=1
        ),
    ]
    condor = RiskService().strategy_risk(_strategy(*legs), underlying_price=500)

    assert condor.max_profit == 200
    assert condor.max_loss == 300
    assert condor.breakevens == [493, 507]
    assert condor.defined_risk is True

    mini_put = _leg(
        symbol="MINI P100",
        underlying="MINI",
        strike=100,
        option_type="P",
        direction=QuantityDirection.SHORT,
        mark=2,
        multiplier=10,
    )
    mini = RiskService().strategy_risk(_strategy(mini_put), underlying_price=105)
    assert mini.max_profit == 20
    assert mini.max_loss == 980
    assert mini.breakevens == [98]


def test_stress_scenarios_include_price_iv_theta_and_expiration() -> None:
    short_call = _leg(
        symbol="SPY  260821C00500000",
        strike=500,
        option_type="C",
        direction=QuantityDirection.SHORT,
        mark=2.0,
        market_value=200,
        delta=0.2,
        gamma=0.01,
        theta=-0.05,
        vega=0.1,
        iv=0.2,
    )
    scenarios = RiskService().stress_scenarios(
        _strategy(short_call),
        underlying_price=500,
    )
    names = {scenario.name for scenario in scenarios}
    assert "price_down_10" in names
    assert "price_up_10" in names
    assert "iv_up_25" in names
    assert "theta_1d" in names
    assert "expiration" in names
    theta = next(item for item in scenarios if item.name == "theta_1d")
    # Short option: negative theta becomes positive daily decay benefit
    assert theta.estimated_pnl_change > 0


def test_portfolio_risk_aggregates_concentration_and_greeks() -> None:
    equity = PositionSnapshot(
        symbol="AAPL",
        underlying_symbol="AAPL",
        quantity=100,
        quantity_direction=QuantityDirection.LONG,
        position_type=PositionType.EQUITY,
        mark_price=200,
        market_value=20_000,
        unrealized_pnl=500,
        delta=1.0,
        multiplier=1,
        horizon=PositionHorizon.STRATEGIC,
    )
    option = _leg(symbol="SPY  260821C00500000", market_value=500)
    snapshot = PortfolioSnapshot(
        snapshot_id="snap-1",
        captured_at=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
        state=SnapshotState.LIVE,
        freshness=DataFreshness(
            as_of=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
            provider="tastytrade",
        ),
        accounts=[
            AccountSnapshot(
                account_id="acct-1",
                label="Individual 1",
                account_type="Individual",
                net_liquidating_value=50_000,
                cash_balance=5_000,
                buying_power=10_000,
                positions=[equity, option],
            )
        ],
        strategies=[
            StrategySnapshot(
                strategy_id="eq-1",
                account_id="acct-1",
                underlying="AAPL",
                strategy_type="Long Stock",
                quantity=100,
                strikes="",
                unrealized_pnl=500,
                horizon=PositionHorizon.STRATEGIC,
                legs=[equity],
            ),
            _strategy(option),
        ],
        totals=PortfolioTotals(
            net_liquidating_value=50_000,
            cash_balance=5_000,
            buying_power=10_000,
            unrealized_pnl=550,
        ),
    )

    risk = RiskService().portfolio_risk(snapshot)

    assert risk.total_delta != 0
    assert risk.total_delta == 80
    assert risk.concentration
    assert risk.concentration[0].underlying in {"AAPL", "SPY"}
    assert risk.unrealized_pnl == 550
    assert risk.account_count == 1
    assert risk.strategy_count == 2
    assert {scenario.label for scenario in risk.stress} >= {
        "Portfolio −1 point",
        "Portfolio +1 point",
    }
