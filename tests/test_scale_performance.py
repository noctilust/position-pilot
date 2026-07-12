"""Scale and performance budgets for Phase 7 targets.

Targets (PRD §17): 5 accounts, 500 legs, 200 strategies, 100 watchlist symbols.
These tests are deterministic and offline.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from position_pilot.analysis.strategies import detect_strategies
from position_pilot.domain.market import MarketService
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
from position_pilot.domain.watchlist import WatchlistService
from position_pilot.models import Position, PositionType
from position_pilot.persistence.sqlite import PositionPilotDatabase


def _now() -> datetime:
    return datetime(2026, 7, 11, 16, 30, tzinfo=UTC)


def test_risk_engine_handles_500_legs_within_budget() -> None:
    legs: list[PositionSnapshot] = []
    strategies: list[StrategySnapshot] = []
    for index in range(500):
        underlying = f"S{index % 100}"
        leg = PositionSnapshot(
            symbol=f"{underlying}{index}",
            underlying_symbol=underlying,
            quantity=-1 if index % 2 else 1,
            quantity_direction=QuantityDirection.SHORT if index % 2 else QuantityDirection.LONG,
            position_type=PositionType.EQUITY_OPTION,
            strike_price=100 + (index % 50),
            option_type="C" if index % 2 else "P",
            expiration_date="2026-09-18",
            days_to_expiration=60,
            market_value=150,
            unrealized_pnl=5,
            delta=0.1 if index % 2 == 0 else -0.1,
            gamma=0.01,
            theta=-0.02,
            vega=0.05,
            multiplier=100,
            horizon=PositionHorizon.TACTICAL,
        )
        legs.append(leg)
        if index < 200:
            strategies.append(
                StrategySnapshot(
                    strategy_id=f"acct:{underlying}:{index}",
                    account_id="acct",
                    underlying=underlying,
                    strategy_type="Vertical",
                    quantity=1,
                    strikes=str(leg.strike_price),
                    unrealized_pnl=5,
                    total_delta=leg.delta or 0,
                    total_theta=leg.theta or 0,
                    horizon=PositionHorizon.TACTICAL,
                    legs=[leg],
                )
            )

    accounts = [
        AccountSnapshot(
            account_id=f"acct-{i}",
            label=f"Account {i}",
            account_type="Margin",
            net_liquidating_value=100_000,
            positions=legs[i * 100 : (i + 1) * 100],
        )
        for i in range(5)
    ]
    snapshot = PortfolioSnapshot(
        snapshot_id="perf",
        captured_at=_now(),
        state=SnapshotState.LIVE,
        freshness=DataFreshness(as_of=_now(), provider="test", state=FreshnessState.FRESH),
        accounts=accounts,
        strategies=strategies,
        totals=PortfolioTotals(net_liquidating_value=500_000, unrealized_pnl=2500),
    )

    started = time.perf_counter()
    risk = RiskService().portfolio_risk(snapshot)
    elapsed = time.perf_counter() - started

    assert risk.account_count == 5
    assert risk.position_count == 500
    assert risk.strategy_count == 200
    # Local budget: portfolio risk over 500 legs should stay well under 1s.
    assert elapsed < 1.0, f"portfolio_risk took {elapsed:.3f}s"


def test_strategy_detection_exactly_200_groups() -> None:
    positions: list[Position] = []
    for index in range(200):
        # Four-leg iron condor units → exactly 200 strategies when detected independently.
        base = 400 + index
        for strike, option_type, direction in (
            (base, "P", "Long"),
            (base + 5, "P", "Short"),
            (base + 30, "C", "Short"),
            (base + 35, "C", "Long"),
        ):
            positions.append(
                Position(
                    symbol=f"T{index}{option_type}{strike}",
                    underlying_symbol=f"T{index}",
                    quantity=1 if direction == "Long" else -1,
                    quantity_direction=direction,
                    position_type=PositionType.EQUITY_OPTION,
                    strike_price=float(strike),
                    option_type=option_type,
                    expiration_date=datetime(2026, 9, 18).date(),
                )
            )

    started = time.perf_counter()
    groups = detect_strategies(positions)
    elapsed = time.perf_counter() - started
    assert len(positions) == 800
    assert len(groups) == 200
    assert elapsed < 3.0, f"detect_strategies took {elapsed:.3f}s"


class _CountingBatchSource:
    """Deterministic market source that counts batch vs per-symbol calls."""

    def __init__(self) -> None:
        self.quote_batch_calls = 0
        self.metrics_batch_calls = 0
        self.quote_calls = 0
        self.metrics_calls = 0

    def get_quote(self, symbol: str, force_refresh: bool = False) -> dict | None:
        self.quote_calls += 1
        return {"mark": 100.0, "bid": 99.5, "ask": 100.5, "__provenance__": {"provider": "test"}}

    def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict[str, dict]:
        self.quote_batch_calls += 1
        return {
            symbol: {
                "mark": 100.0 + (index % 5),
                "bid": 99.5,
                "ask": 100.5,
                "__provenance__": {"provider": "test"},
            }
            for index, symbol in enumerate(symbols)
        }

    def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
        self.metrics_calls += 1
        return {"iv_rank": 40.0, "implied_volatility": 0.2, "iv_percentile": 55.0}

    def get_market_metrics_batch(
        self, symbols: list[str], force_refresh: bool = False
    ) -> dict[str, dict]:
        self.metrics_batch_calls += 1
        return {
            symbol: {"iv_rank": 40.0, "implied_volatility": 0.2, "iv_percentile": 55.0}
            for symbol in symbols
        }


def test_watchlist_100_symbols_uses_bounded_batch_enrichment(tmp_path) -> None:
    symbols = [f"S{index:03d}" for index in range(100)]
    assert len(symbols) == 100
    database = PositionPilotDatabase(tmp_path / "scale.sqlite3")
    database.set_setting("watchlist", symbols)
    source = _CountingBatchSource()
    market = MarketService(source=source, clock=_now)
    service = WatchlistService(database, market)

    started = time.perf_counter()
    snapshot = service.snapshot()
    elapsed = time.perf_counter() - started

    assert len(snapshot.symbols) == 100
    assert len(snapshot.quotes) == 100
    # Bounded provider calls: one batch quotes + one batch metrics (not 100+100).
    assert source.quote_batch_calls == 1
    assert source.metrics_batch_calls == 1
    assert source.quote_calls == 0
    assert source.metrics_calls == 0
    assert elapsed < 2.0, f"watchlist snapshot took {elapsed:.3f}s"
