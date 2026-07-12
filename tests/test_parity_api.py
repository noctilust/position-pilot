"""Phase 4 portfolio parity API coverage."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from position_pilot.domain.market import IVEnvironment, MarketSnapshot
from position_pilot.domain.orders import FillSnapshot, OrderSnapshot
from position_pilot.domain.plans import PlansService
from position_pilot.domain.risk import RiskService
from position_pilot.domain.rolls import (
    HeatmapCell,
    RollChainSnapshot,
    RollEventSnapshot,
    RollHeatmapSnapshot,
    RollPatternsSnapshot,
)
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
from position_pilot.domain.watchlist import WatchlistSnapshot
from position_pilot.models import PositionType
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.web.app import WebSettings, create_app


def _auth_client(app) -> TestClient:
    client = TestClient(app)
    client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})
    return client


def _settings() -> WebSettings:
    return WebSettings(
        launch_token="launch-secret",
        session_token="session-secret",
        enforce_loopback=False,
        enable_streaming=False,
    )


class ParityPortfolioService:
    def __init__(self) -> None:
        leg = PositionSnapshot(
            symbol="SPY  260821P00500000",
            underlying_symbol="SPY",
            quantity=1,
            quantity_direction=QuantityDirection.SHORT,
            position_type=PositionType.EQUITY_OPTION,
            strike_price=500,
            option_type="P",
            expiration_date="2026-08-21",
            days_to_expiration=21,
            mark_price=2.5,
            market_value=250,
            unrealized_pnl=40,
            delta=-0.2,
            theta=-0.04,
            multiplier=100,
            horizon=PositionHorizon.TACTICAL,
        )
        self.snapshot = PortfolioSnapshot(
            snapshot_id="snap-parity",
            captured_at=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
            state=SnapshotState.LIVE,
            freshness=DataFreshness(
                as_of=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
                provider="tastytrade",
            ),
            accounts=[
                AccountSnapshot(
                    account_id="acct-public",
                    label="Individual 1",
                    account_type="Individual",
                    net_liquidating_value=25_000,
                    cash_balance=4_000,
                    buying_power=8_000,
                    positions=[leg],
                )
            ],
            strategies=[
                StrategySnapshot(
                    strategy_id="strat-public",
                    account_id="acct-public",
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
                    legs=[leg],
                )
            ],
            totals=PortfolioTotals(
                net_liquidating_value=25_000,
                cash_balance=4_000,
                buying_power=8_000,
                unrealized_pnl=40,
            ),
        )

    def latest(self, account_id: str = "all") -> PortfolioSnapshot | None:
        return self.snapshot.for_account(account_id)

    def refresh(self) -> PortfolioSnapshot:
        return self.snapshot

    def primary_account_id(self) -> str:
        return "all"

    def set_primary_account(self, account_id: str) -> None:
        return None

    def set_strategy_horizon(self, strategy_id: str, horizon: PositionHorizon) -> StrategySnapshot:
        strategy = next(s for s in self.snapshot.strategies if s.strategy_id == strategy_id)
        return strategy.model_copy(update={"horizon": horizon})


def test_parity_endpoints_are_authenticated_and_redacted(monkeypatch, tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    plans = PlansService(database)
    portfolio = ParityPortfolioService()

    class StubMarket:
        def snapshot(self, symbol, *, force_refresh=False):
            return MarketSnapshot(
                symbol=symbol.upper(),
                price=500,
                iv_rank=42,
                iv_environment=IVEnvironment.NORMAL,
                freshness=DataFreshness(
                    as_of=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
                    provider="tastytrade",
                ),
            )

        def overview(self, symbols=None, *, force_refresh=False):
            from position_pilot.domain.market import MarketOverview

            snap = self.snapshot("SPY")
            return MarketOverview(
                captured_at=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
                quotes=[snap],
                iv_summary={"normal": 1},
            )

        def chart(self, symbol, **kwargs):
            from position_pilot.domain.market import ChartSnapshot, MarketBar

            return ChartSnapshot(
                symbol=symbol.upper(),
                bars=[
                    MarketBar(
                        timestamp=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
                        open=499,
                        high=501,
                        low=498,
                        close=500,
                        volume=1_000_000,
                    )
                ],
                source="tastytrade",
                prior_close=kwargs.get("prior_close", 500.0),
                include_extended_hours=kwargs.get("include_extended_hours", True),
                event_markers=kwargs.get("event_markers") or [],
            )

    class StubRolls:
        def chains(self, account_id, *, symbol=None):
            return [
                RollChainSnapshot(
                    chain_id="chain-1",
                    account_id=account_id,
                    underlying="SPY",
                    strategy_type="Short Put",
                    original_open_credit=2.5,
                    rolls=[
                        RollEventSnapshot(
                            roll_id="r1",
                            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
                            underlying="SPY",
                            strategy_type="Short Put",
                            old_symbol="SPY  260620P00500000",
                            old_strike=500,
                            old_expiration="2026-06-20",
                            old_dte=14,
                            new_symbol="SPY  260718P00500000",
                            new_strike=500,
                            new_expiration="2026-07-18",
                            new_dte=35,
                            premium_effect=0.3,
                            roll_pnl=50,
                        )
                    ],
                )
            ]

        def patterns(self, account_id, *, symbol=None):
            return RollPatternsSnapshot(account_id=account_id, symbol=symbol, total_rolls=1)

        def heatmap(self, account_id, *, symbol):
            return RollHeatmapSnapshot(
                account_id=account_id,
                underlying=symbol,
                cells=[HeatmapCell(strike=500, dte_bucket="8-14", dte_min=8, dte_max=14, count=1)],
                strikes=[500],
                buckets=["8-14"],
                total_rolls=1,
            )

    class StubOrders:
        def list_orders(self, account_id, *, limit=100):
            return [
                OrderSnapshot(
                    order_id="public-order",
                    account_id=account_id,
                    symbol="SPY  260821P00500000",
                    underlying_symbol="SPY",
                    action="Sell to Open",
                    quantity=1,
                    order_type="Limit",
                    status="filled",
                    created_at=datetime(2026, 7, 10, tzinfo=UTC),
                    updated_at=datetime(2026, 7, 10, tzinfo=UTC),
                    filled_quantity=1,
                    average_fill_price=2.5,
                    fills=[
                        FillSnapshot(
                            fill_id="public-fill",
                            filled_at=datetime(2026, 7, 10, tzinfo=UTC),
                            symbol="SPY  260821P00500000",
                            quantity=1,
                            price=2.5,
                            amount=250,
                        )
                    ],
                )
            ]

    class StubWatchlist:
        def snapshot(self, *, force_refresh=False):
            return WatchlistSnapshot(
                symbols=["SPY"],
                quotes=[StubMarket().snapshot("SPY")],
            )

        def set_symbols(self, symbols):
            return [s.upper() for s in symbols]

        def list_symbols(self):
            return ["SPY"]

    class StubCatalysts:
        def analyze_symbol(self, symbol: str):
            from position_pilot.domain.catalysts import (
                AttributionLevel,
                CatalystConfidence,
                CoverageState,
                SymbolCatalystResult,
            )

            return SymbolCatalystResult(
                symbol=symbol.upper(),
                confidence=CatalystConfidence.NO_CONFIRMED_CATALYST,
                attribution=AttributionLevel.NONE,
                summary="No confirmed catalyst found",
                freshness=DataFreshness(
                    as_of=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
                    provider="catalyst-service",
                ),
                coverage=CoverageState.COMPLETE,
                prior_close=500.0,
                last_price=500.0,
                move_percent=0.0,
                quiet=True,
            )

        def event_markers(self, symbol: str):
            return []

        def public_settings(self):
            return {
                "stock_move_threshold_pct": 2.0,
                "etf_move_threshold_pct": 1.0,
                "news_cadence_seconds": 300,
                "benzinga": {"enabled": False, "status": "disabled"},
                "scheduled_window_hours": 72,
            }

    monkeypatch.setattr("position_pilot.web.app.get_market_service", lambda: StubMarket())
    monkeypatch.setattr("position_pilot.web.app.get_roll_service", lambda: StubRolls())
    monkeypatch.setattr("position_pilot.web.app.get_order_service", lambda: StubOrders())
    monkeypatch.setattr("position_pilot.web.app.get_plans_service", lambda: plans)
    monkeypatch.setattr("position_pilot.web.app.get_watchlist_service", lambda: StubWatchlist())
    monkeypatch.setattr("position_pilot.web.app.get_risk_service", lambda: RiskService())
    monkeypatch.setattr("position_pilot.web.app.get_database", lambda: database)
    monkeypatch.setattr("position_pilot.web.app.get_catalyst_service", lambda: StubCatalysts())
    monkeypatch.setattr(
        "position_pilot.web.app.get_field_router",
        lambda: type(
            "R",
            (),
            {"health": lambda self: {}},
        )(),
    )

    app = create_app(_settings(), portfolio_service=portfolio)
    client = _auth_client(app)

    portfolio_response = client.get("/api/v1/portfolio")
    assert portfolio_response.status_code == 200
    body = portfolio_response.json()
    assert body["strategies"][0]["strategy_id"] == "strat-public"
    assert body["accounts"][0]["buying_power"] == 8000
    assert "5WT" not in portfolio_response.text

    risk = client.get("/api/v1/portfolio/risk")
    assert risk.status_code == 200
    assert "total_delta" in risk.json()

    detail = client.get("/api/v1/strategies/strat-public")
    assert detail.status_code == 200
    assert detail.json()["strategy"]["underlying"] == "SPY"
    assert "risk" in detail.json()
    assert "chart" in detail.json()

    thesis = client.put(
        "/api/v1/strategies/strat-public/thesis",
        json={"purpose": "Income", "expected_duration": "Tactical"},
    )
    assert thesis.status_code == 200
    assert thesis.json()["purpose"] == "Income"

    plan = client.put(
        "/api/v1/strategies/strat-public/trade-plan",
        json={"entry_thesis": "Sell put", "profit_target": "50%"},
    )
    assert plan.status_code == 200

    audit = client.get("/api/v1/strategies/strat-public/audit")
    assert audit.status_code == 200
    assert len(audit.json()) >= 2

    orphan = client.put(
        "/api/v1/strategies/not-a-current-strategy/thesis",
        json={"purpose": "orphan"},
    )
    assert orphan.status_code == 404
    assert plans.get_thesis("not-a-current-strategy") is None

    markets = client.get("/api/v1/markets")
    assert markets.status_code == 200
    assert markets.json()["quotes"][0]["symbol"] == "SPY"

    quote = client.get("/api/v1/markets/SPY")
    assert quote.status_code == 200

    chart = client.get("/api/v1/markets/SPY/chart")
    assert chart.status_code == 200
    assert chart.json()["bars"]

    watchlist = client.get("/api/v1/watchlist")
    assert watchlist.status_code == 200
    assert "SPY" in watchlist.json()["symbols"]

    orders = client.get("/api/v1/accounts/acct-public/orders")
    assert orders.status_code == 200
    assert orders.json()[0]["order_id"] == "public-order"
    assert orders.json()[0]["fills"]

    rolls = client.get("/api/v1/accounts/acct-public/rolls")
    assert rolls.status_code == 200
    assert rolls.json()[0]["chain_total_credit"] == 2.8

    patterns = client.get("/api/v1/accounts/acct-public/rolls/patterns?symbol=SPY")
    assert patterns.status_code == 200
    assert patterns.json()["total_rolls"] == 1

    heatmap = client.get("/api/v1/accounts/acct-public/rolls/heatmap?symbol=SPY")
    assert heatmap.status_code == 200
    assert heatmap.json()["cells"]

    bootstrap = client.get("/api/v1/bootstrap")
    assert bootstrap.json()["application"]["phase"] == "hardening-retirement"
