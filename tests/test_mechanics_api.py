"""API surface for tasty mechanics settings and strategy evaluation."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from position_pilot.domain.mechanics import PLAYBOOK_ID_V1, MechanicsService
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
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.web.app import WebSettings, create_app


def _settings() -> WebSettings:
    return WebSettings(
        launch_token="launch-secret",
        session_token="session-secret",
        enforce_loopback=False,
        single_use_launch_token=False,
    )


def _auth_client(app) -> TestClient:
    client = TestClient(app)
    response = client.post(
        "/api/v1/session/exchange",
        json={"launch_token": "launch-secret"},
    )
    assert response.status_code == 204
    return client


def _strategy() -> StrategySnapshot:
    short = PositionSnapshot(
        symbol="SPY  260821P00500000",
        underlying_symbol="SPY",
        quantity=1,
        quantity_direction=QuantityDirection.SHORT,
        position_type=PositionType.EQUITY_OPTION,
        strike_price=500,
        option_type="P",
        expiration_date="2026-08-21",
        days_to_expiration=21,
        mark_price=1.5,
        market_value=150,
        cost_basis=300,
        unrealized_pnl=120,
        pnl_open=120,
        delta=-0.18,
        multiplier=100,
        horizon=PositionHorizon.TACTICAL,
    )
    long = PositionSnapshot(
        symbol="SPY  260821P00490000",
        underlying_symbol="SPY",
        quantity=1,
        quantity_direction=QuantityDirection.LONG,
        position_type=PositionType.EQUITY_OPTION,
        strike_price=490,
        option_type="P",
        expiration_date="2026-08-21",
        days_to_expiration=21,
        mark_price=0.5,
        market_value=50,
        cost_basis=100,
        unrealized_pnl=40,
        pnl_open=40,
        delta=-0.08,
        multiplier=100,
        horizon=PositionHorizon.TACTICAL,
    )
    return StrategySnapshot(
        strategy_id="strat-mech-1",
        account_id="public-acct",
        underlying="SPY",
        strategy_type="Bull Put Spread",
        expiration_date="2026-08-21",
        days_to_expiration=21,
        quantity=1,
        strikes="$490/$500",
        unrealized_pnl=160,
        pnl_open=160,
        total_delta=-0.1,
        total_theta=0.05,
        horizon=PositionHorizon.TACTICAL,
        legs=[short, long],
    )


def _portfolio(strategy: StrategySnapshot) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        snapshot_id="snap-mech",
        captured_at=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
        state=SnapshotState.LIVE,
        freshness=DataFreshness(
            as_of=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
            provider="test",
        ),
        accounts=[
            AccountSnapshot(
                account_id="public-acct",
                label="Individual",
                account_type="Individual",
                net_liquidating_value=100_000,
                cash_balance=20_000,
                buying_power=40_000,
                positions=list(strategy.legs),
            )
        ],
        strategies=[strategy],
        totals=PortfolioTotals(
            net_liquidating_value=100_000,
            cash_balance=20_000,
            buying_power=40_000,
            unrealized_pnl=160,
        ),
    )


def test_mechanics_settings_and_strategy_endpoint(tmp_path, monkeypatch) -> None:
    database = PositionPilotDatabase(tmp_path / "api.sqlite3", backup_directory=tmp_path / "b")
    strategy = _strategy()
    portfolio = _portfolio(strategy)

    class StubPortfolio:
        def latest(self, account_id: str = "all"):
            return portfolio

        def refresh(self):
            return portfolio

        def primary_account_id(self):
            return "all"

        def set_primary_account(self, account_id: str) -> None:
            return None

    class StubMarket:
        def snapshot(self, symbol, force_refresh=False):
            from position_pilot.domain.market import MarketSnapshot
            from position_pilot.domain.snapshots import FreshnessState

            return MarketSnapshot(
                symbol=symbol.upper(),
                price=505.0,
                bid=504.9,
                ask=505.1,
                iv=0.15,
                iv_rank=30.0,
                spread_percent=0.04,
                freshness=DataFreshness(
                    as_of=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
                    provider="test",
                    state=FreshnessState.FRESH,
                ),
            )

        def chart(self, symbol, **kwargs):
            from position_pilot.domain.market import ChartSnapshot, MarketBar

            return ChartSnapshot(
                symbol=symbol.upper(),
                bars=[
                    MarketBar(
                        timestamp=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
                        open=504,
                        high=506,
                        low=503,
                        close=505,
                        volume=1_000_000,
                    )
                ],
                source="test",
                prior_close=kwargs.get("prior_close", 500.0),
                include_extended_hours=kwargs.get("include_extended_hours", True),
                event_markers=kwargs.get("event_markers") or [],
            )

        def overview(self):
            from position_pilot.domain.market import MarketOverview

            return MarketOverview(
                captured_at=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
                quotes=[self.snapshot("SPY")],
                iv_summary={"normal": 1},
            )

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
                    as_of=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
                    provider="test",
                ),
                coverage=CoverageState.COMPLETE,
            )

        def event_markers(self, symbol: str):
            return []

    from position_pilot.domain.plans import PlansService

    plans = PlansService(database)
    mechanics = MechanicsService(database, risk_service=RiskService(), plans_service=plans)

    monkeypatch.setattr("position_pilot.web.app.get_portfolio_service", lambda: StubPortfolio())
    monkeypatch.setattr("position_pilot.web.app.get_market_service", lambda: StubMarket())
    monkeypatch.setattr("position_pilot.web.app.get_risk_service", lambda: RiskService())
    monkeypatch.setattr("position_pilot.web.app.get_plans_service", lambda: plans)
    monkeypatch.setattr("position_pilot.web.app.get_mechanics_service", lambda: mechanics)
    monkeypatch.setattr("position_pilot.web.app.get_database", lambda: database)
    monkeypatch.setattr("position_pilot.web.app.get_catalyst_service", lambda: StubCatalysts())
    monkeypatch.setattr(
        "position_pilot.web.app.get_roll_service",
        lambda: type("R", (), {"chains": lambda self, *a, **k: []})(),
    )
    monkeypatch.setattr(
        "position_pilot.web.app.get_recommendation_service",
        lambda: type(
            "Rec",
            (),
            {
                "get": lambda self, *a, **k: None,
                "history": lambda self, *a, **k: [],
                "list_decisions": lambda self, *a, **k: [],
            },
        )(),
    )

    app = create_app(_settings(), portfolio_service=StubPortfolio())
    client = _auth_client(app)

    settings = client.get("/api/v1/settings/mechanics")
    assert settings.status_code == 200
    body = settings.json()
    assert body["shadow_mode"] is True
    assert body["playbook_id"] == PLAYBOOK_ID_V1
    assert "execution_boundary" in body

    updated = client.put(
        "/api/v1/settings/mechanics",
        json={"profit_target_pct": 0.45, "manage_at_dte": 24},
    )
    assert updated.status_code == 200
    assert updated.json()["profit_target_pct"] == 0.45
    assert updated.json()["manage_at_dte"] == 24

    # Request-model range validation (422) or service fail-closed (400).
    bad = client.put("/api/v1/settings/mechanics", json={"profit_target_pct": 2.0})
    assert bad.status_code in {400, 422}
    bad_playbook = client.put(
        "/api/v1/settings/mechanics",
        json={"playbook_id": "not-a-real-playbook"},
    )
    assert bad_playbook.status_code == 400

    eval_resp = client.get("/api/v1/strategies/strat-mech-1/mechanics")
    assert eval_resp.status_code == 200
    payload = eval_resp.json()
    assert payload["playbook_id"] == PLAYBOOK_ID_V1
    assert payload["shadow_mode"] is True
    assert "rules" in payload and payload["rules"]
    assert "candidates" in payload
    assert "sources" in payload
    assert all("http" in s["url"] for s in payload["sources"])
    boundary = payload["execution_boundary"].lower()
    assert "manual" in boundary
    assert "cannot create" in boundary or "cannot" in boundary

    detail = client.get("/api/v1/strategies/strat-mech-1")
    assert detail.status_code == 200
    assert detail.json().get("mechanics") is not None
    assert detail.json()["mechanics"]["playbook_id"] == PLAYBOOK_ID_V1


def test_mechanics_api_catalyst_failure_is_unknown(tmp_path, monkeypatch) -> None:
    """Catalyst loader exception must not become known-empty catalysts=[]."""
    database = PositionPilotDatabase(tmp_path / "api2.sqlite3", backup_directory=tmp_path / "b2")
    strategy = _strategy()
    portfolio = _portfolio(strategy)

    class StubPortfolio:
        def latest(self, account_id: str = "all"):
            return portfolio

        def refresh(self):
            return portfolio

        def primary_account_id(self):
            return "all"

        def set_primary_account(self, account_id: str) -> None:
            return None

    class StubMarket:
        def snapshot(self, symbol, force_refresh=False):
            from position_pilot.domain.market import MarketSnapshot
            from position_pilot.domain.snapshots import FreshnessState

            return MarketSnapshot(
                symbol=symbol.upper(),
                price=505.0,
                bid=504.9,
                ask=505.1,
                iv=0.15,
                iv_rank=30.0,
                spread_percent=0.04,
                freshness=DataFreshness(
                    as_of=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
                    provider="test",
                    state=FreshnessState.FRESH,
                ),
            )

    class FailingCatalysts:
        def analyze_symbol(self, symbol: str):
            raise RuntimeError("catalyst provider offline")

        def event_markers(self, symbol: str):
            return []

    from position_pilot.domain.plans import PlansService

    plans = PlansService(database)
    mechanics = MechanicsService(database, risk_service=RiskService(), plans_service=plans)

    monkeypatch.setattr("position_pilot.web.app.get_portfolio_service", lambda: StubPortfolio())
    monkeypatch.setattr("position_pilot.web.app.get_market_service", lambda: StubMarket())
    monkeypatch.setattr("position_pilot.web.app.get_risk_service", lambda: RiskService())
    monkeypatch.setattr("position_pilot.web.app.get_plans_service", lambda: plans)
    monkeypatch.setattr("position_pilot.web.app.get_mechanics_service", lambda: mechanics)
    monkeypatch.setattr("position_pilot.web.app.get_database", lambda: database)
    monkeypatch.setattr("position_pilot.web.app.get_catalyst_service", lambda: FailingCatalysts())

    app = create_app(_settings(), portfolio_service=StubPortfolio())
    client = _auth_client(app)
    eval_resp = client.get("/api/v1/strategies/strat-mech-1/mechanics")
    assert eval_resp.status_code == 200
    payload = eval_resp.json()
    assert payload["facts"]["catalyst_availability"] == "unknown"
    assert payload["facts"]["high_impact_catalyst"] is None
    event = next(r for r in payload["rules"] if r["rule_id"] == "gate.event_exposure")
    assert event["status"] == "watch"
    assert event["reason_code"] == "catalysts_unknown"


def test_market_snapshot_to_mechanics_payload_includes_freshness() -> None:
    from position_pilot.domain.market import MarketSnapshot
    from position_pilot.domain.snapshots import FreshnessState
    from position_pilot.web.app import market_snapshot_to_mechanics_payload

    snap = MarketSnapshot(
        symbol="SPY",
        price=500.0,
        bid=499.9,
        ask=500.1,
        spread_percent=0.04,
        freshness=DataFreshness(
            as_of=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
            provider="test",
            state=FreshnessState.FRESH,
        ),
    )
    payload = market_snapshot_to_mechanics_payload(snap)
    assert payload is not None
    assert payload["price"] == 500.0
    assert payload["bid"] == 499.9
    assert payload["as_of"] is not None
    assert payload["freshness"] == "fresh"
    assert market_snapshot_to_mechanics_payload(None) is None
