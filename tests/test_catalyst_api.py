"""Web API coverage for Phase 5 catalyst intelligence."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from position_pilot.domain.catalysts import (
    AttributionLevel,
    CatalystConfidence,
    CatalystFeedbackEvent,
    CatalystScanSnapshot,
    CoverageState,
    SymbolCatalystResult,
)
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
from position_pilot.web.app import WebSettings, create_app

FIXED = datetime(2026, 7, 11, 18, 30, tzinfo=UTC)


class PortfolioStub:
    def __init__(self) -> None:
        leg = PositionSnapshot(
            symbol="AAPL 260821C00200000",
            underlying_symbol="AAPL",
            quantity=1,
            quantity_direction=QuantityDirection.LONG,
            position_type=PositionType.EQUITY_OPTION,
            strike_price=200,
            option_type="C",
            expiration_date="2026-08-21",
            days_to_expiration=41,
            mark_price=5,
            market_value=500,
            unrealized_pnl=50,
            multiplier=100,
            horizon=PositionHorizon.TACTICAL,
        )
        self.snapshot = PortfolioSnapshot(
            snapshot_id="snap-cat",
            captured_at=FIXED,
            state=SnapshotState.LIVE,
            freshness=DataFreshness(as_of=FIXED, provider="tastytrade"),
            accounts=[
                AccountSnapshot(
                    account_id="acct-1",
                    label="Individual 1",
                    account_type="Individual",
                    net_liquidating_value=20_000,
                    cash_balance=2_000,
                    buying_power=5_000,
                    positions=[leg],
                )
            ],
            strategies=[
                StrategySnapshot(
                    strategy_id="strat-1",
                    account_id="acct-1",
                    underlying="AAPL",
                    strategy_type="Long Call",
                    expiration_date="2026-08-21",
                    days_to_expiration=41,
                    quantity=1,
                    strikes="$200",
                    unrealized_pnl=50,
                    total_delta=40,
                    total_theta=-5,
                    horizon=PositionHorizon.TACTICAL,
                    legs=[leg],
                )
            ],
            totals=PortfolioTotals(
                net_liquidating_value=20_000,
                cash_balance=2_000,
                buying_power=5_000,
                unrealized_pnl=50,
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
        return self.snapshot.strategies[0]


class CatalystStub:
    def __init__(self) -> None:
        self.feedback: list[CatalystFeedbackEvent] = []
        self.settings = {
            "stock_move_threshold_pct": 2.0,
            "etf_move_threshold_pct": 1.0,
            "news_cadence_seconds": 300,
            "benzinga": {"enabled": True, "status": "enabled"},
            "scheduled_window_hours": 72,
        }

    def scan_held(self, symbols):
        results = [self.analyze_symbol(symbol) for symbol in symbols]
        return CatalystScanSnapshot(
            captured_at=FIXED,
            results=results,
            settings=self.settings,
            freshness=DataFreshness(as_of=FIXED, provider="catalyst-service"),
            coverage=CoverageState.COMPLETE,
        )

    def analyze_symbol(self, symbol: str) -> SymbolCatalystResult:
        return SymbolCatalystResult(
            symbol=symbol.upper(),
            confidence=CatalystConfidence.NO_CONFIRMED_CATALYST,
            attribution=AttributionLevel.NONE,
            summary="No confirmed catalyst found",
            freshness=DataFreshness(
                as_of=FIXED,
                provider="catalyst-service",
                state=FreshnessState.FRESH,
            ),
            coverage=CoverageState.COMPLETE,
            quiet=True,
            prior_close=200.0,
            last_price=201.0,
            move_percent=0.5,
        )

    def event_markers(self, symbol: str):
        return []

    def submit_feedback(self, catalyst_id, kind, *, symbol=None, note=""):
        event = CatalystFeedbackEvent(
            feedback_id="fb-1",
            kind=kind,
            catalyst_id=catalyst_id,
            symbol=symbol,
            note=note,
            recorded_at=FIXED,
        )
        self.feedback.append(event)
        return event

    def public_settings(self):
        return self.settings

    def update_settings(self, payload):
        self.settings.update({k: v for k, v in payload.items() if k != "benzinga"})
        if "benzinga_enabled" in payload:
            enabled = bool(payload["benzinga_enabled"])
            self.settings["benzinga"] = {
                "enabled": enabled,
                "status": "enabled" if enabled else "disabled",
            }
        return self.settings


def _client(monkeypatch, portfolio=None, catalysts=None) -> TestClient:
    catalyst_service = catalysts or CatalystStub()
    monkeypatch.setattr(
        "position_pilot.web.app.get_catalyst_service",
        lambda: catalyst_service,
    )
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
            enable_streaming=False,
        ),
        portfolio_service=portfolio or PortfolioStub(),
    )
    client = TestClient(app)
    client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})
    return client


def test_catalyst_endpoints_scan_and_feedback(monkeypatch) -> None:
    catalysts = CatalystStub()
    client = _client(monkeypatch, catalysts=catalysts)

    scan = client.get("/api/v1/catalysts")
    assert scan.status_code == 200
    body = scan.json()
    assert body["results"][0]["symbol"] == "AAPL"
    assert body["results"][0]["summary"] == "No confirmed catalyst found"
    assert "api_key" not in scan.text.lower()
    assert "token" not in body["settings"]

    single = client.get("/api/v1/catalysts/AAPL")
    assert single.status_code == 200
    assert single.json()["confidence"] == "no_confirmed_catalyst_found"

    feedback = client.post(
        "/api/v1/catalysts/feedback",
        json={
            "kind": "missing_catalyst",
            "symbol": "AAPL",
            "note": "Supplier outage omitted",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["kind"] == "missing_catalyst"
    assert len(catalysts.feedback) == 1

    settings = client.get("/api/v1/settings/catalysts")
    assert settings.status_code == 200
    assert settings.json()["news_cadence_seconds"] == 300

    updated = client.put(
        "/api/v1/settings/catalysts",
        json={"news_cadence_seconds": 180, "benzinga_enabled": False},
    )
    assert updated.status_code == 200
    assert updated.json()["benzinga"]["status"] == "disabled"


def test_bootstrap_reports_catalyst_phase(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/api/v1/bootstrap")
    assert response.status_code == 200
    payload = response.json()
    assert payload["application"]["phase"] == "catalyst-intelligence"
    assert "catalysts" in payload
    assert payload["catalysts"]["stock_move_threshold_pct"] == 2.0
