"""FastAPI smoke for recommendations, alerts, and monitoring."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from position_pilot.domain.alerts import AlertCategory, AlertService, AlertSeverity
from position_pilot.domain.monitoring import MonitoringService
from position_pilot.domain.notifications import NotificationService
from position_pilot.domain.recommendations import RecommendationService
from position_pilot.domain.snapshots import (
    AccountSnapshot,
    DataFreshness,
    FreshnessState,
    PortfolioSnapshot,
    PortfolioTotals,
    PositionHorizon,
    SnapshotState,
    StrategySnapshot,
)
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.providers.codex import (
    SCHEMA_VERSION,
    CodexInvocationResult,
    CodexProviderStatus,
    CodexStructuredOutput,
    RecommendationAction,
    RecommendationRisk,
)
from position_pilot.web.app import WebSettings, create_app


class FakeCodex:
    def public_status(self) -> str:
        return "configured"

    def complete_recommendation(self, context: dict) -> CodexInvocationResult:
        return CodexInvocationResult(
            status=CodexProviderStatus.OK,
            output=CodexStructuredOutput(
                schema_version=SCHEMA_VERSION,
                action=RecommendationAction.HOLD,
                urgency=2,
                risk=RecommendationRisk.LOW,
                reasoning="Hold for theta.",
                evidence=["dte"],
                catalyst_refs=[],
            ),
        )


def _strategy() -> StrategySnapshot:
    return StrategySnapshot(
        strategy_id="strat-browser",
        account_id="public-account-id",
        underlying="SPY",
        strategy_type="Short Put",
        expiration_date="2026-08-21",
        days_to_expiration=21,
        quantity=1,
        strikes="$500",
        unrealized_pnl=40,
        unrealized_pnl_percent=10,
        total_delta=-20,
        total_theta=4,
        horizon=PositionHorizon.TACTICAL,
        legs=[],
    )


def _portfolio_snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        snapshot_id="snapshot-1",
        captured_at=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
        state=SnapshotState.LIVE,
        freshness=DataFreshness(
            as_of=datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
            provider="tastytrade",
            state=FreshnessState.FRESH,
        ),
        accounts=[
            AccountSnapshot(
                account_id="public-account-id",
                label="Individual 1",
                account_type="Individual",
                net_liquidating_value=25_000,
            )
        ],
        strategies=[_strategy()],
        totals=PortfolioTotals(net_liquidating_value=25_000, unrealized_pnl=40),
        selected_account_id="all",
    )


class PortfolioService:
    def __init__(self) -> None:
        self.snapshot = _portfolio_snapshot()

    def latest(self, account_id: str = "all") -> PortfolioSnapshot | None:
        return self.snapshot.for_account(account_id)

    def refresh(self) -> PortfolioSnapshot:
        return self.snapshot

    def primary_account_id(self) -> str:
        return "all"

    def set_primary_account(self, account_id: str) -> None:
        return None


def _client(tmp_path, monkeypatch):
    db = PositionPilotDatabase(tmp_path / "api.sqlite3", backup_directory=tmp_path / "backups")
    recs = RecommendationService(db, provider=FakeCodex(), clock=lambda: datetime.now(UTC))
    alerts = AlertService(db)
    notes = NotificationService(enabled=False)
    portfolio = PortfolioService()
    monitoring = MonitoringService(
        db,
        recs,
        alerts,
        notes,
        portfolio_loader=lambda: portfolio.latest(),
    )

    monkeypatch.setattr("position_pilot.web.app.get_recommendation_service", lambda: recs)
    monkeypatch.setattr("position_pilot.web.app.get_alert_service", lambda: alerts)
    monkeypatch.setattr("position_pilot.web.app.get_monitoring_service", lambda: monitoring)
    monkeypatch.setattr(
        "position_pilot.web.app.get_catalyst_service",
        lambda: type(
            "C",
            (),
            {
                "public_settings": lambda self: {
                    "stock_move_threshold_pct": 2.0,
                    "etf_move_threshold_pct": 1.0,
                    "news_cadence_seconds": 300,
                    "benzinga": {"enabled": False, "status": "disabled"},
                    "scheduled_window_hours": 72,
                },
                "analyze_symbol": lambda self, symbol: type(
                    "R",
                    (),
                    {
                        "catalysts": [],
                        "prior_close": 100,
                        "lookback_start": datetime.now(UTC),
                        "lookback_end": datetime.now(UTC),
                        "model_dump": lambda **_: {},
                    },
                )(),
                "event_markers": lambda self, symbol: [],
                "apply_retention": lambda self: None,
            },
        )(),
    )

    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
            enable_streaming=False,
        ),
        portfolio_service=portfolio,
    )
    client = TestClient(app)
    client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})
    return client, recs, alerts, monitoring


def test_bootstrap_reports_codex_monitoring_phase(tmp_path, monkeypatch) -> None:
    client, _, _, _ = _client(tmp_path, monkeypatch)
    response = client.get("/api/v1/bootstrap")
    assert response.status_code == 200
    payload = response.json()
    assert payload["application"]["phase"] == "codex-monitoring"
    assert "enabled" in payload["monitoring"]
    assert payload["recommendations"]["selected_provider"] == "codex-cli"
    assert payload["recommendations"].get("api_key_fallback_available") is False
    assert "private" not in response.text.lower() or "private-client" not in response.text


def test_monitoring_consent_and_status_endpoints(tmp_path, monkeypatch) -> None:
    client, _, _, monitoring = _client(tmp_path, monkeypatch)
    status = client.get("/api/v1/monitoring")
    assert status.status_code == 200
    assert status.json()["enabled"] is False

    consent = client.put("/api/v1/monitoring/consent", json={"enabled": True})
    assert consent.status_code == 200
    assert consent.json()["consent"]["enabled"] is True
    assert monitoring.get_consent().enabled is True


def test_recommendation_evaluate_and_decision_flow(tmp_path, monkeypatch) -> None:
    client, recs, _, _ = _client(tmp_path, monkeypatch)
    record = recs.evaluate_strategy(_strategy())
    listed = client.get("/api/v1/recommendations")
    assert listed.status_code == 200
    assert any(row["recommendation_id"] == record.recommendation_id for row in listed.json())

    decision = client.post(
        f"/api/v1/recommendations/{record.recommendation_id}/decisions",
        json={"decision": "deferred", "note": "Wait for open"},
    )
    assert decision.status_code == 200
    assert decision.json()["decision"] == "deferred"


def test_alert_acknowledge_snooze_resolve_mute(tmp_path, monkeypatch) -> None:
    client, _, alerts, _ = _client(tmp_path, monkeypatch)
    alert = alerts.raise_alert(
        category=AlertCategory.PROVIDER_HEALTH,
        severity=AlertSeverity.WARNING,
        alert_type="provider_signed_out",
        title="Codex signed out",
        summary="Sign in to Codex CLI",
        source="codex-cli",
        symbol="SPY",
        strategy_type="Short Put",
    )
    assert alert is not None
    ack = client.post(f"/api/v1/alerts/{alert.alert_id}/acknowledge")
    assert ack.status_code == 200
    assert ack.json()["resolution"] == "acknowledged"

    snooze = client.post(f"/api/v1/alerts/{alert.alert_id}/snooze", json={"minutes": 30})
    assert snooze.status_code == 200
    assert snooze.json()["resolution"] == "snoozed"

    resolve = client.post(f"/api/v1/alerts/{alert.alert_id}/resolve")
    assert resolve.status_code == 200
    assert resolve.json()["resolution"] == "resolved"

    mute = client.post(
        "/api/v1/alerts/mute",
        json={"category": "provider_health", "alert_type": "provider_signed_out"},
    )
    assert mute.status_code == 200


def test_recommendation_settings_default_disables_fallback(tmp_path, monkeypatch) -> None:
    client, _, _, _ = _client(tmp_path, monkeypatch)
    settings = client.get("/api/v1/settings/recommendations")
    assert settings.status_code == 200
    assert settings.json()["api_key_fallback_enabled"] is False
    assert settings.json().get("api_key_fallback_available") is False
    assert settings.json()["selected_provider"] == "codex-cli"

    updated = client.put(
        "/api/v1/settings/recommendations",
        json={
            "rich_notification_preview": True,
            "api_key_fallback_enabled": True,
            "selected_provider": "api-key-fallback",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["rich_notification_preview"] is True
    assert updated.json()["selected_provider"] == "codex-cli"
    assert updated.json()["api_key_fallback_enabled"] is False


def test_strategy_recommend_endpoint(tmp_path, monkeypatch) -> None:
    client, _, _, _ = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "position_pilot.web.app.get_plans_service",
        lambda: type(
            "P",
            (),
            {
                "get_thesis": lambda self, sid: None,
                "get_trade_plan": lambda self, sid: None,
            },
        )(),
    )
    response = client.post(
        "/api/v1/strategies/strat-browser/recommend",
        json={"force": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider_status"] in {"ok", "skipped_unchanged"}
    assert body["action"] == "hold"
