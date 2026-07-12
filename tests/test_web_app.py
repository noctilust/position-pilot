from fastapi.testclient import TestClient

from position_pilot.domain.snapshots import (
    AccountSnapshot,
    DataFreshness,
    PortfolioSnapshot,
    SnapshotState,
)
from position_pilot.web.app import WebSettings, create_app


def test_health_endpoint_describes_the_local_service() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        )
    )

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "position-pilot",
        "version": "0.1.0",
    }
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-security-policy"].startswith("default-src 'self'")


def test_launch_token_is_exchanged_for_a_private_session_cookie() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/session/exchange",
        json={"launch_token": "launch-secret"},
    )

    assert response.status_code == 204
    assert client.cookies.get("position_pilot_session") == "session-secret"
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]

    replay = TestClient(app).post(
        "/api/v1/session/exchange",
        json={"launch_token": "launch-secret"},
    )
    assert replay.status_code == 401


def test_bootstrap_rejects_a_browser_without_a_local_session() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        )
    )

    response = TestClient(app).get("/api/v1/bootstrap")

    assert response.status_code == 401


def test_authenticated_bootstrap_reports_capabilities_without_secrets(monkeypatch) -> None:
    monkeypatch.setenv("TASTYTRADE_CLIENT_SECRET", "private-client-secret")
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "private-refresh-token")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)

    class CatalystSettingsStub:
        def public_settings(self):
            return {
                "stock_move_threshold_pct": 2.0,
                "etf_move_threshold_pct": 1.0,
                "news_cadence_seconds": 300,
                "benzinga": {"enabled": False, "status": "disabled"},
                "scheduled_window_hours": 72,
            }

    class MonitoringStub:
        def public_bootstrap(self):
            return {
                "market_timezone": "America/New_York",
                "window_start": "07:30",
                "window_end": "18:00",
                "evaluation_minutes": 30,
                "risk_refresh_seconds": 60,
                "enabled": False,
                "consented": False,
                "inside_window": False,
                "is_trading_day": True,
                "is_holiday": False,
                "is_early_close": False,
                "provider_status": "not_checked",
                "running": False,
                "notice": "Monitoring is disabled until you grant onboarding consent.",
                "last_evaluation_at": None,
            }

        async def start(self):
            return None

        async def stop(self):
            return None

    class RecommendationStub:
        def provider_public_status(self):
            return "not_checked"

        def settings(self):
            return {
                "selected_provider": "codex-cli",
                "api_key_fallback_available": False,
                "api_key_fallback_enabled": False,
                "rich_notification_preview": False,
            }

    monkeypatch.setattr(
        "position_pilot.web.app.get_catalyst_service",
        lambda: CatalystSettingsStub(),
    )
    monkeypatch.setattr(
        "position_pilot.web.app.get_monitoring_service",
        lambda: MonitoringStub(),
    )
    monkeypatch.setattr(
        "position_pilot.web.app.get_recommendation_service",
        lambda: RecommendationStub(),
    )
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
            enable_streaming=False,
        )
    )
    client = TestClient(app)
    client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})

    response = client.get("/api/v1/bootstrap")

    assert response.status_code == 200
    payload = response.json()
    assert payload["application"] == {
        "name": "Position Pilot",
        "version": "0.1.0",
        "phase": "hardening-retirement",
    }
    assert payload["providers"] == {
        "tastytrade": "configured",
        "codex": "not_checked",
        "massive": "not_configured",
        "benzinga": "not_configured",
    }
    assert payload["monitoring"]["market_timezone"] == "America/New_York"
    assert payload["monitoring"]["enabled"] is False
    assert payload["recommendations"]["selected_provider"] == "codex-cli"
    assert payload["recommendations"].get("api_key_fallback_available") is False
    assert payload["data_state"] == "awaiting_portfolio_snapshot"
    assert "private-client-secret" not in response.text
    assert "private-refresh-token" not in response.text


def test_web_app_loads_tastytrade_configuration_from_the_working_directory(
    monkeypatch,
    tmp_path,
) -> None:
    (tmp_path / ".env").write_text(
        "TASTYTRADE_CLIENT_SECRET=file-client-secret\nTASTYTRADE_REFRESH_TOKEN=file-refresh-token\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TASTYTRADE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TASTYTRADE_REFRESH_TOKEN", raising=False)
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        )
    )
    client = TestClient(app)
    client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})

    response = client.get("/api/v1/bootstrap")

    assert response.json()["providers"]["tastytrade"] == "configured"
    assert "file-client-secret" not in response.text
    assert "file-refresh-token" not in response.text


def test_non_loopback_clients_are_rejected() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=True,
        )
    )

    response = TestClient(app, client=("203.0.113.8", 49152)).get("/api/v1/health")

    assert response.status_code == 403
    assert response.json() == {"detail": "Position Pilot accepts local connections only."}


def test_untrusted_host_headers_are_rejected_even_from_loopback() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=True,
        )
    )

    response = TestClient(app, client=("127.0.0.1", 49152)).get(
        "/api/v1/health",
        headers={"host": "attacker.example"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid local dashboard host."}


def test_root_serves_the_packaged_dashboard() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        )
    )

    response = TestClient(app).get("/?launch_token=launch-secret")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Position Pilot</title>" in response.text


class StubPortfolioService:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.snapshot = PortfolioSnapshot(
            snapshot_id="snapshot-1",
            captured_at="2026-07-11T16:30:00Z",
            state=SnapshotState.LIVE,
            freshness=DataFreshness(
                as_of="2026-07-11T16:30:00Z",
                provider="tastytrade",
            ),
            accounts=[
                AccountSnapshot(
                    account_id="public-account-id",
                    label="Individual 1",
                    account_type="Individual",
                    net_liquidating_value=25_000,
                )
            ],
        )

    def latest(self, account_id: str = "all") -> PortfolioSnapshot | None:
        return self.snapshot.for_account(account_id)

    def refresh(self) -> PortfolioSnapshot:
        self.refresh_count += 1
        return self.snapshot


def test_authenticated_portfolio_endpoint_returns_scoped_browser_safe_snapshot() -> None:
    service = StubPortfolioService()
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        ),
        portfolio_service=service,
    )
    client = TestClient(app)
    client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})

    response = client.get("/api/v1/portfolio?account_id=public-account-id")

    assert response.status_code == 200
    assert response.json()["selected_account_id"] == "public-account-id"
    assert response.json()["accounts"][0]["label"] == "Individual 1"
    assert "5WT" not in response.text
    assert service.refresh_count == 0


def test_portfolio_endpoint_can_request_a_live_refresh() -> None:
    service = StubPortfolioService()
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        ),
        portfolio_service=service,
    )
    client = TestClient(app)
    client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})

    response = client.get("/api/v1/portfolio?refresh=true")

    assert response.status_code == 200
    assert service.refresh_count == 1


def test_streaming_status_is_explicit_when_runtime_is_disabled() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
            enable_streaming=False,
        )
    )
    with TestClient(app) as client:
        client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})

        response = client.get("/api/v1/streaming/status")

    assert response.status_code == 200
    assert response.json() == {
        "market": {"state": "disabled", "error": None},
        "account": {"state": "disabled", "error": None},
    }


def test_live_events_requires_a_local_session() -> None:
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
            enable_streaming=False,
        )
    )

    denied = TestClient(app).get("/api/v1/events")
    assert denied.status_code == 401


def test_broker_outage_does_not_prevent_cached_dashboard_startup(monkeypatch) -> None:
    class FailingPortfolioService(StubPortfolioService):
        def latest(self, account_id: str = "all") -> PortfolioSnapshot | None:
            raise ConnectionError("offline")

    class FailingClient:
        def get_accounts(self):
            raise ConnectionError("offline")

    monkeypatch.setenv("TASTYTRADE_CLIENT_SECRET", "configured")
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "configured")
    monkeypatch.setattr("position_pilot.web.app.get_database", lambda: object())
    monkeypatch.setattr("position_pilot.web.app.get_client", lambda: FailingClient())
    app = create_app(
        WebSettings(
            launch_token="launch-secret",
            session_token="session-secret",
            enforce_loopback=False,
        ),
        portfolio_service=FailingPortfolioService(),
    )

    with TestClient(app) as client:
        client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})
        health = client.get("/api/v1/health")
        streaming = client.get("/api/v1/streaming/status")

    assert health.status_code == 200
    assert streaming.json() == {
        "market": {"state": "degraded", "error": "BrokerStateUnavailable"},
        "account": {"state": "degraded", "error": "BrokerStateUnavailable"},
    }
