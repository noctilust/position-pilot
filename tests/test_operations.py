"""Phase 7 operations: exports, diagnostics, backups, retention, update readiness."""

from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from position_pilot import __version__
from position_pilot.domain.operations import (
    DISCLAIMER,
    PORTABLE_BROKER_ACCOUNT_SENTINEL,
    PORTABLE_FULL_TEXT_SENTINEL,
    OperationsService,
    _simple_pdf,
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
from position_pilot.persistence.sqlite import CURRENT_SCHEMA_VERSION, PositionPilotDatabase
from position_pilot.web.app import WebSettings, create_app


def _snapshot(*, strategy_count: int = 1) -> PortfolioSnapshot:
    now = datetime(2026, 7, 11, 16, 0, tzinfo=UTC)
    strategies: list[StrategySnapshot] = []
    legs: list[PositionSnapshot] = []
    for index in range(strategy_count):
        leg = PositionSnapshot(
            symbol=f"SPY260821P{500 + index:05d}000",
            underlying_symbol="SPY" if index % 2 == 0 else f"T{index}",
            quantity=-1,
            quantity_direction=QuantityDirection.SHORT,
            position_type=PositionType.EQUITY_OPTION,
            strike_price=500 + index,
            option_type="P",
            expiration_date="2026-08-21",
            days_to_expiration=41,
            market_value=250,
            unrealized_pnl=100,
            delta=-0.2,
            theta=-0.05,
            gamma=0.01,
            vega=0.08,
            implied_volatility=0.18,
            multiplier=100,
            horizon=PositionHorizon.TACTICAL,
        )
        legs.append(leg)
        strategies.append(
            StrategySnapshot(
                strategy_id=f"acct-1:U{index}:short-put",
                account_id="acct-1",
                underlying=leg.underlying_symbol,
                strategy_type="Short Put",
                expiration_date="2026-08-21",
                days_to_expiration=41,
                quantity=1,
                strikes=f"{leg.strike_price}P",
                unrealized_pnl=100,
                total_delta=-0.2,
                total_theta=-0.05,
                horizon=PositionHorizon.TACTICAL,
                legs=[leg],
            )
        )
    account = AccountSnapshot(
        account_id="acct-1",
        label="Primary",
        account_type="Margin",
        net_liquidating_value=100_000,
        cash_balance=50_000,
        buying_power=80_000,
        positions=legs,
    )
    return PortfolioSnapshot(
        schema_version=1,
        snapshot_id="snap-ops-1",
        captured_at=now,
        state=SnapshotState.LIVE,
        freshness=DataFreshness(
            as_of=now,
            provider="tastytrade",
            state=FreshnessState.FRESH,
        ),
        accounts=[account],
        strategies=strategies,
        totals=PortfolioTotals(
            net_liquidating_value=100_000,
            cash_balance=50_000,
            buying_power=80_000,
            unrealized_pnl=100 * strategy_count,
        ),
    )


class _StubPortfolio:
    def __init__(self, snapshot: PortfolioSnapshot) -> None:
        self._snapshot = snapshot

    def latest(self, account_id: str = "all") -> PortfolioSnapshot | None:
        if account_id == "all":
            return self._snapshot
        return self._snapshot.for_account(account_id)

    def refresh(self) -> PortfolioSnapshot:
        return self._snapshot


@pytest.fixture()
def ops_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OperationsService:
    monkeypatch.setenv("POSITION_PILOT_DATA_DIR", str(tmp_path / "data"))
    db_path = tmp_path / "data" / "position-pilot.sqlite3"
    database = PositionPilotDatabase(db_path, backup_directory=tmp_path / "backups")
    snapshot = _snapshot()
    database.save_portfolio_snapshot(snapshot)
    # Seed a sensitive setting that must be redacted / excluded from allowlist.
    database.set_setting(
        "demo.secrets",
        {"refresh_token": "super-secret-token-value-0123456789", "theme": "dark"},
    )
    database.set_setting("default_account", PORTABLE_BROKER_ACCOUNT_SENTINEL)
    database.set_setting(
        "account_number",
        PORTABLE_BROKER_ACCOUNT_SENTINEL,
    )
    database.set_setting("operations.retention", {"portfolio_snapshots_days": 365})
    # Seed broker identity + licensed full text for portable backup tests.
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO account_identities(
                account_number, account_id, label, account_type, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                PORTABLE_BROKER_ACCOUNT_SENTINEL,
                "acct-1",
                "Primary",
                "Margin",
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO catalyst_articles(
                article_id, symbol, provider, url, title, source_name,
                published_at, excerpt, full_text, stored_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "art-1",
                "SPY",
                "benzinga",
                "https://example.test/a",
                "Headline",
                "Benzinga",
                datetime.now(UTC).isoformat(),
                "excerpt only",
                PORTABLE_FULL_TEXT_SENTINEL,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO roll_chains(
                account_id, chain_key, underlying, payload_json, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "acct-1",
                "spy-1",
                "SPY",
                json.dumps({"account_number": PORTABLE_BROKER_ACCOUNT_SENTINEL, "rolls": 1}),
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO audit_events(
                event_id, strategy_id, action, summary, payload_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "aud-1",
                "s1",
                "note",
                "kept",
                json.dumps({"ok": True}),
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO trader_decisions(
                decision_id, recommendation_id, subject_type, subject_id,
                decision, note, recorded_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "dec-1",
                "rec-1",
                "strategy",
                "s1",
                "accepted",
                "note",
                datetime.now(UTC).isoformat(),
                json.dumps({"ok": True}),
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO recommendation_history(
                history_id, recommendation_id, subject_type, subject_id,
                kind, recorded_at, day_key, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hist-1",
                "rec-1",
                "strategy",
                "s1",
                "evaluation",
                datetime.now(UTC).isoformat(),
                None,
                json.dumps({"action": "hold"}),
            ),
        )
        connection.commit()

    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(".env\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("TASTYTRADE_REFRESH_TOKEN=should-not-leak\n", encoding="utf-8")
    env_path.chmod(0o644)

    service = OperationsService(
        database=database,
        portfolio=_StubPortfolio(snapshot),  # type: ignore[arg-type]
        data_directory=tmp_path / "data",
        project_root=tmp_path,
        monitoring_active_fn=lambda: False,
        provider_status_fn=lambda: {
            "tastytrade": "configured",
            "codex": "configured",
            "codex_identity": "user@secret.example",
        },
        clock=lambda: datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
    )
    return service


def test_portfolio_csv_export_includes_disclaimer(ops_env: OperationsService) -> None:
    filename, body = ops_env.export_portfolio_csv()
    assert filename.endswith(".csv")
    assert "SPY" in body
    assert "acct-1" in body
    assert DISCLAIMER in body
    assert "super-secret" not in body


def test_history_csv_export(ops_env: OperationsService) -> None:
    filename, body = ops_env.export_history_csv()
    assert "portfolio-history" in filename
    assert "snap-ops-1" in body
    assert DISCLAIMER in body


def test_diagnostic_bundle_strict_allowlist_and_redaction(ops_env: OperationsService) -> None:
    bundle = ops_env.diagnostic_bundle()
    payload = bundle.model_dump(mode="json")
    text = str(payload)
    assert "super-secret-token-value" not in text
    assert "should-not-leak" not in text
    assert PORTABLE_BROKER_ACCOUNT_SENTINEL not in text
    assert "user@secret.example" not in text
    assert "/Users/" not in text
    assert "/home/" not in text
    # Strict allowlist: non-allowlisted secret settings never appear.
    assert "demo.secrets" not in payload["settings_redacted"]
    assert "default_account" not in payload["settings_redacted"]
    assert "account_number" not in payload["settings_redacted"]
    assert "operations.retention" in payload["settings_redacted"]
    assert payload["env_diagnostics"]["path"] == ".env"
    assert bundle.env_diagnostics.broadly_readable is True
    assert bundle.env_diagnostics.warnings
    assert "allowlist" in payload["redaction"]
    assert "credential values" in " ".join(payload["redaction"]["excluded"]).lower() or (
        "credential values" in str(payload["redaction"]).lower()
    )


def test_env_diagnostics_path_is_logical(ops_env: OperationsService) -> None:
    report = ops_env.env_diagnostics()
    assert report.path == ".env"
    assert str(ops_env.project_root) not in report.path
    dumped = report.model_dump(mode="json")
    assert dumped["path"] == ".env"
    assert "/Users/" not in str(dumped)
    assert "/home/" not in str(dumped)


def test_printable_pdf_covers_200_strategies_multipage(ops_env: OperationsService) -> None:
    ops_env.portfolio = _StubPortfolio(_snapshot(strategy_count=200))  # type: ignore[assignment]
    filename, data = ops_env.printable_pdf()
    assert filename.endswith(".pdf")
    assert data.startswith(b"%PDF-1.4")
    assert b"%%EOF" in data
    assert b"/MediaBox [0 0 612 792]" in data
    assert b"/Count " in data
    # Multi-page: more than one page object for 200 strategies.
    assert data.count(b"/Type /Page ") >= 2
    # Attribution / schema / disclaimer markers.
    assert b"Schema:" in data or b"Schema" in data
    assert b"tastytrade" in data or b"Provider" in data
    assert b"decision support" in data.lower() or b"Decision support" in data
    # All strategies present in content streams (underlying markers).
    assert data.count(b"Short Put") >= 200
    # Helper stays valid.
    assert _simple_pdf(["line"]).startswith(b"%PDF")


def test_html_snapshot_attributed(ops_env: OperationsService) -> None:
    filename, html = ops_env.printable_html()
    assert filename.endswith(".html")
    assert "Position Pilot" in html
    assert "Decision support" in html or DISCLAIMER.split(".")[0] in html
    assert "snap-ops-1" in html
    assert "Provider:" in html
    assert "tastytrade" in html
    assert "As-of:" in html
    assert "Schema" in html
    assert DISCLAIMER.split(".")[0] in html or "decision support" in html.lower()


def test_backup_create_list_uses_basename_and_unified_sidecar(
    ops_env: OperationsService,
) -> None:
    created = ops_env.create_backup(reason="manual")
    assert created.integrity_ok
    assert created.backup_id.startswith("position-pilot-")
    assert created.path == created.filename == created.backup_id
    assert "/" not in created.path
    assert "\\" not in created.path
    assert created.app_version == __version__
    assert created.schema_version == CURRENT_SCHEMA_VERSION
    assert created.excludes
    meta_path = ops_env.database.backup_directory / f"{created.backup_id}.meta.json"
    # Sidecar is next to file as .sqlite3.meta.json
    meta_path = ops_env.backup_path(created.backup_id).with_suffix(
        ops_env.backup_path(created.backup_id).suffix + ".meta.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["app_version"] == __version__
    assert meta["schema_version"] == CURRENT_SCHEMA_VERSION
    assert "integrity" in meta
    assert "excludes" in meta
    listed = ops_env.list_backups()
    assert any(item.backup_id == created.backup_id for item in listed)
    for item in listed:
        assert item.path == item.backup_id
        assert "/Users/" not in item.path


def test_portable_browser_backup_strips_sentinels(ops_env: OperationsService) -> None:
    created = ops_env.create_backup(reason="manual")
    # Faithful server-local backup still has sentinels.
    faithful = ops_env.backup_path(created.backup_id).read_bytes()
    assert PORTABLE_FULL_TEXT_SENTINEL.encode() in faithful
    assert PORTABLE_BROKER_ACCOUNT_SENTINEL.encode() in faithful

    filename, archive = ops_env.portable_backup_archive(created.backup_id)
    assert filename.endswith(".zip")
    assert PORTABLE_FULL_TEXT_SENTINEL.encode() not in archive
    assert PORTABLE_BROKER_ACCOUNT_SENTINEL.encode() not in archive
    assert b"super-secret" not in archive
    assert b"should-not-leak" not in archive

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = set(zf.namelist())
        assert "position-pilot-portable.sqlite3" in names
        assert "position-pilot-portable.meta.json" in names
        meta = json.loads(zf.read("position-pilot-portable.meta.json"))
        assert meta["portable"] is True
        assert "catalyst full_text" in " ".join(meta["excludes"])
        db_bytes = zf.read("position-pilot-portable.sqlite3")
        assert PORTABLE_FULL_TEXT_SENTINEL.encode() not in db_bytes
        assert PORTABLE_BROKER_ACCOUNT_SENTINEL.encode() not in db_bytes
        # Load portable DB and verify full_text null + account pseudonymized.
        tmp = ops_env.data_directory / "check-portable.sqlite3"
        tmp.write_bytes(db_bytes)
        with sqlite3.connect(tmp) as connection:
            full = connection.execute(
                "SELECT full_text FROM catalyst_articles WHERE article_id = 'art-1'"
            ).fetchone()
            assert full is not None
            assert full[0] in (None, "")
            accounts = connection.execute(
                "SELECT account_number FROM account_identities"
            ).fetchall()
            assert accounts
            assert all(PORTABLE_BROKER_ACCOUNT_SENTINEL not in str(row[0]) for row in accounts)
            assert all(str(row[0]).startswith("acct-") for row in accounts)


def test_backup_create_list_restore_roundtrip(ops_env: OperationsService) -> None:
    created = ops_env.create_backup(reason="manual")
    assert created.integrity_ok
    ops_env.database.set_setting("marker", "after-backup")
    assert ops_env.database.get_setting("marker") == "after-backup"

    result = ops_env.restore_backup(created.backup_id, confirm=True)
    assert result.restored is True
    assert result.pre_restore_backup_id is not None
    assert ops_env.database.get_setting("marker") is None


def test_restore_blocked_while_monitoring_active(ops_env: OperationsService) -> None:
    ops_env.monitoring_active_fn = lambda: True
    created = ops_env.create_backup(reason="manual")
    with pytest.raises(RuntimeError, match="monitoring"):
        ops_env.restore_backup(created.backup_id, confirm=True)


def test_restore_and_update_fail_closed_on_probe_error(ops_env: OperationsService) -> None:
    def boom() -> bool:
        raise RuntimeError("status unavailable")

    ops_env.monitoring_active_fn = boom
    created = ops_env.create_backup(reason="manual")
    with pytest.raises(RuntimeError, match="monitoring"):
        ops_env.restore_backup(created.backup_id, confirm=True)
    readiness = ops_env.update_readiness()
    assert readiness.monitoring_active is True
    assert readiness.blocked_reason


def test_restore_rejects_newer_backup_schema(ops_env: OperationsService) -> None:
    created = ops_env.create_backup(reason="manual")
    path = ops_env.backup_path(created.backup_id)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    future = CURRENT_SCHEMA_VERSION + 99
    # Authoritative SQLite schema_migrations must be future; keep sidecar in sync.
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (future, datetime.now(UTC).isoformat()),
        )
        connection.commit()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema_version"] = future
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        ops_env.restore_backup(created.backup_id, confirm=True)
    # Live DB must remain intact.
    assert ops_env.database.get_setting("operations.retention") is not None


def test_restore_rejects_future_db_schema_with_stale_sidecar(ops_env: OperationsService) -> None:
    """Actual SQLite schema_migrations is authoritative; stale sidecar mismatch is rejected."""

    created = ops_env.create_backup(reason="manual")
    path = ops_env.backup_path(created.backup_id)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    future = CURRENT_SCHEMA_VERSION + 42
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (future, datetime.now(UTC).isoformat()),
        )
        connection.commit()
    # Sidecar left at the original (stale) version.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert int(meta["schema_version"]) == CURRENT_SCHEMA_VERSION
    with pytest.raises(ValueError, match="mismatch"):
        ops_env.restore_backup(created.backup_id, confirm=True)
    assert ops_env.database.get_setting("operations.retention") is not None


def test_restore_requires_confirm(ops_env: OperationsService) -> None:
    created = ops_env.create_backup(reason="manual")
    with pytest.raises(ValueError, match="confirm"):
        ops_env.restore_backup(created.backup_id, confirm=False)


def test_backup_path_rejects_traversal(ops_env: OperationsService) -> None:
    with pytest.raises(ValueError):
        ops_env.backup_path("../etc/passwd")
    with pytest.raises(ValueError):
        ops_env.backup_path("position-pilot-manual-x.sqlite3/../../secret")


def test_retention_never_clears_audit_critical_and_compacts_snapshots(
    ops_env: OperationsService,
) -> None:
    # Seed old 30-minute snapshots beyond the retention window (valid full payloads).
    old = datetime.now(UTC) - timedelta(days=400)
    day_key = old.date().isoformat()
    source_snap = _snapshot()
    with sqlite3.connect(ops_env.database.path) as connection:
        for index in range(3):
            payload = source_snap.model_copy(
                update={
                    "snapshot_id": f"old-snap-{index}",
                    "captured_at": old + timedelta(hours=index),
                    "totals": PortfolioTotals(
                        net_liquidating_value=100_000 + index,
                        cash_balance=50_000,
                        buying_power=80_000,
                        unrealized_pnl=250 + index,
                    ),
                }
            )
            connection.execute(
                """
                INSERT INTO portfolio_snapshots(
                    snapshot_id, schema_version, captured_at, state, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"old-snap-{index}",
                    payload.schema_version,
                    (old + timedelta(hours=index)).isoformat(),
                    "live",
                    payload.model_dump_json(),
                    old.isoformat(),
                ),
            )
        connection.commit()

    # clear_confirmed must be rejected.
    with pytest.raises(ValueError, match="indefinite"):
        ops_env.update_retention_settings({"transaction_history": "clear_confirmed"})

    preview = ops_env.retention_preview()
    assert preview.settings.transaction_history == "indefinite"
    assert "recommendation history" in " ".join(preview.audit_critical_preserved).lower()
    assert preview.would_delete.get("recommendation_history", 0) == 0
    assert preview.would_delete.get("roll_chains", 0) == 0
    assert preview.would_compact.get("portfolio_snapshots_to_daily", 0) >= 1

    with pytest.raises(ValueError, match="confirm"):
        ops_env.apply_retention(confirm=False)

    applied = ops_env.apply_retention(confirm=True)
    assert applied["deleted"].get("recommendation_history", 0) == 0
    assert (
        applied["deleted"].get("roll_chains") is None
        or applied["deleted"].get("roll_chains", 0) == 0
    )

    with sqlite3.connect(ops_env.database.path) as connection:
        rolls = connection.execute("SELECT COUNT(*) FROM roll_chains").fetchone()[0]
        audits = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
        decisions = connection.execute("SELECT COUNT(*) FROM trader_decisions").fetchone()[0]
        recs = connection.execute("SELECT COUNT(*) FROM recommendation_history").fetchone()[0]
        daily = connection.execute(
            "SELECT COUNT(*) FROM portfolio_snapshots WHERE state = 'daily'"
        ).fetchone()[0]
        old_live = connection.execute(
            "SELECT COUNT(*) FROM portfolio_snapshots WHERE snapshot_id LIKE 'old-snap-%'"
        ).fetchone()[0]
        daily_payload = connection.execute(
            "SELECT payload_json FROM portfolio_snapshots WHERE state = 'daily' LIMIT 1"
        ).fetchone()[0]
    assert rolls >= 1
    assert audits >= 1
    assert decisions >= 1
    assert recs >= 1
    assert daily >= 1
    assert old_live == 0

    # Readers must accept compacted daily rows without Pydantic failures.
    payloads = ops_env.database.list_portfolio_snapshot_payloads(limit=50)
    daily_models = [row for row in payloads if row.state.value == "daily"]
    assert daily_models
    daily_model = daily_models[0]
    assert daily_model.snapshot_id == f"daily-{day_key}"
    assert daily_model.compaction is not None
    assert daily_model.compaction.kind == "daily_summary"
    assert daily_model.totals.net_liquidating_value == 100_002  # last source of the day
    assert daily_model.totals.unrealized_pnl == 252
    assert len(daily_model.accounts) == 1
    assert len(daily_model.strategies) == 1
    parsed = json.loads(daily_payload)
    assert parsed["state"] == "daily"
    assert "accounts" in parsed and "freshness" in parsed

    filename, history_csv = ops_env.export_history_csv()
    assert filename.endswith(".csv")
    assert f"daily-{day_key}" in history_csv
    assert "100002" in history_csv or "100002.0" in history_csv
    assert "252" in history_csv or "252.0" in history_csv


def test_update_readiness_never_auto_installs(ops_env: OperationsService) -> None:
    readiness = ops_env.update_readiness()
    assert readiness.auto_install is False
    assert readiness.backup_required_before_update is True
    assert readiness.reversible_instructions
    assert "never" in readiness.note.lower() or "Never" in readiness.note


def test_operations_api_requires_session_and_exports_csv(
    ops_env: OperationsService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "position_pilot.web.app.get_operations_service",
        lambda: ops_env,
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
    denied = client.get("/api/v1/exports/portfolio.csv")
    assert denied.status_code == 401

    exchange = client.post("/api/v1/session/exchange", json={"launch_token": "launch-secret"})
    assert exchange.status_code == 204

    csv_response = client.get("/api/v1/exports/portfolio.csv")
    assert csv_response.status_code == 200
    assert "text/csv" in csv_response.headers["content-type"]
    assert "SPY" in csv_response.text

    pdf_response = client.get("/api/v1/exports/snapshot.pdf")
    assert pdf_response.status_code == 200
    assert pdf_response.content.startswith(b"%PDF")

    diag = client.get("/api/v1/diagnostics/bundle")
    assert diag.status_code == 200
    body = diag.json()
    assert "super-secret" not in str(body)
    assert body["schema_version"] >= 1
    assert body["env_diagnostics"]["path"] == ".env"
    assert PORTABLE_BROKER_ACCOUNT_SENTINEL not in str(body)

    env = client.get("/api/v1/diagnostics/env")
    assert env.status_code == 200
    assert env.json()["path"] == ".env"
    assert "should-not-leak" not in str(env.json())

    update = client.get("/api/v1/update/status")
    assert update.status_code == 200
    assert update.json()["auto_install"] is False

    backup = client.post("/api/v1/backups")
    assert backup.status_code == 200
    backup_id = backup.json()["backup_id"]
    assert backup.json()["path"] == backup_id
    listed = client.get("/api/v1/backups")
    assert listed.status_code == 200
    assert any(item["backup_id"] == backup_id for item in listed.json())

    download = client.get(f"/api/v1/backups/{backup_id}")
    assert download.status_code == 200
    assert "zip" in download.headers["content-type"]
    assert PORTABLE_FULL_TEXT_SENTINEL.encode() not in download.content
    assert PORTABLE_BROKER_ACCOUNT_SENTINEL.encode() not in download.content
