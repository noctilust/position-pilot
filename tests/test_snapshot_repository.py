import sqlite3
from datetime import UTC, datetime

from position_pilot.domain.snapshots import (
    AccountSnapshot,
    DataFreshness,
    PortfolioSnapshot,
    SnapshotState,
)
from position_pilot.persistence.sqlite import PositionPilotDatabase


def portfolio_snapshot(snapshot_id: str, captured_at: datetime) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        snapshot_id=snapshot_id,
        captured_at=captured_at,
        state=SnapshotState.LIVE,
        freshness=DataFreshness(as_of=captured_at, provider="tastytrade"),
        accounts=[
            AccountSnapshot(
                account_id="account-public-id",
                label="Individual 1",
                account_type="Individual",
                net_liquidating_value=10_000,
                cash_balance=2_000,
                buying_power=4_500,
            )
        ],
    )


def test_latest_portfolio_snapshot_is_replaced_atomically(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    first = portfolio_snapshot("snapshot-1", datetime(2026, 7, 11, 16, 0, tzinfo=UTC))
    second = portfolio_snapshot("snapshot-2", datetime(2026, 7, 11, 16, 5, tzinfo=UTC))

    database.save_portfolio_snapshot(first)
    database.save_portfolio_snapshot(second)

    assert database.latest_portfolio_snapshot() == second
    assert database.schema_version == 5


def test_account_alias_is_stable_without_exposing_broker_number(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")

    first = database.account_identity("5WT12345", "Individual")
    second = database.account_identity("5WT12345", "Individual")

    assert first == second
    assert first.account_id != "5WT12345"
    assert "12345" not in first.label


def test_legacy_json_settings_are_imported_once(tmp_path) -> None:
    legacy = tmp_path / "config.json"
    legacy.write_text('{"watchlist":["SPY","AAPL"],"default_account":"5WT12345"}')
    database = PositionPilotDatabase(
        tmp_path / "position-pilot.sqlite3",
        legacy_config_path=legacy,
    )

    assert database.get_setting("watchlist") == ["SPY", "AAPL"]
    assert database.get_setting("default_account") == "5WT12345"

    legacy.write_text('{"watchlist":["TSLA"]}')
    reopened = PositionPilotDatabase(
        tmp_path / "position-pilot.sqlite3",
        legacy_config_path=legacy,
    )
    assert reopened.get_setting("watchlist") == ["SPY", "AAPL"]


def test_daily_backup_runs_once_and_excludes_environment_file(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "data" / "position-pilot.sqlite3")
    (tmp_path / "data" / ".env").write_text("SECRET=do-not-copy")
    now = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    first = database.ensure_daily_backup(now)
    second = database.ensure_daily_backup(now)

    assert first is not None and first.is_file()
    assert second is None
    assert not list(database.backup_directory.glob("*.env"))
    assert len(list(database.backup_directory.glob("position-pilot-daily-*.sqlite3"))) == 1
    assert len(list(database.backup_directory.glob("position-pilot-weekly-*.sqlite3"))) == 1


def test_legacy_market_cache_is_imported_for_migration_compatibility(tmp_path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    (cache_directory / "quote_SPY.json").write_text(
        '{"value":{"mark":550.25},"timestamp":1783769000}'
    )

    database = PositionPilotDatabase(
        tmp_path / "position-pilot.sqlite3",
        legacy_cache_directory=cache_directory,
    )

    assert database.get_legacy_cache("quote_SPY") == {"mark": 550.25}


def test_existing_schema_is_backed_up_before_versioned_migration(tmp_path) -> None:
    path = tmp_path / "position-pilot.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, '2026-07-01T00:00:00Z')"
        )

    database = PositionPilotDatabase(path)

    assert database.schema_version == 5
    assert len(list(database.backup_directory.glob("position-pilot-pre-migration-*.sqlite3"))) == 1


def test_provider_health_is_durable_and_independent(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    database.save_provider_health("tastytrade", {"state": "healthy"})
    database.save_provider_health("massive-options", {"state": "unavailable"})

    assert database.provider_health() == {
        "tastytrade": {"state": "healthy"},
        "massive-options": {"state": "unavailable"},
    }
