"""SQLite migrations, settings, identities, snapshots, and backups."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..domain.snapshots import PortfolioSnapshot

CURRENT_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class AccountIdentity:
    """Stable opaque identity safe to expose to the local browser."""

    account_id: str
    label: str


class PositionPilotDatabase:
    """Public repository boundary for local versioned application state."""

    def __init__(
        self,
        path: Path,
        *,
        legacy_config_path: Path | None = None,
        legacy_cache_directory: Path | None = None,
        backup_directory: Path | None = None,
    ) -> None:
        self.path = path
        self.legacy_config_path = legacy_config_path
        self.legacy_cache_directory = legacy_cache_directory
        self.backup_directory = backup_directory or path.parent / "backups"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()
        self._import_legacy_settings_once()
        self._import_legacy_cache_once()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _migrate(self) -> None:
        existing_version = 0
        if self.path.exists():
            with sqlite3.connect(self.path) as connection:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
                ).fetchone()
                if table:
                    row = connection.execute(
                        "SELECT MAX(version) FROM schema_migrations"
                    ).fetchone()
                    existing_version = int(row[0] or 0)
        if self.path.exists() and existing_version < CURRENT_SCHEMA_VERSION:
            self.backup(reason="pre-migration")

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS account_identities (
                    account_number TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL UNIQUE,
                    label TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_captured
                    ON portfolio_snapshots(captured_at DESC);
                CREATE TABLE IF NOT EXISTS legacy_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    source_timestamp REAL,
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS roll_chains (
                    account_id TEXT NOT NULL,
                    chain_key TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(account_id, chain_key)
                );
                CREATE TABLE IF NOT EXISTS provider_health (
                    provider TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS strategy_documents (
                    doc_type TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(doc_type, strategy_id)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    strategy_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_events_strategy
                    ON audit_events(strategy_id, recorded_at DESC);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (CURRENT_SCHEMA_VERSION, datetime.now(UTC).isoformat()),
            )

    @property
    def schema_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Commit one complete portfolio generation as a single transaction."""

        payload = snapshot.model_dump_json()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO portfolio_snapshots(
                    snapshot_id, schema_version, captured_at, state, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.schema_version,
                    snapshot.captured_at.isoformat(),
                    snapshot.state.value,
                    payload,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def latest_portfolio_snapshot(self) -> PortfolioSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM portfolio_snapshots
                ORDER BY captured_at DESC, rowid DESC LIMIT 1
                """
            ).fetchone()
        return PortfolioSnapshot.model_validate_json(row[0]) if row else None

    def account_identity(self, account_number: str, account_type: str) -> AccountIdentity:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT account_id, label FROM account_identities WHERE account_number = ?",
                (account_number,),
            ).fetchone()
            if row:
                return AccountIdentity(account_id=row[0], label=row[1])
            ordinal = connection.execute(
                "SELECT COUNT(*) FROM account_identities WHERE account_type = ?",
                (account_type,),
            ).fetchone()[0]
            identity = AccountIdentity(
                account_id=str(uuid4()),
                label=f"{account_type or 'Account'} {ordinal + 1}",
            )
            connection.execute(
                """
                INSERT INTO account_identities(
                    account_number, account_id, label, account_type, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    account_number,
                    identity.account_id,
                    identity.label,
                    account_type,
                    datetime.now(UTC).isoformat(),
                ),
            )
            return identity

    def account_id_for_broker_number(self, account_number: str) -> str | None:
        """Resolve an internal broker identifier to its browser-safe identity."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT account_id FROM account_identities WHERE account_number = ?",
                (account_number,),
            ).fetchone()
        return str(row[0]) if row else None

    def broker_number_for_account_id(self, account_id: str) -> str | None:
        """Resolve a public account identity to the internal broker number (server-only)."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT account_number FROM account_identities WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def save_strategy_document(self, doc_type: str, strategy_id: str, payload: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO strategy_documents(doc_type, strategy_id, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_type, strategy_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (doc_type, strategy_id, json.dumps(payload), datetime.now(UTC).isoformat()),
            )

    def get_strategy_document(self, doc_type: str, strategy_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM strategy_documents
                WHERE doc_type = ? AND strategy_id = ?
                """,
                (doc_type, strategy_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def append_audit_event(self, payload: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events(
                    event_id, strategy_id, action, summary, payload_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["event_id"],
                    payload["strategy_id"],
                    payload["action"],
                    payload["summary"],
                    json.dumps(payload),
                    payload.get("recorded_at") or datetime.now(UTC).isoformat(),
                ),
            )

    def list_audit_events(self, strategy_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM audit_events
                WHERE strategy_id = ?
                ORDER BY recorded_at DESC, rowid DESC
                """,
                (strategy_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def set_setting(self, key: str, value: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), datetime.now(UTC).isoformat()),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key = ?",
                (key,),
            ).fetchone()
        return json.loads(row[0]) if row else default

    def _import_legacy_settings_once(self) -> None:
        if self.get_setting("migration.legacy_config_imported", False):
            return
        if self.legacy_config_path and self.legacy_config_path.is_file():
            try:
                legacy = json.loads(self.legacy_config_path.read_text())
            except (OSError, json.JSONDecodeError):
                legacy = {}
            for key, value in legacy.items():
                self.set_setting(key, value)
        self.set_setting("migration.legacy_config_imported", True)

    def _import_legacy_cache_once(self) -> None:
        if self.get_setting("migration.legacy_cache_imported", False):
            return
        if self.legacy_cache_directory and self.legacy_cache_directory.is_dir():
            with self._connect() as connection:
                for cache_file in self.legacy_cache_directory.glob("*.json"):
                    try:
                        payload = json.loads(cache_file.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO legacy_cache(
                            cache_key, payload_json, source_timestamp, imported_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            cache_file.stem,
                            json.dumps(payload.get("value", payload)),
                            payload.get("timestamp"),
                            datetime.now(UTC).isoformat(),
                        ),
                    )
        self.set_setting("migration.legacy_cache_imported", True)

    def get_legacy_cache(self, key: str) -> Any:
        """Read an imported legacy cache value during migration compatibility."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM legacy_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_roll_chain(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO roll_chains(
                    account_id, chain_key, underlying, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account_id, chain_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    payload["account_id"],
                    payload["chain_id"],
                    payload["underlying"],
                    json.dumps(payload),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def roll_chains(self, account_id: str, *, symbol: str | None = None) -> list[dict]:
        query = "SELECT payload_json FROM roll_chains WHERE account_id = ?"
        parameters: list[Any] = [account_id]
        if symbol:
            query += " AND underlying = ?"
            parameters.append(symbol.upper())
        query += " ORDER BY updated_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_provider_health(self, provider: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO provider_health(provider, payload_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (provider, json.dumps(payload), datetime.now(UTC).isoformat()),
            )

    def provider_health(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT provider, payload_json FROM provider_health"
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def backup(self, *, reason: str = "daily") -> Path | None:
        """Create a credentials-free SQLite backup and apply retention."""

        if not self.path.exists():
            return None
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = self.backup_directory / f"position-pilot-{reason}-{timestamp}.sqlite3"
        with sqlite3.connect(self.path) as source, sqlite3.connect(destination) as target:
            source.backup(target)
        os.chmod(destination, 0o600)
        self._prune_backups()
        return destination

    def ensure_daily_backup(self, now: datetime | None = None) -> Path | None:
        """Create at most one daily backup and a weekly copy every Monday."""

        current = now or datetime.now(UTC)
        day = current.date().isoformat()
        if self.get_setting("backup.last_daily") == day:
            return None
        daily = self.backup(reason="daily")
        self.set_setting("backup.last_daily", day)
        if current.weekday() == 0:
            self.backup(reason="weekly")
        return daily

    def _prune_backups(self) -> None:
        retention = {"daily": 7, "weekly": 4, "pre-migration": 2}
        for reason, keep in retention.items():
            backups = sorted(
                self.backup_directory.glob(f"position-pilot-{reason}-*.sqlite3"),
                reverse=True,
            )
            for expired in backups[keep:]:
                expired.unlink(missing_ok=True)
