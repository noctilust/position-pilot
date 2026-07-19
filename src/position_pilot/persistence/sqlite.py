"""SQLite migrations, settings, identities, snapshots, and backups."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from ..domain.snapshots import (
    AccountSnapshot,
    DataFreshness,
    FreshnessState,
    PortfolioSnapshot,
    PortfolioTotals,
    PositionSnapshot,
    SnapshotCompaction,
    SnapshotState,
    StrategySnapshot,
)

CURRENT_SCHEMA_VERSION = 6


@contextmanager
def managed_sqlite_connection(
    path: str | Path,
    *,
    uri: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection that is always closed on exit.

    ``sqlite3.Connection`` as a context manager only commits/rollbacks; it does
    not close the underlying file descriptors. This helper commits on success,
    rolls back on error, and closes in ``finally`` so callers cannot leak FDs.
    It does not force WAL checkpoints — that remains SQLite's normal behavior.
    """

    connection = sqlite3.connect(path, uri=uri)
    try:
        yield connection
        connection.commit()
    except Exception:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()


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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection that is closed on every exit path."""

        with managed_sqlite_connection(self.path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection

    def _migrate(self) -> None:
        existing_version = 0
        if self.path.exists():
            with managed_sqlite_connection(self.path) as connection:
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
                CREATE TABLE IF NOT EXISTS catalyst_events (
                    catalyst_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    taxonomy TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    attribution TEXT NOT NULL,
                    evidence_kind TEXT NOT NULL,
                    event_at TEXT NOT NULL,
                    rank_score REAL NOT NULL,
                    high_impact INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_catalyst_events_symbol
                    ON catalyst_events(symbol, event_at DESC);
                CREATE TABLE IF NOT EXISTS catalyst_sources (
                    source_id TEXT PRIMARY KEY,
                    catalyst_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    url TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    excerpt TEXT,
                    FOREIGN KEY(catalyst_id) REFERENCES catalyst_events(catalyst_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS catalyst_source_links (
                    catalyst_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    PRIMARY KEY(catalyst_id, source_id)
                );
                CREATE TABLE IF NOT EXISTS catalyst_articles (
                    article_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    excerpt TEXT,
                    full_text TEXT,
                    stored_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_catalyst_articles_symbol
                    ON catalyst_articles(symbol, stored_at DESC);
                CREATE TABLE IF NOT EXISTS catalyst_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    catalyst_id TEXT,
                    symbol TEXT,
                    note TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_catalyst_feedback_catalyst
                    ON catalyst_feedback(catalyst_id, recorded_at DESC);
                CREATE INDEX IF NOT EXISTS idx_catalyst_feedback_symbol
                    ON catalyst_feedback(symbol, recorded_at DESC);
                CREATE TABLE IF NOT EXISTS symbol_catalyst_snapshots (
                    symbol TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    captured_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recommendations (
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    recommendation_id TEXT NOT NULL UNIQUE,
                    account_id TEXT,
                    payload_json TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    last_evaluated_at TEXT NOT NULL,
                    recommendation_updated_at TEXT,
                    provider_status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(subject_type, subject_id)
                );
                CREATE INDEX IF NOT EXISTS idx_recommendations_account
                    ON recommendations(account_id, last_evaluated_at DESC);
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    history_id TEXT PRIMARY KEY,
                    recommendation_id TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    day_key TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_recommendation_history_subject
                    ON recommendation_history(subject_type, subject_id, recorded_at DESC);
                CREATE INDEX IF NOT EXISTS idx_recommendation_history_day
                    ON recommendation_history(subject_type, subject_id, kind, day_key);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_recommendation_daily_summary
                    ON recommendation_history(subject_type, subject_id, kind, day_key)
                    WHERE kind = 'daily_summary' AND day_key IS NOT NULL;
                CREATE TABLE IF NOT EXISTS trader_decisions (
                    decision_id TEXT PRIMARY KEY,
                    recommendation_id TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trader_decisions_subject
                    ON trader_decisions(subject_type, subject_id, recorded_at DESC);
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    account_id TEXT,
                    symbol TEXT,
                    resolution TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    snoozed_until TEXT,
                    dedupe_key TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_open
                    ON alerts(resolution, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_alerts_dedupe
                    ON alerts(dedupe_key, resolution);
                CREATE TABLE IF NOT EXISTS alert_mute_rules (
                    rule_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
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

    def list_portfolio_snapshots(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return lightweight snapshot metadata newest-first (payload excluded)."""

        capped = min(max(limit, 1), 5_000)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT snapshot_id, schema_version, captured_at, state, created_at
                FROM portfolio_snapshots
                ORDER BY captured_at DESC, rowid DESC
                LIMIT ?
                """,
                (capped,),
            ).fetchall()
        return [
            {
                "snapshot_id": row[0],
                "schema_version": int(row[1]),
                "captured_at": row[2],
                "state": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def list_portfolio_snapshot_payloads(self, *, limit: int = 100) -> list[PortfolioSnapshot]:
        """Return full snapshot payloads newest-first for history export."""

        capped = min(max(limit, 1), 1_000)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM portfolio_snapshots
                ORDER BY captured_at DESC, rowid DESC
                LIMIT ?
                """,
                (capped,),
            ).fetchall()
        return [PortfolioSnapshot.model_validate_json(row[0]) for row in rows]

    def count_table(self, table: str) -> int:
        """Return row count for a known application table (whitelist)."""

        allowed = {
            "portfolio_snapshots",
            "catalyst_events",
            "catalyst_articles",
            "catalyst_sources",
            "recommendation_history",
            "recommendations",
            "alerts",
            "audit_events",
            "roll_chains",
            "provider_health",
            "trader_decisions",
        }
        if table not in allowed:
            raise ValueError(f"Unsupported table count: {table}")
        with self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0] or 0)

    def integrity_ok(self) -> tuple[bool, str]:
        """Run SQLite integrity_check against the live database file."""

        if not self.path.exists():
            return False, "database file missing"
        with self._connect() as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        message = str(row[0]) if row else "unknown"
        return message.lower() == "ok", message

    def list_portfolio_snapshot_summaries(self, *, limit: int = 365) -> list[dict[str, Any]]:
        """Return compact history rows for export without full payloads."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT snapshot_id, schema_version, captured_at, state, payload_json
                FROM portfolio_snapshots
                ORDER BY captured_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        summaries: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                payload = {}
            accounts = payload.get("accounts") or []
            strategies = payload.get("strategies") or []
            positions = sum(len(account.get("positions") or []) for account in accounts)
            totals = payload.get("totals") or {}
            summaries.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "schema_version": row["schema_version"],
                    "captured_at": row["captured_at"],
                    "state": row["state"],
                    "account_count": len(accounts),
                    "strategy_count": len(strategies),
                    "position_count": positions,
                    "net_liquidating_value": totals.get("net_liquidating_value"),
                    "unrealized_pnl": totals.get("unrealized_pnl"),
                }
            )
        return summaries

    def all_settings(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value_json FROM settings ORDER BY key"
            ).fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value_json"])
            except (TypeError, json.JSONDecodeError):
                result[row["key"]] = None
        return result

    def operational_counts(self) -> dict[str, int]:
        tables = {
            "portfolio_snapshots": "portfolio_snapshots",
            "catalyst_events": "catalyst_events",
            "catalyst_articles": "catalyst_articles",
            "recommendations": "recommendations",
            "recommendation_history": "recommendation_history",
            "alerts": "alerts",
            "roll_chains": "roll_chains",
            "audit_events": "audit_events",
            "trader_decisions": "trader_decisions",
        }
        counts: dict[str, int] = {}
        with self._connect() as connection:
            for label, table in tables.items():
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if not exists:
                    counts[label] = 0
                    continue
                counts[label] = int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
        return counts

    def retention_candidates(
        self,
        *,
        portfolio_snapshots_days: int,
        catalyst_events_days: int,
        article_metadata_days: int,
        recommendation_history_days: int,
    ) -> dict[str, int]:
        now = datetime.now(UTC)
        candidates: dict[str, int] = {}
        with self._connect() as connection:
            snap_cutoff = now.timestamp() - portfolio_snapshots_days * 86400
            # captured_at is ISO text; compare lexicographically with ISO cutoff.
            snap_iso = datetime.fromtimestamp(snap_cutoff, tz=UTC).isoformat()
            candidates["portfolio_snapshots"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM portfolio_snapshots WHERE captured_at < ?",
                    (snap_iso,),
                ).fetchone()[0]
            )
            # Non-daily older snapshots compact into one durable daily summary per day.
            candidates["portfolio_snapshots_compactable"] = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM portfolio_snapshots
                    WHERE captured_at < ? AND state != 'daily'
                    """,
                    (snap_iso,),
                ).fetchone()[0]
            )
            cat_iso = datetime.fromtimestamp(
                now.timestamp() - catalyst_events_days * 86400, tz=UTC
            ).isoformat()
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='catalyst_events'"
            ).fetchone():
                candidates["catalyst_events"] = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM catalyst_events WHERE event_at < ?",
                        (cat_iso,),
                    ).fetchone()[0]
                )
            else:
                candidates["catalyst_events"] = 0
            art_iso = datetime.fromtimestamp(
                now.timestamp() - article_metadata_days * 86400, tz=UTC
            ).isoformat()
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='catalyst_articles'"
            ).fetchone():
                candidates["catalyst_articles"] = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM catalyst_articles WHERE stored_at < ?",
                        (art_iso,),
                    ).fetchone()[0]
                )
            else:
                candidates["catalyst_articles"] = 0
            # Recommendation history is audit-critical: report count but ordinary
            # apply never purges it (recommendation_history_days ignored for delete).
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_history'"
            ).fetchone():
                candidates["recommendation_history"] = int(
                    connection.execute("SELECT COUNT(*) FROM recommendation_history").fetchone()[0]
                )
            else:
                candidates["recommendation_history"] = 0
            _ = recommendation_history_days  # retained for API compatibility
            for table, label in (
                ("roll_chains", "roll_chains"),
                ("audit_events", "audit_events"),
                ("trader_decisions", "trader_decisions"),
            ):
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone():
                    candidates[label] = int(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    )
                else:
                    candidates[label] = 0
        return candidates

    def apply_retention(
        self,
        *,
        portfolio_snapshots_days: int,
        catalyst_events_days: int,
        article_metadata_days: int,
        recommendation_history_days: int,
        clear_transactions: bool = False,
        compact_portfolio_snapshots: bool = True,
    ) -> dict[str, int]:
        """Apply ordinary retention. Never touches credentials or audit-critical tables.

        Portfolio snapshots older than the retention window are compacted into one
        durable daily summary per calendar day instead of deleting all history.
        ``clear_transactions`` is accepted for API compatibility but is ignored —
        roll chains, trader decisions, audit events, recommendation history, and
        transactions are never cleared by ordinary apply.
        """

        now = datetime.now(UTC)
        deleted: dict[str, int] = {}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            snap_iso = datetime.fromtimestamp(
                now.timestamp() - portfolio_snapshots_days * 86400, tz=UTC
            ).isoformat()
            # Keep at least the newest snapshot even if outside the window.
            newest = connection.execute(
                """
                SELECT snapshot_id FROM portfolio_snapshots
                ORDER BY captured_at DESC, rowid DESC LIMIT 1
                """
            ).fetchone()
            newest_id = newest[0] if newest else None

            compacted = 0
            if compact_portfolio_snapshots:
                compacted = self._compact_old_portfolio_snapshots(
                    connection,
                    older_than_iso=snap_iso,
                    protect_snapshot_id=newest_id,
                )
            deleted["portfolio_snapshots_compacted"] = compacted

            # After compaction, no older non-daily rows should remain; do not hard-delete
            # daily summaries or the newest snapshot.
            if newest_id:
                cursor = connection.execute(
                    """
                    DELETE FROM portfolio_snapshots
                    WHERE captured_at < ? AND snapshot_id != ? AND state != 'daily'
                    """,
                    (snap_iso, newest_id),
                )
            else:
                cursor = connection.execute(
                    """
                    DELETE FROM portfolio_snapshots
                    WHERE captured_at < ? AND state != 'daily'
                    """,
                    (snap_iso,),
                )
            deleted["portfolio_snapshots_intraday"] = cursor.rowcount
            deleted["portfolio_snapshots"] = compacted + cursor.rowcount

            cat_iso = datetime.fromtimestamp(
                now.timestamp() - catalyst_events_days * 86400, tz=UTC
            ).isoformat()
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='catalyst_events'"
            ).fetchone():
                expired_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT catalyst_id FROM catalyst_events WHERE event_at < ?",
                        (cat_iso,),
                    ).fetchall()
                ]
                for catalyst_id in expired_ids:
                    connection.execute(
                        "DELETE FROM catalyst_sources WHERE catalyst_id = ?",
                        (catalyst_id,),
                    )
                    connection.execute(
                        "DELETE FROM catalyst_source_links WHERE catalyst_id = ?",
                        (catalyst_id,),
                    )
                cursor = connection.execute(
                    "DELETE FROM catalyst_events WHERE event_at < ?",
                    (cat_iso,),
                )
                deleted["catalyst_events"] = cursor.rowcount
            else:
                deleted["catalyst_events"] = 0

            art_iso = datetime.fromtimestamp(
                now.timestamp() - article_metadata_days * 86400, tz=UTC
            ).isoformat()
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='catalyst_articles'"
            ).fetchone():
                cursor = connection.execute(
                    "DELETE FROM catalyst_articles WHERE stored_at < ?",
                    (art_iso,),
                )
                deleted["catalyst_articles"] = cursor.rowcount
            else:
                deleted["catalyst_articles"] = 0

            # Ordinary apply never purges recommendation history / rolls / audit / decisions.
            deleted["recommendation_history"] = 0
            _ = recommendation_history_days
            _ = clear_transactions
            connection.commit()
        return deleted

    def _compact_old_portfolio_snapshots(
        self,
        connection: sqlite3.Connection,
        *,
        older_than_iso: str,
        protect_snapshot_id: str | None,
    ) -> int:
        """Replace older 30-minute snapshots with one durable daily summary per day.

        Migration-compatible: reuses ``portfolio_snapshots`` with ``state='daily'``.
        Payload is always a fully valid ``PortfolioSnapshot`` plus compaction metadata
        (never a sparse summary dict that would fail Pydantic readers).
        """

        rows = connection.execute(
            """
            SELECT snapshot_id, schema_version, captured_at, state, payload_json, created_at
            FROM portfolio_snapshots
            WHERE captured_at < ?
            ORDER BY captured_at ASC, rowid ASC
            """,
            (older_than_iso,),
        ).fetchall()
        by_day: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            if protect_snapshot_id and row["snapshot_id"] == protect_snapshot_id:
                continue
            day_key = str(row["captured_at"])[:10]
            by_day.setdefault(day_key, []).append(row)

        compacted = 0
        for day_key, day_rows in by_day.items():
            # Prefer an existing daily summary; otherwise build from the last snapshot of the day.
            existing_daily = [r for r in day_rows if str(r["state"]) == "daily"]
            source = existing_daily[-1] if existing_daily else day_rows[-1]
            try:
                payload = json.loads(source["payload_json"])
            except (TypeError, json.JSONDecodeError):
                payload = {}
            daily_snapshot = self._build_daily_portfolio_snapshot(
                day_key=day_key,
                source_snapshot_id=str(source["snapshot_id"]),
                schema_version=int(source["schema_version"] or 1),
                payload=payload if isinstance(payload, dict) else {},
                source_count=len(day_rows),
            )
            daily_id = daily_snapshot.snapshot_id
            connection.execute(
                """
                INSERT INTO portfolio_snapshots(
                    snapshot_id, schema_version, captured_at, state, payload_json, created_at
                ) VALUES (?, ?, ?, 'daily', ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    captured_at = excluded.captured_at,
                    state = 'daily',
                    payload_json = excluded.payload_json
                """,
                (
                    daily_id,
                    daily_snapshot.schema_version,
                    daily_snapshot.captured_at.isoformat(),
                    daily_snapshot.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )
            # Remove non-daily rows for this day (history retained via daily summary).
            for row in day_rows:
                if row["snapshot_id"] == daily_id:
                    continue
                if str(row["state"]) == "daily" and row["snapshot_id"] != daily_id:
                    connection.execute(
                        "DELETE FROM portfolio_snapshots WHERE snapshot_id = ?",
                        (row["snapshot_id"],),
                    )
                    compacted += 1
                    continue
                if str(row["state"]) != "daily":
                    connection.execute(
                        "DELETE FROM portfolio_snapshots WHERE snapshot_id = ?",
                        (row["snapshot_id"],),
                    )
                    compacted += 1
        return compacted

    def _build_daily_portfolio_snapshot(
        self,
        *,
        day_key: str,
        source_snapshot_id: str,
        schema_version: int,
        payload: dict[str, Any],
        source_count: int,
    ) -> PortfolioSnapshot:
        """Build a fully valid daily PortfolioSnapshot from a source row payload."""

        captured_at = datetime.fromisoformat(f"{day_key}T23:59:59+00:00")
        compaction = SnapshotCompaction(
            kind="daily_summary",
            day=day_key,
            source_snapshot_id=source_snapshot_id,
            source_count=source_count,
        )
        try:
            full = PortfolioSnapshot.model_validate(payload)
            notice_parts = [
                part for part in (full.notice, f"Compacted daily summary for {day_key}") if part
            ]
            return full.model_copy(
                update={
                    "snapshot_id": f"daily-{day_key}",
                    "schema_version": schema_version or full.schema_version,
                    "captured_at": captured_at,
                    "state": SnapshotState.DAILY,
                    "compaction": compaction,
                    "notice": " · ".join(notice_parts),
                }
            )
        except ValidationError:
            pass

        accounts: list[AccountSnapshot] = []
        for index, raw_account in enumerate(payload.get("accounts") or []):
            if not isinstance(raw_account, dict):
                continue
            positions = []
            for raw_position in raw_account.get("positions") or []:
                if not isinstance(raw_position, dict):
                    continue
                try:
                    positions.append(PositionSnapshot.model_validate(raw_position))
                except ValidationError:
                    continue
            accounts.append(
                AccountSnapshot(
                    account_id=str(raw_account.get("account_id") or f"account-{index + 1}"),
                    label=str(raw_account.get("label") or "Account"),
                    account_type=str(raw_account.get("account_type") or "unknown"),
                    net_liquidating_value=float(raw_account.get("net_liquidating_value") or 0),
                    cash_balance=float(raw_account.get("cash_balance") or 0),
                    buying_power=float(raw_account.get("buying_power") or 0),
                    positions=positions,
                )
            )
        strategies: list[StrategySnapshot] = []
        for raw_strategy in payload.get("strategies") or []:
            if not isinstance(raw_strategy, dict):
                continue
            try:
                strategies.append(StrategySnapshot.model_validate(raw_strategy))
            except ValidationError:
                continue
        totals_raw = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
        totals = PortfolioTotals(
            net_liquidating_value=float(totals_raw.get("net_liquidating_value") or 0),
            cash_balance=float(totals_raw.get("cash_balance") or 0),
            buying_power=float(totals_raw.get("buying_power") or 0),
            unrealized_pnl=float(totals_raw.get("unrealized_pnl") or 0),
        )
        raw_freshness = payload.get("freshness")
        freshness_raw = raw_freshness if isinstance(raw_freshness, dict) else {}
        try:
            freshness = DataFreshness.model_validate(freshness_raw) if freshness_raw else None
        except ValidationError:
            freshness = None
        if freshness is None:
            freshness = DataFreshness(
                as_of=captured_at,
                provider=str(freshness_raw.get("provider") or "position-pilot-compaction"),
                state=FreshnessState.STALE,
            )
        return PortfolioSnapshot(
            schema_version=schema_version,
            snapshot_id=f"daily-{day_key}",
            captured_at=captured_at,
            state=SnapshotState.DAILY,
            freshness=freshness,
            accounts=accounts,
            strategies=strategies,
            totals=totals,
            selected_account_id=str(payload.get("selected_account_id") or "all"),
            notice=f"Compacted daily summary for {day_key}",
            compaction=compaction,
        )

    def reopen_after_restore(self) -> None:
        """Re-run migrations after an atomic file restore (upgrade only)."""

        self._migrate()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

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

    def has_setting(self, key: str) -> bool:
        """True when a settings row exists for key (even if value is null/empty/false)."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM settings WHERE key = ?",
                (key,),
            ).fetchone()
        return row is not None

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

    def upsert_catalyst_event(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalyst_events(
                    catalyst_id, symbol, fingerprint, headline, summary, taxonomy,
                    confidence, attribution, evidence_kind, event_at, rank_score,
                    high_impact, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(catalyst_id) DO UPDATE SET
                    headline = excluded.headline,
                    summary = excluded.summary,
                    taxonomy = excluded.taxonomy,
                    confidence = excluded.confidence,
                    attribution = excluded.attribution,
                    evidence_kind = excluded.evidence_kind,
                    event_at = excluded.event_at,
                    rank_score = excluded.rank_score,
                    high_impact = excluded.high_impact,
                    payload_json = excluded.payload_json
                """,
                (
                    payload["catalyst_id"],
                    payload["symbol"],
                    payload["fingerprint"],
                    payload["headline"],
                    payload["summary"],
                    payload["taxonomy"],
                    payload["confidence"],
                    payload["attribution"],
                    payload["evidence_kind"],
                    payload["event_at"],
                    payload["rank_score"],
                    1 if payload.get("high_impact") else 0,
                    json.dumps(payload.get("payload_json") or payload),
                    payload.get("created_at") or datetime.now(UTC).isoformat(),
                ),
            )

    def upsert_catalyst_source(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalyst_sources(
                    source_id, catalyst_id, name, tier, url, provider, published_at, excerpt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    name = excluded.name,
                    tier = excluded.tier,
                    url = excluded.url,
                    provider = excluded.provider,
                    published_at = excluded.published_at,
                    excerpt = excluded.excerpt
                """,
                (
                    payload["source_id"],
                    payload["catalyst_id"],
                    payload["name"],
                    payload["tier"],
                    payload["url"],
                    payload["provider"],
                    payload["published_at"],
                    payload.get("excerpt"),
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO catalyst_source_links(catalyst_id, source_id)
                VALUES (?, ?)
                """,
                (payload["catalyst_id"], payload["source_id"]),
            )

    def upsert_catalyst_article(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalyst_articles(
                    article_id, symbol, provider, url, title, source_name,
                    published_at, excerpt, full_text, stored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(article_id) DO UPDATE SET
                    title = excluded.title,
                    source_name = excluded.source_name,
                    published_at = excluded.published_at,
                    excerpt = excluded.excerpt,
                    full_text = COALESCE(excluded.full_text, catalyst_articles.full_text),
                    stored_at = excluded.stored_at
                """,
                (
                    payload["article_id"],
                    payload["symbol"],
                    payload["provider"],
                    payload["url"],
                    payload["title"],
                    payload["source_name"],
                    payload["published_at"],
                    payload.get("excerpt"),
                    payload.get("full_text"),
                    payload.get("stored_at") or datetime.now(UTC).isoformat(),
                ),
            )

    def list_catalyst_articles(self, symbol: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM catalyst_articles"
        parameters: list[Any] = []
        if symbol:
            query += " WHERE symbol = ?"
            parameters.append(symbol.upper())
        query += " ORDER BY published_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def append_catalyst_feedback(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalyst_feedback(
                    feedback_id, kind, catalyst_id, symbol, note, recorded_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["feedback_id"],
                    payload["kind"],
                    payload.get("catalyst_id"),
                    payload.get("symbol"),
                    payload.get("note") or "",
                    payload.get("recorded_at") or datetime.now(UTC).isoformat(),
                    json.dumps(payload),
                ),
            )

    def list_catalyst_feedback(
        self,
        *,
        catalyst_id: str | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if catalyst_id:
            clauses.append("catalyst_id = ?")
            parameters.append(catalyst_id)
        if symbol:
            clauses.append("symbol = ?")
            parameters.append(symbol.upper())
        query = "SELECT payload_json FROM catalyst_feedback"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY recorded_at ASC, rowid ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_symbol_catalyst_snapshot(
        self,
        symbol: str,
        payload: dict[str, Any],
        *,
        captured_at: datetime | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO symbol_catalyst_snapshots(symbol, payload_json, captured_at)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    captured_at = excluded.captured_at
                """,
                (
                    symbol.upper(),
                    json.dumps(payload),
                    (captured_at or datetime.now(UTC)).isoformat(),
                ),
            )

    def get_latest_symbol_catalyst(self, symbol: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM symbol_catalyst_snapshots WHERE symbol = ?",
                (symbol.upper(),),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def clear_catalyst_full_text(self, *, provider: str, url: str) -> int:
        return self.apply_catalyst_removal(provider=provider, url=url)

    def clear_stored_catalyst_full_text(self, *, provider: str | None = None) -> int:
        """Null retained full article text for one provider or all providers.

        Public domain API for consent revocation — does not delete articles or
        mutate feedback; only clears the licensed full_text column.
        """

        with self._connect() as connection:
            if provider:
                cursor = connection.execute(
                    "UPDATE catalyst_articles SET full_text = NULL WHERE lower(provider) = ?",
                    (provider.lower(),),
                )
            else:
                cursor = connection.execute(
                    "UPDATE catalyst_articles SET full_text = NULL WHERE full_text IS NOT NULL"
                )
            return int(cursor.rowcount)

    def apply_catalyst_removal(self, *, provider: str, url: str) -> int:
        """Remove licensed content and any unsupported derived presentation for a URL.

        Immutable feedback rows are never mutated.
        """

        cleared = 0
        with self._connect() as connection:
            affected_ids = {
                str(row[0])
                for row in connection.execute(
                    "SELECT catalyst_id FROM catalyst_sources WHERE provider = ? AND url = ?",
                    (provider, url),
                ).fetchall()
            }
            article_cursor = connection.execute(
                "DELETE FROM catalyst_articles WHERE provider = ? AND url = ?",
                (provider, url),
            )
            cleared += int(article_cursor.rowcount)
            source_ids = [
                str(row[0])
                for row in connection.execute(
                    "SELECT source_id FROM catalyst_sources WHERE provider = ? AND url = ?",
                    (provider, url),
                ).fetchall()
            ]
            for source_id in source_ids:
                connection.execute(
                    "DELETE FROM catalyst_source_links WHERE source_id = ?", (source_id,)
                )
            source_cursor = connection.execute(
                "DELETE FROM catalyst_sources WHERE provider = ? AND url = ?",
                (provider, url),
            )
            cleared += int(source_cursor.rowcount)
            for catalyst_id in affected_ids:
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM catalyst_sources WHERE catalyst_id = ?",
                    (catalyst_id,),
                ).fetchone()[0]
                if not remaining:
                    connection.execute(
                        "DELETE FROM catalyst_source_links WHERE catalyst_id = ?", (catalyst_id,)
                    )
                    cleared += int(
                        connection.execute(
                            "DELETE FROM catalyst_events WHERE catalyst_id = ?", (catalyst_id,)
                        ).rowcount
                    )
                else:
                    event_row = connection.execute(
                        "SELECT headline, payload_json FROM catalyst_events WHERE catalyst_id = ?",
                        (catalyst_id,),
                    ).fetchone()
                    if event_row:
                        try:
                            event_payload = json.loads(event_row[1])
                        except json.JSONDecodeError:
                            event_payload = {}
                        event_payload["sources"] = [
                            source
                            for source in event_payload.get("sources") or []
                            if not (source.get("provider") == provider and source.get("url") == url)
                        ]
                        event_payload["summary"] = str(event_row[0])
                        connection.execute(
                            """
                            UPDATE catalyst_events SET summary = ?, payload_json = ?
                            WHERE catalyst_id = ?
                            """,
                            (event_row[0], json.dumps(event_payload), catalyst_id),
                        )
            # Strip derived presentation from live symbol snapshots without touching feedback.
            rows = connection.execute(
                "SELECT symbol, payload_json FROM symbol_catalyst_snapshots"
            ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(row[1])
                except json.JSONDecodeError:
                    continue
                changed = False
                catalysts = payload.get("catalysts") or []
                retained_catalysts = []
                for catalyst in catalysts:
                    sources = catalyst.get("sources") or []
                    retained_sources = [
                        source
                        for source in sources
                        if not (source.get("provider") == provider and source.get("url") == url)
                    ]
                    if len(retained_sources) != len(sources):
                        changed = True
                    if retained_sources:
                        catalyst["sources"] = retained_sources
                        catalyst["summary"] = catalyst.get("headline", "Catalyst source updated")
                        retained_catalysts.append(catalyst)
                    elif catalyst.get("catalyst_id") not in affected_ids:
                        retained_catalysts.append(catalyst)
                if len(retained_catalysts) != len(catalysts):
                    changed = True
                payload["catalysts"] = retained_catalysts
                notes = payload.get("social_side_notes") or []
                retained_notes = [note for note in notes if note.get("url") != url]
                if len(retained_notes) != len(notes):
                    payload["social_side_notes"] = retained_notes
                    changed = True
                if changed and not retained_catalysts:
                    mechanisms = payload.get("option_mechanisms") or []
                    if mechanisms:
                        payload["confidence"] = "likely"
                        payload["attribution"] = "options_market"
                        payload["summary"] = mechanisms[0].get(
                            "summary", "Options-market mechanism"
                        )
                        payload["promoted"] = True
                        payload["quiet"] = False
                    else:
                        payload["confidence"] = "no_confirmed_catalyst_found"
                        payload["attribution"] = "none"
                        payload["summary"] = "No confirmed catalyst found"
                        payload["promoted"] = False
                        payload["quiet"] = True
                if changed:
                    connection.execute(
                        """
                        UPDATE symbol_catalyst_snapshots
                        SET payload_json = ?
                        WHERE symbol = ?
                        """,
                        (json.dumps(payload), row[0]),
                    )
                    cleared += 1
        return cleared

    def catalyst_exists(self, catalyst_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM catalyst_events WHERE catalyst_id = ?",
                (catalyst_id,),
            ).fetchone()
        return row is not None

    def symbol_for_catalyst(self, catalyst_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT symbol FROM catalyst_events WHERE catalyst_id = ?",
                (catalyst_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def count_catalyst_sources(self, catalyst_id: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM catalyst_sources"
        params: list[Any] = []
        if catalyst_id:
            query += " WHERE catalyst_id = ?"
            params.append(catalyst_id)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return int(row[0] if row else 0)

    def prune_catalysts(
        self,
        *,
        event_cutoff: datetime,
        article_cutoff: datetime,
    ) -> dict[str, int]:
        event_iso = event_cutoff.isoformat()
        article_iso = article_cutoff.isoformat()
        with self._connect() as connection:
            expired_events = connection.execute(
                "SELECT catalyst_id FROM catalyst_events WHERE created_at < ?",
                (event_iso,),
            ).fetchall()
            event_ids = [row[0] for row in expired_events]
            for catalyst_id in event_ids:
                connection.execute(
                    "DELETE FROM catalyst_source_links WHERE catalyst_id = ?",
                    (catalyst_id,),
                )
                connection.execute(
                    "DELETE FROM catalyst_sources WHERE catalyst_id = ?",
                    (catalyst_id,),
                )
                connection.execute(
                    "DELETE FROM catalyst_events WHERE catalyst_id = ?",
                    (catalyst_id,),
                )
            article_cursor = connection.execute(
                "DELETE FROM catalyst_articles WHERE stored_at < ?",
                (article_iso,),
            )
            snapshot_cursor = connection.execute(
                "DELETE FROM symbol_catalyst_snapshots WHERE captured_at < ?",
                (event_iso,),
            )
        return {
            "events": len(event_ids),
            "articles": int(article_cursor.rowcount),
            "snapshots": int(snapshot_cursor.rowcount),
        }

    # --- Recommendations / alerts (schema v6) ---------------------------------

    def upsert_recommendation(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            self._upsert_recommendation_on(connection, payload)

    def _upsert_recommendation_on(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
    ) -> None:
        now = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT INTO recommendations(
                subject_type, subject_id, recommendation_id, account_id,
                payload_json, input_fingerprint, last_evaluated_at,
                recommendation_updated_at, provider_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_type, subject_id) DO UPDATE SET
                recommendation_id=excluded.recommendation_id,
                account_id=excluded.account_id,
                payload_json=excluded.payload_json,
                input_fingerprint=excluded.input_fingerprint,
                last_evaluated_at=excluded.last_evaluated_at,
                recommendation_updated_at=excluded.recommendation_updated_at,
                provider_status=excluded.provider_status,
                updated_at=excluded.updated_at
            """,
            (
                payload["subject_type"],
                payload["subject_id"],
                payload["recommendation_id"],
                payload.get("account_id"),
                json.dumps(payload),
                payload["input_fingerprint"],
                payload["last_evaluated_at"],
                payload.get("recommendation_updated_at"),
                payload["provider_status"],
                now,
            ),
        )

    def upsert_recommendation_atomic(
        self,
        recommendation: dict[str, Any],
        *,
        history_entry: dict[str, Any] | None = None,
        daily_summary: dict[str, Any] | None = None,
    ) -> None:
        """Atomically persist current recommendation plus optional history/summary."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_recommendation_on(connection, recommendation)
                if history_entry is not None:
                    self._append_history_on(connection, history_entry)
                if daily_summary is not None:
                    self._upsert_daily_summary_on(connection, daily_summary)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get_recommendation(self, subject_type: str, subject_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM recommendations
                WHERE subject_type = ? AND subject_id = ?
                """,
                (subject_type, subject_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_recommendation_by_id(self, recommendation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM recommendations WHERE recommendation_id = ?",
                (recommendation_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_recommendations(self, *, account_id: str = "all") -> list[dict[str, Any]]:
        with self._connect() as connection:
            if account_id == "all":
                rows = connection.execute(
                    """
                    SELECT payload_json FROM recommendations
                    ORDER BY last_evaluated_at DESC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT payload_json FROM recommendations
                    WHERE account_id = ? OR account_id IS NULL
                    ORDER BY last_evaluated_at DESC
                    """,
                    (account_id,),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def append_recommendation_history(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            self._append_history_on(connection, payload)

    def _day_key_for_history(self, payload: dict[str, Any]) -> str | None:
        if payload.get("kind") != "daily_summary":
            return None
        recorded = payload.get("recorded_at") or ""
        day_key = str(recorded)[:10]
        if payload.get("diff") and isinstance(payload["diff"], dict):
            day_key = str(payload["diff"].get("day") or day_key)
        return day_key

    def _append_history_on(self, connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        day_key = self._day_key_for_history(payload)
        connection.execute(
            """
            INSERT INTO recommendation_history(
                history_id, recommendation_id, subject_type, subject_id,
                kind, recorded_at, day_key, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["history_id"],
                payload["recommendation_id"],
                payload["subject_type"],
                payload["subject_id"],
                payload["kind"],
                payload["recorded_at"],
                day_key,
                json.dumps(payload),
            ),
        )

    def _upsert_daily_summary_on(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
    ) -> None:
        day_key = self._day_key_for_history(payload)
        existing = connection.execute(
            """
            SELECT history_id, payload_json FROM recommendation_history
            WHERE subject_type = ? AND subject_id = ?
              AND kind = 'daily_summary' AND day_key = ?
            LIMIT 1
            """,
            (payload["subject_type"], payload["subject_id"], day_key),
        ).fetchone()
        if existing:
            prior = json.loads(existing[1])
            count = int(prior.get("evaluation_count") or 1) + 1
            payload = {
                **payload,
                "history_id": existing[0],
                "evaluation_count": count,
                "summary": (
                    f"Unchanged evaluation ×{count} on {day_key} ({payload.get('action') or 'n/a'})"
                ),
            }
            connection.execute(
                """
                UPDATE recommendation_history
                SET recorded_at = ?, payload_json = ?
                WHERE history_id = ?
                """,
                (payload["recorded_at"], json.dumps(payload), existing[0]),
            )
            return
        connection.execute(
            """
            INSERT INTO recommendation_history(
                history_id, recommendation_id, subject_type, subject_id,
                kind, recorded_at, day_key, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["history_id"],
                payload["recommendation_id"],
                payload["subject_type"],
                payload["subject_id"],
                payload["kind"],
                payload["recorded_at"],
                day_key,
                json.dumps(payload),
            ),
        )

    def update_recommendation_history(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE recommendation_history
                SET recorded_at = ?, payload_json = ?
                WHERE history_id = ?
                """,
                (payload["recorded_at"], json.dumps(payload), payload["history_id"]),
            )

    def get_daily_history_summary(
        self,
        subject_type: str,
        subject_id: str,
        *,
        day: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM recommendation_history
                WHERE subject_type = ? AND subject_id = ?
                  AND kind = 'daily_summary' AND day_key = ?
                ORDER BY recorded_at DESC LIMIT 1
                """,
                (subject_type, subject_id, day),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_recommendation_history(
        self,
        subject_type: str,
        subject_id: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM recommendation_history
                WHERE subject_type = ? AND subject_id = ?
                ORDER BY recorded_at DESC LIMIT ?
                """,
                (subject_type, subject_id, limit),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def append_trader_decision(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trader_decisions(
                    decision_id, recommendation_id, subject_type, subject_id,
                    decision, note, recorded_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["decision_id"],
                    payload["recommendation_id"],
                    payload["subject_type"],
                    payload["subject_id"],
                    payload["decision"],
                    payload.get("note") or "",
                    payload["recorded_at"],
                    json.dumps(payload),
                ),
            )

    def list_trader_decisions(
        self,
        *,
        subject_type: str | None = None,
        subject_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if subject_type and subject_id:
                rows = connection.execute(
                    """
                    SELECT payload_json FROM trader_decisions
                    WHERE subject_type = ? AND subject_id = ?
                    ORDER BY recorded_at DESC LIMIT ?
                    """,
                    (subject_type, subject_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT payload_json FROM trader_decisions
                    ORDER BY recorded_at DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def insert_alert(self, payload: dict[str, Any]) -> None:
        dedupe_key = None
        if isinstance(payload.get("payload"), dict):
            dedupe_key = payload["payload"].get("dedupe_key")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alerts(
                    alert_id, category, severity, alert_type, account_id, symbol,
                    resolution, created_at, updated_at, snoozed_until, dedupe_key, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["alert_id"],
                    payload["category"],
                    payload["severity"],
                    payload["alert_type"],
                    payload.get("account_id"),
                    payload.get("symbol"),
                    payload["resolution"],
                    payload["created_at"],
                    payload["updated_at"],
                    payload.get("snoozed_until"),
                    dedupe_key,
                    json.dumps(payload),
                ),
            )

    def update_alert(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE alerts
                SET resolution = ?, updated_at = ?, snoozed_until = ?, payload_json = ?
                WHERE alert_id = ?
                """,
                (
                    payload["resolution"],
                    payload["updated_at"],
                    payload.get("snoozed_until"),
                    json.dumps(payload),
                    payload["alert_id"],
                ),
            )

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM alerts WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def find_open_alert(self, dedupe_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM alerts
                WHERE dedupe_key = ? AND resolution IN ('open', 'acknowledged', 'snoozed')
                ORDER BY created_at DESC LIMIT 1
                """,
                (dedupe_key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_alerts(
        self,
        *,
        account_id: str = "all",
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if include_resolved:
                if account_id == "all":
                    rows = connection.execute(
                        """
                        SELECT payload_json FROM alerts
                        ORDER BY created_at DESC LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT payload_json FROM alerts
                        WHERE account_id = ? OR account_id IS NULL
                        ORDER BY created_at DESC LIMIT ?
                        """,
                        (account_id, limit),
                    ).fetchall()
            else:
                if account_id == "all":
                    rows = connection.execute(
                        """
                        SELECT payload_json FROM alerts
                        WHERE resolution NOT IN ('resolved', 'muted')
                        ORDER BY created_at DESC LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT payload_json FROM alerts
                        WHERE (account_id = ? OR account_id IS NULL)
                          AND resolution NOT IN ('resolved', 'muted')
                        ORDER BY created_at DESC LIMIT ?
                        """,
                        (account_id, limit),
                    ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def insert_mute_rule(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alert_mute_rules(rule_id, payload_json, created_at)
                VALUES (?, ?, ?)
                """,
                (payload["rule_id"], json.dumps(payload), payload["created_at"]),
            )

    def list_mute_rules(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM alert_mute_rules ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def backup(self, *, reason: str = "daily") -> Path | None:
        """Create a credentials-free SQLite backup and apply retention.

        Backups contain application state only (SQLite). Credentials live in
        ``.env`` and are never copied into backup files. Licensed full article
        text may exist in the DB under provider license rules; diagnostics
        exports deliberately omit it even when present here.
        """

        if not self.path.exists():
            return None
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in reason)[:40]
        destination = self.backup_directory / f"position-pilot-{safe_reason}-{timestamp}.sqlite3"
        with (
            managed_sqlite_connection(self.path) as source,
            managed_sqlite_connection(destination) as target,
        ):
            source.backup(target)
        os.chmod(destination, 0o600)
        self._write_backup_sidecar(destination, reason=safe_reason)
        self._prune_backups()
        return destination

    def write_backup_sidecar(
        self,
        destination: Path,
        *,
        reason: str,
        created_at: str | None = None,
        app_version: str | None = None,
    ) -> None:
        """Write unified backup sidecar metadata (schema, version, integrity, exclusions)."""

        from .. import __version__ as package_version

        version = app_version or package_version
        meta = {
            "schema_version": self.schema_version,
            "app_version": version,
            "application_version": version,  # backward-compatible alias
            "reason": reason,
            "created_at": created_at or datetime.now(UTC).isoformat(),
            "source_db": self.path.name,
            "integrity": self._file_integrity(destination),
            "excludes": [
                "credentials",
                ".env",
                "oauth_tokens",
                "session_cookies",
                "launch_tokens",
                "prompts",
            ],
        }
        sidecar = destination.with_suffix(destination.suffix + ".meta.json")
        sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            os.chmod(sidecar, 0o600)
        except OSError:
            pass

    def _write_backup_sidecar(self, destination: Path, *, reason: str) -> None:
        self.write_backup_sidecar(destination, reason=reason)

    def _file_integrity(self, path: Path) -> str:
        try:
            with managed_sqlite_connection(path) as connection:
                row = connection.execute("PRAGMA integrity_check").fetchone()
            return str(row[0]) if row else "unknown"
        except sqlite3.Error as error:
            return f"error:{error}"

    def list_backups(self) -> list[dict[str, Any]]:
        """List backup files under the controlled backup directory only."""

        if not self.backup_directory.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(self.backup_directory.glob("position-pilot-*.sqlite3"), reverse=True):
            if not path.is_file():
                continue
            # Reject path traversal / unexpected names.
            if path.parent.resolve() != self.backup_directory.resolve():
                continue
            if ".." in path.name or "/" in path.name or "\\" in path.name:
                continue
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta: dict[str, Any] = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    meta = {}
            stat = path.stat()
            items.append(
                {
                    "backup_id": path.name,
                    "path_name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    "schema_version": meta.get("schema_version"),
                    "application_version": meta.get("application_version"),
                    "reason": meta.get("reason"),
                    "integrity": meta.get("integrity") or self._file_integrity(path),
                    "excludes": meta.get("excludes", []),
                }
            )
        return items

    def resolve_backup_path(self, backup_id: str) -> Path:
        """Resolve a backup id to a file strictly inside the backup directory."""

        if not backup_id or backup_id != Path(backup_id).name:
            raise ValueError("Invalid backup identifier")
        if not backup_id.startswith("position-pilot-") or not backup_id.endswith(".sqlite3"):
            raise ValueError("Invalid backup identifier")
        if ".." in backup_id or "/" in backup_id or "\\" in backup_id:
            raise ValueError("Invalid backup identifier")
        candidate = (self.backup_directory / backup_id).resolve()
        backup_root = self.backup_directory.resolve()
        if not str(candidate).startswith(str(backup_root) + os.sep) and candidate != backup_root:
            raise ValueError("Backup path escapes backup directory")
        if not candidate.is_file():
            raise FileNotFoundError(backup_id)
        return candidate

    def restore_backup(self, backup_id: str) -> Path:
        """Atomically replace the live database with a verified backup.

        Creates a pre-restore backup first. Caller must ensure monitoring is
        stopped before invoking this method.
        """

        source = self.resolve_backup_path(backup_id)
        message = self._file_integrity(source)
        if message.lower() != "ok":
            raise ValueError(f"Backup failed integrity check: {message}")

        # Pre-restore safety copy of the current DB.
        if self.path.exists():
            self.backup(reason="pre-restore")

        staging = self.path.with_suffix(".restore-staging.sqlite3")
        if staging.exists():
            staging.unlink()
        with (
            managed_sqlite_connection(source) as src,
            managed_sqlite_connection(staging) as dst,
        ):
            src.backup(dst)
        staging_ok = self._file_integrity(staging)
        if staging_ok.lower() != "ok":
            staging.unlink(missing_ok=True)
            raise ValueError(f"Staged restore failed integrity check: {staging_ok}")

        # Atomic replace: move current aside, promote staging, remove old.
        previous = self.path.with_suffix(".pre-restore-live.sqlite3")
        if previous.exists():
            previous.unlink()
        if self.path.exists():
            self.path.replace(previous)
        staging.replace(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        if previous.exists():
            previous.unlink(missing_ok=True)
        # Ensure migrations are current after reopen.
        self._migrate()
        return self.path

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
        retention = {
            "daily": 7,
            "weekly": 4,
            "pre-migration": 2,
            "pre-restore": 3,
            "manual": 10,
        }
        for reason, keep in retention.items():
            backups = sorted(
                self.backup_directory.glob(f"position-pilot-{reason}-*.sqlite3"),
                reverse=True,
            )
            for expired in backups[keep:]:
                expired.unlink(missing_ok=True)
                meta = expired.with_suffix(expired.suffix + ".meta.json")
                meta.unlink(missing_ok=True)
