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

CURRENT_SCHEMA_VERSION = 5


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
                            if not (
                                source.get("provider") == provider and source.get("url") == url
                            )
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
