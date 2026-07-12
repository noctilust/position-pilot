"""Local operations: exports, diagnostics, backups, retention, and updates.

All outputs are decision-support artifacts for a local, read-only workstation.
Nothing in this module places trades, reads credential values for export, or
stores licensed full article text in diagnostics/exports.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from .. import __version__
from ..persistence.sqlite import CURRENT_SCHEMA_VERSION, PositionPilotDatabase
from .portfolio import PortfolioService
from .snapshots import PortfolioSnapshot

# Secrets and sensitive material never leave the local service via these tools.
_SENSITIVE_ENV_PREFIXES = (
    "TASTYTRADE_",
    "ANTHROPIC_",
    "OPENAI_",
    "XAI_",
    "MASSIVE_",
    "BENZINGA_",
    "POSITION_PILOT_LAUNCH",
    "POSITION_PILOT_SESSION",
)
_SENSITIVE_KEY_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "cookie",
    "api_key",
    "apikey",
    "refresh",
    "client_secret",
    "authorization",
    "session",
    "launch_token",
    "credential",
    "private_key",
    "prompt",
    "account_number",
    "default_account",
    "broker_number",
)
# Diagnostics emit only this allowlisted surface (plus nested safe fields).
_DIAGNOSTIC_SETTINGS_ALLOWLIST = frozenset(
    {
        "operations.retention",
        "watchlist",
        "theme",
        "backup.last_daily",
        "catalysts.news_cadence_seconds",
        "catalysts.stock_move_threshold_pct",
        "catalysts.etf_move_threshold_pct",
        "catalysts.benzinga_enabled",
        "catalysts",
        "recommendations.rich_notification_preview",
        "monitoring.enabled",
        "monitoring.consented",
        "primary_account_id",
    }
)
_ACCOUNT_KEY_FRAGMENTS = (
    "account_number",
    "default_account",
    "broker_number",
    "broker_account",
    "account",
)
_HOME_PATH_RE = re.compile(r"(?i)(/Users/[^/\s\"']+|/home/[^/\s\"']+|\\\\Users\\\\[^\\\s\"']+)")
_EXCLUDED_DIAGNOSTIC_FIELDS = (
    "credential values",
    "refresh/client tokens",
    "cookies",
    "session/launch tokens",
    "full licensed article text",
    "prompts",
    "local AI account identity",
    "raw environment variables",
    "absolute filesystem paths",
    "raw broker account numbers",
)

# Retention: audit-critical classes are never purged by ordinary apply.
_RETENTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "portfolio_snapshots_days": {
        "default": 365,
        "min": 30,
        "max": 3650,
        "label": "Thirty-minute portfolio snapshots (days before daily compaction)",
        "audit_critical": False,
    },
    "catalyst_events_days": {
        "default": 365,
        "min": 30,
        "max": 3650,
        "label": "Catalyst events and source links (days)",
        "audit_critical": False,
    },
    "article_metadata_days": {
        "default": 90,
        "min": 7,
        "max": 365,
        "label": "Article metadata and permitted excerpts (days)",
        "audit_critical": False,
    },
    "recommendation_history_days": {
        "default": 0,
        "min": 0,
        "max": 3650,
        "label": "Recommendation history retention display (ordinary apply never purges)",
        "audit_critical": True,
    },
    "transaction_history": {
        "default": "indefinite",
        "label": "Transaction and roll-chain history (always indefinite)",
        "audit_critical": True,
    },
}

_PORTABLE_BACKUP_EXCLUSIONS = (
    "credentials",
    ".env",
    "oauth_tokens",
    "session_cookies",
    "launch_tokens",
    "prompts",
    "local AI identity",
    "catalyst full_text",
    "raw broker account numbers",
)

# Sentinel strings used in portable backup regression tests.
PORTABLE_FULL_TEXT_SENTINEL = "LICENSE_FULL_TEXT_MUST_NOT_EXPORT"
PORTABLE_BROKER_ACCOUNT_SENTINEL = "5WX12345"

DISCLAIMER = (
    "Position Pilot provides decision support only. It does not place, stage, "
    "or cancel orders and is not autonomous trading software."
)


class EnvDiagnostic(BaseModel):
    """Env file presence/permission report. Path is always the logical name `.env`."""

    path: str = ".env"
    exists: bool
    gitignored: bool
    tracked_by_git: bool | None = None
    permission_mode: str | None = None
    broadly_readable: bool = False
    warnings: list[str] = Field(default_factory=list)
    note: str = "Credential values are never read or returned."


class RetentionSettings(BaseModel):
    portfolio_snapshots_days: int = 365
    catalyst_events_days: int = 365
    article_metadata_days: int = 90
    recommendation_history_days: int = 0
    transaction_history: str = "indefinite"
    updated_at: str | None = None


class RetentionPreview(BaseModel):
    settings: RetentionSettings
    candidates: dict[str, int]
    audit_critical_preserved: list[str]
    would_delete: dict[str, int]
    would_compact: dict[str, int] = Field(default_factory=dict)
    disclaimer: str = DISCLAIMER


class BackupInfo(BaseModel):
    backup_id: str
    filename: str
    path: str  # basename / id only — never an absolute filesystem path
    size_bytes: int
    created_at: str
    reason: str
    schema_version: int | None = None
    app_version: str | None = None
    sha256: str
    integrity_ok: bool = True
    excludes: list[str] = Field(default_factory=list)


class RestoreResult(BaseModel):
    restored: bool
    backup_id: str
    pre_restore_backup_id: str | None = None
    schema_version: int | None = None
    message: str
    disclaimer: str = DISCLAIMER


class UpdateReadiness(BaseModel):
    current_version: str
    latest_version: str | None = None
    update_available: bool = False
    schema_version: int
    schema_migrations_pending: bool = False
    backup_required_before_update: bool = True
    monitoring_active: bool = False
    blocked_reason: str | None = None
    reversible_instructions: list[str] = Field(default_factory=list)
    auto_install: bool = False
    note: str = "Updates are never installed automatically. Package managers are never executed."
    disclaimer: str = DISCLAIMER


class DiagnosticBundle(BaseModel):
    generated_at: str
    app_version: str
    schema_version: int
    provider_status: dict[str, str]
    settings_redacted: dict[str, Any]
    env_diagnostics: EnvDiagnostic
    monitoring: dict[str, Any]
    counts: dict[str, int]
    redaction: dict[str, Any]
    disclaimer: str = DISCLAIMER


@dataclass(slots=True)
class OperationsService:
    """Cohesive local operations boundary for export, backup, and diagnostics."""

    database: PositionPilotDatabase
    portfolio: PortfolioService
    data_directory: Path
    project_root: Path | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    monitoring_active_fn: Callable[[], bool] | None = None
    provider_status_fn: Callable[[], dict[str, str]] | None = None
    version_probe_fn: Callable[[], str | None] | None = None

    # ── Portfolio export ──────────────────────────────────────────────────

    def export_portfolio_csv(self, account_id: str = "all") -> tuple[str, str]:
        """Return (filename, csv_text) for the current portfolio snapshot."""

        snapshot = self.portfolio.latest(account_id)
        if snapshot is None:
            snapshot = self.database.latest_portfolio_snapshot()
        if snapshot is None:
            raise LookupError("No portfolio snapshot is available to export.")
        if account_id != "all":
            snapshot = snapshot.for_account(account_id)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "account_id",
                "account_label",
                "strategy_id",
                "strategy_type",
                "underlying",
                "horizon",
                "symbol",
                "quantity",
                "position_type",
                "strike",
                "option_type",
                "expiration",
                "market_value",
                "unrealized_pnl",
                "delta",
                "theta",
                "gamma",
                "vega",
                "iv",
                "snapshot_id",
                "captured_at",
                "disclaimer",
            ]
        )
        for account in snapshot.accounts:
            strategies = [
                strategy
                for strategy in snapshot.strategies
                if strategy.account_id == account.account_id
            ]
            for strategy in strategies:
                for leg in strategy.legs:
                    writer.writerow(self._leg_row(account, strategy, leg, snapshot))
            # Standalone legs not in a strategy (should be rare after grouping)
            strategy_symbols = {leg.symbol for strategy in strategies for leg in strategy.legs}
            for leg in account.positions:
                if leg.symbol not in strategy_symbols:
                    writer.writerow(self._leg_row(account, None, leg, snapshot))
        filename = f"portfolio-{account_id}-{self._stamp()}.csv"
        return filename, buffer.getvalue()

    def export_history_csv(self, *, limit: int = 365) -> tuple[str, str]:
        """Return (filename, csv_text) summarizing historical portfolio snapshots."""

        rows = self.database.list_portfolio_snapshot_summaries(limit=max(1, min(limit, 2000)))
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "snapshot_id",
                "captured_at",
                "state",
                "schema_version",
                "account_count",
                "strategy_count",
                "position_count",
                "net_liquidating_value",
                "unrealized_pnl",
                "disclaimer",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("snapshot_id"),
                    row.get("captured_at"),
                    row.get("state"),
                    row.get("schema_version"),
                    row.get("account_count"),
                    row.get("strategy_count"),
                    row.get("position_count"),
                    row.get("net_liquidating_value"),
                    row.get("unrealized_pnl"),
                    DISCLAIMER,
                ]
            )
        return f"portfolio-history-{self._stamp()}.csv", buffer.getvalue()

    # ── Diagnostics ───────────────────────────────────────────────────────

    def env_diagnostics(self) -> EnvDiagnostic:
        """Report .env presence/ignore/permissions without reading values.

        The returned path is always the logical name ``.env`` — never an absolute
        user/home path.
        """

        root = self.project_root or Path.cwd()
        env_path = root / ".env"
        warnings: list[str] = []
        gitignored = self._is_gitignored(env_path, root)
        tracked: bool | None = None
        if env_path.exists():
            tracked = self._is_tracked_by_git(env_path, root)
            if tracked:
                warnings.append(
                    ".env appears tracked by git; remove it from version control immediately."
                )
            if not gitignored:
                warnings.append(".env does not appear in .gitignore.")
        else:
            gitignored = self._path_matches_gitignore(env_path, root)
            if not gitignored:
                warnings.append(".env is missing and is not listed in .gitignore.")

        mode_text: str | None = None
        broadly = False
        if env_path.exists():
            try:
                mode = stat.S_IMODE(env_path.stat().st_mode)
                mode_text = oct(mode)
                # Group or other read/write bits are considered broad.
                if mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
                    broadly = True
                    warnings.append(
                        f".env permissions {mode_text} are broader than owner-only "
                        f"(0600 recommended)."
                    )
            except OSError as error:
                # Never echo absolute paths from OSError messages.
                warnings.append(f"Could not stat .env: {type(error).__name__}")

        return EnvDiagnostic(
            path=".env",
            exists=env_path.exists(),
            gitignored=gitignored,
            tracked_by_git=tracked,
            permission_mode=mode_text,
            broadly_readable=broadly,
            warnings=warnings,
        )

    def diagnostic_bundle(self) -> DiagnosticBundle:
        """Build a redacted JSON diagnostics payload safe for local download."""

        settings_raw = self.database.all_settings()
        allowlisted = self._allowlisted_settings(settings_raw)
        redacted_settings = self._redact_mapping(allowlisted)
        # Strip any residual absolute paths that slipped through non-path fields.
        redacted_settings = self._strip_home_paths(redacted_settings)
        providers = (
            self.provider_status_fn()
            if self.provider_status_fn is not None
            else {"tastytrade": "unknown", "codex": "unknown"}
        )
        # Never surface local AI identity strings — only coarse status labels
        # for known provider keys.
        allowed_provider_keys = {
            "tastytrade",
            "codex",
            "massive",
            "benzinga",
            "openai",
            "anthropic",
        }
        safe_providers = {
            key: self._safe_provider_status(str(value))
            for key, value in providers.items()
            if str(key).lower() in allowed_provider_keys
        }
        monitoring_active = self._monitoring_active()
        counts = self.database.operational_counts()
        return DiagnosticBundle(
            generated_at=self.clock().isoformat(),
            app_version=__version__,
            schema_version=self.database.schema_version,
            provider_status=safe_providers,
            settings_redacted=redacted_settings,
            env_diagnostics=self.env_diagnostics(),
            monitoring={
                "active": monitoring_active,
                "note": "Monitoring status only; no account identity or AI tokens.",
            },
            counts=counts,
            redaction={
                "excluded": list(_EXCLUDED_DIAGNOSTIC_FIELDS),
                "allowlist": sorted(_DIAGNOSTIC_SETTINGS_ALLOWLIST),
                "redacted_keys": sorted(self._collect_redacted_keys(allowlisted)),
                "policy": (
                    "Strict allowlist only. Credentials, tokens, cookies, prompts, "
                    "licensed full text, raw env, absolute paths, broker numbers, "
                    "and local AI identity are excluded."
                ),
            },
        )

    # ── Printable snapshot ────────────────────────────────────────────────

    def printable_html(self, account_id: str = "all") -> tuple[str, str]:
        snapshot = self._require_snapshot(account_id)
        generated = self.clock().isoformat()
        provider = snapshot.freshness.provider if snapshot.freshness is not None else "unknown"
        freshness_as_of = (
            snapshot.freshness.as_of.isoformat()
            if snapshot.freshness is not None and snapshot.freshness.as_of is not None
            else snapshot.captured_at.isoformat()
        )
        rows = []
        for strategy in snapshot.strategies:
            horizon = (
                strategy.horizon.value
                if hasattr(strategy.horizon, "value")
                else str(strategy.horizon)
            )
            dte = strategy.days_to_expiration if strategy.days_to_expiration is not None else "—"
            rows.append(
                "<tr>"
                f"<td>{_escape(strategy.underlying)}</td>"
                f"<td>{_escape(strategy.strategy_type)}</td>"
                f"<td>{_escape(horizon)}</td>"
                f"<td>{dte}</td>"
                f"<td>{strategy.unrealized_pnl:+.2f}</td>"
                f"<td>{len(strategy.legs)}</td>"
                "</tr>"
            )
        body_rows = "\n".join(rows) or "<tr><td colspan='6'>No strategies</td></tr>"
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Position Pilot Portfolio Snapshot</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; color: #111; }}
    h1 {{ font-size: 1.4rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{
      border: 1px solid #999; padding: 0.4rem 0.6rem; text-align: left;
      font-variant-numeric: tabular-nums;
    }}
    th {{ background: #eee; }}
    .meta {{ color: #444; font-size: 0.9rem; }}
    .disclaimer {{ margin-top: 1.5rem; font-size: 0.85rem; color: #333; max-width: 48rem; }}
    @media print {{ body {{ margin: 0.5in; }} }}
  </style>
</head>
<body>
  <h1>Position Pilot — Portfolio Snapshot</h1>
  <p class="meta">
    Generated: {_escape(generated)} · Snapshot: {_escape(snapshot.snapshot_id)} ·
    Captured: {_escape(snapshot.captured_at.isoformat())} · Scope: {_escape(account_id)} ·
    App {_escape(__version__)} · Schema {self.database.schema_version}
  </p>
  <p class="meta">
    Provider: {_escape(provider)} · As-of: {_escape(freshness_as_of)} ·
    Attribution: local Position Pilot export · Decision support only · Read-only workstation
  </p>
  <table>
    <thead>
      <tr>
        <th>Underlying</th><th>Strategy</th><th>Horizon</th>
        <th>DTE</th><th>Unrealized P/L</th><th>Legs</th>
      </tr>
    </thead>
    <tbody>
      {body_rows}
    </tbody>
  </table>
  <p class="disclaimer">{_escape(DISCLAIMER)}</p>
</body>
</html>
"""
        return f"portfolio-snapshot-{self._stamp()}.html", html

    def printable_pdf(self, account_id: str = "all") -> tuple[str, bytes]:
        """Return a multi-page valid PDF covering every strategy in the snapshot."""

        snapshot = self._require_snapshot(account_id)
        generated = self.clock().isoformat()
        provider = snapshot.freshness.provider if snapshot.freshness is not None else "unknown"
        freshness_as_of = (
            snapshot.freshness.as_of.isoformat()
            if snapshot.freshness is not None and snapshot.freshness.as_of is not None
            else snapshot.captured_at.isoformat()
        )
        header = [
            "Position Pilot — Portfolio Snapshot",
            f"Generated: {generated}",
            f"Snapshot: {snapshot.snapshot_id}",
            f"Captured: {snapshot.captured_at.isoformat()}",
            f"Provider: {provider}  As-of: {freshness_as_of}",
            f"Scope: {account_id}",
            f"App: {__version__}  Schema: {self.database.schema_version}",
            f"Strategies: {len(snapshot.strategies)}  Accounts: {len(snapshot.accounts)}",
            "Attribution: local Position Pilot export (decision support only)",
            "",
            "Underlying | Strategy | Horizon | DTE | P/L | Legs",
        ]
        body: list[str] = []
        for strategy in snapshot.strategies:
            horizon = (
                strategy.horizon.value
                if hasattr(strategy.horizon, "value")
                else str(strategy.horizon)
            )
            dte = (
                str(strategy.days_to_expiration) if strategy.days_to_expiration is not None else "-"
            )
            body.append(
                f"{strategy.underlying} | {strategy.strategy_type} | {horizon} | "
                f"{dte} | {strategy.unrealized_pnl:+.2f} | {len(strategy.legs)}"
            )
        if not snapshot.strategies:
            body.append("(no strategies in scope)")
        footer = ["", DISCLAIMER]
        pdf_bytes = _simple_pdf(header + body + footer)
        return f"portfolio-snapshot-{self._stamp()}.pdf", pdf_bytes

    # ── Retention ─────────────────────────────────────────────────────────

    def retention_settings(self) -> RetentionSettings:
        stored = self.database.get_setting("operations.retention", {}) or {}
        return RetentionSettings(
            portfolio_snapshots_days=int(
                stored.get(
                    "portfolio_snapshots_days",
                    _RETENTION_DEFAULTS["portfolio_snapshots_days"]["default"],
                )
            ),
            catalyst_events_days=int(
                stored.get(
                    "catalyst_events_days",
                    _RETENTION_DEFAULTS["catalyst_events_days"]["default"],
                )
            ),
            article_metadata_days=int(
                stored.get(
                    "article_metadata_days",
                    _RETENTION_DEFAULTS["article_metadata_days"]["default"],
                )
            ),
            recommendation_history_days=int(
                stored.get(
                    "recommendation_history_days",
                    _RETENTION_DEFAULTS["recommendation_history_days"]["default"],
                )
            ),
            transaction_history=str(stored.get("transaction_history", "indefinite")),
            updated_at=stored.get("updated_at"),
        )

    def update_retention_settings(self, payload: dict[str, Any]) -> RetentionSettings:
        current = self.retention_settings().model_dump()
        for key in (
            "portfolio_snapshots_days",
            "catalyst_events_days",
            "article_metadata_days",
            "recommendation_history_days",
        ):
            if key in payload and payload[key] is not None:
                meta = _RETENTION_DEFAULTS[key]
                value = int(payload[key])
                if value < meta["min"] or value > meta["max"]:
                    raise ValueError(f"{key} must be between {meta['min']} and {meta['max']}")
                current[key] = value
        # Destructive clear_confirmed mode is intentionally unsupported.
        if "transaction_history" in payload and payload["transaction_history"] is not None:
            value = str(payload["transaction_history"])
            if value != "indefinite":
                raise ValueError(
                    "transaction_history is always indefinite; "
                    "destructive clear modes are not supported."
                )
            current["transaction_history"] = "indefinite"
        else:
            current["transaction_history"] = "indefinite"
        current["updated_at"] = self.clock().isoformat()
        self.database.set_setting("operations.retention", current)
        return self.retention_settings()

    def retention_preview(self) -> RetentionPreview:
        settings = self.retention_settings()
        # Force indefinite transaction policy in preview regardless of stale storage.
        settings.transaction_history = "indefinite"
        candidates = self.database.retention_candidates(
            portfolio_snapshots_days=settings.portfolio_snapshots_days,
            catalyst_events_days=settings.catalyst_events_days,
            article_metadata_days=settings.article_metadata_days,
            recommendation_history_days=0,  # never purge recommendation history
        )
        # Audit-critical classes are candidates for awareness but never deleted.
        audit_critical_keys = {
            "recommendation_history",
            "roll_chains",
            "audit_events",
            "trader_decisions",
            "transactions",
        }
        would_delete = {
            key: count
            for key, count in candidates.items()
            if key not in audit_critical_keys and key != "portfolio_snapshots_compactable"
        }
        would_compact = {
            "portfolio_snapshots_to_daily": int(
                candidates.get("portfolio_snapshots_compactable", 0)
            )
        }
        # Snapshots older than the window are compacted, not fully deleted.
        if "portfolio_snapshots" in would_delete:
            # Rows that become daily summaries are not pure deletes.
            compactable = would_compact["portfolio_snapshots_to_daily"]
            pure_delete = max(0, would_delete["portfolio_snapshots"] - max(compactable, 0))
            # Preview truth: report compaction + any residual purge of intra-day rows.
            would_delete["portfolio_snapshots_intraday"] = pure_delete
            would_delete.pop("portfolio_snapshots", None)
        audit_critical = [
            "transaction and roll-chain history (indefinite)",
            "trader decisions",
            "audit events",
            "recommendation history (indefinite)",
        ]
        return RetentionPreview(
            settings=settings,
            candidates=candidates,
            audit_critical_preserved=audit_critical,
            would_delete=would_delete,
            would_compact=would_compact,
        )

    def apply_retention(self, *, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            raise ValueError("Retention apply requires explicit confirm=true.")
        preview = self.retention_preview()
        settings = preview.settings
        # Ordinary apply never clears audit-critical classes or recommendation history.
        deleted = self.database.apply_retention(
            portfolio_snapshots_days=settings.portfolio_snapshots_days,
            catalyst_events_days=settings.catalyst_events_days,
            article_metadata_days=settings.article_metadata_days,
            recommendation_history_days=0,
            clear_transactions=False,
            compact_portfolio_snapshots=True,
        )
        return {
            "deleted": deleted,
            "compacted": deleted.get("portfolio_snapshots_compacted", 0),
            "settings": settings.model_dump(),
            "applied_at": self.clock().isoformat(),
            "audit_critical_preserved": preview.audit_critical_preserved,
            "disclaimer": DISCLAIMER,
        }

    # ── Backups ───────────────────────────────────────────────────────────

    def list_backups(self) -> list[BackupInfo]:
        directory = self.database.backup_directory
        if not directory.exists():
            return []
        items: list[BackupInfo] = []
        for path in sorted(directory.glob("position-pilot-*.sqlite3"), reverse=True):
            info = self._backup_info(path)
            if info is not None:
                items.append(info)
        return items

    def create_backup(self, *, reason: str = "manual") -> BackupInfo:
        """Create a faithful server-local SQLite backup (used by restore)."""

        safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "-", reason).strip("-")[:40] or "manual"
        path = self.database.backup(reason=safe_reason)
        if path is None:
            raise RuntimeError("Backup failed because the database file is missing.")
        # Database.backup already writes the unified sidecar; re-stamp with clock.
        self.database.write_backup_sidecar(
            path,
            reason=safe_reason,
            created_at=self.clock().isoformat(),
            app_version=__version__,
        )
        info = self._backup_info(path)
        assert info is not None
        return info

    def backup_path(self, backup_id: str) -> Path:
        """Resolve the faithful server-local backup file (restore only)."""

        path = self._resolve_backup_path(backup_id)
        if not path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")
        return path

    def portable_backup_archive(self, backup_id: str) -> tuple[str, bytes]:
        """Build a sanitized portable archive for browser download.

        Server-local full backups remain faithful for restore. Browser downloads
        never include credentials, tokens, prompts, cookies, licensed full article
        text, or raw broker account numbers.
        """

        source = self.backup_path(backup_id)
        portable_bytes = self._build_portable_sqlite(source)
        try:
            portable_schema = self._read_backup_schema_version(source)
        except ValueError:
            # Portable download still works for mismatched metadata; restore remains blocked.
            portable_schema = self.database.schema_version
        meta = {
            "schema_version": portable_schema,
            "app_version": __version__,
            "reason": "portable-download",
            "created_at": self.clock().isoformat(),
            "source_backup_id": Path(backup_id).name,
            "integrity": "ok" if self._verify_sqlite_bytes(portable_bytes) else "unknown",
            "excludes": list(_PORTABLE_BACKUP_EXCLUSIONS),
            "portable": True,
            "disclaimer": DISCLAIMER,
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("position-pilot-portable.sqlite3", portable_bytes)
            archive.writestr(
                "position-pilot-portable.meta.json",
                json.dumps(meta, indent=2),
            )
            archive.writestr(
                "README.txt",
                (
                    "Position Pilot portable backup (sanitized).\n"
                    "Broker account numbers are pseudonymized.\n"
                    "Catalyst full_text, credentials, tokens, prompts, and cookies "
                    "are excluded.\n"
                    "Do not use this archive for server restore; restore uses the "
                    "faithful server-local backup.\n"
                    f"{DISCLAIMER}\n"
                ),
            )
        filename = f"{Path(backup_id).stem}-portable.zip"
        return filename, buffer.getvalue()

    def restore_backup(self, backup_id: str, *, confirm: bool = False) -> RestoreResult:
        if not confirm:
            raise ValueError("Restore requires explicit confirm=true.")
        if self._monitoring_active():
            raise RuntimeError(
                "Restore is blocked while monitoring is active. Disable monitoring first."
            )
        source = self._resolve_backup_path(backup_id)
        if not source.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")
        if not self._verify_sqlite(source):
            raise ValueError("Backup failed integrity check and cannot be restored.")

        backup_schema = self._read_backup_schema_version(source)
        if backup_schema is not None and backup_schema > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Backup schema version {backup_schema} is newer than this "
                f"application's schema {CURRENT_SCHEMA_VERSION} and cannot be restored."
            )

        pre = self.create_backup(reason="pre-restore")
        # Atomic replace: copy to temp beside live DB, then replace.
        live = self.database.path
        temp = live.with_name(f".restore-{uuid4().hex}.sqlite3")
        try:
            shutil.copy2(source, temp)
            os.chmod(temp, 0o600)
            if not self._verify_sqlite(temp):
                raise ValueError("Copied backup failed integrity check.")
            os.replace(temp, live)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)
        # Re-open / re-migrate if needed (schema upgrades only; never auto-downgrade).
        self.database.reopen_after_restore()
        return RestoreResult(
            restored=True,
            backup_id=backup_id,
            pre_restore_backup_id=pre.backup_id,
            schema_version=self.database.schema_version,
            message=(
                "Database restored atomically from the faithful server-local backup. "
                "A pre-restore backup was created. "
                "Restart the dashboard if long-lived connections retain stale state."
            ),
        )

    # ── Update readiness ──────────────────────────────────────────────────

    def update_readiness(self) -> UpdateReadiness:
        latest = self.version_probe_fn() if self.version_probe_fn is not None else None
        # Never call package managers. latest remains None unless a probe is injected.
        update_available = bool(latest and latest != __version__)
        monitoring = self._monitoring_active()
        blocked = None
        if monitoring:
            blocked = "Monitoring is active; stop monitoring before applying an update."
        instructions = [
            "1. Disable monitoring from Settings if it is running.",
            "2. Create a manual backup from Settings → Diagnostics.",
            "3. Review release notes and schema migration notes for the target version.",
            "4. Install the new version yourself (for example: uv sync after git pull).",
            "5. Launch `pilot dashboard` and confirm schema migration completes.",
            "6. If needed, restore the pre-update backup from Settings → Diagnostics.",
        ]
        return UpdateReadiness(
            current_version=__version__,
            latest_version=latest,
            update_available=update_available,
            schema_version=self.database.schema_version,
            schema_migrations_pending=self.database.schema_version < CURRENT_SCHEMA_VERSION,
            backup_required_before_update=True,
            monitoring_active=monitoring,
            blocked_reason=blocked,
            reversible_instructions=instructions,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _require_snapshot(self, account_id: str) -> PortfolioSnapshot:
        snapshot = self.portfolio.latest(account_id)
        if snapshot is None:
            snapshot = self.database.latest_portfolio_snapshot()
        if snapshot is None:
            raise LookupError("No portfolio snapshot is available.")
        if account_id != "all":
            snapshot = snapshot.for_account(account_id)
        return snapshot

    def _stamp(self) -> str:
        return self.clock().strftime("%Y%m%dT%H%M%SZ")

    def _monitoring_active(self) -> bool:
        """Fail closed: unknown/error is treated as active (blocks restore/update)."""

        if self.monitoring_active_fn is None:
            # No probe wired — treat as unknown → fail closed for restore/update guards.
            return True
        try:
            return bool(self.monitoring_active_fn())
        except Exception:
            return True

    def _leg_row(self, account, strategy, leg, snapshot: PortfolioSnapshot) -> list[Any]:
        expiration = leg.expiration_date
        if hasattr(expiration, "isoformat"):
            expiration = expiration.isoformat()
        return [
            account.account_id,
            account.label,
            strategy.strategy_id if strategy else "",
            strategy.strategy_type if strategy else "",
            strategy.underlying if strategy else leg.underlying_symbol,
            (
                strategy.horizon.value
                if strategy and hasattr(strategy.horizon, "value")
                else (str(strategy.horizon) if strategy else "")
            ),
            leg.symbol,
            leg.quantity,
            leg.position_type.value if hasattr(leg.position_type, "value") else leg.position_type,
            leg.strike_price,
            leg.option_type,
            expiration or "",
            leg.market_value,
            leg.unrealized_pnl,
            getattr(leg, "delta", None),
            getattr(leg, "theta", None),
            getattr(leg, "gamma", None),
            getattr(leg, "vega", None),
            getattr(leg, "implied_volatility", None),
            snapshot.snapshot_id,
            snapshot.captured_at.isoformat(),
            DISCLAIMER,
        ]

    def _backup_info(self, path: Path) -> BackupInfo | None:
        if not path.is_file():
            return None
        # Prevent path traversal: only files directly in backup_directory.
        try:
            path.resolve().relative_to(self.database.backup_directory.resolve())
        except ValueError:
            return None
        reason = "unknown"
        match = re.match(r"position-pilot-([a-zA-Z0-9_-]+)-", path.name)
        if match:
            reason = match.group(1)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        schema_version = None
        app_version = None
        excludes: list[str] = []
        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                schema_version = meta.get("schema_version")
                app_version = meta.get("app_version") or meta.get("application_version")
                created_at = meta.get("created_at", created_at)
                excludes = list(meta.get("excludes") or [])
            except (OSError, json.JSONDecodeError):
                pass
        digest = _sha256_file(path)
        ok = self._verify_sqlite(path)
        # path field is basename/id only — never absolute filesystem paths.
        return BackupInfo(
            backup_id=path.name,
            filename=path.name,
            path=path.name,
            size_bytes=path.stat().st_size,
            created_at=created_at,
            reason=reason,
            schema_version=schema_version,
            app_version=app_version,
            sha256=digest,
            integrity_ok=ok,
            excludes=excludes,
        )

    def _resolve_backup_path(self, backup_id: str) -> Path:
        # Strict basename only — no directories or traversal.
        name = Path(backup_id).name
        if name != backup_id or ".." in backup_id or "/" in backup_id or "\\" in backup_id:
            raise ValueError("Invalid backup identifier.")
        if not re.fullmatch(r"position-pilot-[a-zA-Z0-9_.-]+\.sqlite3", name):
            raise ValueError("Invalid backup identifier.")
        path = (self.database.backup_directory / name).resolve()
        root = self.database.backup_directory.resolve()
        if not str(path).startswith(str(root) + os.sep) and path != root:
            raise ValueError("Invalid backup path.")
        return path

    @staticmethod
    def _verify_sqlite(path: Path) -> bool:
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                row = connection.execute("PRAGMA integrity_check").fetchone()
                return bool(row and row[0] == "ok")
        except sqlite3.Error:
            return False

    def _is_gitignored(self, path: Path, root: Path) -> bool:
        if self._path_matches_gitignore(path, root):
            return True
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", str(path)],
                cwd=root,
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return self._path_matches_gitignore(path, root)

    def _is_tracked_by_git(self, path: Path, root: Path) -> bool | None:
        try:
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(path)],
                cwd=root,
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _path_matches_gitignore(path: Path, root: Path) -> bool:
        gitignore = root / ".gitignore"
        if not gitignore.exists():
            return False
        try:
            patterns = gitignore.read_text(encoding="utf-8").splitlines()
        except OSError:
            return False
        name = path.name
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else name
        for line in patterns:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line in {name, ".env", ".env.*", "*.env", rel, f"/{name}"}:
                return True
            if line.endswith(".env") and name.endswith(".env"):
                return True
        return False

    def _allowlisted_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Keep only diagnostic-safe settings keys (strict allowlist)."""

        out: dict[str, Any] = {}
        for key, value in settings.items():
            key_str = str(key)
            if key_str in _DIAGNOSTIC_SETTINGS_ALLOWLIST:
                out[key_str] = value
                continue
            # Nested allowlist prefixes (e.g. catalysts.* already covered by full key).
            if any(
                key_str == allowed or key_str.startswith(f"{allowed}.")
                for allowed in _DIAGNOSTIC_SETTINGS_ALLOWLIST
            ):
                out[key_str] = value
        # Drop account-bearing keys even if they match a broad allowlist entry.
        cleaned: dict[str, Any] = {}
        for key, value in out.items():
            if self._is_account_key(key):
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _safe_provider_status(value: str) -> str:
        lowered = value.lower()
        allowed = {
            "configured",
            "not_configured",
            "unavailable",
            "unknown",
            "ok",
            "error",
            "degraded",
            "disabled",
        }
        if lowered in allowed:
            return lowered
        # Collapse any identity-bearing free text.
        if "error" in lowered or "fail" in lowered:
            return "error"
        if "ok" in lowered or "ready" in lowered:
            return "ok"
        return "unknown"

    def _strip_home_paths(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._strip_home_paths(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._strip_home_paths(item) for item in value]
        if isinstance(value, str):
            return _HOME_PATH_RE.sub("[PATH]", value)
        return value

    def _read_backup_schema_version(self, path: Path) -> int | None:
        """Return the backup schema version.

        The SQLite ``schema_migrations`` table is authoritative whenever readable.
        A present sidecar is always cross-checked; mismatch rejects restore.
        """

        db_schema: int | None = None
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
                ).fetchone()
                if table:
                    row = connection.execute(
                        "SELECT MAX(version) FROM schema_migrations"
                    ).fetchone()
                    if row and row[0] is not None:
                        db_schema = int(row[0])
        except (OSError, TypeError, ValueError, sqlite3.Error):
            db_schema = None

        sidecar_schema: int | None = None
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("schema_version") is not None:
                    sidecar_schema = int(meta["schema_version"])
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                sidecar_schema = None

        if db_schema is not None and sidecar_schema is not None and db_schema != sidecar_schema:
            raise ValueError(
                f"Backup schema mismatch: database reports {db_schema} but "
                f"sidecar reports {sidecar_schema}."
            )
        # Prefer the live SQLite migrations table; fall back to sidecar only if DB unreadable.
        return db_schema if db_schema is not None else sidecar_schema

    def _build_portable_sqlite(self, source: Path) -> bytes:
        """Copy a faithful backup into a sanitized in-memory portable SQLite."""

        temp_dir = self.data_directory / ".portable-tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"portable-{uuid4().hex}.sqlite3"
        try:
            shutil.copy2(source, temp_path)
            os.chmod(temp_path, 0o600)
            with sqlite3.connect(temp_path) as connection:
                connection.row_factory = sqlite3.Row
                # Pseudonymize broker account numbers.
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='account_identities'"
                ).fetchone():
                    rows = connection.execute(
                        "SELECT account_number, account_id FROM account_identities"
                    ).fetchall()
                    for row in rows:
                        pseudo = f"acct-{hashlib.sha256(str(row[0]).encode()).hexdigest()[:12]}"
                        connection.execute(
                            "UPDATE account_identities SET account_number = ? WHERE account_id = ?",
                            (pseudo, row[1]),
                        )
                # Remove licensed full article text.
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='catalyst_articles'"
                ).fetchone():
                    connection.execute("UPDATE catalyst_articles SET full_text = NULL")
                # Strip sensitive settings values.
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='settings'"
                ).fetchone():
                    settings_rows = connection.execute(
                        "SELECT key, value_json FROM settings"
                    ).fetchall()
                    for row in settings_rows:
                        key = str(row[0])
                        if self._is_sensitive_key(key) or self._is_account_key(key):
                            connection.execute(
                                "UPDATE settings SET value_json = ? WHERE key = ?",
                                (json.dumps("[REDACTED]"), key),
                            )
                            continue
                        try:
                            parsed = json.loads(row[1])
                        except (TypeError, json.JSONDecodeError):
                            continue
                        redacted = self._redact_mapping(parsed, key=key)
                        redacted = self._strip_home_paths(redacted)
                        connection.execute(
                            "UPDATE settings SET value_json = ? WHERE key = ?",
                            (json.dumps(redacted), key),
                        )
                # Scrub account numbers inside JSON payloads.
                for table, column in (
                    ("portfolio_snapshots", "payload_json"),
                    ("roll_chains", "payload_json"),
                    ("audit_events", "payload_json"),
                    ("recommendations", "payload_json"),
                    ("recommendation_history", "payload_json"),
                    ("trader_decisions", "payload_json"),
                    ("alerts", "payload_json"),
                    ("legacy_cache", "payload_json"),
                ):
                    if not connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone():
                        continue
                    pk_cols = [
                        r[1]
                        for r in connection.execute(f"PRAGMA table_info({table})").fetchall()
                        if r[5] > 0  # pk ordinal
                    ]
                    if not pk_cols:
                        # Update by rowid.
                        for row in connection.execute(
                            f"SELECT rowid, {column} FROM {table}"
                        ).fetchall():
                            cleaned = self._scrub_json_payload(row[1])
                            if cleaned != row[1]:
                                connection.execute(
                                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                                    (cleaned, row[0]),
                                )
                    else:
                        select_cols = ", ".join([*pk_cols, column])
                        for row in connection.execute(
                            f"SELECT {select_cols} FROM {table}"
                        ).fetchall():
                            payload = row[-1]
                            cleaned = self._scrub_json_payload(payload)
                            if cleaned == payload:
                                continue
                            where = " AND ".join(f"{col} = ?" for col in pk_cols)
                            connection.execute(
                                f"UPDATE {table} SET {column} = ? WHERE {where}",
                                (cleaned, *row[: len(pk_cols)]),
                            )
                connection.commit()
            # VACUUM INTO rewrites a clean file so freelist pages cannot retain secrets.
            vacuumed = temp_path.with_name(temp_path.name + ".vac")
            try:
                with sqlite3.connect(temp_path) as connection:
                    # Path must be single-quoted for SQLite; avoid injection via uuid name only.
                    connection.execute(f"VACUUM INTO '{vacuumed.as_posix()}'")
                return vacuumed.read_bytes()
            finally:
                vacuumed.unlink(missing_ok=True)
        finally:
            temp_path.unlink(missing_ok=True)

    def _scrub_json_payload(self, raw: Any) -> str:
        if raw is None:
            return "null"
        text = raw if isinstance(raw, str) else json.dumps(raw)
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            # Still strip known broker sentinel shapes from plain text.
            return _HOME_PATH_RE.sub(
                "[PATH]",
                re.sub(
                    r"(?i)\"account_number\"\s*:\s*\"[^\"]+\"",
                    '"account_number":"[REDACTED]"',
                    text,
                ),
            )
        scrubbed = self._redact_mapping(data)
        scrubbed = self._strip_home_paths(scrubbed)
        return json.dumps(scrubbed)

    @staticmethod
    def _verify_sqlite_bytes(payload: bytes) -> bool:
        return payload[:16].startswith(b"SQLite format 3")

    def _redact_mapping(self, value: Any, *, key: str = "") -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for child_key, child_value in value.items():
                child_name = str(child_key)
                if self._is_account_key(child_name) or self._is_sensitive_key(child_name):
                    if isinstance(child_value, (dict, list)):
                        out[child_name] = self._redact_mapping(child_value, key=child_name)
                    else:
                        out[child_name] = "[REDACTED]"
                elif isinstance(child_value, (dict, list)):
                    out[child_name] = self._redact_mapping(child_value, key=child_name)
                else:
                    out[child_name] = self._redact_mapping(child_value, key=child_name)
            return out
        if isinstance(value, list):
            return [self._redact_mapping(item, key=key) for item in value]
        if self._is_sensitive_key(key) or self._is_account_key(key):
            return "[REDACTED]"
        if isinstance(value, str) and self._looks_like_secret(value):
            return "[REDACTED]"
        if isinstance(value, str):
            return _HOME_PATH_RE.sub("[PATH]", value)
        return value

    def _collect_redacted_keys(self, value: Any, *, prefix: str = "") -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                path = f"{prefix}.{child_key}" if prefix else str(child_key)
                if self._is_sensitive_key(str(child_key)) or self._is_account_key(str(child_key)):
                    found.add(path)
                found |= self._collect_redacted_keys(child_value, prefix=path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found |= self._collect_redacted_keys(item, prefix=f"{prefix}[{index}]")
        return found

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        lowered = key.lower()
        return any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS)

    @staticmethod
    def _is_account_key(key: str) -> bool:
        lowered = key.lower()
        # Opaque local ids are safe to keep; raw broker numbers are not.
        if lowered in {"account_id", "primary_account_id", "selected_account_id"}:
            return False
        return any(
            fragment == lowered or f"_{fragment}" in lowered or lowered.endswith(fragment)
            for fragment in (
                "account_number",
                "default_account",
                "broker_number",
                "broker_account",
            )
        ) or lowered in {"account", "accounts"}

    @staticmethod
    def _looks_like_secret(value: str) -> bool:
        if len(value) >= 32 and re.fullmatch(r"[A-Za-z0-9_\-+/=]+", value):
            return True
        return False


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _simple_pdf(lines: list[str], *, lines_per_page: int = 48) -> bytes:
    """Build a multi-page PDF with Helvetica text (no third-party deps).

    Each page uses MediaBox [0 0 612 792]. Content is paginated so large
    portfolios (200+ strategies) fit within page bounds.
    """

    def pdf_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # Soft-wrap long lines to stay inside page width (~100 chars at 10pt).
    wrapped: list[str] = []
    for line in lines:
        text = line if line else " "
        while len(text) > 100:
            wrapped.append(text[:100])
            text = text[100:]
        wrapped.append(text)

    pages: list[list[str]] = []
    for index in range(0, max(len(wrapped), 1), lines_per_page):
        pages.append(wrapped[index : index + lines_per_page])
    if not pages:
        pages = [["(empty)"]]

    page_streams: list[bytes] = []
    for page_lines in pages:
        content_lines = ["BT", "/F1 10 Tf", "50 780 Td", "14 TL"]
        first = True
        for line in page_lines:
            safe = pdf_escape(line[:110])
            if first:
                content_lines.append(f"({safe}) Tj")
                first = False
            else:
                content_lines.append("T*")
                content_lines.append(f"({safe}) Tj")
        content_lines.append("ET")
        page_streams.append("\n".join(content_lines).encode("latin-1", errors="replace"))

    # Object layout:
    # 1 Catalog, 2 Pages, 3 Font, then pairs of Page + Content per page.
    objects: list[bytes] = []
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    # Placeholder for Pages — filled after we know kids.
    objects.append(b"")  # index 1
    objects.append(b"3 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")

    page_object_numbers: list[int] = []
    for stream in page_streams:
        page_obj_num = len(objects) + 1  # 1-based
        content_obj_num = page_obj_num + 1
        page_object_numbers.append(page_obj_num)
        objects.append(
            (
                f"{page_obj_num} 0 obj<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 612 792] /Contents {content_obj_num} 0 R "
                f"/Resources << /Font << /F1 3 0 R >> >> >>endobj\n"
            ).encode("ascii")
        )
        objects.append(
            f"{content_obj_num} 0 obj<< /Length {len(stream)} >>stream\n".encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        )

    kids = " ".join(f"{num} 0 R" for num in page_object_numbers)
    objects[1] = (
        f"2 0 obj<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>endobj\n"
    ).encode("ascii")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)
    xref_pos = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)


def package_diagnostic_zip(bundle: DiagnosticBundle) -> bytes:
    """Optional zip wrapper for the diagnostic JSON (still fully redacted)."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "diagnostic-bundle.json",
            json.dumps(bundle.model_dump(mode="json"), indent=2),
        )
        archive.writestr(
            "README.txt",
            (
                "Position Pilot redacted diagnostic bundle.\n"
                f"{DISCLAIMER}\n"
                "This archive never includes credentials, tokens, cookies, prompts, "
                "licensed full article text, or raw environment values.\n"
            ),
        )
    return buffer.getvalue()
