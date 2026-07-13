"""Account-scoped roll history, pattern analytics, and heatmap service."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, computed_field

from ..analysis.roll_analytics import analyze_patterns
from ..analysis.roll_tracker import RollTracker, normalize_occ_symbol
from ..models.position import Position
from ..models.roll import RollChain, RollEvent
from ..models.transaction import Transaction
from ..persistence.sqlite import PositionPilotDatabase
from .snapshots import PositionSnapshot, StrategySnapshot

logger = logging.getLogger(__name__)

DEFAULT_TRANSACTION_LOOKBACK_DAYS = 730  # ~2 years
DEFAULT_TRANSACTION_LIMIT = 5_000
DEFAULT_SYNC_TTL_SECONDS = 15 * 60  # 15 minutes
LEGACY_IMPORT_SETTING = "rolls.legacy_file_imported"
SOURCE_BROKER = "broker"
SOURCE_LEGACY = "legacy"


class TransactionSource(Protocol):
    """Broker boundary used to refresh roll-relevant transaction history."""

    def get_transactions(
        self,
        account_number: str,
        *,
        start_date=None,
        end_date=None,
        limit: int = 250,
        force_refresh: bool = False,
    ) -> list[Transaction]: ...


class RollEventSnapshot(BaseModel):
    roll_id: str
    timestamp: datetime
    underlying: str
    strategy_type: str
    old_symbol: str
    old_strike: float
    old_expiration: date
    old_dte: int
    new_symbol: str
    new_strike: float
    new_expiration: date
    new_dte: int
    old_quantity: float = 1
    old_delta: float | None = None
    new_quantity: float = 1
    new_delta: float | None = None
    roll_pnl: float = 0
    premium_effect: float = 0
    commission: float = 0
    reason: str | None = None
    notes: str | None = None


class RollChainSnapshot(BaseModel):
    chain_id: str
    account_id: str
    underlying: str
    strategy_type: str
    rolls: list[RollEventSnapshot] = Field(default_factory=list)
    original_open_date: datetime | None = None
    original_open_credit: float | None = None
    history_complete: bool = True
    is_open: bool = False
    terminal_symbol: str | None = None
    root_symbol: str | None = None
    # broker = authoritative for P/L Open; legacy = display/history only.
    # Missing/unknown defaults to legacy (non-authoritative) for safety.
    source: str = SOURCE_LEGACY

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chain_total_credit(self) -> float | None:
        if self.original_open_credit is None:
            return None
        return self.original_open_credit + sum(roll.premium_effect for roll in self.rolls)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_roll_pnl(self) -> float:
        return sum(roll.roll_pnl for roll in self.rolls)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def roll_count(self) -> int:
        return len(self.rolls)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def history_status(self) -> str:
        if not self.history_complete:
            return "partial"
        return "complete" if self.rolls else "none"

    @property
    def is_broker_authoritative(self) -> bool:
        return self.source == SOURCE_BROKER


class RollPatternsSnapshot(BaseModel):
    account_id: str
    symbol: str | None = None
    avg_dte_at_roll: float = 0
    typical_roll_days: list[int] = Field(default_factory=list)
    min_dte_at_roll: int = 0
    max_dte_at_roll: int = 0
    avg_strike_adjustment: float = 0
    avg_roll_pnl: float = 0
    avg_premium_effect: float = 0
    total_pnl: float = 0
    win_rate: float = 0
    rolls_per_month: float = 0
    avg_days_between_rolls: float = 0
    best_dte_window: tuple[int, int] = (0, 0)
    best_strike_range: tuple[float, float] = (0.0, 0.0)
    total_rolls: int = 0
    avg_roll_duration_days: float = 0


class HeatmapCell(BaseModel):
    strike: float
    dte_bucket: str
    dte_min: int
    dte_max: int
    count: int


class RollHeatmapSnapshot(BaseModel):
    account_id: str
    underlying: str
    cells: list[HeatmapCell] = Field(default_factory=list)
    strikes: list[float] = Field(default_factory=list)
    buckets: list[str] = Field(default_factory=list)
    total_rolls: int = 0


class RollSyncResult(BaseModel):
    """Outcome of a non-fatal transaction/roll sync attempt."""

    account_id: str
    attempted: bool = False
    refreshed: bool = False
    chain_count: int = 0
    status: str = "skipped"  # skipped | refreshed | cached | unavailable | partial
    error: str | None = None
    synced_at: datetime | None = None


DTE_BUCKETS: list[tuple[int, int, str]] = [
    (0, 7, "0-7"),
    (8, 14, "8-14"),
    (15, 21, "15-21"),
    (22, 35, "22-35"),
    (36, 999, "36+"),
]


class RollService:
    """Persist and retrieve normalized roll chains using opaque account identities."""

    def __init__(
        self,
        database: PositionPilotDatabase,
        *,
        legacy_history_path: Path | None = None,
        transaction_source: TransactionSource | None = None,
        sync_ttl_seconds: int = DEFAULT_SYNC_TTL_SECONDS,
        lookback_days: int = DEFAULT_TRANSACTION_LOOKBACK_DAYS,
        transaction_limit: int = DEFAULT_TRANSACTION_LIMIT,
        clock=None,
    ) -> None:
        self.database = database
        self.legacy_history_path = legacy_history_path
        self.transaction_source = transaction_source
        self.sync_ttl_seconds = sync_ttl_seconds
        self.lookback_days = lookback_days
        self.transaction_limit = transaction_limit
        self.clock = clock or (lambda: datetime.now(UTC))
        self.tracker = RollTracker()

    @staticmethod
    def stable_chain_id(account_id: str, chain: RollChain) -> str:
        """Stable per-lineage identity (does not include the changing terminal).

        Anchors on account + underlying + root OCC symbol + first roll ID only.
        Deliberately omits ``original_open_date`` / completion state so a
        partial-history row keeps the same ID when older fills later complete it.
        """
        first_roll_id = chain.rolls[0].roll_id if chain.rolls else "unrolled"
        root = (
            chain.root_symbol or (chain.rolls[0].old_symbol if chain.rolls else None) or "unknown"
        )
        identity = "|".join(
            (
                account_id,
                chain.underlying,
                normalize_occ_symbol(root) or "unknown",
                first_roll_id,
            )
        )
        return hashlib.sha256(identity.encode()).hexdigest()[:20]

    def save_chain(
        self,
        account_id: str,
        chain: RollChain,
        *,
        source: str = SOURCE_BROKER,
    ) -> str:
        chain_id = self.stable_chain_id(account_id, chain)
        if source == SOURCE_LEGACY:
            existing = self._payload_by_chain_id(account_id, chain_id)
            if existing is not None and existing.get("source") == SOURCE_BROKER:
                # Never overwrite a broker-authoritative row with legacy data.
                return chain_id

        root = chain.root_symbol or (chain.rolls[0].old_symbol if chain.rolls else None) or None
        snapshot = RollChainSnapshot(
            chain_id=chain_id,
            account_id=account_id,
            underlying=chain.underlying,
            strategy_type=chain.strategy_type,
            original_open_date=chain.original_open_date,
            original_open_credit=chain.original_open_credit,
            history_complete=chain.history_complete,
            is_open=chain.is_open,
            terminal_symbol=chain.resolved_terminal_symbol(),
            root_symbol=root,
            source=source if source in {SOURCE_BROKER, SOURCE_LEGACY} else SOURCE_LEGACY,
            rolls=[
                RollEventSnapshot.model_validate(
                    {key: value for key, value in roll.to_dict().items() if key != "account_number"}
                )
                for roll in chain.rolls
            ],
        )
        self.database.save_roll_chain(snapshot.model_dump(mode="json"))
        return chain_id

    def chains(self, account_id: str, *, symbol: str | None = None) -> list[RollChainSnapshot]:
        # Import legacy file at most once; never re-upsert on every read.
        self.ensure_legacy_history_imported()
        return [
            RollChainSnapshot.model_validate(payload)
            for payload in self.database.roll_chains(account_id, symbol=symbol)
        ]

    def patterns(self, account_id: str, *, symbol: str | None = None) -> RollPatternsSnapshot:
        chains = self.chains(account_id, symbol=symbol)
        domain_chains = [self._to_domain_chain(chain) for chain in chains]
        patterns = analyze_patterns(domain_chains)
        return RollPatternsSnapshot(
            account_id=account_id,
            symbol=symbol,
            avg_dte_at_roll=patterns.avg_dte_at_roll,
            typical_roll_days=list(patterns.typical_roll_days),
            min_dte_at_roll=patterns.min_dte_at_roll,
            max_dte_at_roll=patterns.max_dte_at_roll,
            avg_strike_adjustment=patterns.avg_strike_adjustment,
            avg_roll_pnl=patterns.avg_roll_pnl,
            avg_premium_effect=patterns.avg_premium_effect,
            total_pnl=patterns.total_pnl,
            win_rate=patterns.win_rate,
            rolls_per_month=patterns.rolls_per_month,
            avg_days_between_rolls=patterns.avg_days_between_rolls,
            best_dte_window=patterns.best_dte_window,
            best_strike_range=patterns.best_strike_range,
            total_rolls=patterns.total_rolls,
            avg_roll_duration_days=patterns.avg_roll_duration_days,
        )

    def heatmap(self, account_id: str, *, symbol: str) -> RollHeatmapSnapshot:
        chains = self.chains(account_id, symbol=symbol)
        counts: dict[tuple[float, str], int] = {}
        strikes: set[float] = set()
        total = 0
        for chain in chains:
            for roll in chain.rolls:
                bucket = self._bucket_for_dte(roll.old_dte)
                key = (roll.old_strike, bucket[2])
                counts[key] = counts.get(key, 0) + 1
                strikes.add(roll.old_strike)
                total += 1
        cells = [
            HeatmapCell(
                strike=strike,
                dte_bucket=label,
                dte_min=low,
                dte_max=high,
                count=counts.get((strike, label), 0),
            )
            for strike in sorted(strikes, reverse=True)
            for low, high, label in DTE_BUCKETS
        ]
        return RollHeatmapSnapshot(
            account_id=account_id,
            underlying=symbol.upper(),
            cells=cells,
            strikes=sorted(strikes, reverse=True),
            buckets=[label for _, _, label in DTE_BUCKETS],
            total_rolls=total,
        )

    def migrate_legacy_cache(self) -> int:
        legacy = self.database.get_legacy_cache("roll_history") or {}
        migrated = 0
        for account_number, account_data in legacy.get("accounts", {}).items():
            account_id = self.database.account_id_for_broker_number(account_number)
            if account_id is None:
                continue
            for chain_data in account_data.get("chains", {}).values():
                self.save_chain(account_id, RollChain.from_dict(chain_data), source=SOURCE_LEGACY)
                migrated += 1
        return migrated

    def ensure_legacy_history_imported(self) -> int:
        """Import the CLI roll cache file once; skip subsequent reads."""
        if self.database.get_setting(LEGACY_IMPORT_SETTING, False):
            return 0
        migrated = self.sync_legacy_history()
        self.database.set_setting(LEGACY_IMPORT_SETTING, True)
        return migrated

    def sync_legacy_history(self) -> int:
        """Ingest the CLI roll cache after account IDs exist (legacy source only)."""
        path = self.legacy_history_path
        if path is None or not path.is_file():
            return 0
        try:
            payload = json.loads(path.read_text())
        except (OSError, TypeError, ValueError):
            return 0
        migrated = 0
        for account_number, account_data in payload.get("accounts", {}).items():
            account_id = self.database.account_id_for_broker_number(account_number)
            if account_id is None:
                continue
            for chain_data in account_data.get("chains", {}).values():
                try:
                    self.save_chain(
                        account_id,
                        RollChain.from_dict(chain_data),
                        source=SOURCE_LEGACY,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                migrated += 1
        return migrated

    def sync_account_transactions(
        self,
        account_id: str,
        account_number: str,
        positions: list[Position],
        *,
        force: bool = False,
    ) -> RollSyncResult:
        """Fetch broker transactions (TTL-bounded) and upsert detected roll chains.

        Non-fatal: failures return status=unavailable and never raise to callers.
        Incomplete lineages surface status=partial while still being persisted.
        """
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        result = RollSyncResult(account_id=account_id, synced_at=now)
        if self.transaction_source is None:
            result.status = "skipped"
            return result

        setting_key = f"rolls.sync.{account_id}"
        last_raw = self.database.get_setting(setting_key)
        last_sync: datetime | None = None
        if isinstance(last_raw, str) and last_raw:
            try:
                last_sync = datetime.fromisoformat(last_raw)
                if last_sync.tzinfo is None:
                    last_sync = last_sync.replace(tzinfo=UTC)
            except ValueError:
                last_sync = None

        ttl = timedelta(seconds=self.sync_ttl_seconds)
        within_ttl = last_sync is not None and (now - last_sync) < ttl
        if within_ttl and not force:
            self._relabel_open_status(account_id, positions)
            result.attempted = False
            # Cached read: still surface partial if stored broker chains are incomplete.
            stored = [
                c
                for c in self.chains(account_id)
                if c.source == SOURCE_BROKER and not c.history_complete
            ]
            result.status = "partial" if stored else "cached"
            result.chain_count = len(self.database.roll_chains(account_id))
            result.synced_at = last_sync
            return result

        result.attempted = True
        try:
            start_date = now - timedelta(days=self.lookback_days)
            transactions = self.transaction_source.get_transactions(
                account_number,
                start_date=start_date,
                end_date=None,
                limit=self.transaction_limit,
                force_refresh=True,
            )
        except Exception as exc:  # noqa: BLE001 — must not fail portfolio refresh
            logger.warning("Roll transaction sync failed for %s: %s", account_id, exc)
            result.status = "unavailable"
            result.error = str(exc)
            self._relabel_open_status(account_id, positions)
            return result

        try:
            detected = self.tracker.detect_rolls(
                transactions,
                positions,
                account_number,
                include_closed=True,
            )
            for chain in detected:
                chain.source = SOURCE_BROKER
                self.save_chain(account_id, chain, source=SOURCE_BROKER)
            self._relabel_open_status(account_id, positions)
            self.database.set_setting(setting_key, now.isoformat())
            result.refreshed = True
            result.chain_count = len(detected)
            result.synced_at = now
            if any(not chain.history_complete for chain in detected):
                result.status = "partial"
            else:
                result.status = "refreshed"
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Roll detection failed for %s: %s", account_id, exc)
            result.status = "partial"
            result.error = str(exc)
            return result

    def apply_roll_adjustments(
        self,
        account_id: str,
        positions: list[PositionSnapshot],
        strategies: list[StrategySnapshot],
    ) -> tuple[list[PositionSnapshot], list[StrategySnapshot]]:
        """Attach complete open *broker* lineages to matching current legs.

        Only broker-sourced chains whose terminal matches a current open symbol and
        whose history is complete may adjust that leg. Legacy rows never adjust.
        """
        open_by_symbol = self._open_broker_chains_by_terminal(account_id)
        adjusted_positions = [
            self._adjust_position(
                position,
                open_by_symbol.get(normalize_occ_symbol(position.symbol)),
            )
            for position in positions
        ]
        by_symbol = {normalize_occ_symbol(p.symbol): p for p in adjusted_positions}
        adjusted_strategies: list[StrategySnapshot] = []
        for strategy in strategies:
            if strategy.account_id != account_id:
                adjusted_strategies.append(strategy)
                continue
            legs = [by_symbol.get(normalize_occ_symbol(leg.symbol), leg) for leg in strategy.legs]
            pnl_open = sum(leg.effective_pnl_open() for leg in legs)
            roll_adjustment = sum(leg.roll_adjustment for leg in legs)
            roll_count = sum(leg.roll_count for leg in legs)
            # Sum every current leg's basis once (unrolled cost + rolled lifetime).
            bases = [leg.pnl_open_basis for leg in legs if leg.pnl_open_basis is not None]
            basis_sum = sum(bases) if bases else None
            if basis_sum is not None and abs(basis_sum) > 1e-9:
                pnl_open_percent = (pnl_open / abs(basis_sum)) * 100
            else:
                pnl_open_percent = strategy.unrealized_pnl_percent
            adjusted_strategies.append(
                strategy.model_copy(
                    update={
                        "legs": legs,
                        "pnl_open": pnl_open,
                        "pnl_open_percent": pnl_open_percent,
                        "pnl_open_basis": abs(basis_sum) if basis_sum is not None else None,
                        "roll_adjustment": roll_adjustment,
                        "roll_count": roll_count,
                    }
                )
            )
        return adjusted_positions, adjusted_strategies

    def _payload_by_chain_id(self, account_id: str, chain_id: str) -> dict | None:
        for payload in self.database.roll_chains(account_id):
            if payload.get("chain_id") == chain_id:
                return payload
        return None

    def _open_broker_chains_by_terminal(self, account_id: str) -> dict[str, RollChainSnapshot]:
        """Map terminal OCC symbol -> open *broker* chain eligible to adjust that leg."""
        by_terminal: dict[str, RollChainSnapshot] = {}
        for chain in self.chains(account_id):
            if chain.source != SOURCE_BROKER:
                continue
            if not chain.is_open or not chain.rolls:
                continue
            terminal = normalize_occ_symbol(chain.terminal_symbol or chain.rolls[-1].new_symbol)
            if not terminal:
                continue
            existing = by_terminal.get(terminal)
            if existing is None:
                by_terminal[terminal] = chain
                continue
            if not existing.history_complete and chain.history_complete:
                by_terminal[terminal] = chain
        return by_terminal

    def _adjust_position(
        self,
        position: PositionSnapshot,
        chain: RollChainSnapshot | None,
    ) -> PositionSnapshot:
        raw_basis = abs(position.cost_basis) if position.cost_basis else None

        if chain is None or chain.source != SOURCE_BROKER:
            # Unrolled / legacy-only: basis is raw absolute cost when known.
            basis = raw_basis
            percent = (
                (position.unrealized_pnl / basis) * 100
                if basis is not None and basis > 1e-9
                else position.unrealized_pnl_percent
            )
            return position.model_copy(
                update={
                    "pnl_open": position.unrealized_pnl,
                    "pnl_open_percent": percent,
                    "pnl_open_basis": basis,
                    "roll_adjustment": 0.0,
                    "roll_count": 0,
                    "roll_chain_id": None,
                    "roll_history_status": "none",
                    "lifetime_net_credit": None,
                }
            )

        history_status = "complete" if chain.history_complete else "partial"
        lifetime = chain.chain_total_credit
        if chain.history_complete:
            adjustment = sum(roll.roll_pnl for roll in chain.rolls)
            pnl_open = position.unrealized_pnl + adjustment
            # Rolled complete: use absolute lifetime/net premium as basis.
            basis = abs(lifetime) if lifetime is not None else raw_basis
            if basis is not None and basis > 1e-9:
                pnl_open_percent = (pnl_open / basis) * 100
            else:
                pnl_open_percent = position.unrealized_pnl_percent
        else:
            adjustment = 0.0
            pnl_open = position.unrealized_pnl
            basis = raw_basis
            pnl_open_percent = (
                (pnl_open / basis) * 100
                if basis is not None and basis > 1e-9
                else position.unrealized_pnl_percent
            )

        return position.model_copy(
            update={
                "pnl_open": pnl_open,
                "pnl_open_percent": pnl_open_percent,
                "pnl_open_basis": basis,
                "roll_adjustment": adjustment,
                "roll_count": len(chain.rolls),
                "roll_chain_id": chain.chain_id,
                "roll_history_status": history_status,
                "lifetime_net_credit": lifetime,
            }
        )

    def _relabel_open_status(self, account_id: str, positions: list[Position]) -> None:
        """Mark chains open only when terminal exactly matches a current position."""
        current = {normalize_occ_symbol(p.symbol) for p in positions if p.symbol}
        for payload in self.database.roll_chains(account_id):
            try:
                chain = RollChainSnapshot.model_validate(payload)
            except Exception:  # noqa: BLE001
                continue
            terminal = normalize_occ_symbol(
                chain.terminal_symbol or (chain.rolls[-1].new_symbol if chain.rolls else "")
            )
            is_open = bool(terminal) and terminal in current
            if chain.is_open == is_open and chain.terminal_symbol == (terminal or None):
                continue
            updated = chain.model_copy(
                update={
                    "is_open": is_open,
                    "terminal_symbol": terminal or chain.terminal_symbol,
                }
            )
            self.database.save_roll_chain(updated.model_dump(mode="json"))

    @staticmethod
    def _bucket_for_dte(dte: int) -> tuple[int, int, str]:
        for low, high, label in DTE_BUCKETS:
            if low <= dte <= high:
                return low, high, label
        return DTE_BUCKETS[-1]

    @staticmethod
    def _to_domain_chain(chain: RollChainSnapshot) -> RollChain:
        return RollChain(
            underlying=chain.underlying,
            strategy_type=chain.strategy_type,
            account_number="",
            original_open_date=chain.original_open_date,
            original_open_credit=chain.original_open_credit,
            history_complete=chain.history_complete,
            is_open=chain.is_open,
            terminal_symbol=chain.terminal_symbol,
            root_symbol=chain.root_symbol,
            source=chain.source,
            rolls=[
                RollEvent(
                    roll_id=roll.roll_id,
                    timestamp=roll.timestamp,
                    account_number="",
                    underlying=roll.underlying,
                    strategy_type=roll.strategy_type,
                    old_symbol=roll.old_symbol,
                    old_strike=roll.old_strike,
                    old_expiration=roll.old_expiration,
                    old_dte=roll.old_dte,
                    new_symbol=roll.new_symbol,
                    new_strike=roll.new_strike,
                    new_expiration=roll.new_expiration,
                    new_dte=roll.new_dte,
                    old_quantity=roll.old_quantity,
                    old_delta=roll.old_delta,
                    new_quantity=roll.new_quantity,
                    new_delta=roll.new_delta,
                    roll_pnl=roll.roll_pnl,
                    premium_effect=roll.premium_effect,
                    commission=roll.commission,
                    reason=roll.reason,
                    notes=roll.notes,
                )
                for roll in chain.rolls
            ],
        )
