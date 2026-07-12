"""Account-scoped roll history, pattern analytics, and heatmap service."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from ..analysis.roll_analytics import analyze_patterns
from ..models.roll import RollChain, RollEvent
from ..persistence.sqlite import PositionPilotDatabase


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chain_total_credit(self) -> float | None:
        if self.original_open_credit is None:
            return None
        return self.original_open_credit + sum(roll.premium_effect for roll in self.rolls)


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
    ) -> None:
        self.database = database
        self.legacy_history_path = legacy_history_path

    def save_chain(self, account_id: str, chain: RollChain) -> None:
        first_roll_id = chain.rolls[0].roll_id if chain.rolls else "unrolled"
        identity = "|".join(
            (
                account_id,
                chain.underlying,
                chain.strategy_type,
                chain.original_open_date.isoformat() if chain.original_open_date else "unknown",
                first_roll_id,
            )
        )
        snapshot = RollChainSnapshot(
            chain_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
            account_id=account_id,
            underlying=chain.underlying,
            strategy_type=chain.strategy_type,
            original_open_date=chain.original_open_date,
            original_open_credit=chain.original_open_credit,
            rolls=[
                RollEventSnapshot.model_validate(
                    {key: value for key, value in roll.to_dict().items() if key != "account_number"}
                )
                for roll in chain.rolls
            ],
        )
        self.database.save_roll_chain(snapshot.model_dump(mode="json"))

    def chains(self, account_id: str, *, symbol: str | None = None) -> list[RollChainSnapshot]:
        self.sync_legacy_history()
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
        # Keep zero cells so the grid is complete for the UI.
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
                self.save_chain(account_id, RollChain.from_dict(chain_data))
                migrated += 1
        return migrated

    def sync_legacy_history(self) -> int:
        """Continuously ingest the CLI/TUI roll cache after account IDs exist."""

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
                    self.save_chain(account_id, RollChain.from_dict(chain_data))
                except (KeyError, TypeError, ValueError):
                    continue
                migrated += 1
        return migrated

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
