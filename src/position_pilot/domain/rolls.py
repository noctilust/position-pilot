"""Account-scoped roll history service with redacted public models."""

from __future__ import annotations

import hashlib
from datetime import date, datetime

from pydantic import BaseModel, Field

from ..models.roll import RollChain
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

    @property
    def chain_total_credit(self) -> float | None:
        if self.original_open_credit is None:
            return None
        return self.original_open_credit + sum(roll.premium_effect for roll in self.rolls)


class RollService:
    """Persist and retrieve normalized roll chains using opaque account identities."""

    def __init__(self, database: PositionPilotDatabase) -> None:
        self.database = database

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
        return [
            RollChainSnapshot.model_validate(payload)
            for payload in self.database.roll_chains(account_id, symbol=symbol)
        ]

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
