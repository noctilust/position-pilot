"""Versioned, browser-safe portfolio snapshot models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ..models import PositionType


class SnapshotState(StrEnum):
    """Whether a snapshot came from a live refresh or durable cache."""

    LIVE = "live"
    CACHED = "cached"


class FreshnessState(StrEnum):
    """Freshness classification for an independently updated dataset."""

    FRESH = "fresh"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class PositionHorizon(StrEnum):
    """Expected decision horizon for a position or strategy."""

    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    UNCLASSIFIED = "unclassified"


class QuantityDirection(StrEnum):
    """Broker position direction."""

    LONG = "Long"
    SHORT = "Short"


class FieldProvenance(BaseModel):
    """Origin and observation time for a provider-backed field."""

    provider: str
    observed_at: datetime
    field: str


class DataFreshness(BaseModel):
    """Provider timestamp and freshness for a coherent data panel."""

    as_of: datetime
    provider: str
    state: FreshnessState = FreshnessState.FRESH


class PositionSnapshot(BaseModel):
    """A browser-safe open position leg."""

    symbol: str
    underlying_symbol: str
    quantity: int
    quantity_direction: QuantityDirection
    position_type: PositionType
    strike_price: float | None = None
    option_type: str | None = None
    expiration_date: str | None = None
    days_to_expiration: int | None = None
    mark_price: float | None = None
    market_value: float = 0
    unrealized_pnl: float = 0
    unrealized_pnl_percent: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    implied_volatility: float | None = None
    multiplier: int = 1
    horizon: PositionHorizon = PositionHorizon.UNCLASSIFIED
    provenance: dict[str, FieldProvenance] = Field(default_factory=dict)


class AccountSnapshot(BaseModel):
    """Public account identity, balances, and legs without broker identifiers."""

    account_id: str
    label: str
    account_type: str
    net_liquidating_value: float = 0
    cash_balance: float = 0
    buying_power: float = 0
    maintenance_excess: float | None = None
    day_trading_buying_power: float | None = None
    pnl_today: float = 0
    positions: list[PositionSnapshot] = Field(default_factory=list)
    provenance: dict[str, FieldProvenance] = Field(default_factory=dict)


class StrategySnapshot(BaseModel):
    """A detected strategy scoped to exactly one public account identity."""

    strategy_id: str
    account_id: str
    underlying: str
    strategy_type: str
    expiration_date: str | None = None
    days_to_expiration: int | None = None
    quantity: int
    strikes: str
    unrealized_pnl: float
    unrealized_pnl_percent: float | None = None
    total_delta: float = 0
    total_theta: float = 0
    horizon: PositionHorizon
    legs: list[PositionSnapshot]
    provenance: dict[str, FieldProvenance] = Field(default_factory=dict)


class PortfolioTotals(BaseModel):
    """Consolidated totals across the selected account scope."""

    net_liquidating_value: float = 0
    cash_balance: float = 0
    buying_power: float = 0
    unrealized_pnl: float = 0


class PortfolioSnapshot(BaseModel):
    """Atomic versioned state consumed by CLI and web surfaces."""

    schema_version: int = 1
    snapshot_id: str
    captured_at: datetime
    state: SnapshotState
    freshness: DataFreshness
    freshness_by_panel: dict[str, DataFreshness] = Field(default_factory=dict)
    accounts: list[AccountSnapshot]
    strategies: list[StrategySnapshot] = Field(default_factory=list)
    totals: PortfolioTotals = Field(default_factory=PortfolioTotals)
    selected_account_id: str = "all"
    notice: str | None = None

    def for_account(self, account_id: str) -> "PortfolioSnapshot":
        """Return one account view while preserving the snapshot identity."""

        if account_id == "all":
            return self
        accounts = [account for account in self.accounts if account.account_id == account_id]
        strategies = [strategy for strategy in self.strategies if strategy.account_id == account_id]
        if not accounts:
            raise KeyError(account_id)
        account = accounts[0]
        return self.model_copy(
            update={
                "accounts": accounts,
                "strategies": strategies,
                "selected_account_id": account_id,
                "totals": PortfolioTotals(
                    net_liquidating_value=account.net_liquidating_value,
                    cash_balance=account.cash_balance,
                    buying_power=account.buying_power,
                    unrealized_pnl=sum(position.unrealized_pnl for position in account.positions),
                ),
            }
        )
