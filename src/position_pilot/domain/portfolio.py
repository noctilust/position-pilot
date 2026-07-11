"""Portfolio application service shared by CLI and web consumers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from ..analysis.strategies import StrategyGroup, detect_strategies
from ..models import Account, Position
from ..persistence.sqlite import PositionPilotDatabase
from .snapshots import (
    AccountSnapshot,
    DataFreshness,
    FieldProvenance,
    PortfolioSnapshot,
    PortfolioTotals,
    PositionHorizon,
    PositionSnapshot,
    SnapshotState,
    StrategySnapshot,
)


class PortfolioSource(Protocol):
    """Read-only broker boundary needed to assemble a portfolio snapshot."""

    def get_accounts(self) -> list[Account]: ...

    def get_account_balances(self, account_number: str) -> dict | None: ...

    def get_positions(self, account_number: str) -> list[Position]: ...

    def enrich_positions_greeks_batch(self, positions: list[Position]) -> list[Position]: ...


class PortfolioService:
    """Build, persist, scope, and recover atomic portfolio snapshots."""

    def __init__(
        self,
        *,
        database: PositionPilotDatabase,
        source: PortfolioSource,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self.source = source
        self.clock = clock or (lambda: datetime.now(UTC))
        self._current_snapshot: PortfolioSnapshot | None = None

    def refresh(self, *, enrich: bool = True) -> PortfolioSnapshot:
        captured_at = self.clock()
        try:
            snapshot = self._build_snapshot(captured_at, enrich=enrich)
        except (ConnectionError, OSError, TimeoutError):
            cached = self.database.latest_portfolio_snapshot()
            if cached is None:
                raise
            return self._as_cached(cached, network_unreachable=True)
        self.database.save_portfolio_snapshot(snapshot)
        self._current_snapshot = snapshot
        return snapshot

    def latest(self, account_id: str = "all") -> PortfolioSnapshot | None:
        snapshot = self._current_snapshot
        if snapshot is None:
            stored = self.database.latest_portfolio_snapshot()
            snapshot = self._as_cached(stored) if stored else None
        return snapshot.for_account(account_id) if snapshot else None

    def primary_account_id(self) -> str:
        """Return the saved browser account scope."""

        return self.database.get_setting("primary_account_id", "all")

    def set_primary_account(self, account_id: str) -> None:
        """Persist the preferred browser account scope after validation."""

        snapshot = self.latest()
        valid_ids = {"all"} | (
            {account.account_id for account in snapshot.accounts} if snapshot else set()
        )
        if account_id not in valid_ids:
            raise KeyError(account_id)
        self.database.set_setting("primary_account_id", account_id)

    def set_strategy_horizon(
        self,
        strategy_id: str,
        horizon: PositionHorizon,
    ) -> StrategySnapshot:
        """Persist an editable strategy horizon and update current state."""

        snapshot = self.latest()
        if snapshot is None:
            raise KeyError(strategy_id)
        strategy = next(
            (item for item in snapshot.strategies if item.strategy_id == strategy_id),
            None,
        )
        if strategy is None:
            raise KeyError(strategy_id)
        updated = strategy.model_copy(update={"horizon": horizon})
        self.database.set_setting(f"horizon.strategy.{strategy_id}", horizon.value)
        if self._current_snapshot is not None:
            self._current_snapshot = self._current_snapshot.model_copy(
                update={
                    "strategies": [
                        updated if item.strategy_id == strategy_id else item
                        for item in self._current_snapshot.strategies
                    ]
                }
            )
        return updated

    @staticmethod
    def _as_cached(
        snapshot: PortfolioSnapshot,
        *,
        network_unreachable: bool = False,
    ) -> PortfolioSnapshot:
        timestamp = snapshot.captured_at.isoformat()
        prefix = (
            "Network is currently unreachable. Dashboard content is"
            if network_unreachable
            else "Dashboard content is"
        )
        return snapshot.model_copy(
            update={
                "state": SnapshotState.CACHED,
                "freshness": snapshot.freshness.model_copy(update={"state": "stale"}),
                "freshness_by_panel": {
                    panel: freshness.model_copy(update={"state": "stale"})
                    for panel, freshness in snapshot.freshness_by_panel.items()
                },
                "notice": (f"{prefix} a cached snapshot from {timestamp}."),
            }
        )

    def _build_snapshot(self, captured_at: datetime, *, enrich: bool) -> PortfolioSnapshot:
        accounts: list[AccountSnapshot] = []
        strategies: list[StrategySnapshot] = []
        broker_accounts = self.source.get_accounts()
        if not broker_accounts:
            raise ConnectionError("Broker returned no account data")
        for broker_account in broker_accounts:
            identity = self.database.account_identity(
                broker_account.account_number,
                broker_account.account_type,
            )
            balances = self.source.get_account_balances(broker_account.account_number) or {}
            positions = self.source.get_positions(broker_account.account_number)
            enrich_positions = getattr(self.source, "enrich_positions_greeks_batch", None)
            if enrich and positions and callable(enrich_positions):
                positions = enrich_positions(positions)
            position_snapshots = [
                self._position_snapshot(position, captured_at) for position in positions
            ]
            account = AccountSnapshot(
                account_id=identity.account_id,
                label=identity.label,
                account_type=broker_account.account_type,
                net_liquidating_value=balances.get("net_liquidating_value") or 0,
                cash_balance=balances.get("cash_balance") or 0,
                buying_power=balances.get("buying_power") or 0,
                maintenance_excess=balances.get("maintenance_excess"),
                day_trading_buying_power=balances.get("day_trading_buying_power"),
                pnl_today=balances.get("pnl_today") or 0,
                positions=position_snapshots,
                provenance={
                    field: FieldProvenance(
                        provider="tastytrade",
                        observed_at=captured_at,
                        field=field,
                    )
                    for field in (
                        "net_liquidating_value",
                        "cash_balance",
                        "buying_power",
                        "maintenance_excess",
                        "day_trading_buying_power",
                        "pnl_today",
                    )
                },
            )
            accounts.append(account)
            for group in detect_strategies(positions):
                strategies.append(self._strategy_snapshot(identity.account_id, group, captured_at))

        totals = PortfolioTotals(
            net_liquidating_value=sum(account.net_liquidating_value for account in accounts),
            cash_balance=sum(account.cash_balance for account in accounts),
            buying_power=sum(account.buying_power for account in accounts),
            unrealized_pnl=sum(
                position.unrealized_pnl for account in accounts for position in account.positions
            ),
        )
        return PortfolioSnapshot(
            snapshot_id=str(uuid4()),
            captured_at=captured_at,
            state=SnapshotState.LIVE,
            freshness=DataFreshness(as_of=captured_at, provider="tastytrade"),
            freshness_by_panel={
                "portfolio": DataFreshness(as_of=captured_at, provider="tastytrade")
            },
            accounts=accounts,
            strategies=strategies,
            totals=totals,
        )

    @staticmethod
    def _default_horizon(position: Position) -> PositionHorizon:
        if not position.is_option:
            return PositionHorizon.STRATEGIC
        if not position.is_short and (position.days_to_expiration or 0) >= 180:
            return PositionHorizon.STRATEGIC
        return PositionHorizon.TACTICAL

    @classmethod
    def _position_snapshot(cls, position: Position, captured_at: datetime) -> PositionSnapshot:
        provenance = {
            field: FieldProvenance(
                provider="tastytrade",
                observed_at=captured_at,
                field=field,
            )
            for field in ("quantity", "mark_price", "market_value", "unrealized_pnl")
        }
        greeks = position.greeks
        return PositionSnapshot(
            symbol=position.symbol,
            underlying_symbol=position.underlying_symbol,
            quantity=position.quantity,
            quantity_direction=position.quantity_direction,
            position_type=position.position_type.value,
            strike_price=position.strike_price,
            option_type=position.option_type,
            expiration_date=(
                position.expiration_date.isoformat() if position.expiration_date else None
            ),
            days_to_expiration=position.days_to_expiration,
            mark_price=position.mark_price,
            market_value=position.market_value,
            unrealized_pnl=position.unrealized_pnl,
            unrealized_pnl_percent=position.unrealized_pnl_percent,
            delta=greeks.delta if greeks else None,
            gamma=greeks.gamma if greeks else None,
            theta=greeks.theta if greeks else None,
            vega=greeks.vega if greeks else None,
            implied_volatility=greeks.implied_volatility if greeks else None,
            multiplier=position.multiplier,
            horizon=cls._default_horizon(position),
            provenance=provenance,
        )

    def _strategy_snapshot(
        self,
        account_id: str,
        group: StrategyGroup,
        captured_at: datetime,
    ) -> StrategySnapshot:
        identity_material = "|".join(sorted(position.symbol for position in group.positions))
        strategy_id = hashlib.sha256(
            f"{account_id}|{group.strategy_type.value}|{identity_material}".encode()
        ).hexdigest()[:20]
        legs = [self._position_snapshot(position, captured_at) for position in group.positions]
        horizon = (
            PositionHorizon.STRATEGIC
            if len(group.positions) == 1 and legs[0].horizon is PositionHorizon.STRATEGIC
            else PositionHorizon.TACTICAL
        )
        saved_horizon = self.database.get_setting(f"horizon.strategy.{strategy_id}")
        if saved_horizon:
            horizon = PositionHorizon(saved_horizon)
        return StrategySnapshot(
            strategy_id=strategy_id,
            account_id=account_id,
            underlying=group.underlying,
            strategy_type=group.strategy_type.value,
            expiration_date=group.expiration.isoformat() if group.expiration else None,
            days_to_expiration=group.days_to_expiration,
            quantity=group.total_quantity,
            strikes=group.strikes_display,
            unrealized_pnl=group.unrealized_pnl,
            unrealized_pnl_percent=group.unrealized_pnl_percent,
            total_delta=group.total_delta,
            total_theta=group.total_theta,
            horizon=horizon,
            legs=legs,
            provenance={
                "strategy_grouping": FieldProvenance(
                    provider="position-pilot",
                    observed_at=captured_at,
                    field="strategy_grouping",
                )
            },
        )
