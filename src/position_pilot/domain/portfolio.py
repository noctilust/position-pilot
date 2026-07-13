"""Portfolio application service shared by CLI and web consumers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from ..models import Account, Greeks, Position
from ..persistence.sqlite import PositionPilotDatabase
from ..providers.router import FieldRouter
from .accounts import AccountService
from .rolls import RollService
from .snapshots import (
    AccountSnapshot,
    DataFreshness,
    FieldProvenance,
    FreshnessState,
    PortfolioSnapshot,
    PortfolioTotals,
    PositionHorizon,
    SnapshotState,
    StrategySnapshot,
)
from .strategies import StrategyService

logger = logging.getLogger(__name__)


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
        field_router: FieldRouter | None = None,
        clock: Callable[[], datetime] | None = None,
        roll_service: RollService | None = None,
    ) -> None:
        self.database = database
        self.source = source
        self.field_router = field_router
        self.clock = clock or (lambda: datetime.now(UTC))
        self.roll_service = roll_service
        self.accounts = AccountService(database)
        self.strategy_detector = StrategyService(database)
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
                "freshness": snapshot.freshness.model_copy(update={"state": FreshnessState.STALE}),
                "freshness_by_panel": {
                    panel: freshness.model_copy(update={"state": FreshnessState.STALE})
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
        # unavailable ranks above partial/stale; never downgrade unavailable.
        roll_panel_state = FreshnessState.FRESH
        roll_panel_provider = "position-pilot"
        # Track newest roll sync timestamp separately; fall back to captured_at only
        # when no sync timestamp is available (do not initialize to captured_at).
        newest_roll_sync: datetime | None = None
        for broker_account in broker_accounts:
            balances = self.source.get_account_balances(broker_account.account_number) or {}
            positions = self.source.get_positions(broker_account.account_number)
            enrich_positions = getattr(self.source, "enrich_positions_greeks_batch", None)
            if enrich and positions and callable(enrich_positions):
                positions = enrich_positions(positions)
            positions, position_provenance = self._apply_option_fallbacks(positions)
            account = self.accounts.snapshot(
                broker_account,
                balances,
                positions,
                captured_at,
                position_provenance,
            )
            account_strategies = self.strategy_detector.snapshots(
                account.account_id,
                positions,
                captured_at,
                position_provenance,
            )

            # Automatic roll ledger sync (TTL-bounded, non-fatal).
            if self.roll_service is not None:
                try:
                    sync_result = self.roll_service.sync_account_transactions(
                        account.account_id,
                        broker_account.account_number,
                        positions,
                    )
                    if sync_result.synced_at is not None and (
                        newest_roll_sync is None or sync_result.synced_at > newest_roll_sync
                    ):
                        newest_roll_sync = sync_result.synced_at
                    if sync_result.status == "unavailable":
                        roll_panel_state = FreshnessState.UNAVAILABLE
                    elif sync_result.status == "partial":
                        if roll_panel_state is not FreshnessState.UNAVAILABLE:
                            roll_panel_state = FreshnessState.STALE
                    adjusted_positions, account_strategies = (
                        self.roll_service.apply_roll_adjustments(
                            account.account_id,
                            account.positions,
                            account_strategies,
                        )
                    )
                    account = account.model_copy(update={"positions": adjusted_positions})
                except Exception as exc:  # noqa: BLE001 — positions must stay available
                    logger.warning("Roll adjustment failed for %s: %s", account.account_id, exc)
                    roll_panel_state = FreshnessState.UNAVAILABLE

            accounts.append(account)
            strategies.extend(account_strategies)

        totals = PortfolioTotals(
            net_liquidating_value=sum(account.net_liquidating_value for account in accounts),
            cash_balance=sum(account.cash_balance for account in accounts),
            buying_power=sum(account.buying_power for account in accounts),
            # Keep broker raw unrealized semantics for risk/account totals.
            unrealized_pnl=sum(
                position.unrealized_pnl for account in accounts for position in account.positions
            ),
        )
        roll_as_of = newest_roll_sync if newest_roll_sync is not None else captured_at
        freshness_by_panel = {
            "portfolio": DataFreshness(as_of=captured_at, provider="tastytrade"),
            "rolls": DataFreshness(
                as_of=roll_as_of,
                provider=roll_panel_provider,
                state=roll_panel_state,
            ),
        }
        return PortfolioSnapshot(
            snapshot_id=str(uuid4()),
            captured_at=captured_at,
            state=SnapshotState.LIVE,
            freshness=DataFreshness(as_of=captured_at, provider="tastytrade"),
            freshness_by_panel=freshness_by_panel,
            accounts=accounts,
            strategies=strategies,
            totals=totals,
        )

    def _apply_option_fallbacks(
        self,
        positions: list[Position],
    ) -> tuple[list[Position], dict[str, dict[str, FieldProvenance]]]:
        if self.field_router is None:
            return positions, {}
        enriched: list[Position] = []
        provenance: dict[str, dict[str, FieldProvenance]] = {}
        for position in positions:
            updates: dict = {}
            field_provenance: dict[str, FieldProvenance] = {}
            if position.is_option and position.mark_price is None:
                mark = self.field_router.resolve("option.mark", position.symbol)
                if mark is not None and isinstance(mark.value, (int, float)):
                    mark_price = float(mark.value)
                    market_value = mark_price * abs(position.quantity) * position.multiplier
                    unrealized_pnl = (
                        position.cost_basis - market_value
                        if position.is_short
                        else market_value - position.cost_basis
                    )
                    unrealized_pnl_percent = (
                        (unrealized_pnl / abs(position.cost_basis)) * 100
                        if position.cost_basis
                        else None
                    )
                    updates.update(
                        {
                            "mark_price": mark_price,
                            "market_value": market_value,
                            "unrealized_pnl": unrealized_pnl,
                            "unrealized_pnl_percent": unrealized_pnl_percent,
                        }
                    )
                    field_provenance.update(
                        {
                            field: FieldProvenance(
                                provider=mark.provider,
                                observed_at=mark.observed_at,
                                field=field,
                                fallback_reason=mark.fallback_reason,
                            )
                            for field in (
                                "mark_price",
                                "market_value",
                                "unrealized_pnl",
                                "unrealized_pnl_percent",
                            )
                        }
                    )
            missing_greeks = position.greeks is None or any(
                getattr(position.greeks, field) is None
                for field in ("delta", "gamma", "theta", "vega", "implied_volatility")
            )
            if position.is_option and missing_greeks:
                current = position.greeks.model_dump() if position.greeks else {}
                required_greeks = {
                    field
                    for field in ("delta", "gamma", "theta", "vega", "implied_volatility")
                    if current.get(field) is None
                }
                greek_value = self.field_router.resolve(
                    "option.greeks",
                    position.symbol,
                    required_keys=required_greeks,
                )
                if greek_value is not None and isinstance(greek_value.value, dict):
                    supplied = {
                        field: value
                        for field, value in greek_value.value.items()
                        if field in Greeks.model_fields
                        and value is not None
                        and current.get(field) is None
                    }
                    updates["greeks"] = Greeks(**{**current, **supplied})
                    field_provenance.update(
                        {
                            field: FieldProvenance(
                                provider=greek_value.provider,
                                observed_at=greek_value.observed_at,
                                field=field,
                                fallback_reason=greek_value.fallback_reason,
                            )
                            for field in supplied
                        }
                    )
            updated = position.model_copy(update=updates) if updates else position
            enriched.append(updated)
            if field_provenance:
                provenance[position.symbol] = field_provenance
        return enriched, provenance
