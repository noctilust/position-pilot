"""Account-scoped strategy detection and snapshot assembly."""

from __future__ import annotations

import hashlib
from datetime import datetime

from ..analysis.strategies import StrategyGroup, StrategyType, detect_strategies
from ..models import Position
from ..persistence.sqlite import PositionPilotDatabase
from .accounts import position_snapshot
from .risk import RiskService
from .snapshots import FieldProvenance, PositionHorizon, StrategySnapshot


class StrategyService:
    """Detect strategies inside one account boundary and apply durable horizons."""

    def __init__(self, database: PositionPilotDatabase) -> None:
        self.database = database

    def snapshots(
        self,
        account_id: str,
        positions: list[Position],
        captured_at: datetime,
        position_provenance: dict[str, dict[str, FieldProvenance]] | None = None,
    ) -> list[StrategySnapshot]:
        return [
            self._snapshot(account_id, group, captured_at, position_provenance or {})
            for group in detect_strategies(positions)
        ]

    def _snapshot(
        self,
        account_id: str,
        group: StrategyGroup,
        captured_at: datetime,
        position_provenance: dict[str, dict[str, FieldProvenance]],
    ) -> StrategySnapshot:
        identity_material = "|".join(sorted(position.symbol for position in group.positions))
        strategy_id = hashlib.sha256(
            f"{account_id}|{group.strategy_type.value}|{identity_material}".encode()
        ).hexdigest()[:20]
        legs = [
            position_snapshot(
                position,
                captured_at,
                position_provenance.get(position.symbol),
            )
            for position in group.positions
        ]
        horizon = (
            PositionHorizon.STRATEGIC
            if (len(group.positions) == 1 and legs[0].horizon is PositionHorizon.STRATEGIC)
            or group.strategy_type in {StrategyType.PROTECTIVE_PUT, StrategyType.COLLAR}
            else PositionHorizon.TACTICAL
        )
        saved_horizon = self.database.get_setting(f"horizon.strategy.{strategy_id}")
        if saved_horizon:
            horizon = PositionHorizon(saved_horizon)
        combined = RiskService().combined_greeks(legs)
        pnl_open = sum(leg.effective_pnl_open() for leg in legs)
        roll_adjustment = sum(leg.roll_adjustment for leg in legs)
        roll_count = sum(leg.roll_count for leg in legs)
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
            pnl_open=pnl_open,
            pnl_open_percent=group.unrealized_pnl_percent,
            roll_adjustment=roll_adjustment,
            roll_count=roll_count,
            total_delta=combined.delta or 0,
            total_theta=combined.theta or 0,
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
