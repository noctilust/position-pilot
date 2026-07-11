"""Account and position snapshot assembly."""

from __future__ import annotations

from datetime import datetime

from ..models import Account, Position
from ..persistence.sqlite import PositionPilotDatabase
from .snapshots import (
    AccountSnapshot,
    FieldProvenance,
    PositionHorizon,
    PositionSnapshot,
)


def default_horizon(position: Position) -> PositionHorizon:
    if not position.is_option:
        return PositionHorizon.STRATEGIC
    if position.days_to_expiration is None:
        return PositionHorizon.UNCLASSIFIED
    if not position.is_short and position.days_to_expiration >= 180:
        return PositionHorizon.STRATEGIC
    return PositionHorizon.TACTICAL


def position_snapshot(position: Position, captured_at: datetime) -> PositionSnapshot:
    provider_fields = {
        "quantity": position.quantity,
        "mark_price": position.mark_price,
        "market_value": position.market_value,
        "unrealized_pnl": position.unrealized_pnl,
        "expiration_date": position.expiration_date,
        "days_to_expiration": position.days_to_expiration,
        "delta": position.greeks.delta if position.greeks else None,
        "gamma": position.greeks.gamma if position.greeks else None,
        "theta": position.greeks.theta if position.greeks else None,
        "vega": position.greeks.vega if position.greeks else None,
        "implied_volatility": position.greeks.implied_volatility if position.greeks else None,
    }
    provenance = {
        field: FieldProvenance(provider="tastytrade", observed_at=captured_at, field=field)
        for field, value in provider_fields.items()
        if value is not None
    }
    greeks = position.greeks
    return PositionSnapshot(
        symbol=position.symbol,
        underlying_symbol=position.underlying_symbol,
        quantity=position.quantity,
        quantity_direction=position.quantity_direction,
        position_type=position.position_type,
        strike_price=position.strike_price,
        option_type=position.option_type,
        expiration_date=position.expiration_date.isoformat() if position.expiration_date else None,
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
        horizon=default_horizon(position),
        provenance=provenance,
    )


class AccountService:
    """Build redacted account snapshots with stable public identities."""

    def __init__(self, database: PositionPilotDatabase) -> None:
        self.database = database

    def snapshot(
        self,
        account: Account,
        balances: dict,
        positions: list[Position],
        captured_at: datetime,
    ) -> AccountSnapshot:
        identity = self.database.account_identity(account.account_number, account.account_type)
        return AccountSnapshot(
            account_id=identity.account_id,
            label=identity.label,
            account_type=account.account_type,
            net_liquidating_value=balances.get("net_liquidating_value") or 0,
            cash_balance=balances.get("cash_balance") or 0,
            buying_power=balances.get("buying_power") or 0,
            maintenance_excess=balances.get("maintenance_excess"),
            day_trading_buying_power=balances.get("day_trading_buying_power"),
            pnl_today=balances.get("pnl_today") or 0,
            positions=[position_snapshot(position, captured_at) for position in positions],
            provenance={
                field: FieldProvenance(provider="tastytrade", observed_at=captured_at, field=field)
                for field in (
                    "net_liquidating_value",
                    "cash_balance",
                    "buying_power",
                    "maintenance_excess",
                    "day_trading_buying_power",
                    "pnl_today",
                )
                if balances.get(field) is not None
            },
        )
