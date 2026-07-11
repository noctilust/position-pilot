"""Read-only order activity with fill linkage and redacted identities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel, Field

from ..models.transaction import Order, Transaction
from ..persistence.sqlite import PositionPilotDatabase


class OrderSource(Protocol):
    def get_orders(self, account_number: str, *, limit: int = 100) -> list[Order]: ...

    def get_transactions(
        self,
        account_number: str,
        *,
        start_date=None,
        end_date=None,
    ) -> list[Transaction]: ...


class FillSnapshot(BaseModel):
    fill_id: str
    filled_at: datetime
    symbol: str
    quantity: float | None = None
    price: float | None = None
    amount: float = 0
    commission: float | None = None
    action: str | None = None


class OrderSnapshot(BaseModel):
    order_id: str
    account_id: str
    symbol: str
    underlying_symbol: str | None = None
    action: str
    quantity: float
    order_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    filled_quantity: float = 0
    average_fill_price: float | None = None
    commissions: float = 0
    fills: list[FillSnapshot] = Field(default_factory=list)


class OrderService:
    """Expose brokerage order history without raw account or order identifiers."""

    def __init__(self, *, database: PositionPilotDatabase, source: OrderSource) -> None:
        self.database = database
        self.source = source

    def list_orders(self, account_id: str, *, limit: int = 100) -> list[OrderSnapshot]:
        account_number = self.database.broker_number_for_account_id(account_id)
        if account_number is None:
            raise KeyError(account_id)
        orders = self.source.get_orders(account_number, limit=limit)
        if not orders:
            return []
        start_date = min(order.created_at for order in orders) - timedelta(days=1)
        transactions = self.source.get_transactions(account_number, start_date=start_date)
        fills_by_order: dict[str, list[Transaction]] = {}
        for transaction in transactions:
            if not transaction.order_id:
                continue
            fills_by_order.setdefault(transaction.order_id, []).append(transaction)

        snapshots: list[OrderSnapshot] = []
        for order in orders:
            public_order_id = self._public_id(account_id, "order", order.order_id)
            fills = [
                FillSnapshot(
                    fill_id=self._public_id(account_id, "fill", transaction.transaction_id),
                    filled_at=transaction.transaction_date,
                    symbol=transaction.symbol or order.symbol,
                    quantity=transaction.quantity,
                    price=transaction.price,
                    amount=transaction.amount,
                    commission=transaction.commission,
                    action=transaction.action,
                )
                for transaction in fills_by_order.get(order.order_id, [])
            ]
            snapshots.append(
                OrderSnapshot(
                    order_id=public_order_id,
                    account_id=account_id,
                    symbol=order.symbol,
                    underlying_symbol=order.underlying_symbol,
                    action=order.action,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    status=(
                        order.status.value
                        if hasattr(order.status, "value")
                        else str(order.status)
                    ),
                    created_at=order.created_at,
                    updated_at=order.updated_at,
                    filled_quantity=order.filled_quantity,
                    average_fill_price=order.average_fill_price,
                    commissions=order.commissions,
                    fills=fills,
                )
            )
        return snapshots

    @staticmethod
    def _public_id(account_id: str, kind: str, raw: str) -> str:
        return hashlib.sha256(f"{account_id}|{kind}|{raw}".encode()).hexdigest()[:20]
