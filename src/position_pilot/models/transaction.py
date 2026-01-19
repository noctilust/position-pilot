"""Data models for transactions and roll tracking."""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    """Type of transaction."""

    ORDER_FILL = "order-fill"
    DIVIDEND = "dividend"
    FEE = "fee"
    CASH_DEPOSIT = "cash-deposit"
    WITHDRAWAL = "withdrawal"
    INTEREST = "interest"
    ADJUSTMENT = "adjustment"
    TRANSFER = "transfer"


class Transaction(BaseModel):
    """A transaction from Tastytrade."""

    transaction_id: str
    transaction_type: TransactionType
    transaction_date: datetime
    description: str
    amount: float
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    commission: Optional[float] = None
    order_id: Optional[str] = None
    account_number: str = Field(default="", alias="account-number")

    class Config:
        populate_by_name = True


class OrderStatus(str, Enum):
    """Order status."""

    RECEIVED = "received"
    CANCELLED = "cancelled"
    FILLED = "filled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(BaseModel):
    """An order from Tastytrade."""

    order_id: str
    account_number: str
    symbol: str
    action: str  # "buy", "sell", "buy_to_open", "sell_to_open", etc.
    quantity: float
    order_type: str  # "limit", "market", "stop", "stop_limit"
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commissions: float = 0.0
    underlying_symbol: Optional[str] = None

    class Config:
        populate_by_name = True
