"""Data models for positions and accounts."""

from datetime import date, datetime
from enum import Enum
from math import isfinite
from typing import Any, Optional

from pydantic import BaseModel, Field


def is_valid_mark_price(mark: Any) -> bool:
    """True when mark is a usable market price: finite and >= 0 (zero is valid)."""
    if isinstance(mark, bool):
        return False
    if isinstance(mark, (int, float)):
        return isfinite(mark) and mark >= 0
    return False


class PositionType(str, Enum):
    """Type of position."""

    EQUITY = "Equity"
    EQUITY_OPTION = "Equity Option"
    FUTURE = "Future"
    FUTURE_OPTION = "Future Option"
    CRYPTOCURRENCY = "Cryptocurrency"


class Greeks(BaseModel):
    """Option Greeks."""

    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None


class Position(BaseModel):
    """A single position in an account."""

    symbol: str
    underlying_symbol: str
    quantity: int
    quantity_direction: str = "Long"  # "Long" or "Short"
    position_type: PositionType

    # Option-specific fields
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # "C" or "P"
    expiration_date: Optional[date] = None
    days_to_expiration: Optional[int] = None

    # Pricing
    average_open_price: float = 0.0
    close_price: float = 0.0
    mark_price: Optional[float] = None
    underlying_price: Optional[float] = None  # Current price of underlying stock

    # P/L
    cost_basis: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: Optional[float] = None
    realized_pnl: float = 0.0

    # Greeks (for options)
    greeks: Optional[Greeks] = None

    # Multiplier (100 for options, 1 for stocks)
    multiplier: int = 1

    @property
    def is_option(self) -> bool:
        return self.position_type in (PositionType.EQUITY_OPTION, PositionType.FUTURE_OPTION)

    @property
    def is_short(self) -> bool:
        return self.quantity_direction == "Short" or self.quantity < 0

    def apply_mark_price(self, mark: float) -> bool:
        """Set mark and recompute market value plus raw unrealized P/L fields.

        Shared by single-position and batch enrichment so mark updates cannot
        leave market_value / unrealized_pnl stale relative to cost basis.

        Returns True when the mark was applied. Invalid marks (missing, NaN,
        infinity, negative) leave the model unchanged and return False.
        Zero is a valid mark and recomputes accounting.
        """
        if not is_valid_mark_price(mark):
            return False
        price = float(mark)
        self.mark_price = price
        self.market_value = price * abs(self.quantity) * self.multiplier
        if self.is_short:
            self.unrealized_pnl = self.cost_basis - self.market_value
        else:
            self.unrealized_pnl = self.market_value - self.cost_basis
        if self.cost_basis != 0:
            self.unrealized_pnl_percent = (self.unrealized_pnl / abs(self.cost_basis)) * 100
        else:
            self.unrealized_pnl_percent = None
        return True

    @property
    def display_quantity(self) -> str:
        """Display quantity with direction."""
        sign = "-" if self.is_short else "+"
        return f"{sign}{abs(self.quantity)}"

    @property
    def intrinsic_value(self) -> Optional[float]:
        """Calculate intrinsic value of option."""
        if not self.is_option or not self.strike_price or not self.underlying_price:
            return None

        # Calculate intrinsic value
        if self.option_type == "C":
            # Call: intrinsic = max(0, underlying - strike)
            intrinsic = max(0, self.underlying_price - self.strike_price)
        elif self.option_type == "P":
            # Put: intrinsic = max(0, strike - underlying)
            intrinsic = max(0, self.strike_price - self.underlying_price)
        else:
            return None

        return intrinsic

    @property
    def extrinsic_value(self) -> Optional[float]:
        """Calculate extrinsic (time) value of option."""
        if (
            not self.is_option
            or not self.mark_price
            or not self.strike_price
            or not self.underlying_price
        ):
            return None

        intrinsic = self.intrinsic_value or 0

        # Extrinsic = option price - intrinsic
        extrinsic = self.mark_price - intrinsic
        return max(0, extrinsic)  # Ensure non-negative


class Account(BaseModel):
    """A trading account."""

    account_number: str
    account_type: str = ""  # "Individual", "IRA", etc.
    nickname: Optional[str] = None

    # Balances
    net_liquidating_value: float = 0.0
    cash_balance: float = 0.0
    buying_power: float = 0.0
    maintenance_excess: Optional[float] = None

    # Day trading
    day_trading_buying_power: Optional[float] = None
    day_trade_count: int = 0

    # P/L
    pnl_today: float = 0.0
    pnl_today_percent: Optional[float] = None

    # Positions
    positions: list[Position] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        if self.nickname:
            return f"{self.nickname} ({self.account_number[-4:]})"
        return f"Account ...{self.account_number[-4:]}"

    @property
    def total_positions(self) -> int:
        return len(self.positions)

    @property
    def option_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_option]

    @property
    def equity_positions(self) -> list[Position]:
        return [p for p in self.positions if not p.is_option]


class Signal(str, Enum):
    """Trading signal type."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    ROLL = "roll"
    CLOSE = "close"


class Recommendation(BaseModel):
    """A recommendation for a position."""

    position: Position
    signal: Signal
    reason: str
    urgency: int = Field(ge=1, le=5, default=3)  # 1=low, 5=critical
    suggested_action: Optional[str] = None
    risk_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
