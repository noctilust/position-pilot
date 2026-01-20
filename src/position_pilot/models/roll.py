"""Data models for roll tracking."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


def parse_option_type(occ_symbol: str) -> str:
    """Extract option type (CALL/PUT) from OCC symbol format.

    OCC format: SYMBOL  YYMMDD[C/P]SSSSSSS
    The C/P character is 9 positions from the right.

    Args:
        occ_symbol: OCC option symbol

    Returns:
        "CALL" or "PUT" or "UNKNOWN"
    """
    if not occ_symbol or len(occ_symbol) < 10:
        return "UNKNOWN"
    # The option type indicator is the 9th character from the right
    try:
        indicator = occ_symbol[-9]
        if indicator == "C":
            return "CALL"
        elif indicator == "P":
            return "PUT"
    except (IndexError, TypeError):
        pass
    return "UNKNOWN"


@dataclass
class RollEvent:
    """A single roll operation."""

    # Identity
    roll_id: str
    timestamp: datetime
    underlying: str
    strategy_type: str
    account_number: str

    # Old position (closed)
    old_symbol: str
    old_strike: float
    old_expiration: date
    old_dte: int

    # New position (opened)
    new_symbol: str
    new_strike: float
    new_expiration: date
    new_dte: int

    # Optional fields (must come last)
    old_quantity: float = 1.0
    old_delta: Optional[float] = None
    new_quantity: float = 1.0
    new_delta: Optional[float] = None
    roll_pnl: float = 0.0  # P/L from closing old position
    premium_effect: float = 0.0  # Net debit/credit from roll
    commission: float = 0.0
    reason: Optional[str] = None  # Manually tagged or AI-inferred
    notes: Optional[str] = None

    @property
    def dte_change(self) -> int:
        """Days rolled forward (+) or backward (-)."""
        return self.new_dte - self.old_dte

    @property
    def strike_change(self) -> float:
        """Strike price change."""
        return self.new_strike - self.old_strike

    @property
    def option_type(self) -> str:
        """Option type (CALL/PUT) from symbol."""
        return parse_option_type(self.old_symbol)

    @property
    def option_indicator(self) -> str:
        """Single character option indicator (C/P) from symbol."""
        if not self.old_symbol or len(self.old_symbol) < 10:
            return "?"
        try:
            indicator = self.old_symbol[-9]
            if indicator in ("C", "P"):
                return indicator
        except (IndexError, TypeError):
            pass
        return "?"

    @property
    def pnl_per_contract(self) -> float:
        """P/L per contract (normalized for quantity)."""
        if self.old_quantity and self.old_quantity > 0:
            return self.roll_pnl / self.old_quantity
        return self.roll_pnl

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "roll_id": self.roll_id,
            "timestamp": self.timestamp.isoformat(),
            "underlying": self.underlying,
            "strategy_type": self.strategy_type,
            "account_number": self.account_number,
            "old_symbol": self.old_symbol,
            "old_strike": self.old_strike,
            "old_expiration": self.old_expiration.isoformat(),
            "old_dte": self.old_dte,
            "old_delta": self.old_delta,
            "old_quantity": self.old_quantity,
            "new_symbol": self.new_symbol,
            "new_strike": self.new_strike,
            "new_expiration": self.new_expiration.isoformat(),
            "new_dte": self.new_dte,
            "new_delta": self.new_delta,
            "new_quantity": self.new_quantity,
            "roll_pnl": self.roll_pnl,
            "premium_effect": self.premium_effect,
            "commission": self.commission,
            "reason": self.reason,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RollEvent":
        """Create from dictionary."""
        return cls(
            roll_id=data["roll_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            underlying=data["underlying"],
            strategy_type=data["strategy_type"],
            account_number=data["account_number"],
            old_symbol=data["old_symbol"],
            old_strike=data["old_strike"],
            old_expiration=date.fromisoformat(data["old_expiration"]),
            old_dte=data["old_dte"],
            old_delta=data.get("old_delta"),
            old_quantity=data.get("old_quantity", 1.0),
            new_symbol=data["new_symbol"],
            new_strike=data["new_strike"],
            new_expiration=date.fromisoformat(data["new_expiration"]),
            new_dte=data["new_dte"],
            new_delta=data.get("new_delta"),
            new_quantity=data.get("new_quantity", 1.0),
            roll_pnl=data.get("roll_pnl", 0.0),
            premium_effect=data.get("premium_effect", 0.0),
            commission=data.get("commission", 0.0),
            reason=data.get("reason"),
            notes=data.get("notes"),
        )


class RollChain(BaseModel):
    """Complete history of a rolled position."""

    underlying: str
    strategy_type: str
    account_number: str
    rolls: list[RollEvent] = Field(default_factory=list)
    original_open_date: Optional[datetime] = None

    @property
    def roll_count(self) -> int:
        """Total number of rolls."""
        return len(self.rolls)

    @property
    def total_roll_pnl(self) -> float:
        """Cumulative P/L from all rolls."""
        return sum(roll.roll_pnl for roll in self.rolls)

    @property
    def total_commission(self) -> float:
        """Total commission paid across all rolls."""
        return sum(roll.commission for roll in self.rolls)

    @property
    def net_pnl(self) -> float:
        """Net P/L after commissions."""
        return self.total_roll_pnl - self.total_commission

    @property
    def pl_open(self) -> float:
        """P/L from original position open through all rolls (without current position)."""
        return self.net_pnl

    def get_strike_history(self) -> list[float]:
        """List of strikes rolled through (old to new)."""
        return [roll.old_strike for roll in self.rolls] + (
            [self.rolls[-1].new_strike] if self.rolls else []
        )

    def get_dte_history(self) -> list[int]:
        """List of DTEs at each roll."""
        return [roll.old_dte for roll in self.rolls] + (
            [self.rolls[-1].new_dte] if self.rolls else []
        )

    def get_delta_history(self) -> list[Optional[float]]:
        """List of deltas at each roll."""
        return [roll.old_delta for roll in self.rolls] + (
            [self.rolls[-1].new_delta] if self.rolls else []
        )

    def add_roll(self, roll: RollEvent) -> None:
        """Add a roll event to the chain."""
        self.rolls.append(roll)
        # Sort by timestamp
        self.rolls.sort(key=lambda r: r.timestamp)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "underlying": self.underlying,
            "strategy_type": self.strategy_type,
            "account_number": self.account_number,
            "original_open_date": self.original_open_date.isoformat() if self.original_open_date else None,
            "rolls": [roll.to_dict() for roll in self.rolls],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RollChain":
        """Create from dictionary."""
        rolls = [RollEvent.from_dict(r) for r in data.get("rolls", [])]
        return cls(
            underlying=data["underlying"],
            strategy_type=data["strategy_type"],
            account_number=data["account_number"],
            rolls=rolls,
            original_open_date=(
                datetime.fromisoformat(data["original_open_date"])
                if data.get("original_open_date")
                else None
            ),
        )
