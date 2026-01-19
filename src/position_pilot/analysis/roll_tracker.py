"""Roll detection and tracking from transaction history."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from ..models.position import Position
from ..models.roll import RollEvent, RollChain
from ..models.transaction import Transaction, TransactionType

logger = logging.getLogger(__name__)


class RollTracker:
    """Detects and tracks roll operations from transaction history."""

    def __init__(self, time_window_hours: int = 48):
        """Initialize the roll tracker.

        Args:
            time_window_hours: Maximum time between close and open to consider it a roll (default: 48 hours)
        """
        self.time_window = timedelta(hours=time_window_hours)

    def detect_rolls(
        self,
        transactions: list[Transaction],
        positions: list[Position],
        account_number: str
    ) -> list[RollChain]:
        """Detect roll operations from transaction history.

        Args:
            transactions: List of transactions to analyze
            positions: Current positions (for context)
            account_number: Account number

        Returns:
            List of RollChain objects representing detected rolls
        """
        # Filter to only order-fill transactions
        order_fills = [t for t in transactions if t.transaction_type == TransactionType.ORDER_FILL]

        # Group by underlying symbol
        underlying_groups = self._group_by_underlying(order_fills)

        # Detect rolls within each underlying group
        roll_chains = []

        for underlying, fills in underlying_groups.items():
            chains = self._detect_rolls_for_underlying(fills, positions, account_number)
            roll_chains.extend(chains)

        logger.info(f"Detected {len(roll_chains)} roll chains from {len(order_fills)} transactions")
        return roll_chains

    def _group_by_underlying(self, transactions: list[Transaction]) -> dict[str, list[Transaction]]:
        """Group transactions by underlying symbol."""
        groups = {}
        for tx in transactions:
            if tx.symbol:
                # Extract underlying from option symbol (OCC format)
                underlying = self._extract_underlying(tx.symbol)
                if underlying not in groups:
                    groups[underlying] = []
                groups[underlying].append(tx)
        return groups

    def _extract_underlying(self, symbol: str) -> str:
        """Extract underlying symbol from option symbol."""
        # OCC format: SYMBOL (6 chars) + ...
        if len(symbol) >= 6:
            return symbol[:6].strip()
        return symbol

    def _detect_rolls_for_underlying(
        self,
        fills: list[Transaction],
        current_positions: list[Position],
        account_number: str
    ) -> list[RollChain]:
        """Detect roll operations for a single underlying.

        A roll is detected when:
        1. Same underlying symbol
        2. Opposite position directions (closing then opening)
        3. Same strategy type (e.g., closing put spread, opening new put spread)
        4. Within time window
        5. Similar quantity (allowing for partial rolls)
        """
        # Sort by date
        fills.sort(key=lambda t: t.transaction_date)

        # Group potential roll pairs (close followed by open)
        roll_candidates = []

        for i, close_tx in enumerate(fills):
            # Look for opening transaction after this close
            for open_tx in fills[i + 1:]:
                # Check time window
                time_diff = open_tx.transaction_date - close_tx.transaction_date
                if time_diff > self.time_window:
                    break  # Too far apart, stop looking

                # Check if this looks like a roll pair
                if self._is_roll_pair(close_tx, open_tx):
                    roll_candidates.append((close_tx, open_tx))

        # Build roll chains from candidates
        chains = self._build_roll_chains(roll_candidates, account_number)

        return chains

    def _is_roll_pair(self, close_tx: Transaction, open_tx: Transaction) -> bool:
        """Check if two transactions form a roll pair.

        A roll pair has:
        - Same underlying
        - Opposite directions (one closing, one opening)
        - Similar option type (both calls or both puts)
        - Similar quantities
        """
        if not close_tx.symbol or not open_tx.symbol:
            return False

        # Extract underlying symbols
        close_underlying = self._extract_underlying(close_tx.symbol)
        open_underlying = self._extract_underlying(open_tx.symbol)

        if close_underlying != open_underlying:
            return False

        # Check option type
        close_opt = self._get_option_type(close_tx.symbol)
        open_opt = self._get_option_type(open_tx.symbol)

        if close_opt != open_opt:
            return False

        # Check quantities (allow for partial rolls, within 20%)
        if close_tx.quantity and open_tx.quantity:
            qty_ratio = abs(open_tx.quantity / close_tx.quantity)
            if not 0.8 <= qty_ratio <= 1.2:
                return False

        # Check opposite directions via transaction amounts
        # Closing: positive amount (credit received), Opening: negative amount (debit paid)
        # Or vice versa depending on position direction
        if close_tx.amount and open_tx.amount:
            # One should be opposite sign of the other
            return (close_tx.amount > 0 and open_tx.amount < 0) or (close_tx.amount < 0 and open_tx.amount > 0)

        return False

    def _get_option_type(self, symbol: str) -> Optional[str]:
        """Extract option type (C/P) from OCC symbol."""
        if len(symbol) >= 15:
            # OCC format: SYMBOL(6) + YYMMDD(6) + C/P(1) + STRIKE(8)
            opt_char = symbol[-9].upper()
            if opt_char in ("C", "P"):
                return opt_char
        return None

    def _build_roll_chains(
        self,
        roll_pairs: list[tuple[Transaction, Transaction]],
        account_number: str
    ) -> list[RollChain]:
        """Build roll chains from detected roll pairs."""
        # For now, create a simple chain per underlying
        # In the future, this could link rolls together based on continuation

        chains_by_underlying = {}

        for close_tx, open_tx in roll_pairs:
            underlying = self._extract_underlying(close_tx.symbol)

            if underlying not in chains_by_underlying:
                chains_by_underlying[underlying] = RollChain(
                    underlying=underlying,
                    strategy_type=self._infer_strategy_type(close_tx, open_tx),
                    account_number=account_number,
                    original_open_date=close_tx.transaction_date
                )

            # Create roll event
            roll = self._create_roll_event(close_tx, open_tx, account_number)
            chains_by_underlying[underlying].add_roll(roll)

        return list(chains_by_underlying.values())

    def _infer_strategy_type(self, close_tx: Transaction, open_tx: Transaction) -> str:
        """Infer the strategy type from the transactions."""
        # For now, simple inference based on option type
        # In the future, could analyze multi-leg orders
        close_opt = self._get_option_type(close_tx.symbol)
        open_opt = self._get_option_type(open_tx.symbol)

        if close_opt == "P":
            return "put_spread"
        elif close_opt == "C":
            return "call_spread"
        else:
            return "unknown"

    def _create_roll_event(
        self,
        close_tx: Transaction,
        open_tx: Transaction,
        account_number: str
    ) -> RollEvent:
        """Create a RollEvent from a pair of transactions."""
        # Extract details from symbols
        close_strike = self._extract_strike(close_tx.symbol)
        open_strike = self._extract_strike(open_tx.symbol)

        close_exp = self._extract_expiration(close_tx.symbol)
        open_exp = self._extract_expiration(open_tx.symbol)

        # Calculate DTE
        today = datetime.now().date()
        close_dte = (close_exp - today).days if close_exp else 0
        open_dte = (open_exp - today).days if open_exp else 0

        # Generate roll ID
        roll_id = f"roll_{close_tx.transaction_id}_{open_tx.transaction_id}"

        return RollEvent(
            roll_id=roll_id,
            timestamp=open_tx.transaction_date,
            underlying=self._extract_underlying(close_tx.symbol),
            strategy_type=self._infer_strategy_type(close_tx, open_tx),
            account_number=account_number,
            old_symbol=close_tx.symbol,
            old_strike=close_strike,
            old_expiration=close_exp,
            old_dte=close_dte,
            old_delta=None,  # Would need to fetch from market data
            old_quantity=abs(close_tx.quantity) if close_tx.quantity else 1.0,
            new_symbol=open_tx.symbol,
            new_strike=open_strike,
            new_expiration=open_exp,
            new_dte=open_dte,
            new_delta=None,  # Would need to fetch from market data
            new_quantity=abs(open_tx.quantity) if open_tx.quantity else 1.0,
            roll_pnl=close_tx.amount or 0.0,
            premium_effect=open_tx.amount or 0.0,
            commission=(close_tx.commission or 0.0) + (open_tx.commission or 0.0),
            reason=None,
            notes=None,
        )

    def _extract_strike(self, symbol: str) -> float:
        """Extract strike price from OCC option symbol."""
        try:
            if len(symbol) >= 15:
                # OCC format: SYMBOL(6) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
                strike_str = symbol[-8:]
                strike = int(strike_str) / 1000.0
                return float(strike)
        except (ValueError, IndexError):
            pass
        return 0.0

    def _extract_expiration(self, symbol: str) -> Optional[datetime]:
        """Extract expiration date from OCC option symbol."""
        try:
            if len(symbol) >= 15:
                exp_str = symbol[-15:-9]  # YYMMDD
                exp_date = datetime.strptime(exp_str, "%y%m%d")
                return exp_date
        except (ValueError, IndexError):
            pass
        return None
