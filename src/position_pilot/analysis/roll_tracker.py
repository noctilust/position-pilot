"""Roll detection and tracking from transaction history.

This module uses an order-based approach to detect roll operations for multi-leg
strategies like short strangles, iron condors, and vertical spreads.

Algorithm:
1. Group transactions by order-id to handle multi-leg orders
2. Use the action field to classify legs as opening or closing
3. Detect single orders with both closing and opening legs (roll orders)
4. Match closing legs with opening legs within the same order
5. Build roll chains from matched leg pairs
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from ..models.position import Position
from ..models.roll import RollEvent, RollChain
from ..models.transaction import Transaction, TransactionType
from .strategies import StrategyType

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
        # Handle empty transaction list
        if not transactions:
            return []

        # Filter to only order-fill transactions with error handling
        try:
            order_fills = [t for t in transactions if t.transaction_type == TransactionType.ORDER_FILL]
        except Exception as e:
            logger.error(f"Error filtering transactions: {e}")
            return []

        if not order_fills:
            return []

        logger.info(f"Analyzing {len(order_fills)} order-fill transactions for rolls")

        # Group by underlying symbol with error handling
        try:
            underlying_groups = self._group_by_underlying(order_fills)
        except Exception as e:
            logger.error(f"Error grouping transactions by underlying: {e}")
            return []

        # Detect rolls within each underlying group
        roll_chains = []

        for underlying, fills in underlying_groups.items():
            try:
                chains = self._detect_rolls_for_underlying(fills, positions, account_number)
                roll_chains.extend(chains)
            except Exception as e:
                logger.error(f"Error detecting rolls for {underlying}: {e}")
                continue

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
        """Detect roll operations for a single underlying using order-level matching.

        Algorithm:
        1. Group transactions by order-id
        2. For each order, classify legs as closing or opening based on action field
        3. Detect single orders with both closing and opening legs (roll orders)
        4. Match closing legs with opening legs within the same order
        5. Build roll chains from matched leg pairs
        """
        # Sort by date
        fills.sort(key=lambda t: t.transaction_date)

        # Group by order-id
        order_groups = self._group_by_order_id(fills)

        # Detect roll orders (single orders with both close and open legs)
        roll_orders = []

        for order_id, transactions in order_groups.items():
            # Classify legs as closing or opening based on action field
            closing_legs, opening_legs = self._classify_order_legs(transactions)

            # Only consider orders with both closing and opening legs as roll orders
            if closing_legs and opening_legs:
                roll_orders.append((order_id, closing_legs, opening_legs))

        # Build roll chains from roll orders
        return self._build_roll_chains_from_roll_orders(roll_orders, account_number)

    def _group_by_order_id(self, transactions: list[Transaction]) -> dict[str, list[Transaction]]:
        """Group transactions by order-id."""
        groups = {}
        for tx in transactions:
            order_id = tx.order_id or f"no-order-{tx.transaction_id}"
            if order_id not in groups:
                groups[order_id] = []
            groups[order_id].append(tx)
        return groups

    def _classify_order_legs(self, transactions: list[Transaction]) -> tuple[list[Transaction], list[Transaction]]:
        """Classify order legs as closing or opening based on action field.

        Uses the action field from Tastytrade API:
        - Closing legs: "Buy to Close", "Sell to Close"
        - Opening legs: "Buy to Open", "Sell to Open"

        Returns:
            Tuple of (closing_legs, opening_legs)
        """
        closing_legs = []
        opening_legs = []

        for tx in transactions:
            action = (tx.action or "").lower()

            # Classify based on action field
            if "close" in action:
                closing_legs.append(tx)
            elif "open" in action:
                opening_legs.append(tx)
            else:
                # Fallback: use quantity sign (positive = sell/open, negative = buy/close)
                # This is less reliable than the action field
                if tx.quantity and tx.quantity < 0:
                    closing_legs.append(tx)
                elif tx.quantity and tx.quantity > 0:
                    opening_legs.append(tx)

        return closing_legs, opening_legs

    def _build_roll_chains_from_roll_orders(
        self,
        roll_orders: list[tuple[str, list[Transaction], list[Transaction]]],
        account_number: str
    ) -> list[RollChain]:
        """Build roll chains from roll orders.

        Each roll order contains both closing and opening legs.
        We match closing legs with opening legs by option type (call/put).

        Args:
            roll_orders: List of (order_id, closing_legs, opening_legs) tuples
            account_number: Account number

        Returns:
            List of RollChain objects
        """
        chains_by_underlying = {}

        for order_id, close_legs, open_legs in roll_orders:
            # Get underlying from first closing leg
            underlying = self._extract_underlying(close_legs[0].symbol) if close_legs else None

            if not underlying:
                logger.warning(f"Could not extract underlying from roll order {order_id}")
                continue

            # Create new chain if needed
            if underlying not in chains_by_underlying:
                # Infer strategy type from legs
                opt_types = set()
                for tx in close_legs:
                    if tx.symbol:
                        opt_type = self._get_option_type(tx.symbol)
                        if opt_type:
                            opt_types.add(opt_type)

                strategy_type = self._infer_strategy_type_from_option_types(opt_types)

                chains_by_underlying[underlying] = RollChain(
                    underlying=underlying,
                    strategy_type=strategy_type,
                    account_number=account_number,
                    original_open_date=close_legs[0].transaction_date
                )

            # Create roll events by matching close and open legs
            # Group by option type for matching
            close_by_type = self._group_by_option_type(close_legs)
            open_by_type = self._group_by_option_type(open_legs)

            for opt_type in ("C", "P"):
                close_type_legs = close_by_type.get(opt_type, [])
                open_type_legs = open_by_type.get(opt_type, [])

                if not close_type_legs or not open_type_legs:
                    continue

                # Match close legs with open legs (should be 1-to-1 or close)
                min_legs = min(len(close_type_legs), len(open_type_legs))
                for i in range(min_legs):
                    close_tx = close_type_legs[i]
                    open_tx = open_type_legs[i]
                    roll = self._create_roll_event(close_tx, open_tx, account_number)
                    chains_by_underlying[underlying].add_roll(roll)

        return list(chains_by_underlying.values())

    def _group_by_option_type(self, transactions: list[Transaction]) -> dict[str, list[Transaction]]:
        """Group transactions by option type (call/put)."""
        groups: dict[str, list[Transaction]] = {"C": [], "P": []}

        for tx in transactions:
            if not tx.symbol:
                continue
            opt_type = self._get_option_type(tx.symbol)
            if opt_type in groups:
                groups[opt_type].append(tx)

        return groups

    def _get_option_type(self, symbol: str) -> Optional[str]:
        """Extract option type (C/P) from OCC symbol."""
        if len(symbol) >= 15:
            # OCC format: SYMBOL(6) + YYMMDD(6) + C/P(1) + STRIKE(8)
            opt_char = symbol[-9].upper()
            if opt_char in ("C", "P"):
                return opt_char
        return None

    def _infer_strategy_type_from_option_types(self, opt_types: set[str]) -> str:
        """Infer strategy type from option types present.

        Args:
            opt_types: Set of option types ("C" and/or "P")

        Returns:
            Strategy type string
        """
        # Multi-leg strategies with both calls and puts
        if opt_types == {"C", "P"}:
            # Has both calls and puts - default to short strangle
            # Could be: strangle, straddle, iron condor, etc.
            return StrategyType.SHORT_STRANGLE.value
        elif "P" in opt_types:
            return StrategyType.BULL_PUT_SPREAD.value
        elif "C" in opt_types:
            return StrategyType.BEAR_CALL_SPREAD.value
        else:
            return StrategyType.CUSTOM.value

    def _infer_strategy_type(self, close_tx: Transaction, open_tx: Transaction) -> str:
        """Infer the strategy type from the transactions.

        Uses StrategyType enum values for consistency with the dashboard.
        For put spreads, defaults to Bull Put Spread (most common rolling strategy).
        For call spreads, defaults to Bear Call Spread (most common rolling strategy).

        DEPRECATED: Use _infer_strategy_type_from_option_types instead.
        """
        close_opt = self._get_option_type(close_tx.symbol)

        if close_opt == "P":
            return StrategyType.BULL_PUT_SPREAD.value
        elif close_opt == "C":
            return StrategyType.BEAR_CALL_SPREAD.value
        else:
            return StrategyType.CUSTOM.value

    def _create_roll_event(
        self,
        close_tx: Transaction,
        open_tx: Transaction,
        account_number: str
    ) -> RollEvent:
        """Create a RollEvent from a pair of transactions."""
        try:
            # Extract details from symbols with error handling
            close_strike = self._extract_strike(close_tx.symbol) if close_tx.symbol else 0.0
            open_strike = self._extract_strike(open_tx.symbol) if open_tx.symbol else 0.0

            # Validate strikes were extracted successfully
            if close_strike <= 0 or open_strike <= 0:
                logger.warning(f"Invalid strikes for roll event: close={close_strike}, open={open_strike}")

            close_exp = self._extract_expiration(close_tx.symbol) if close_tx.symbol else None
            open_exp = self._extract_expiration(open_tx.symbol) if open_tx.symbol else None

            # Calculate DTE with validation
            today = datetime.now().date()
            close_dte = 0
            open_dte = 0

            if close_exp:
                try:
                    close_dte = (close_exp.date() - today).days
                    if close_dte < 0:
                        close_dte = 0  # Expired
                except Exception as e:
                    close_dte = 0

            if open_exp:
                try:
                    open_dte = (open_exp.date() - today).days
                    if open_dte < 0:
                        open_dte = 0  # Expired
                except Exception as e:
                    open_dte = 0

            # Generate roll ID with validation
            roll_id = f"roll_{close_tx.transaction_id}_{open_tx.transaction_id}"

            # Validate transaction IDs exist
            if not close_tx.transaction_id or not open_tx.transaction_id:
                logger.warning("Missing transaction ID for roll event")
                roll_id = f"roll_{datetime.now().timestamp()}"

            # Extract underlying with fallback
            underlying = self._extract_underlying(close_tx.symbol) if close_tx.symbol else "UNKNOWN"
            strategy_type = self._infer_strategy_type(close_tx, open_tx)

            # Safely extract quantities with defaults
            close_qty = abs(close_tx.quantity) if close_tx.quantity and close_tx.quantity != 0 else 1.0
            open_qty = abs(open_tx.quantity) if open_tx.quantity and open_tx.quantity != 0 else 1.0

            # Safely extract amounts with defaults
            # P/L from closing old position (positive = profit)
            roll_pnl = close_tx.amount if close_tx.amount is not None else 0.0

            # Net premium effect: opening premium - closing cost
            # For credit spreads: opening > closing (positive net credit)
            # For debit spreads: opening < closing (negative net debit)
            premium_effect = (open_tx.amount if open_tx.amount is not None else 0.0) - (close_tx.amount if close_tx.amount is not None else 0.0)

            # Safely extract commissions with defaults
            close_commission = close_tx.commission if close_tx.commission is not None else 0.0
            open_commission = open_tx.commission if open_tx.commission is not None else 0.0

            return RollEvent(
                roll_id=roll_id,
                timestamp=open_tx.transaction_date if open_tx.transaction_date else datetime.now(),
                underlying=underlying,
                strategy_type=strategy_type,
                account_number=account_number,
                old_symbol=close_tx.symbol or "UNKNOWN",
                old_strike=close_strike,
                old_expiration=close_exp.date() if close_exp else datetime.now().date(),
                old_dte=close_dte,
                old_delta=None,  # Would need to fetch from market data
                old_quantity=close_qty,
                new_symbol=open_tx.symbol or "UNKNOWN",
                new_strike=open_strike,
                new_expiration=open_exp.date() if open_exp else datetime.now().date(),
                new_dte=open_dte,
                new_delta=None,  # Would need to fetch from market data
                new_quantity=open_qty,
                roll_pnl=roll_pnl,
                premium_effect=premium_effect,
                commission=close_commission + open_commission,
                reason=None,
                notes=None,
            )
        except Exception as e:
            logger.error(f"Error creating roll event: {e}")
            # Return minimal valid roll event
            return RollEvent(
                roll_id=f"roll_error_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                underlying="ERROR",
                strategy_type="unknown",
                account_number=account_number,
                old_symbol="ERROR",
                old_strike=0.0,
                old_expiration=datetime.now().date(),
                old_dte=0,
                new_symbol="ERROR",
                new_strike=0.0,
                new_expiration=datetime.now().date(),
                new_dte=0,
            )

    def _extract_strike(self, symbol: str) -> float:
        """Extract strike price from OCC option symbol."""
        try:
            if not symbol or len(symbol) < 15:
                return 0.0

            # OCC format: SYMBOL(6) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
            strike_str = symbol[-8:]

            # Validate strike string is numeric
            if not strike_str.isdigit():
                return 0.0

            strike = int(strike_str) / 1000.0

            # Validate strike is reasonable (between 0 and 100,000)
            if strike <= 0 or strike > 100000:
                return 0.0

            return float(strike)
        except (ValueError, IndexError, AttributeError) as e:
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error extracting strike from {symbol}: {e}")
            return 0.0

    def _extract_expiration(self, symbol: str) -> Optional[datetime]:
        """Extract expiration date from OCC option symbol."""
        try:
            if not symbol or len(symbol) < 15:
                return None

            exp_str = symbol[-15:-9]  # YYMMDD

            # Validate expiration string format
            if not exp_str.isdigit():
                return None

            exp_date = datetime.strptime(exp_str, "%y%m%d")

            # Validate expiration is reasonable (between 1900 and 2100)
            if exp_date.year < 1900 or exp_date.year > 2100:
                return None

            return exp_date
        except (ValueError, IndexError, AttributeError) as e:
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting expiration from {symbol}: {e}")
            return None
