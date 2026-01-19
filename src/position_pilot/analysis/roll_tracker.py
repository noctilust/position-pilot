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
        # Handle empty transaction list
        if not transactions:
            logger.debug("No transactions provided for roll detection")
            return []

        # Filter to only order-fill transactions with error handling
        try:
            order_fills = [t for t in transactions if t.transaction_type == TransactionType.ORDER_FILL]
        except Exception as e:
            logger.error(f"Error filtering transactions: {e}")
            return []

        if not order_fills:
            logger.debug("No order-fill transactions found")
            return []

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
        try:
            # Validate both transactions have symbols
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

            # Both must be same type (both calls or both puts)
            if close_opt != open_opt or close_opt is None:
                return False

            # Check quantities (allow for partial rolls, within 20%)
            if close_tx.quantity and open_tx.quantity:
                # Avoid division by zero
                if close_tx.quantity == 0:
                    logger.debug(f"Close quantity is zero for {close_tx.symbol}")
                    return False

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
        except Exception as e:
            logger.debug(f"Error checking roll pair: {e}")
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
                    logger.debug(f"Error calculating close DTE: {e}")
                    close_dte = 0

            if open_exp:
                try:
                    open_dte = (open_exp.date() - today).days
                    if open_dte < 0:
                        open_dte = 0  # Expired
                except Exception as e:
                    logger.debug(f"Error calculating open DTE: {e}")
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
            roll_pnl = close_tx.amount if close_tx.amount is not None else 0.0
            premium_effect = open_tx.amount if open_tx.amount is not None else 0.0

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
                logger.debug(f"Invalid symbol format for strike extraction: {symbol}")
                return 0.0

            # OCC format: SYMBOL(6) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
            strike_str = symbol[-8:]

            # Validate strike string is numeric
            if not strike_str.isdigit():
                logger.debug(f"Strike value is not numeric: {strike_str}")
                return 0.0

            strike = int(strike_str) / 1000.0

            # Validate strike is reasonable (between 0 and 100,000)
            if strike <= 0 or strike > 100000:
                logger.debug(f"Strike value out of reasonable range: {strike}")
                return 0.0

            return float(strike)
        except (ValueError, IndexError, AttributeError) as e:
            logger.debug(f"Error extracting strike from {symbol}: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error extracting strike from {symbol}: {e}")
            return 0.0

    def _extract_expiration(self, symbol: str) -> Optional[datetime]:
        """Extract expiration date from OCC option symbol."""
        try:
            if not symbol or len(symbol) < 15:
                logger.debug(f"Invalid symbol format for expiration extraction: {symbol}")
                return None

            exp_str = symbol[-15:-9]  # YYMMDD

            # Validate expiration string format
            if not exp_str.isdigit():
                logger.debug(f"Expiration value is not numeric: {exp_str}")
                return None

            exp_date = datetime.strptime(exp_str, "%y%m%d")

            # Validate expiration is reasonable (between 1900 and 2100)
            if exp_date.year < 1900 or exp_date.year > 2100:
                logger.debug(f"Expiration year out of reasonable range: {exp_date.year}")
                return None

            return exp_date
        except (ValueError, IndexError, AttributeError) as e:
            logger.debug(f"Error extracting expiration from {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting expiration from {symbol}: {e}")
            return None
