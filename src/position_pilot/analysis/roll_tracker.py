"""Roll detection and tracking from transaction history.

Builds independent option-leg lineages linked by exact OCC symbols
(old_symbol -> new_symbol). Multi-leg roll orders create multiple chains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ..models.position import Position
from ..models.roll import RollChain, RollEvent
from ..models.transaction import Transaction, TransactionType
from .strategies import StrategyType

logger = logging.getLogger(__name__)


def normalize_occ_symbol(symbol: str | None) -> str:
    """Normalize OCC symbols for exact lineage matching."""
    if not symbol:
        return ""
    return " ".join(str(symbol).split()).upper()


def extract_occ_root(symbol: str | None) -> str:
    """Extract the equity root from padded OCC or normalized/compact OCC symbols.

    Padded OCC: ``MU    260710P00800000`` (6-char root field, may include spaces).
    Normalized: ``MU 260710P00800000`` (root before YYMMDD+C/P+strike).
    """
    raw = str(symbol or "")
    if not raw.strip():
        return ""

    # Standard 21-char-style padded OCC: root(6) + YYMMDD(6) + C/P + strike(8).
    if len(raw) >= 15 and raw[6:12].isdigit() and raw[12].upper() in {"C", "P"}:
        return raw[:6].strip().upper()

    normalized = normalize_occ_symbol(raw)
    if len(normalized) >= 15 and normalized[-8:].isdigit() and normalized[-9] in {"C", "P"}:
        date_part = normalized[-15:-9]
        if date_part.isdigit():
            root = normalized[:-15].strip()
            if root:
                return root.upper()

    # Compact fallback: alphabetic/punct root before first digit run of length >= 6.
    buf: list[str] = []
    for ch in normalized:
        if ch.isdigit() and len(buf) > 0:
            break
        if ch == " " and buf:
            # space between root and date in normalized form
            break
        if ch.isalnum() or ch in {".", "/", "^", "-"}:
            buf.append(ch)
    root = "".join(buf).strip().upper()
    return root or normalized[:6].strip().upper()


def signed_cash(tx: Transaction) -> float:
    """Signed cash flow for a fill.

    STO/STC are credits (+); BTO/BTC are debits (−).
    Uses absolute transaction amount (total cash, not unit price).
    """
    amount = float(tx.amount or 0.0)
    action = (tx.action or "").lower()
    if "sell" in action:
        return abs(amount)
    if "buy" in action:
        return -abs(amount)
    return amount


def fill_quantity(tx: Transaction) -> float:
    """Absolute contract quantity for a fill (default 1)."""
    qty = abs(float(tx.quantity or 0.0))
    return qty if qty > 0 else 1.0


def position_side_from_open_action(action: str | None) -> str:
    """Side of a lineage from the *opening* action: short (STO) or long (BTO)."""
    text = (action or "").lower()
    if "sell" in text and "open" in text:
        return "short"
    if "buy" in text and "open" in text:
        return "long"
    if "sell" in text:
        return "short"
    if "buy" in text:
        return "long"
    return "unknown"


def position_side_from_close_action(action: str | None) -> str:
    """Side implied by a closing action: BTC closes short, STC closes long."""
    text = (action or "").lower()
    if "buy" in text and "close" in text:
        return "short"
    if "sell" in text and "close" in text:
        return "long"
    if "buy" in text:
        return "short"
    if "sell" in text:
        return "long"
    return "unknown"


@dataclass
class _OpenLot:
    symbol: str
    cash: float
    quantity: float
    date: datetime
    transaction_id: str
    side: str  # short | long


@dataclass
class _ChainBuilder:
    underlying: str
    strategy_type: str
    account_number: str
    rolls: list[RollEvent] = field(default_factory=list)
    original_open_date: datetime | None = None
    original_open_credit: float | None = None
    history_complete: bool = True
    closed: bool = False
    # Signed open cash of the current terminal symbol (predecessor on next close).
    terminal_open_cash: float | None = None
    terminal_symbol: str = ""
    terminal_quantity: float = 1.0
    side: str = "short"
    root_symbol: str = ""


class RollTracker:
    """Detects and tracks roll operations from transaction history."""

    def __init__(self, time_window_hours: int = 48):
        """Initialize the roll tracker.

        Args:
            time_window_hours: Reserved for compatibility; lineage uses order-id
                matching rather than a close/open wall-clock window.
        """
        self.time_window = timedelta(hours=time_window_hours)

    def detect_rolls(
        self,
        transactions: list[Transaction],
        positions: list[Position],
        account_number: str,
        *,
        include_closed: bool = True,
    ) -> list[RollChain]:
        """Detect roll operations from transaction history."""
        if not transactions:
            return []

        try:
            order_fills = [
                t for t in transactions if t.transaction_type == TransactionType.ORDER_FILL
            ]
        except Exception as e:
            logger.error(f"Error filtering transactions: {e}")
            return []

        if not order_fills:
            return []

        logger.info("Analyzing %s order-fill transactions for rolls", len(order_fills))

        try:
            underlying_groups = self._group_by_underlying(order_fills)
        except Exception as e:
            logger.error(f"Error grouping transactions by underlying: {e}")
            return []

        roll_chains: list[RollChain] = []
        current_symbols = {normalize_occ_symbol(p.symbol) for p in positions if p.symbol}

        for underlying, fills in underlying_groups.items():
            try:
                chains = self._detect_rolls_for_underlying(
                    fills,
                    account_number,
                    current_symbols=current_symbols,
                    include_closed=include_closed,
                )
                roll_chains.extend(chains)
            except Exception as e:
                logger.error(f"Error detecting rolls for {underlying}: {e}")
                continue

        logger.info(
            "Detected %s roll chains from %s transactions",
            len(roll_chains),
            len(order_fills),
        )
        return roll_chains

    def _group_by_underlying(self, transactions: list[Transaction]) -> dict[str, list[Transaction]]:
        groups: dict[str, list[Transaction]] = {}
        for tx in transactions:
            if tx.symbol:
                underlying = self._extract_underlying(tx.symbol)
                groups.setdefault(underlying, []).append(tx)
        return groups

    def _extract_underlying(self, symbol: str) -> str:
        return extract_occ_root(symbol)

    def _detect_rolls_for_underlying(
        self,
        fills: list[Transaction],
        account_number: str,
        *,
        current_symbols: set[str],
        include_closed: bool,
    ) -> list[RollChain]:
        fills = sorted(fills, key=lambda t: t.transaction_date)
        order_groups = self._group_by_order_id(fills)

        all_orders: list[dict] = []
        for order_id, transactions in order_groups.items():
            closing_legs, opening_legs = self._classify_order_legs(transactions)
            date = min(t.transaction_date for t in transactions)
            if closing_legs and opening_legs:
                order_type = "roll"
            elif opening_legs:
                order_type = "open"
            elif closing_legs:
                order_type = "close"
            else:
                continue
            all_orders.append(
                {
                    "order_id": order_id,
                    "type": order_type,
                    "date": date,
                    "closing": closing_legs,
                    "opening": opening_legs,
                }
            )
        all_orders.sort(key=lambda o: o["date"])

        open_lots: dict[str, list[_OpenLot]] = {}
        active: dict[str, _ChainBuilder] = {}
        finished: list[_ChainBuilder] = []

        for order in all_orders:
            if order["type"] == "open":
                for tx in order["opening"]:
                    self._record_open_lot(open_lots, tx)
            elif order["type"] == "roll":
                pairs, unmatched_closes, unmatched_opens = self._pair_roll_legs(
                    order["closing"], order["opening"]
                )
                for close_tx, open_tx in pairs:
                    chain = self._extend_or_start_chain(
                        active=active,
                        open_lots=open_lots,
                        close_tx=close_tx,
                        open_tx=open_tx,
                        account_number=account_number,
                    )
                    if chain is not None:
                        new_symbol = normalize_occ_symbol(open_tx.symbol)
                        active[new_symbol] = chain
                # Unmatched opens remain ordinary opening lots for later exact-symbol rolls.
                for open_tx in unmatched_opens:
                    self._record_open_lot(open_lots, open_tx)
                # Unmatched closes terminate an active lineage (or consume free lots).
                for close_tx in unmatched_closes:
                    symbol = normalize_occ_symbol(close_tx.symbol)
                    qty = fill_quantity(close_tx)
                    chain = active.pop(symbol, None)
                    if chain is not None:
                        if abs(chain.terminal_quantity - qty) > 1e-6:
                            chain.history_complete = False
                        chain.closed = True
                        chain.terminal_symbol = symbol
                        finished.append(chain)
                    else:
                        self._consume_open_lots(open_lots, symbol, qty)
            elif order["type"] == "close":
                for close_tx in order["closing"]:
                    symbol = normalize_occ_symbol(close_tx.symbol)
                    qty = fill_quantity(close_tx)
                    self._consume_open_lots(open_lots, symbol, qty)
                    chain = active.pop(symbol, None)
                    if chain is not None:
                        if abs(chain.terminal_quantity - qty) > 1e-6:
                            chain.history_complete = False
                        chain.closed = True
                        chain.terminal_symbol = symbol
                        finished.append(chain)

        finished.extend(active.values())

        result: list[RollChain] = []
        for builder in finished:
            if not builder.rolls:
                continue
            if builder.closed and not include_closed:
                continue
            terminal = normalize_occ_symbol(builder.terminal_symbol or builder.rolls[-1].new_symbol)
            is_open = (not builder.closed) and terminal in current_symbols
            if not is_open and not include_closed:
                continue
            chain = RollChain(
                underlying=builder.underlying,
                strategy_type=builder.strategy_type,
                account_number=account_number,
                rolls=list(builder.rolls),
                original_open_date=builder.original_open_date,
                original_open_credit=builder.original_open_credit,
                history_complete=builder.history_complete,
                is_open=is_open,
                terminal_symbol=terminal or None,
                root_symbol=builder.root_symbol or None,
            )
            result.append(chain)
        return result

    def _record_open_lot(self, open_lots: dict[str, list[_OpenLot]], tx: Transaction) -> None:
        symbol = normalize_occ_symbol(tx.symbol)
        if not symbol:
            return
        open_lots.setdefault(symbol, []).append(
            _OpenLot(
                symbol=symbol,
                cash=signed_cash(tx),
                quantity=fill_quantity(tx),
                date=tx.transaction_date,
                transaction_id=tx.transaction_id or "",
                side=position_side_from_open_action(tx.action),
            )
        )

    def _consume_open_lots(
        self,
        open_lots: dict[str, list[_OpenLot]],
        symbol: str,
        quantity: float,
    ) -> tuple[float | None, float, bool, datetime | None]:
        """FIFO-consume opening lots for ``quantity`` contracts.

        Returns ``(cash, consumed_qty, complete, first_open_date)``. Cash is signed
        open cash for the consumed quantity. Incomplete when insufficient quantity.
        """
        need = abs(quantity) if quantity else 1.0
        lots = open_lots.get(symbol)
        if not lots:
            return None, 0.0, False, None

        cash = 0.0
        consumed = 0.0
        remaining = need
        first_open_date: datetime | None = lots[0].date if lots else None
        while remaining > 1e-9 and lots:
            lot = lots[0]
            if lot.quantity <= remaining + 1e-9:
                cash += lot.cash
                consumed += lot.quantity
                remaining -= lot.quantity
                lots.pop(0)
            else:
                fraction = remaining / lot.quantity
                cash += lot.cash * fraction
                lot.cash *= 1.0 - fraction
                lot.quantity -= remaining
                consumed += remaining
                remaining = 0.0

        if not lots:
            open_lots.pop(symbol, None)
        complete = remaining <= 1e-9
        return (cash if complete else None), consumed, complete, first_open_date

    def _extend_or_start_chain(
        self,
        *,
        active: dict[str, _ChainBuilder],
        open_lots: dict[str, list[_OpenLot]],
        close_tx: Transaction,
        open_tx: Transaction,
        account_number: str,
    ) -> _ChainBuilder | None:
        close_symbol = normalize_occ_symbol(close_tx.symbol)
        open_symbol = normalize_occ_symbol(open_tx.symbol)
        if not close_symbol or not open_symbol or close_symbol == open_symbol:
            return None

        close_cash = signed_cash(close_tx)
        open_cash = signed_cash(open_tx)
        close_qty = fill_quantity(close_tx)
        open_qty = fill_quantity(open_tx)
        side = position_side_from_open_action(open_tx.action)
        if side == "unknown":
            side = position_side_from_close_action(close_tx.action)

        chain = active.pop(close_symbol, None)
        predecessor_cash: float | None
        if chain is not None:
            # Active lineage: require quantity match against terminal size.
            if abs(chain.terminal_quantity - close_qty) > 1e-6:
                chain.history_complete = False
                predecessor_cash = None
            else:
                predecessor_cash = chain.terminal_open_cash
        else:
            cash, _consumed, complete, open_date = self._consume_open_lots(
                open_lots, close_symbol, close_qty
            )
            if complete and cash is not None:
                predecessor_cash = cash
                chain = _ChainBuilder(
                    underlying=self._extract_underlying(close_symbol),
                    strategy_type=self._infer_strategy_type(close_tx, open_tx),
                    account_number=account_number,
                    original_open_date=open_date or close_tx.transaction_date,
                    original_open_credit=cash,
                    history_complete=True,
                    terminal_open_cash=cash,
                    terminal_symbol=close_symbol,
                    terminal_quantity=close_qty,
                    side=side,
                    root_symbol=close_symbol,
                )
            else:
                predecessor_cash = None
                chain = _ChainBuilder(
                    underlying=self._extract_underlying(close_symbol),
                    strategy_type=self._infer_strategy_type(close_tx, open_tx),
                    account_number=account_number,
                    original_open_date=None,
                    original_open_credit=None,
                    history_complete=False,
                    terminal_open_cash=None,
                    terminal_symbol=close_symbol,
                    terminal_quantity=close_qty,
                    side=side,
                    root_symbol=close_symbol,
                )

        if predecessor_cash is None:
            roll_pnl = 0.0
            chain.history_complete = False
        else:
            roll_pnl = predecessor_cash + close_cash

        # Quantity mismatch between close and successor open → incomplete.
        if abs(close_qty - open_qty) > 1e-6:
            chain.history_complete = False

        premium_effect = close_cash + open_cash
        roll = self._create_roll_event(
            close_tx,
            open_tx,
            account_number,
            roll_pnl=roll_pnl,
            premium_effect=premium_effect,
        )
        if not chain.rolls:
            chain.root_symbol = close_symbol
            # Do not invent original_open_date from the close fill when history is incomplete.
        chain.rolls.append(roll)
        chain.rolls.sort(key=lambda r: r.timestamp)
        chain.terminal_open_cash = open_cash
        chain.terminal_symbol = open_symbol
        chain.terminal_quantity = open_qty
        chain.side = side
        chain.strategy_type = self._infer_strategy_type(close_tx, open_tx)
        return chain

    def _pair_roll_legs(
        self,
        close_legs: list[Transaction],
        open_legs: list[Transaction],
    ) -> tuple[
        list[tuple[Transaction, Transaction]],
        list[Transaction],
        list[Transaction],
    ]:
        """Pair closes with opens by option type and position side (BTC↔STO, STC↔BTO).

        Returns ``(pairs, unmatched_closes, unmatched_opens)``.
        """
        close_buckets: dict[tuple[str, str], list[Transaction]] = {}
        open_buckets: dict[tuple[str, str], list[Transaction]] = {}

        for tx in close_legs:
            opt = self._get_option_type(tx.symbol or "") or "?"
            side = position_side_from_close_action(tx.action)
            close_buckets.setdefault((opt, side), []).append(tx)
        for tx in open_legs:
            opt = self._get_option_type(tx.symbol or "") or "?"
            side = position_side_from_open_action(tx.action)
            open_buckets.setdefault((opt, side), []).append(tx)

        pairs: list[tuple[Transaction, Transaction]] = []
        paired_close_ids: set[int] = set()
        paired_open_ids: set[int] = set()
        for key, closes in close_buckets.items():
            opens = open_buckets.get(key, [])
            closes_sorted = sorted(
                closes,
                key=lambda t: (
                    self._extract_strike(t.symbol or "") or 0.0,
                    normalize_occ_symbol(t.symbol),
                    t.transaction_id or "",
                ),
            )
            opens_sorted = sorted(
                opens,
                key=lambda t: (
                    self._extract_strike(t.symbol or "") or 0.0,
                    normalize_occ_symbol(t.symbol),
                    t.transaction_id or "",
                ),
            )
            for close_tx, open_tx in zip(closes_sorted, opens_sorted, strict=False):
                pairs.append((close_tx, open_tx))
                paired_close_ids.add(id(close_tx))
                paired_open_ids.add(id(open_tx))

        unmatched_closes = [tx for tx in close_legs if id(tx) not in paired_close_ids]
        unmatched_opens = [tx for tx in open_legs if id(tx) not in paired_open_ids]
        return pairs, unmatched_closes, unmatched_opens

    def _group_by_order_id(self, transactions: list[Transaction]) -> dict[str, list[Transaction]]:
        groups: dict[str, list[Transaction]] = {}
        for tx in transactions:
            order_id = tx.order_id or f"no-order-{tx.transaction_id}"
            groups.setdefault(order_id, []).append(tx)
        return groups

    def _classify_order_legs(
        self, transactions: list[Transaction]
    ) -> tuple[list[Transaction], list[Transaction]]:
        closing_legs: list[Transaction] = []
        opening_legs: list[Transaction] = []
        for tx in transactions:
            action = (tx.action or "").lower()
            if "close" in action:
                closing_legs.append(tx)
            elif "open" in action:
                opening_legs.append(tx)
            else:
                if tx.quantity and tx.quantity < 0:
                    closing_legs.append(tx)
                elif tx.quantity and tx.quantity > 0:
                    opening_legs.append(tx)
        return closing_legs, opening_legs

    def _get_option_type(self, symbol: str) -> Optional[str]:
        normalized = normalize_occ_symbol(symbol)
        if len(normalized) >= 15:
            opt_char = normalized[-9].upper()
            if opt_char in ("C", "P"):
                return opt_char
        for i, ch in enumerate(normalized):
            if ch in ("C", "P") and i + 1 < len(normalized) and normalized[i + 1 :].isdigit():
                if len(normalized) - i <= 12:
                    return ch
        if len(normalized) >= 9:
            opt_char = normalized[-9].upper()
            if opt_char in ("C", "P"):
                return opt_char
        return None

    def _infer_strategy_type(self, close_tx: Transaction, open_tx: Transaction) -> str:
        """Infer strategy type from the successor *opening* action (not the close)."""
        close_opt = self._get_option_type(close_tx.symbol or "")
        side = position_side_from_open_action(open_tx.action)
        if side == "unknown":
            side = position_side_from_close_action(close_tx.action)
        is_short = side != "long"
        if close_opt == "P":
            return StrategyType.SHORT_PUT.value if is_short else StrategyType.LONG_PUT.value
        if close_opt == "C":
            return StrategyType.SHORT_CALL.value if is_short else StrategyType.LONG_CALL.value
        return StrategyType.CUSTOM.value

    def _create_roll_event(
        self,
        close_tx: Transaction,
        open_tx: Transaction,
        account_number: str,
        *,
        roll_pnl: float,
        premium_effect: float,
    ) -> RollEvent:
        try:
            close_strike = self._extract_strike(close_tx.symbol) if close_tx.symbol else 0.0
            open_strike = self._extract_strike(open_tx.symbol) if open_tx.symbol else 0.0
            close_exp = self._extract_expiration(close_tx.symbol) if close_tx.symbol else None
            open_exp = self._extract_expiration(open_tx.symbol) if open_tx.symbol else None

            # DTE at roll execution (not relative to "now", which zeroes historical rolls).
            roll_ts = (
                open_tx.transaction_date
                if open_tx.transaction_date
                else close_tx.transaction_date
                if close_tx.transaction_date
                else datetime.now()
            )
            roll_day = roll_ts.date() if hasattr(roll_ts, "date") else roll_ts
            close_dte = 0
            open_dte = 0
            if close_exp:
                try:
                    close_dte = (close_exp.date() - roll_day).days
                except Exception:
                    close_dte = 0
            if open_exp:
                try:
                    open_dte = (open_exp.date() - roll_day).days
                except Exception:
                    open_dte = 0

            roll_id = f"roll_{close_tx.transaction_id}_{open_tx.transaction_id}"
            if not close_tx.transaction_id or not open_tx.transaction_id:
                roll_id = f"roll_{datetime.now().timestamp()}"

            underlying = self._extract_underlying(close_tx.symbol) if close_tx.symbol else "UNKNOWN"
            strategy_type = self._infer_strategy_type(close_tx, open_tx)
            close_qty = fill_quantity(close_tx)
            open_qty = fill_quantity(open_tx)
            close_commission = close_tx.commission if close_tx.commission is not None else 0.0
            open_commission = open_tx.commission if open_tx.commission is not None else 0.0

            return RollEvent(
                roll_id=roll_id,
                timestamp=open_tx.transaction_date if open_tx.transaction_date else datetime.now(),
                underlying=underlying,
                strategy_type=strategy_type,
                account_number=account_number,
                old_symbol=normalize_occ_symbol(close_tx.symbol) or "UNKNOWN",
                old_strike=close_strike,
                old_expiration=close_exp.date() if close_exp else datetime.now().date(),
                old_dte=close_dte,
                old_delta=None,
                old_quantity=close_qty,
                new_symbol=normalize_occ_symbol(open_tx.symbol) or "UNKNOWN",
                new_strike=open_strike,
                new_expiration=open_exp.date() if open_exp else datetime.now().date(),
                new_dte=open_dte,
                new_delta=None,
                new_quantity=open_qty,
                roll_pnl=roll_pnl,
                premium_effect=premium_effect,
                commission=close_commission + open_commission,
                reason=None,
                notes=None,
            )
        except Exception as e:
            logger.error(f"Error creating roll event: {e}")
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
        try:
            normalized = normalize_occ_symbol(symbol)
            if not normalized or len(normalized) < 15:
                digits = ""
                for ch in reversed(normalized):
                    if ch.isdigit():
                        digits = ch + digits
                    else:
                        break
                if len(digits) >= 5:
                    strike = int(digits) / 1000.0
                    if 0 < strike <= 100000:
                        return float(strike)
                return 0.0
            strike_str = normalized[-8:]
            if not strike_str.isdigit():
                return 0.0
            strike = int(strike_str) / 1000.0
            if strike <= 0 or strike > 100000:
                return 0.0
            return float(strike)
        except (ValueError, IndexError, AttributeError):
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error extracting strike from {symbol}: {e}")
            return 0.0

    def _extract_expiration(self, symbol: str) -> Optional[datetime]:
        try:
            normalized = normalize_occ_symbol(symbol)
            if not normalized or len(normalized) < 15:
                return None
            exp_str = normalized[-15:-9]
            if not exp_str.isdigit():
                return None
            exp_date = datetime.strptime(exp_str, "%y%m%d")
            if exp_date.year < 1900 or exp_date.year > 2100:
                return None
            return exp_date
        except (ValueError, IndexError, AttributeError):
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting expiration from {symbol}: {e}")
            return None
