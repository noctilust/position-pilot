"""Strategy detection and grouping for options positions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import date

from ..models import Position, PositionType


class StrategyType(Enum):
    """Types of options strategies."""
    # Single leg
    LONG_CALL = "Long Call"
    SHORT_CALL = "Short Call"
    LONG_PUT = "Long Put"
    SHORT_PUT = "Short Put"

    # Vertical spreads
    BULL_CALL_SPREAD = "Bull Call Spread"
    BEAR_CALL_SPREAD = "Bear Call Spread"
    BULL_PUT_SPREAD = "Bull Put Spread"
    BEAR_PUT_SPREAD = "Bear Put Spread"

    # Iron strategies
    IRON_CONDOR = "Iron Condor"
    IRON_BUTTERFLY = "Iron Butterfly"

    # Straddles/Strangles
    LONG_STRADDLE = "Long Straddle"
    SHORT_STRADDLE = "Short Straddle"
    LONG_STRANGLE = "Long Strangle"
    SHORT_STRANGLE = "Short Strangle"

    # Calendar/Diagonal
    CALENDAR_SPREAD = "Calendar Spread"
    DIAGONAL_SPREAD = "Diagonal Spread"

    # Stock + Options
    COVERED_CALL = "Covered Call"
    PROTECTIVE_PUT = "Protective Put"
    COLLAR = "Collar"

    # Complex
    JADE_LIZARD = "Jade Lizard"
    BIG_LIZARD = "Big Lizard"
    RATIO_SPREAD = "Ratio Spread"
    BUTTERFLY = "Butterfly"

    # Equity
    LONG_STOCK = "Long Stock"
    SHORT_STOCK = "Short Stock"

    # Unknown/custom
    CUSTOM = "Custom"


@dataclass
class StrategyGroup:
    """A group of positions forming a strategy."""
    strategy_type: StrategyType
    underlying: str
    positions: list[Position] = field(default_factory=list)
    expiration: Optional[date] = None

    @property
    def total_quantity(self) -> int:
        """Number of strategy units (based on smallest leg)."""
        if not self.positions:
            return 0
        return min(abs(p.quantity) for p in self.positions)

    @property
    def is_credit(self) -> bool:
        """Whether the strategy was opened for a credit."""
        return self.total_cost_basis < 0

    @property
    def total_cost_basis(self) -> float:
        """Total cost basis of all legs."""
        total = 0.0
        for pos in self.positions:
            if pos.is_short:
                total -= pos.cost_basis
            else:
                total += pos.cost_basis
        return total

    @property
    def total_market_value(self) -> float:
        """Total market value of all legs."""
        total = 0.0
        for pos in self.positions:
            if pos.is_short:
                total -= pos.market_value
            else:
                total += pos.market_value
        return total

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P/L."""
        return sum(p.unrealized_pnl for p in self.positions)

    @property
    def unrealized_pnl_percent(self) -> Optional[float]:
        """P/L as percentage of max risk or premium received."""
        if abs(self.total_cost_basis) < 0.01:
            return None
        return (self.unrealized_pnl / abs(self.total_cost_basis)) * 100

    @property
    def total_delta(self) -> float:
        """Net delta of the strategy."""
        total = 0.0
        for pos in self.positions:
            if pos.greeks and pos.greeks.delta:
                multiplier = -1 if pos.is_short else 1
                total += pos.greeks.delta * pos.quantity * multiplier
        return total

    @property
    def total_theta(self) -> float:
        """Net theta of the strategy (daily)."""
        total = 0.0
        for pos in self.positions:
            if pos.greeks and pos.greeks.theta:
                total += pos.greeks.theta * abs(pos.quantity) * pos.multiplier
        return total

    @property
    def days_to_expiration(self) -> Optional[int]:
        """DTE of the nearest expiration."""
        dtes = [p.days_to_expiration for p in self.positions if p.days_to_expiration is not None]
        return min(dtes) if dtes else None

    @property
    def display_name(self) -> str:
        """Human-readable strategy name."""
        exp_str = ""
        if self.expiration:
            exp_str = f" {self.expiration.strftime('%m/%d')}"
        elif self.days_to_expiration is not None:
            exp_str = f" ({self.days_to_expiration}d)"

        return f"{self.underlying}{exp_str} {self.strategy_type.value}"

    @property
    def strikes_display(self) -> str:
        """Display strikes involved in the strategy."""
        strikes = sorted(set(
            p.strike_price for p in self.positions
            if p.strike_price is not None
        ))
        if not strikes:
            return ""
        if len(strikes) == 1:
            return f"${strikes[0]:.0f}"
        return "/".join(f"${s:.0f}" for s in strikes)


class StrategyDetector:
    """Detects and groups options strategies from positions."""

    def __init__(self):
        self.strategies: list[StrategyGroup] = []
        self._used_positions: set[str] = set()  # Track by symbol

    def detect_strategies(self, positions: list[Position]) -> list[StrategyGroup]:
        """
        Analyze positions and group them into strategies.

        Returns list of StrategyGroup objects.
        """
        self.strategies = []
        self._used_positions = set()

        # Separate by underlying
        by_underlying: dict[str, list[Position]] = {}
        for pos in positions:
            underlying = pos.underlying_symbol
            if underlying not in by_underlying:
                by_underlying[underlying] = []
            by_underlying[underlying].append(pos)

        # Process each underlying
        for underlying, pos_list in by_underlying.items():
            self._detect_for_underlying(underlying, pos_list)

        # Sort strategies by underlying, then by expiration
        self.strategies.sort(key=lambda s: (
            s.underlying,
            s.expiration or date.max,
            s.strategy_type.value
        ))

        return self.strategies

    def _detect_for_underlying(self, underlying: str, positions: list[Position]) -> None:
        """Detect strategies for a single underlying."""
        # Separate options and stock
        options = [p for p in positions if p.is_option]
        stocks = [p for p in positions if p.position_type == PositionType.EQUITY]

        # Group options by expiration
        by_expiration: dict[date, list[Position]] = {}
        for opt in options:
            exp = opt.expiration_date or date.max
            if exp not in by_expiration:
                by_expiration[exp] = []
            by_expiration[exp].append(opt)

        # Detect multi-leg strategies first (most specific)
        for exp, opts in by_expiration.items():
            self._detect_iron_condor(underlying, opts, exp)
            self._detect_iron_butterfly(underlying, opts, exp)
            self._detect_butterfly(underlying, opts, exp)
            self._detect_straddle(underlying, opts, exp)
            self._detect_strangle(underlying, opts, exp)
            self._detect_vertical_spreads(underlying, opts, exp)
            self._detect_jade_lizard(underlying, opts, exp)

        # Detect calendar/diagonal spreads (cross-expiration)
        self._detect_calendar_diagonal(underlying, options)

        # Detect stock + option strategies
        if stocks:
            self._detect_covered_strategies(underlying, stocks, options)

        # Handle remaining single-leg options
        for opt in options:
            if opt.symbol not in self._used_positions:
                self._add_single_option(opt)

        # Handle remaining stock positions
        for stock in stocks:
            if stock.symbol not in self._used_positions:
                self._add_single_stock(stock)

    def _mark_used(self, positions: list[Position]) -> None:
        """Mark positions as used in a strategy."""
        for pos in positions:
            self._used_positions.add(pos.symbol)

    def _is_available(self, pos: Position) -> bool:
        """Check if position is available (not used in another strategy)."""
        return pos.symbol not in self._used_positions

    def _detect_iron_condor(self, underlying: str, options: list[Position], exp: date) -> None:
        """Detect iron condor: short strangle + long strangle wings."""
        available = [o for o in options if self._is_available(o)]

        calls = sorted([o for o in available if o.option_type == "C"],
                       key=lambda x: x.strike_price or 0)
        puts = sorted([o for o in available if o.option_type == "P"],
                      key=lambda x: x.strike_price or 0)

        # Need at least 2 calls and 2 puts
        if len(calls) < 2 or len(puts) < 2:
            return

        # Look for pattern: long put < short put < short call < long call
        for i, short_put in enumerate(puts):
            if not short_put.is_short:
                continue

            # Find long put below short put
            long_puts = [p for p in puts[:i] if not p.is_short and
                        p.strike_price and short_put.strike_price and
                        p.strike_price < short_put.strike_price and
                        abs(p.quantity) == abs(short_put.quantity)]

            for long_put in long_puts:
                # Find short call above short put
                for j, short_call in enumerate(calls):
                    if not short_call.is_short:
                        continue
                    if short_call.strike_price and short_put.strike_price:
                        if short_call.strike_price <= short_put.strike_price:
                            continue
                    if abs(short_call.quantity) != abs(short_put.quantity):
                        continue

                    # Find long call above short call
                    long_calls = [c for c in calls[j+1:] if not c.is_short and
                                 c.strike_price and short_call.strike_price and
                                 c.strike_price > short_call.strike_price and
                                 abs(c.quantity) == abs(short_call.quantity)]

                    for long_call in long_calls:
                        # Found an iron condor!
                        legs = [long_put, short_put, short_call, long_call]
                        if all(self._is_available(leg) for leg in legs):
                            group = StrategyGroup(
                                strategy_type=StrategyType.IRON_CONDOR,
                                underlying=underlying,
                                positions=legs,
                                expiration=exp if exp != date.max else None,
                            )
                            self.strategies.append(group)
                            self._mark_used(legs)
                            return  # One IC per expiration

    def _detect_iron_butterfly(self, underlying: str, options: list[Position], exp: date) -> None:
        """Detect iron butterfly: short straddle + long strangle wings."""
        available = [o for o in options if self._is_available(o)]

        calls = [o for o in available if o.option_type == "C"]
        puts = [o for o in available if o.option_type == "P"]

        # Look for short call and short put at same strike (short straddle)
        for short_call in calls:
            if not short_call.is_short:
                continue

            for short_put in puts:
                if not short_put.is_short:
                    continue
                if short_call.strike_price != short_put.strike_price:
                    continue
                if abs(short_call.quantity) != abs(short_put.quantity):
                    continue

                center_strike = short_call.strike_price
                qty = abs(short_call.quantity)

                # Find long put below and long call above
                long_put = next((p for p in puts if not p.is_short and
                               p.strike_price and center_strike and
                               p.strike_price < center_strike and
                               abs(p.quantity) == qty), None)

                long_call = next((c for c in calls if not c.is_short and
                                c.strike_price and center_strike and
                                c.strike_price > center_strike and
                                abs(c.quantity) == qty), None)

                if long_put and long_call:
                    legs = [long_put, short_put, short_call, long_call]
                    if all(self._is_available(leg) for leg in legs):
                        group = StrategyGroup(
                            strategy_type=StrategyType.IRON_BUTTERFLY,
                            underlying=underlying,
                            positions=legs,
                            expiration=exp if exp != date.max else None,
                        )
                        self.strategies.append(group)
                        self._mark_used(legs)
                        return

    def _detect_butterfly(self, underlying: str, options: list[Position], exp: date) -> None:
        """Detect butterfly spread (all calls or all puts)."""
        available = [o for o in options if self._is_available(o)]

        for opt_type in ["C", "P"]:
            opts = sorted([o for o in available if o.option_type == opt_type],
                         key=lambda x: x.strike_price or 0)

            if len(opts) < 3:
                continue

            # Look for 1 long / 2 short / 1 long pattern at equal intervals
            for i in range(len(opts) - 2):
                low = opts[i]
                for j in range(i + 1, len(opts) - 1):
                    mid = opts[j]
                    for k in range(j + 1, len(opts)):
                        high = opts[k]

                        if not (low.strike_price and mid.strike_price and high.strike_price):
                            continue

                        # Check equal spacing
                        low_to_mid = mid.strike_price - low.strike_price
                        mid_to_high = high.strike_price - mid.strike_price
                        if abs(low_to_mid - mid_to_high) > 0.01:
                            continue

                        # Check quantities: wings should be long, middle short (or vice versa)
                        if (not low.is_short and mid.is_short and not high.is_short):
                            if abs(mid.quantity) == 2 * abs(low.quantity) == 2 * abs(high.quantity):
                                legs = [low, mid, high]
                                if all(self._is_available(leg) for leg in legs):
                                    group = StrategyGroup(
                                        strategy_type=StrategyType.BUTTERFLY,
                                        underlying=underlying,
                                        positions=legs,
                                        expiration=exp if exp != date.max else None,
                                    )
                                    self.strategies.append(group)
                                    self._mark_used(legs)
                                    return

    def _detect_straddle(self, underlying: str, options: list[Position], exp: date) -> None:
        """Detect straddle: same strike call and put."""
        available = [o for o in options if self._is_available(o)]

        calls = [o for o in available if o.option_type == "C"]
        puts = [o for o in available if o.option_type == "P"]

        for call in calls:
            for put in puts:
                if call.strike_price != put.strike_price:
                    continue
                if abs(call.quantity) != abs(put.quantity):
                    continue
                if call.is_short != put.is_short:
                    continue

                legs = [put, call]
                if all(self._is_available(leg) for leg in legs):
                    strategy_type = StrategyType.SHORT_STRADDLE if call.is_short else StrategyType.LONG_STRADDLE
                    group = StrategyGroup(
                        strategy_type=strategy_type,
                        underlying=underlying,
                        positions=legs,
                        expiration=exp if exp != date.max else None,
                    )
                    self.strategies.append(group)
                    self._mark_used(legs)

    def _detect_strangle(self, underlying: str, options: list[Position], exp: date) -> None:
        """Detect strangle: OTM call and OTM put."""
        available = [o for o in options if self._is_available(o)]

        calls = [o for o in available if o.option_type == "C"]
        puts = [o for o in available if o.option_type == "P"]

        for call in calls:
            for put in puts:
                if not (call.strike_price and put.strike_price):
                    continue
                if call.strike_price <= put.strike_price:
                    continue  # Call must be higher strike
                if abs(call.quantity) != abs(put.quantity):
                    continue
                if call.is_short != put.is_short:
                    continue

                legs = [put, call]
                if all(self._is_available(leg) for leg in legs):
                    strategy_type = StrategyType.SHORT_STRANGLE if call.is_short else StrategyType.LONG_STRANGLE
                    group = StrategyGroup(
                        strategy_type=strategy_type,
                        underlying=underlying,
                        positions=legs,
                        expiration=exp if exp != date.max else None,
                    )
                    self.strategies.append(group)
                    self._mark_used(legs)

    def _detect_vertical_spreads(self, underlying: str, options: list[Position], exp: date) -> None:
        """Detect vertical spreads (bull/bear call/put spreads)."""
        available = [o for o in options if self._is_available(o)]

        for opt_type in ["C", "P"]:
            opts = sorted([o for o in available if o.option_type == opt_type],
                         key=lambda x: x.strike_price or 0)

            # Look for pairs with one long, one short
            for i in range(len(opts)):
                for j in range(i + 1, len(opts)):
                    low = opts[i]
                    high = opts[j]

                    if low.is_short == high.is_short:
                        continue  # Need one long, one short
                    if abs(low.quantity) != abs(high.quantity):
                        continue

                    legs = [low, high]
                    if not all(self._is_available(leg) for leg in legs):
                        continue

                    # Determine spread type
                    if opt_type == "C":
                        if low.is_short:
                            # Short low call, long high call = Bear Call Spread
                            strategy_type = StrategyType.BEAR_CALL_SPREAD
                        else:
                            # Long low call, short high call = Bull Call Spread
                            strategy_type = StrategyType.BULL_CALL_SPREAD
                    else:  # Put
                        if low.is_short:
                            # Short low put, long high put = Bull Put Spread
                            strategy_type = StrategyType.BULL_PUT_SPREAD
                        else:
                            # Long low put, short high put = Bear Put Spread
                            strategy_type = StrategyType.BEAR_PUT_SPREAD

                    group = StrategyGroup(
                        strategy_type=strategy_type,
                        underlying=underlying,
                        positions=legs,
                        expiration=exp if exp != date.max else None,
                    )
                    self.strategies.append(group)
                    self._mark_used(legs)

    def _detect_jade_lizard(self, underlying: str, options: list[Position], exp: date) -> None:
        """Detect jade lizard: short put + short call spread (bear call)."""
        available = [o for o in options if self._is_available(o)]

        calls = sorted([o for o in available if o.option_type == "C"],
                      key=lambda x: x.strike_price or 0)
        puts = [o for o in available if o.option_type == "P"]

        # Need a short put
        short_puts = [p for p in puts if p.is_short]

        for short_put in short_puts:
            # Look for bear call spread above
            short_calls = [c for c in calls if c.is_short and
                          c.strike_price and short_put.strike_price and
                          c.strike_price > short_put.strike_price]

            for short_call in short_calls:
                # Find long call above short call
                long_calls = [c for c in calls if not c.is_short and
                             c.strike_price and short_call.strike_price and
                             c.strike_price > short_call.strike_price and
                             abs(c.quantity) == abs(short_call.quantity)]

                for long_call in long_calls:
                    if abs(short_put.quantity) != abs(short_call.quantity):
                        continue

                    legs = [short_put, short_call, long_call]
                    if all(self._is_available(leg) for leg in legs):
                        group = StrategyGroup(
                            strategy_type=StrategyType.JADE_LIZARD,
                            underlying=underlying,
                            positions=legs,
                            expiration=exp if exp != date.max else None,
                        )
                        self.strategies.append(group)
                        self._mark_used(legs)
                        return

    def _detect_calendar_diagonal(self, underlying: str, options: list[Position]) -> None:
        """Detect calendar and diagonal spreads (different expirations)."""
        available = [o for o in options if self._is_available(o)]

        for opt_type in ["C", "P"]:
            opts = [o for o in available if o.option_type == opt_type]

            # Group by strike
            by_strike: dict[float, list[Position]] = {}
            for opt in opts:
                if opt.strike_price:
                    if opt.strike_price not in by_strike:
                        by_strike[opt.strike_price] = []
                    by_strike[opt.strike_price].append(opt)

            # Calendar: same strike, different expiration
            for strike, strike_opts in by_strike.items():
                if len(strike_opts) < 2:
                    continue

                # Sort by expiration
                strike_opts.sort(key=lambda x: x.expiration_date or date.max)

                for i in range(len(strike_opts)):
                    for j in range(i + 1, len(strike_opts)):
                        near = strike_opts[i]
                        far = strike_opts[j]

                        if near.expiration_date == far.expiration_date:
                            continue
                        if near.is_short == far.is_short:
                            continue
                        if abs(near.quantity) != abs(far.quantity):
                            continue

                        legs = [near, far]
                        if all(self._is_available(leg) for leg in legs):
                            group = StrategyGroup(
                                strategy_type=StrategyType.CALENDAR_SPREAD,
                                underlying=underlying,
                                positions=legs,
                                expiration=near.expiration_date,
                            )
                            self.strategies.append(group)
                            self._mark_used(legs)

            # Diagonal: different strike, different expiration
            for i, opt1 in enumerate(opts):
                for opt2 in opts[i+1:]:
                    if not self._is_available(opt1) or not self._is_available(opt2):
                        continue
                    if opt1.strike_price == opt2.strike_price:
                        continue
                    if opt1.expiration_date == opt2.expiration_date:
                        continue
                    if opt1.is_short == opt2.is_short:
                        continue
                    if abs(opt1.quantity) != abs(opt2.quantity):
                        continue

                    # Determine near/far
                    if (opt1.expiration_date or date.max) < (opt2.expiration_date or date.max):
                        near, far = opt1, opt2
                    else:
                        near, far = opt2, opt1

                    legs = [near, far]
                    group = StrategyGroup(
                        strategy_type=StrategyType.DIAGONAL_SPREAD,
                        underlying=underlying,
                        positions=legs,
                        expiration=near.expiration_date,
                    )
                    self.strategies.append(group)
                    self._mark_used(legs)

    def _detect_covered_strategies(self, underlying: str, stocks: list[Position],
                                    options: list[Position]) -> None:
        """Detect covered call, protective put, and collar."""
        available_opts = [o for o in options if self._is_available(o)]

        for stock in stocks:
            if not self._is_available(stock):
                continue

            shares = stock.quantity  # Positive = long, negative = short

            if shares > 0:  # Long stock
                # Look for covered call (short call)
                short_calls = [o for o in available_opts if o.option_type == "C" and o.is_short]
                # Look for protective put (long put)
                long_puts = [o for o in available_opts if o.option_type == "P" and not o.is_short]

                # Check for collar first (has both)
                for call in short_calls:
                    for put in long_puts:
                        if (call.expiration_date == put.expiration_date and
                            abs(call.quantity) * 100 <= shares and
                            abs(put.quantity) == abs(call.quantity)):

                            legs = [stock, put, call]
                            if all(self._is_available(leg) for leg in legs):
                                group = StrategyGroup(
                                    strategy_type=StrategyType.COLLAR,
                                    underlying=underlying,
                                    positions=legs,
                                    expiration=call.expiration_date,
                                )
                                self.strategies.append(group)
                                self._mark_used(legs)
                                return

                # Covered call only
                for call in short_calls:
                    if abs(call.quantity) * 100 <= shares:
                        legs = [stock, call]
                        if all(self._is_available(leg) for leg in legs):
                            group = StrategyGroup(
                                strategy_type=StrategyType.COVERED_CALL,
                                underlying=underlying,
                                positions=legs,
                                expiration=call.expiration_date,
                            )
                            self.strategies.append(group)
                            self._mark_used(legs)
                            return

                # Protective put only
                for put in long_puts:
                    if abs(put.quantity) * 100 <= shares:
                        legs = [stock, put]
                        if all(self._is_available(leg) for leg in legs):
                            group = StrategyGroup(
                                strategy_type=StrategyType.PROTECTIVE_PUT,
                                underlying=underlying,
                                positions=legs,
                                expiration=put.expiration_date,
                            )
                            self.strategies.append(group)
                            self._mark_used(legs)
                            return

    def _add_single_option(self, pos: Position) -> None:
        """Add a single-leg option position."""
        if pos.option_type == "C":
            strategy_type = StrategyType.SHORT_CALL if pos.is_short else StrategyType.LONG_CALL
        else:
            strategy_type = StrategyType.SHORT_PUT if pos.is_short else StrategyType.LONG_PUT

        group = StrategyGroup(
            strategy_type=strategy_type,
            underlying=pos.underlying_symbol,
            positions=[pos],
            expiration=pos.expiration_date,
        )
        self.strategies.append(group)
        self._mark_used([pos])

    def _add_single_stock(self, pos: Position) -> None:
        """Add a single stock position."""
        strategy_type = StrategyType.SHORT_STOCK if pos.is_short else StrategyType.LONG_STOCK

        group = StrategyGroup(
            strategy_type=strategy_type,
            underlying=pos.underlying_symbol,
            positions=[pos],
        )
        self.strategies.append(group)
        self._mark_used([pos])


def detect_strategies(positions: list[Position]) -> list[StrategyGroup]:
    """Convenience function to detect strategies from positions."""
    detector = StrategyDetector()
    return detector.detect_strategies(positions)
