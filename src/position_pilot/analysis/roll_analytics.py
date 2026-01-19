"""Roll analytics and pattern analysis."""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from statistics import mean

from ..models.position import Position
from ..models.roll import RollChain, RollEvent

logger = logging.getLogger(__name__)


@dataclass
class RollPatterns:
    """Learned rolling patterns for a symbol/strategy."""

    # Timing patterns
    avg_dte_at_roll: float  # Average DTE when rolls occur
    typical_roll_days: list[int]  # Most common DTE targets
    min_dte_at_roll: int
    max_dte_at_roll: int

    # Strike selection
    avg_strike_adjustment: float  # How far strikes move on rolls
    typical_delta_targets: list[float]  # Target deltas
    min_strike_change: float
    max_strike_change: float

    # Economics
    avg_roll_pnl: float  # Average P/L per roll
    avg_premium_effect: float  # Average debit/credit per roll
    total_pnl: float  # Total P/L from all rolls
    win_rate: float  # % of rolls that were profitable

    # Frequency
    rolls_per_month: float
    avg_days_between_rolls: float

    # Success metrics
    best_dte_window: tuple[int, int]  # Most profitable DTE range
    best_strike_range: tuple[float, float]  # Most profitable strike range
    total_rolls: int

    # Duration tracking
    avg_roll_duration_days: float  # How long positions are typically held


def calculate_pl_open(chain: RollChain, current_position: Optional[Position] = None) -> float:
    """Calculate P/L Open from position inception through all rolls.

    P/L Open = (Current Position Unrealized P/L)
               + Sum of all realized P/L from roll closes
               - Sum of all premiums paid on rolls
               + Sum of all premiums collected on rolls

    Args:
        chain: Roll chain with roll history
        current_position: Current open position (if still open)

    Returns:
        Total P/L from original position open through all rolls
    """
    total = 0.0

    # Add realized P/L from each roll close
    for roll in chain.rolls:
        total += roll.roll_pnl

    # Add premium effects from rolls
    for roll in chain.rolls:
        total += roll.premium_effect

    # Add current unrealized P/L if position is still open
    if current_position:
        total += current_position.unrealized_pnl

    return total


def analyze_patterns(chains: list[RollChain]) -> RollPatterns:
    """Analyze historical rolls to find patterns.

    Args:
        chains: List of roll chains to analyze

    Returns:
        RollPatterns object with discovered patterns
    """
    if not chains:
        return RollPatterns(
            avg_dte_at_roll=0.0,
            typical_roll_days=[],
            min_dte_at_roll=0,
            max_dte_at_roll=0,
            avg_strike_adjustment=0.0,
            typical_delta_targets=[],
            min_strike_change=0.0,
            max_strike_change=0.0,
            avg_roll_pnl=0.0,
            avg_premium_effect=0.0,
            total_pnl=0.0,
            win_rate=0.0,
            rolls_per_month=0.0,
            avg_days_between_rolls=0.0,
            best_dte_window=(0, 0),
            best_strike_range=(0.0, 0.0),
            total_rolls=0,
            avg_roll_duration_days=0.0,
        )

    # Flatten all rolls from all chains
    all_rolls = []
    for chain in chains:
        all_rolls.extend(chain.rolls)

    if not all_rolls:
        return RollPatterns(
            avg_dte_at_roll=0.0,
            typical_roll_days=[],
            min_dte_at_roll=0,
            max_dte_at_roll=0,
            avg_strike_adjustment=0.0,
            typical_delta_targets=[],
            min_strike_change=0.0,
            max_strike_change=0.0,
            avg_roll_pnl=0.0,
            avg_premium_effect=0.0,
            total_pnl=0.0,
            win_rate=0.0,
            rolls_per_month=0.0,
            avg_days_between_rolls=0.0,
            best_dte_window=(0, 0),
            best_strike_range=(0.0, 0.0),
            total_rolls=0,
            avg_roll_duration_days=0.0,
        )

    total_rolls_count = len(all_rolls)

    # Timing patterns
    dte_at_roll = [roll.old_dte for roll in all_rolls]
    avg_dte_at_roll = mean(dte_at_roll) if dte_at_roll else 0.0

    # Most common DTE targets (new DTE after roll)
    new_dte_list = [roll.new_dte for roll in all_rolls]
    dte_counter = Counter(new_dte_list)
    typical_roll_days = [dte for dte, count in dte_counter.most_common(5)]

    # Strike selection patterns
    strike_changes = [roll.strike_change for roll in all_rolls]
    avg_strike_adjustment = mean(strike_changes) if strike_changes else 0.0

    # Economics
    roll_pnls = [roll.roll_pnl for roll in all_rolls]
    premium_effects = [roll.premium_effect for roll in all_rolls]

    avg_roll_pnl = mean(roll_pnls) if roll_pnls else 0.0
    avg_premium_effect = mean(premium_effects) if premium_effects else 0.0
    total_pnl = sum(roll_pnls)

    # Win rate
    profitable_rolls = sum(1 for pnl in roll_pnls if pnl > 0)
    win_rate = profitable_rolls / total_rolls_count if total_rolls_count > 0 else 0.0

    # Frequency analysis
    if total_rolls_count >= 2:
        # Sort rolls by timestamp
        sorted_rolls = sorted(all_rolls, key=lambda r: r.timestamp)

        # Calculate average days between rolls
        days_between = []
        for i in range(1, len(sorted_rolls)):
            days_diff = (sorted_rolls[i].timestamp - sorted_rolls[i-1].timestamp).days
            if 0 < days_diff < 365:  # Filter out unrealistic gaps
                days_between.append(days_diff)

        avg_days_between_rolls = mean(days_between) if days_between else 0.0

        # Calculate span in months for rolls per month
        if len(sorted_rolls) >= 2:
            date_span = (sorted_rolls[-1].timestamp - sorted_rolls[0].timestamp).days
            if date_span > 0:
                rolls_per_month = (total_rolls_count / date_span) * 30
            else:
                rolls_per_month = 0.0
        else:
            rolls_per_month = 0.0
    else:
        avg_days_between_rolls = 0.0
        rolls_per_month = 0.0

    # Find best DTE window (most profitable)
    best_dte_window = _find_best_dte_window(all_rolls)

    # Find best strike range
    best_strike_range = _find_best_strike_range(all_rolls)

    # Duration tracking (how long positions held before roll)
    durations = []
    for chain in chains:
        if len(chain.rolls) >= 1:
            # Time from original open to first roll
            if chain.original_open_date and chain.rolls[0].timestamp:
                duration = (chain.rolls[0].timestamp - chain.original_open_date).days
                if 0 < duration < 365:
                    durations.append(duration)

    avg_roll_duration_days = mean(durations) if durations else 0.0

    return RollPatterns(
        avg_dte_at_roll=avg_dte_at_roll,
        typical_roll_days=typical_roll_days,
        min_dte_at_roll=min(dte_at_roll) if dte_at_roll else 0,
        max_dte_at_roll=max(dte_at_roll) if dte_at_roll else 0,
        avg_strike_adjustment=avg_strike_adjustment,
        typical_delta_targets=[],  # Would need delta data from market
        min_strike_change=min(strike_changes) if strike_changes else 0.0,
        max_strike_change=max(strike_changes) if strike_changes else 0.0,
        avg_roll_pnl=avg_roll_pnl,
        avg_premium_effect=avg_premium_effect,
        total_pnl=total_pnl,
        win_rate=win_rate,
        rolls_per_month=rolls_per_month,
        avg_days_between_rolls=avg_days_between_rolls,
        best_dte_window=best_dte_window,
        best_strike_range=best_strike_range,
        total_rolls=total_rolls_count,
        avg_roll_duration_days=avg_roll_duration_days,
    )


def _find_best_dte_window(rolls: list[RollEvent], window_size: int = 7) -> tuple[int, int]:
    """Find the most profitable DTE window for rolling.

    Args:
        rolls: List of roll events
        window_size: DTE range to consider

    Returns:
        Tuple of (min_dte, max_dte) for best window
    """
    if len(rolls) < 3:
        return (0, 0)

    # Group rolls by old DTE and calculate average P/L
    dte_pnls = {}
    for roll in rolls:
        dte = roll.old_dte
        if dte not in dte_pnls:
            dte_pnls[dte] = []
        dte_pnls[dte].append(roll.roll_pnl)

    # Calculate average P/L per DTE
    dte_avg_pnl = {dte: mean(pnls) for dte, pnls in dte_pnls.items()}

    # Find DTE with highest average P/L
    best_dte = max(dte_avg_pnl, key=dte_avg_pnl.get) if dte_avg_pnl else 0

    # Return window around best DTE
    return (best_dte - window_size // 2, best_dte + window_size // 2)


def _find_best_strike_range(rolls: list[RollEvent], range_size: float = 5.0) -> tuple[float, float]:
    """Find the most profitable strike range for rolling.

    Args:
        rolls: List of roll events
        range_size: Strike price range

    Returns:
        Tuple of (min_strike, max_strike) for best range
    """
    if len(rolls) < 3:
        return (0.0, 0.0)

    # Group rolls by old strike and calculate average P/L
    strike_pnls = {}
    for roll in rolls:
        strike = roll.old_strike
        if strike not in strike_pnls:
            strike_pnls[strike] = []
        strike_pnls[strike].append(roll.roll_pnl)

    # Calculate average P/L per strike
    strike_avg_pnl = {strike: mean(pnls) for strike, pnls in strike_pnls.items()}

    # Find strike with highest average P/L
    best_strike = max(strike_avg_pnl, key=strike_avg_pnl.get) if strike_avg_pnl else 0.0

    # Return window around best strike
    return (best_strike - range_size / 2, best_strike + range_size / 2)


def format_roll_summary(chain: RollChain, current_pnl: Optional[float] = None) -> str:
    """Format a roll chain for display.

    Args:
        chain: Roll chain to format
        current_pnl: Optional current unrealized P/L

    Returns:
        Formatted string representation
    """
    lines = [
        f"{chain.underlying} {chain.strategy_type}",
        f"Rolls: {chain.roll_count}",
    ]

    if chain.roll_count > 0:
        lines.extend([
            f"Total Roll P/L: ${chain.total_roll_pnl:+,.2f}",
            f"Total Commission: ${chain.total_commission:+,.2f}",
            f"Net P/L: ${chain.net_pnl:+,.2f}",
        ])

    if current_pnl is not None:
        lines.append(f"P/L Open: ${current_pnl:+,.2f}")

    # Roll history
    if chain.rolls:
        lines.append("\nRoll History:")
        for i, roll in enumerate(chain.rolls, 1):
            lines.append(
                f"  {i}. {roll.timestamp.strftime('%Y-%m-%d')}: "
                f"${roll.old_strike} → ${roll.new_strike} "
                f"({roll.old_dte} → {roll.new_dte} DTE, "
                f"Δ{roll.dte_change:+d} days) "
                f"P/L: ${roll.roll_pnl:+,.2f}"
            )

    return "\n".join(lines)


def format_patterns_summary(patterns: RollPatterns) -> str:
    """Format roll patterns for display.

    Args:
        patterns: Roll patterns to format

    Returns:
        Formatted string representation
    """
    lines = [
        f"Total Rolls: {patterns.total_rolls}",
        f"Win Rate: {patterns.win_rate:.1%}",
        "",
        "Timing Patterns:",
        f"  Avg DTE at roll: {patterns.avg_dte_at_roll:.1f} days",
        f"  Typical roll targets: {patterns.typical_roll_days if patterns.typical_roll_days else 'N/A'}",
        f"  Best DTE window: {patterns.best_dte_window[0]}-{patterns.best_dte_window[1]} DTE",
        "",
        "Strike Selection:",
        f"  Avg strike change: ${patterns.avg_strike_adjustment:+.2f}",
        f"  Best strike range: ${patterns.best_strike_range[0]:.2f}-${patterns.best_strike_range[1]:.2f}",
        "",
        "Economics:",
        f"  Avg roll P/L: ${patterns.avg_roll_pnl:+,.2f}",
        f"  Total P/L: ${patterns.total_pnl:+,.2f}",
        "",
        "Frequency:",
        f"  {patterns.rolls_per_month:.2f} rolls/month",
        f"  Avg days between rolls: {patterns.avg_days_between_rolls:.1f}",
        f"  Avg position duration: {patterns.avg_roll_duration_days:.1f} days",
    ]

    return "\n".join(lines)
