from datetime import date

from position_pilot.analysis.strategies import StrategyType, detect_strategies
from position_pilot.models import Position, PositionType


def option(symbol: str, strike: float, option_type: str, direction: str) -> Position:
    return Position(
        symbol=symbol,
        underlying_symbol="SPY",
        quantity=1,
        quantity_direction=direction,
        position_type=PositionType.EQUITY_OPTION,
        strike_price=strike,
        option_type=option_type,
        expiration_date=date(2026, 8, 21),
    )


def test_four_leg_condor_remains_grouped_as_one_strategy() -> None:
    positions = [
        option("SPY260821P00500000", 500, "P", "Long"),
        option("SPY260821P00505000", 505, "P", "Short"),
        option("SPY260821C00530000", 530, "C", "Short"),
        option("SPY260821C00535000", 535, "C", "Long"),
    ]

    strategies = detect_strategies(positions)

    assert len(strategies) == 1
    assert strategies[0].strategy_type is StrategyType.IRON_CONDOR
    assert {position.symbol for position in strategies[0].positions} == {
        position.symbol for position in positions
    }
