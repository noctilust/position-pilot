from datetime import date, datetime

from position_pilot.models.roll import RollChain, RollEvent


def test_roll_chain_round_trip_preserves_credit_and_history() -> None:
    event = RollEvent(
        roll_id="roll-1",
        timestamp=datetime(2026, 7, 11, 12, 30),
        underlying="SPY",
        strategy_type="Short Put",
        account_number="5WT00000",
        old_symbol="SPY   260717P00500000",
        old_strike=500,
        old_expiration=date(2026, 7, 17),
        old_dte=6,
        new_symbol="SPY   260821P00495000",
        new_strike=495,
        new_expiration=date(2026, 8, 21),
        new_dte=41,
        roll_pnl=125,
        premium_effect=80,
        commission=2.40,
    )
    chain = RollChain(
        underlying="SPY",
        strategy_type="Short Put",
        account_number="5WT00000",
        original_open_credit=150,
        rolls=[event],
    )

    restored = RollChain.from_dict(chain.to_dict())

    assert restored.chain_total_credit == 230
    assert restored.net_pnl == 122.6
    assert restored.get_strike_history() == [500, 495]
    assert restored.rolls[0].option_type == "PUT"
