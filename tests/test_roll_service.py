from datetime import date, datetime

from position_pilot.domain.rolls import RollService
from position_pilot.models.roll import RollChain, RollEvent
from position_pilot.persistence.sqlite import PositionPilotDatabase


def roll_chain() -> RollChain:
    return RollChain(
        underlying="SPY",
        strategy_type="Short Put",
        account_number="broker-internal",
        original_open_credit=150,
        rolls=[
            RollEvent(
                roll_id="roll-1",
                timestamp=datetime(2026, 7, 10, 14, 30),
                underlying="SPY",
                strategy_type="Short Put",
                account_number="broker-internal",
                old_symbol="SPY   260717P00500000",
                old_strike=500,
                old_expiration=date(2026, 7, 17),
                old_dte=7,
                new_symbol="SPY   260821P00495000",
                new_strike=495,
                new_expiration=date(2026, 8, 21),
                new_dte=42,
                premium_effect=80,
            )
        ],
    )


def test_roll_chains_round_trip_through_public_account_scope(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT12345", "Individual")
    service = RollService(database)

    service.save_chain(identity.account_id, roll_chain())
    restored = service.chains(identity.account_id, symbol="SPY")

    assert len(restored) == 1
    assert restored[0].chain_total_credit == 230
    assert restored[0].rolls[0].roll_id == "roll-1"


def test_distinct_roll_chains_for_same_strategy_do_not_overwrite(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT12345", "Individual")
    first = roll_chain()
    second = roll_chain()
    second.rolls[0].roll_id = "roll-2"
    service = RollService(database)

    service.save_chain(identity.account_id, first)
    service.save_chain(identity.account_id, second)

    assert {chain.rolls[0].roll_id for chain in service.chains(identity.account_id)} == {
        "roll-1",
        "roll-2",
    }


def test_legacy_roll_cache_migrates_after_account_identity_exists(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    legacy_chain = roll_chain().to_dict()
    (cache / "roll_history.json").write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "accounts": {
                    "5WT12345": {
                        "chains": {"SPY:Short Put": legacy_chain},
                        "last_updated": "2026-07-10T14:30:00",
                    }
                },
            }
        )
    )
    database = PositionPilotDatabase(
        tmp_path / "position-pilot.sqlite3",
        legacy_cache_directory=cache,
    )
    identity = database.account_identity("5WT12345", "Individual")

    migrated = RollService(database).migrate_legacy_cache()

    assert migrated == 1
    assert len(RollService(database).chains(identity.account_id)) == 1
