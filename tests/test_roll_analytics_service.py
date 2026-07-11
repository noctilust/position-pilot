"""Roll pattern analytics and heatmap construction."""

import json
from datetime import UTC, date, datetime

from position_pilot.domain.rolls import RollChainSnapshot, RollEventSnapshot, RollService
from position_pilot.persistence.sqlite import PositionPilotDatabase


def _chain(account_id: str = "acct-1") -> RollChainSnapshot:
    return RollChainSnapshot(
        chain_id="chain-1",
        account_id=account_id,
        underlying="SPY",
        strategy_type="Short Strangle",
        original_open_date=datetime(2026, 1, 2, tzinfo=UTC),
        original_open_credit=2.5,
        rolls=[
            RollEventSnapshot(
                roll_id="r1",
                timestamp=datetime(2026, 2, 1, tzinfo=UTC),
                underlying="SPY",
                strategy_type="Short Strangle",
                old_symbol="SPY  260220C00500000",
                old_strike=500,
                old_expiration=date(2026, 2, 20),
                old_dte=14,
                new_symbol="SPY  260320C00505000",
                new_strike=505,
                new_expiration=date(2026, 3, 20),
                new_dte=35,
                roll_pnl=75,
                premium_effect=0.4,
            ),
            RollEventSnapshot(
                roll_id="r2",
                timestamp=datetime(2026, 3, 5, tzinfo=UTC),
                underlying="SPY",
                strategy_type="Short Strangle",
                old_symbol="SPY  260320C00505000",
                old_strike=505,
                old_expiration=date(2026, 3, 20),
                old_dte=12,
                new_symbol="SPY  260417C00510000",
                new_strike=510,
                new_expiration=date(2026, 4, 17),
                new_dte=40,
                roll_pnl=40,
                premium_effect=0.25,
            ),
        ],
    )


def test_patterns_and_heatmap_are_browser_safe(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    service = RollService(database)
    chain = _chain()
    database.save_roll_chain(chain.model_dump(mode="json"))

    patterns = service.patterns("acct-1", symbol="SPY")
    heatmap = service.heatmap("acct-1", symbol="SPY")
    chains = service.chains("acct-1", symbol="SPY")

    assert patterns.total_rolls == 2
    assert patterns.win_rate == 1.0
    assert chains[0].chain_total_credit == 3.15
    assert heatmap.underlying == "SPY"
    assert heatmap.cells
    assert all("5WT" not in cell.model_dump_json() for cell in heatmap.cells)


def test_live_legacy_roll_cache_retries_after_account_identity_exists(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    cache_path = tmp_path / "roll_history.json"
    chain = _chain(account_id="ignored")
    domain_payload = {
        "underlying": chain.underlying,
        "strategy_type": chain.strategy_type,
        "account_number": "5WT12345",
        "original_open_date": chain.original_open_date.isoformat(),
        "original_open_credit": chain.original_open_credit,
        "rolls": [
            {**roll.model_dump(mode="json"), "account_number": "5WT12345"} for roll in chain.rolls
        ],
    }
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "accounts": {
                    "5WT12345": {
                        "chains": {"SPY:Short Strangle": domain_payload},
                    }
                },
            }
        )
    )
    service = RollService(database, legacy_history_path=cache_path)

    assert service.sync_legacy_history() == 0
    identity = database.account_identity("5WT12345", "Individual")
    synced = service.chains(identity.account_id, symbol="SPY")

    assert len(synced) == 1
    assert synced[0].chain_total_credit == 3.15
