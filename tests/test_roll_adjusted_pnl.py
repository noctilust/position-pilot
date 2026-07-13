"""Tastytrade-compatible roll lineage accounting and P/L Open adjustments.

Live MU topology (July 2026 evidence, total contract cash, pre-commission):
- Put: prior ``MU    260710P00800000`` STO +485, BTC -386 →
  current ``MU    260717P00730000`` STO +693 → realized +99, lifetime 792
- Call: prior ``MU    260731C01750000`` STO +762, BTC -401 →
  current ``MU    260731C01400000`` STO +1601 → realized +361, lifetime 1962
- Unrolled strangle put ``MU    260731P00800000`` STO 2238, raw P/L -212
- Unrelated ``MU    260717P00900000`` STO 1670
- Strangle 800P/1400C open P/L = 945 after call carry
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from position_pilot.analysis.roll_tracker import (
    RollTracker,
    extract_occ_root,
    normalize_occ_symbol,
)
from position_pilot.domain.portfolio import PortfolioService
from position_pilot.domain.rolls import SOURCE_BROKER, SOURCE_LEGACY, RollService
from position_pilot.domain.snapshots import (
    FreshnessState,
    PositionHorizon,
    PositionSnapshot,
    QuantityDirection,
    StrategySnapshot,
)
from position_pilot.models import Account, Position, PositionType
from position_pilot.models.roll import RollChain, RollEvent
from position_pilot.models.transaction import Transaction, TransactionType
from position_pilot.persistence.sqlite import PositionPilotDatabase

# Broker-padded OCC (transaction fixtures) and normalized position forms.
MU_800P_OLD = "MU    260710P00800000"
MU_730P = "MU    260717P00730000"
MU_1750C_OLD = "MU    260731C01750000"
MU_1400C = "MU    260731C01400000"
MU_800P = "MU    260731P00800000"
MU_900P = "MU    260717P00900000"

# Normalized forms used after tracker processing / position matching.
MU_800P_OLD_N = normalize_occ_symbol(MU_800P_OLD)
MU_730P_N = normalize_occ_symbol(MU_730P)
MU_1750C_OLD_N = normalize_occ_symbol(MU_1750C_OLD)
MU_1400C_N = normalize_occ_symbol(MU_1400C)
MU_800P_N = normalize_occ_symbol(MU_800P)
MU_900P_N = normalize_occ_symbol(MU_900P)


def _tx(
    *,
    tx_id: str,
    order_id: str,
    symbol: str,
    action: str,
    amount: float,
    when: datetime,
    quantity: float | None = None,
) -> Transaction:
    if quantity is None:
        quantity = -1.0 if "Close" in action else 1.0
    return Transaction(
        id=tx_id,
        transaction_type=TransactionType.ORDER_FILL,
        transaction_date=when,
        description=f"{action} {symbol}",
        amount=abs(amount),
        symbol=symbol,
        quantity=quantity,
        price=abs(amount) / 100.0,
        commission=0.0,
        order_id=order_id,
        account_number="5WT00001",
        action=action,
    )


def mu_transactions() -> list[Transaction]:
    """Broker fills for the live MU multi-lineage topology (padded OCC)."""
    # Chronology from live evidence: June 26 open, July 1 roll, July 2/6 later opens.
    t_open = datetime(2026, 6, 26, 15, 0, 0)
    t_roll = datetime(2026, 7, 1, 15, 30, 0)
    t_800 = datetime(2026, 7, 2, 16, 0, 0)
    t_900 = datetime(2026, 7, 6, 14, 0, 0)

    return [
        _tx(
            tx_id="1",
            order_id="open-strangle",
            symbol=MU_800P_OLD,
            action="Sell to Open",
            amount=485,
            when=t_open,
        ),
        _tx(
            tx_id="2",
            order_id="open-strangle",
            symbol=MU_1750C_OLD,
            action="Sell to Open",
            amount=762,
            when=t_open,
        ),
        # Multi-leg roll → independent put (800→730) and call (1750→1400) lineages.
        _tx(
            tx_id="3",
            order_id="roll-strangle",
            symbol=MU_800P_OLD,
            action="Buy to Close",
            amount=386,
            when=t_roll,
        ),
        _tx(
            tx_id="4",
            order_id="roll-strangle",
            symbol=MU_1750C_OLD,
            action="Buy to Close",
            amount=401,
            when=t_roll,
        ),
        _tx(
            tx_id="5",
            order_id="roll-strangle",
            symbol=MU_730P,
            action="Sell to Open",
            amount=693,
            when=t_roll,
        ),
        _tx(
            tx_id="6",
            order_id="roll-strangle",
            symbol=MU_1400C,
            action="Sell to Open",
            amount=1601,
            when=t_roll,
        ),
        # Current unrolled strangle put (later).
        _tx(
            tx_id="7",
            order_id="open-800p",
            symbol=MU_800P,
            action="Sell to Open",
            amount=2238,
            when=t_800,
        ),
        # Unrelated 900 put.
        _tx(
            tx_id="8",
            order_id="open-900p",
            symbol=MU_900P,
            action="Sell to Open",
            amount=1670,
            when=t_900,
        ),
    ]


def mu_positions() -> list[Position]:
    """Current open MU legs (broker-padded OCC forms)."""
    return [
        Position(
            symbol=MU_730P,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            strike_price=730,
            option_type="P",
            expiration_date=date(2026, 7, 17),
            average_open_price=6.93,
            cost_basis=693,
            market_value=243,
            unrealized_pnl=450,
            multiplier=100,
        ),
        Position(
            symbol=MU_800P,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            strike_price=800,
            option_type="P",
            expiration_date=date(2026, 7, 31),
            average_open_price=22.38,
            cost_basis=2238,
            market_value=2450,
            unrealized_pnl=-212,
            multiplier=100,
        ),
        Position(
            symbol=MU_1400C,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            strike_price=1400,
            option_type="C",
            expiration_date=date(2026, 7, 31),
            average_open_price=16.01,
            cost_basis=1601,
            market_value=805,
            unrealized_pnl=796,
            multiplier=100,
        ),
        Position(
            symbol=MU_900P,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            strike_price=900,
            option_type="P",
            expiration_date=date(2026, 7, 17),
            average_open_price=16.70,
            cost_basis=1670,
            market_value=1570,
            unrealized_pnl=100,
            multiplier=100,
        ),
    ]


def test_extract_occ_root_handles_padded_and_normalized() -> None:
    assert extract_occ_root("MU    260710P00800000") == "MU"
    assert extract_occ_root("MU 260710P00800000") == "MU"
    assert extract_occ_root(normalize_occ_symbol("MU    260710P00800000")) == "MU"
    assert extract_occ_root("SPY   260717P00500000") == "SPY"
    assert extract_occ_root("BRK/B 260717C00100000") == "BRK/B" or extract_occ_root(
        "BRK/B 260717C00100000"
    ).startswith("BRK")


def test_mu_two_independent_lineages_exclude_unrelated_900p(tmp_path) -> None:
    tracker = RollTracker()
    chains = tracker.detect_rolls(mu_transactions(), mu_positions(), "5WT00001")

    terminals = {c.resolved_terminal_symbol() for c in chains}
    assert MU_730P_N in terminals
    assert MU_1400C_N in terminals
    assert MU_900P_N not in terminals
    assert MU_800P_N not in terminals
    assert len(chains) == 2
    assert all(c.underlying == "MU" for c in chains)

    put = next(c for c in chains if c.resolved_terminal_symbol() == MU_730P_N)
    call = next(c for c in chains if c.resolved_terminal_symbol() == MU_1400C_N)

    assert put.rolls[0].old_symbol == MU_800P_OLD_N
    assert put.rolls[0].new_symbol == MU_730P_N
    assert put.rolls[0].roll_pnl == 99
    assert put.chain_total_credit == 792
    assert put.history_complete is True
    assert put.is_open is True
    # DTE at roll execution (2026-07-01): old exp 7/10 → 9d, new exp 7/17 → 16d.
    assert put.rolls[0].old_dte == 9
    assert put.rolls[0].new_dte == 16

    assert call.rolls[0].old_symbol == MU_1750C_OLD_N
    assert call.rolls[0].new_symbol == MU_1400C_N
    assert call.rolls[0].roll_pnl == 361
    assert call.chain_total_credit == 1962
    assert call.history_complete is True
    assert call.is_open is True
    # Call same-day roll on 7/1 with 7/31 exp → 30 DTE both legs.
    assert call.rolls[0].old_dte == 30
    assert call.rolls[0].new_dte == 30

    database = PositionPilotDatabase(tmp_path / "mu-under.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    service = RollService(database)
    for chain in chains:
        service.save_chain(identity.account_id, chain, source=SOURCE_BROKER)
    stored = service.chains(identity.account_id, symbol="MU")
    assert len(stored) == 2
    assert {c.underlying for c in stored} == {"MU"}
    assert {c.terminal_symbol for c in stored} == {MU_730P_N, MU_1400C_N}


def test_mu_roll_adjusted_leg_and_strangle_pnl_open_and_basis(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    service = RollService(database)

    for chain in RollTracker().detect_rolls(mu_transactions(), mu_positions(), "5WT00001"):
        service.save_chain(identity.account_id, chain, source=SOURCE_BROKER)

    cost = {
        MU_730P: 693.0,
        MU_800P: 2238.0,
        MU_1400C: 1601.0,
        MU_900P: 300.0,
    }
    raw = {
        MU_730P: 450.0,
        MU_800P: -212.0,
        MU_1400C: 796.0,
        MU_900P: 100.0,
    }
    positions = [
        PositionSnapshot(
            symbol=symbol,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction=QuantityDirection.SHORT,
            position_type=PositionType.EQUITY_OPTION,
            cost_basis=cost[symbol],
            unrealized_pnl=pnl,
            pnl_open=pnl,
            pnl_open_basis=cost[symbol],
            strike_price=float(symbol[-8:]) / 1000.0 if len(symbol) >= 15 else None,
            option_type="P" if "P" in symbol[-9:-8] or "P" in symbol else "C",
            horizon=PositionHorizon.TACTICAL,
        )
        for symbol, pnl in raw.items()
    ]
    # Fix option_type detection for normalized symbols.
    for p in positions:
        p.option_type = "P" if "P00" in p.symbol or p.symbol[-9] == "P" else "C"

    strangle = StrategySnapshot(
        strategy_id="strangle",
        account_id=identity.account_id,
        underlying="MU",
        strategy_type="Short Strangle",
        expiration_date="2026-02-20",
        days_to_expiration=30,
        quantity=1,
        strikes="$800/$1400",
        unrealized_pnl=-212 + 796,
        pnl_open=-212 + 796,
        horizon=PositionHorizon.TACTICAL,
        legs=[p for p in positions if p.symbol in {MU_800P, MU_1400C}],
    )
    singles = [
        StrategySnapshot(
            strategy_id=f"single-{p.symbol}",
            account_id=identity.account_id,
            underlying="MU",
            strategy_type="Short Put" if p.option_type == "P" else "Short Call",
            expiration_date="2026-02-20",
            days_to_expiration=30,
            quantity=1,
            strikes=f"${p.strike_price}",
            unrealized_pnl=p.unrealized_pnl,
            pnl_open=p.unrealized_pnl,
            horizon=PositionHorizon.TACTICAL,
            legs=[p],
        )
        for p in positions
        if p.symbol in {MU_730P, MU_900P}
    ]

    adj_positions, adj_strategies = service.apply_roll_adjustments(
        identity.account_id,
        positions,
        [strangle, *singles],
    )
    by_symbol = {p.symbol: p for p in adj_positions}

    assert by_symbol[MU_730P].roll_adjustment == 99
    assert by_symbol[MU_730P].pnl_open == 549
    assert by_symbol[MU_1400C].roll_adjustment == 361
    assert by_symbol[MU_1400C].pnl_open == 1157
    assert by_symbol[MU_800P].roll_adjustment == 0
    assert by_symbol[MU_800P].pnl_open == -212
    assert by_symbol[MU_800P].pnl_open_basis == 2238
    assert by_symbol[MU_1400C].pnl_open_basis == 1962

    adj_strangle = next(s for s in adj_strategies if s.strategy_id == "strangle")
    assert adj_strangle.pnl_open == 945
    assert adj_strangle.roll_adjustment == 361
    # Denominator includes unrolled 800P raw basis + rolled call lifetime.
    assert adj_strangle.pnl_open_basis == 2238 + 1962
    expected_pct = (945 / (2238 + 1962)) * 100
    assert adj_strangle.pnl_open_percent == expected_pct

    symbol_total = sum(s.pnl_open or 0 for s in adj_strategies)
    assert symbol_total == 549 + 945 + 100


def test_stable_chain_id_across_second_roll(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    service = RollService(database)

    t0 = datetime(2026, 6, 26, 15, 0, 0)
    t1 = datetime(2026, 7, 1, 15, 30, 0)
    t2 = datetime(2026, 7, 15, 15, 30, 0)
    first = "MU    260710P00800000"
    second = "MU    260717P00730000"
    third = "MU    260814P00700000"

    txs_one = [
        _tx(
            tx_id="a1",
            order_id="o1",
            symbol=first,
            action="Sell to Open",
            amount=485,
            when=t0,
        ),
        _tx(
            tx_id="a2",
            order_id="r1",
            symbol=first,
            action="Buy to Close",
            amount=386,
            when=t1,
        ),
        _tx(
            tx_id="a3",
            order_id="r1",
            symbol=second,
            action="Sell to Open",
            amount=693,
            when=t1,
        ),
    ]
    pos_one = [
        Position(
            symbol=second,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=10,
        )
    ]
    chain_one = RollTracker().detect_rolls(txs_one, pos_one, "5WT00001")[0]
    id_one = service.save_chain(identity.account_id, chain_one, source=SOURCE_BROKER)

    txs_two = txs_one + [
        _tx(
            tx_id="a4",
            order_id="r2",
            symbol=second,
            action="Buy to Close",
            amount=200,
            when=t2,
        ),
        _tx(
            tx_id="a5",
            order_id="r2",
            symbol=third,
            action="Sell to Open",
            amount=250,
            when=t2,
        ),
    ]
    pos_two = [
        Position(
            symbol=third,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=20,
        )
    ]
    chain_two = RollTracker().detect_rolls(txs_two, pos_two, "5WT00001")[0]
    id_two = service.save_chain(identity.account_id, chain_two, source=SOURCE_BROKER)

    assert id_one == id_two
    stored = service.chains(identity.account_id)
    assert len(stored) == 1
    assert stored[0].chain_id == id_one
    assert len(stored[0].rolls) == 2
    assert stored[0].terminal_symbol == normalize_occ_symbol(third)
    assert stored[0].underlying == "MU"


def test_stable_chain_id_survives_partial_to_complete(tmp_path) -> None:
    """Partial history (missing open) and later complete history share one chain_id."""
    database = PositionPilotDatabase(tmp_path / "partial-complete.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    service = RollService(database)

    old_sym = "MU    260710P00800000"
    new_sym = "MU    260717P00730000"
    t_open = datetime(2026, 6, 26, 15, 0, 0)
    t_roll = datetime(2026, 7, 1, 15, 30, 0)
    positions = [
        Position(
            symbol=new_sym,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=450,
        )
    ]
    # Partial: roll only (no original open in window).
    partial_txs = [
        _tx(
            tx_id="c1",
            order_id="r1",
            symbol=old_sym,
            action="Buy to Close",
            amount=386,
            when=t_roll,
        ),
        _tx(
            tx_id="c2",
            order_id="r1",
            symbol=new_sym,
            action="Sell to Open",
            amount=693,
            when=t_roll,
        ),
    ]
    partial = RollTracker().detect_rolls(partial_txs, positions, "5WT00001")[0]
    assert partial.history_complete is False
    assert partial.original_open_credit is None
    id_partial = service.save_chain(identity.account_id, partial, source=SOURCE_BROKER)

    # Complete: older open becomes available; first roll_id stays roll_c1_c2.
    complete_txs = [
        _tx(
            tx_id="c0",
            order_id="o1",
            symbol=old_sym,
            action="Sell to Open",
            amount=485,
            when=t_open,
        ),
        *partial_txs,
    ]
    complete = RollTracker().detect_rolls(complete_txs, positions, "5WT00001")[0]
    assert complete.history_complete is True
    assert complete.original_open_date is not None
    id_complete = service.save_chain(identity.account_id, complete, source=SOURCE_BROKER)

    assert id_partial == id_complete
    stored = service.chains(identity.account_id, symbol="MU")
    assert len(stored) == 1
    assert stored[0].chain_id == id_partial
    assert stored[0].history_complete is True
    assert stored[0].original_open_credit == 485
    assert stored[0].rolls[0].roll_pnl == 99


def test_legacy_row_never_adjusts_live_pnl(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    service = RollService(database)

    # Wrong legacy carry that would corrupt P/L if applied.
    legacy = RollChain(
        underlying="MU",
        strategy_type="Short Put",
        account_number="5WT00001",
        original_open_credit=485,
        history_complete=True,
        is_open=True,
        terminal_symbol=MU_730P,
        root_symbol=MU_800P_OLD,
        source=SOURCE_LEGACY,
        rolls=[
            RollEvent(
                roll_id="legacy-bad",
                timestamp=datetime(2026, 1, 20, 15, 30, 0),
                underlying="MU",
                strategy_type="Short Put",
                account_number="5WT00001",
                old_symbol=MU_800P_OLD,
                old_strike=800,
                old_expiration=date(2026, 1, 16),
                old_dte=5,
                new_symbol=MU_730P,
                new_strike=730,
                new_expiration=date(2026, 2, 20),
                new_dte=30,
                roll_pnl=9999,  # deliberately wrong
                premium_effect=307,
            )
        ],
    )
    service.save_chain(identity.account_id, legacy, source=SOURCE_LEGACY)

    position = PositionSnapshot(
        symbol=MU_730P,
        underlying_symbol="MU",
        quantity=-1,
        quantity_direction=QuantityDirection.SHORT,
        position_type=PositionType.EQUITY_OPTION,
        cost_basis=693,
        unrealized_pnl=450,
        pnl_open=450,
        horizon=PositionHorizon.TACTICAL,
    )
    adj, _ = service.apply_roll_adjustments(identity.account_id, [position], [])
    assert adj[0].roll_adjustment == 0
    assert adj[0].pnl_open == 450

    # Broker chain coexists and wins; legacy cannot overwrite broker row.
    broker_put = next(
        c
        for c in RollTracker().detect_rolls(mu_transactions(), mu_positions(), "5WT00001")
        if c.resolved_terminal_symbol() == MU_730P_N
    )
    service.save_chain(identity.account_id, broker_put, source=SOURCE_BROKER)

    # Attempt legacy re-save against the broker stable id must not clobber broker.
    legacy_clobber = legacy.model_copy(
        update={
            "rolls": [
                legacy.rolls[0].__class__(
                    **{
                        **legacy.rolls[0].__dict__,
                        "roll_id": broker_put.rolls[0].roll_id,
                    }
                )
            ],
            "root_symbol": broker_put.root_symbol,
        }
    )
    service.save_chain(identity.account_id, legacy_clobber, source=SOURCE_LEGACY)
    broker_rows = [c for c in service.chains(identity.account_id) if c.source == SOURCE_BROKER]
    put_broker = next(
        c for c in broker_rows if normalize_occ_symbol(c.terminal_symbol) == MU_730P_N
    )
    assert put_broker.rolls[0].roll_pnl == 99
    assert put_broker.underlying == "MU"

    adj2, _ = service.apply_roll_adjustments(identity.account_id, [position], [])
    assert adj2[0].roll_adjustment == 99
    assert adj2[0].pnl_open == 549


def test_legacy_file_imported_only_once(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    chain = RollChain(
        underlying="SPY",
        strategy_type="Short Put",
        account_number="5WT00001",
        original_open_credit=150,
        rolls=[
            RollEvent(
                roll_id="roll-1",
                timestamp=datetime(2026, 7, 10, 14, 30),
                underlying="SPY",
                strategy_type="Short Put",
                account_number="5WT00001",
                old_symbol="SPY 260717P00500000",
                old_strike=500,
                old_expiration=date(2026, 7, 17),
                old_dte=7,
                new_symbol="SPY 260821P00495000",
                new_strike=495,
                new_expiration=date(2026, 8, 21),
                new_dte=42,
                premium_effect=80,
            )
        ],
    )
    (cache / "roll_history.json").write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "accounts": {
                    "5WT00001": {
                        "chains": {"SPY:Short Put": chain.to_dict()},
                        "last_updated": "2026-07-10T14:30:00",
                    }
                },
            }
        )
    )
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    service = RollService(database, legacy_history_path=cache / "roll_history.json")

    first = service.ensure_legacy_history_imported()
    assert first == 1
    assert len(service.chains(identity.account_id)) == 1
    # Mutate file; second import must not re-run.
    (cache / "roll_history.json").write_text("{}")
    second = service.ensure_legacy_history_imported()
    assert second == 0
    assert len(service.chains(identity.account_id)) == 1
    assert service.chains(identity.account_id)[0].source == SOURCE_LEGACY


def test_long_and_mixed_multi_leg_pairing() -> None:
    t0 = datetime(2026, 3, 1, 15, 0, 0)
    t1 = datetime(2026, 3, 10, 15, 0, 0)
    long_put_old = "IWM 260320P00190000"
    long_put_new = "IWM 260417P00185000"
    short_call_old = "IWM 260320C00220000"
    short_call_new = "IWM 260417C00225000"

    transactions = [
        _tx(
            tx_id="l1",
            order_id="open-mixed",
            symbol=long_put_old,
            action="Buy to Open",
            amount=400,
            when=t0,
        ),
        _tx(
            tx_id="s1",
            order_id="open-mixed",
            symbol=short_call_old,
            action="Sell to Open",
            amount=300,
            when=t0,
        ),
        # Mixed roll: STC long put + BTO new long put; BTC short call + STO new short call.
        _tx(
            tx_id="l2",
            order_id="roll-mixed",
            symbol=long_put_old,
            action="Sell to Close",
            amount=250,
            when=t1,
        ),
        _tx(
            tx_id="s2",
            order_id="roll-mixed",
            symbol=short_call_old,
            action="Buy to Close",
            amount=150,
            when=t1,
        ),
        _tx(
            tx_id="l3",
            order_id="roll-mixed",
            symbol=long_put_new,
            action="Buy to Open",
            amount=380,
            when=t1,
        ),
        _tx(
            tx_id="s3",
            order_id="roll-mixed",
            symbol=short_call_new,
            action="Sell to Open",
            amount=280,
            when=t1,
        ),
    ]
    positions = [
        Position(
            symbol=long_put_new,
            underlying_symbol="IWM",
            quantity=1,
            quantity_direction="Long",
            position_type=PositionType.EQUITY_OPTION,
            option_type="P",
            unrealized_pnl=10,
        ),
        Position(
            symbol=short_call_new,
            underlying_symbol="IWM",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            option_type="C",
            unrealized_pnl=20,
        ),
    ]
    chains = RollTracker().detect_rolls(transactions, positions, "5WT00001")
    assert len(chains) == 2
    by_terminal = {c.resolved_terminal_symbol(): c for c in chains}

    long_chain = by_terminal[long_put_new]
    assert long_chain.strategy_type == "Long Put"
    # Long realized: STC +250 + original BTO -400 = -150
    assert long_chain.rolls[0].roll_pnl == -150

    short_chain = by_terminal[short_call_new]
    assert short_chain.strategy_type == "Short Call"
    # Short realized: STO +300 + BTC -150 = +150
    assert short_chain.rolls[0].roll_pnl == 150


def test_fifo_multi_lot_and_quantity_mismatch_incomplete() -> None:
    t0 = datetime(2026, 4, 1, 15, 0, 0)
    t1 = datetime(2026, 4, 2, 15, 0, 0)
    t2 = datetime(2026, 4, 10, 15, 0, 0)
    sym_a = "QQQ 260417P00400000"
    sym_b = "QQQ 260515P00390000"

    # Two lots of 1 contract each (cash 100 then 200).
    multi_lot = [
        _tx(
            tx_id="m1",
            order_id="o1",
            symbol=sym_a,
            action="Sell to Open",
            amount=100,
            when=t0,
            quantity=1,
        ),
        _tx(
            tx_id="m2",
            order_id="o2",
            symbol=sym_a,
            action="Sell to Open",
            amount=200,
            when=t1,
            quantity=1,
        ),
        # Close both lots in one roll of qty 2.
        _tx(
            tx_id="m3",
            order_id="r1",
            symbol=sym_a,
            action="Buy to Close",
            amount=150,
            when=t2,
            quantity=2,
        ),
        _tx(
            tx_id="m4",
            order_id="r1",
            symbol=sym_b,
            action="Sell to Open",
            amount=180,
            when=t2,
            quantity=2,
        ),
    ]
    pos = [
        Position(
            symbol=sym_b,
            underlying_symbol="QQQ",
            quantity=-2,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=50,
        )
    ]
    chains = RollTracker().detect_rolls(multi_lot, pos, "5WT00001")
    assert len(chains) == 1
    assert chains[0].history_complete is True
    # FIFO cash 100+200=300 + close -150 = 150
    assert chains[0].rolls[0].roll_pnl == 150

    # Quantity mismatch: close 1 of 2 without matching open qty → incomplete.
    mismatch = [
        _tx(
            tx_id="x1",
            order_id="ox",
            symbol=sym_a,
            action="Sell to Open",
            amount=200,
            when=t0,
            quantity=2,
        ),
        _tx(
            tx_id="x2",
            order_id="rx",
            symbol=sym_a,
            action="Buy to Close",
            amount=80,
            when=t2,
            quantity=1,
        ),
        _tx(
            tx_id="x3",
            order_id="rx",
            symbol=sym_b,
            action="Sell to Open",
            amount=90,
            when=t2,
            quantity=1,
        ),
    ]
    pos_m = [
        Position(
            symbol=sym_b,
            underlying_symbol="QQQ",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=5,
        )
    ]
    chains_m = RollTracker().detect_rolls(mismatch, pos_m, "5WT00001")
    assert len(chains_m) == 1
    # Pro-rata half of 200 = 100 open + close -80 = 20, and complete for qty 1 of lot 2.
    # Remaining lot left open — close qty matched pro-rata so complete is True for cash.
    # Quantity close==open (1==1) so not incomplete from qty mismatch.
    assert chains_m[0].rolls[0].roll_pnl == 20
    assert chains_m[0].history_complete is True

    # Explicit open/close qty mismatch marks incomplete.
    bad = [
        _tx(
            tx_id="y1",
            order_id="oy",
            symbol=sym_a,
            action="Sell to Open",
            amount=200,
            when=t0,
            quantity=1,
        ),
        _tx(
            tx_id="y2",
            order_id="ry",
            symbol=sym_a,
            action="Buy to Close",
            amount=80,
            when=t2,
            quantity=1,
        ),
        _tx(
            tx_id="y3",
            order_id="ry",
            symbol=sym_b,
            action="Sell to Open",
            amount=90,
            when=t2,
            quantity=2,  # mismatch vs close qty 1
        ),
    ]
    chains_bad = RollTracker().detect_rolls(bad, pos_m, "5WT00001")
    assert len(chains_bad) == 1
    assert chains_bad[0].history_complete is False
    assert chains_bad[0].roll_adjustment == 0


def test_missing_original_open_is_partial_and_does_not_adjust() -> None:
    t1 = datetime(2026, 1, 20, 15, 30, 0)
    transactions = [
        _tx(
            tx_id="10",
            order_id="roll-only",
            symbol=MU_800P_OLD,
            action="Buy to Close",
            amount=386,
            when=t1,
        ),
        _tx(
            tx_id="11",
            order_id="roll-only",
            symbol=MU_730P,
            action="Sell to Open",
            amount=693,
            when=t1,
        ),
    ]
    positions = [
        Position(
            symbol=MU_730P,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            strike_price=730,
            option_type="P",
            unrealized_pnl=450,
            multiplier=100,
        )
    ]
    chains = RollTracker().detect_rolls(transactions, positions, "5WT00001")
    assert len(chains) == 1
    chain = chains[0]
    assert chain.history_complete is False
    assert chain.original_open_credit is None
    assert chain.chain_total_credit is None
    assert chain.roll_adjustment == 0
    assert chain.is_open is True


def test_closed_stale_chain_does_not_attach_to_current_positions(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    service = RollService(database)

    stale = RollChain(
        underlying="MU",
        strategy_type="Short Put",
        account_number="5WT00001",
        original_open_credit=485,
        history_complete=True,
        is_open=True,
        terminal_symbol=MU_800P_OLD,
        source=SOURCE_BROKER,
        rolls=[
            RollEvent(
                roll_id="stale-1",
                timestamp=datetime(2026, 1, 20, 15, 30, 0),
                underlying="MU",
                strategy_type="Short Put",
                account_number="5WT00001",
                old_symbol=MU_800P_OLD,
                old_strike=800,
                old_expiration=date(2026, 1, 16),
                old_dte=5,
                new_symbol=MU_800P_OLD,
                new_strike=800,
                new_expiration=date(2026, 1, 16),
                new_dte=5,
                roll_pnl=99,
                premium_effect=307,
            )
        ],
    )
    service.save_chain(identity.account_id, stale, source=SOURCE_BROKER)
    service._relabel_open_status(
        identity.account_id,
        [
            Position(
                symbol=MU_800P,
                underlying_symbol="MU",
                quantity=-1,
                quantity_direction="Short",
                position_type=PositionType.EQUITY_OPTION,
                unrealized_pnl=-212,
            )
        ],
    )
    assert service.chains(identity.account_id)[0].is_open is False

    positions = [
        PositionSnapshot(
            symbol=MU_800P,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction=QuantityDirection.SHORT,
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=-212,
            pnl_open=-212,
            horizon=PositionHorizon.TACTICAL,
        )
    ]
    adj, _ = service.apply_roll_adjustments(identity.account_id, positions, [])
    assert adj[0].roll_adjustment == 0
    assert adj[0].pnl_open == -212


def test_existing_roll_payload_without_new_fields_still_validates(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    legacy_payload = {
        "chain_id": "legacy-chain-1",
        "account_id": identity.account_id,
        "underlying": "SPY",
        "strategy_type": "Short Put",
        "original_open_credit": 150,
        "rolls": [
            {
                "roll_id": "roll-1",
                "timestamp": "2026-07-10T14:30:00",
                "underlying": "SPY",
                "strategy_type": "Short Put",
                "old_symbol": "SPY 260717P00500000",
                "old_strike": 500,
                "old_expiration": "2026-07-17",
                "old_dte": 7,
                "new_symbol": "SPY 260821P00495000",
                "new_strike": 495,
                "new_expiration": "2026-08-21",
                "new_dte": 42,
                "roll_pnl": 125,
                "premium_effect": 80,
                "commission": 0,
            }
        ],
    }
    database.save_roll_chain(legacy_payload)
    restored = RollService(database).chains(identity.account_id)
    assert len(restored) == 1
    assert restored[0].history_complete is True
    assert restored[0].source == SOURCE_LEGACY
    assert restored[0].chain_total_credit == 230
    assert restored[0].is_open is False


class _TxSource:
    def __init__(self, transactions: list[Transaction], *, fail: bool = False) -> None:
        self.transactions = transactions
        self.fail = fail
        self.calls = 0
        self.force_refresh_calls = 0

    def get_transactions(
        self,
        account_number: str,
        *,
        start_date=None,
        end_date=None,
        limit: int = 250,
        force_refresh: bool = False,
    ) -> list[Transaction]:
        self.calls += 1
        if force_refresh:
            self.force_refresh_calls += 1
        if self.fail:
            raise ConnectionError("broker transactions unavailable")
        return list(self.transactions)


def test_automatic_sync_ttl_partial_and_errors(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    source = _TxSource(mu_transactions())
    clock_times = [
        datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        datetime(2026, 7, 13, 12, 5, tzinfo=UTC),
        datetime(2026, 7, 13, 12, 20, tzinfo=UTC),
    ]
    clock_iter = iter(clock_times)

    service = RollService(
        database,
        transaction_source=source,
        sync_ttl_seconds=15 * 60,
        clock=lambda: next(clock_iter),
    )
    positions = mu_positions()

    first = service.sync_account_transactions(identity.account_id, "5WT00001", positions)
    assert first.status == "refreshed"
    assert first.refreshed is True
    assert source.force_refresh_calls == 1

    second = service.sync_account_transactions(identity.account_id, "5WT00001", positions)
    assert second.status == "cached"
    assert source.force_refresh_calls == 1

    third = service.sync_account_transactions(identity.account_id, "5WT00001", positions)
    assert third.status == "refreshed"
    assert source.force_refresh_calls == 2

    # Incomplete detected lineages surface as partial (and still persist).
    partial_txs = [
        _tx(
            tx_id="p1",
            order_id="pr",
            symbol=MU_800P_OLD,
            action="Buy to Close",
            amount=386,
            when=datetime(2026, 1, 20, 15, 30, 0),
        ),
        _tx(
            tx_id="p2",
            order_id="pr",
            symbol=MU_730P,
            action="Sell to Open",
            amount=693,
            when=datetime(2026, 1, 20, 15, 30, 0),
        ),
    ]
    partial_service = RollService(
        database,
        transaction_source=_TxSource(partial_txs),
        sync_ttl_seconds=0,
        clock=lambda: datetime(2026, 7, 13, 14, 0, tzinfo=UTC),
    )
    partial = partial_service.sync_account_transactions(identity.account_id, "5WT00001", positions)
    assert partial.status == "partial"
    assert partial.refreshed is True

    failing = RollService(
        database,
        transaction_source=_TxSource([], fail=True),
        sync_ttl_seconds=0,
        clock=lambda: datetime(2026, 7, 13, 15, 0, tzinfo=UTC),
    )
    failed = failing.sync_account_transactions(identity.account_id, "5WT00001", positions)
    assert failed.status == "unavailable"
    assert failed.error is not None


def test_dte_uses_roll_execution_date_not_now() -> None:
    """Historical rolls keep DTE relative to execution day, even after expiry."""
    t_roll = datetime(2026, 7, 1, 12, 0, 0)
    old_sym = "SPY   260703P00500000"  # exp July 3 → 2 DTE at roll
    new_sym = "SPY   260731P00495000"  # exp July 31 → 30 DTE at roll
    txs = [
        _tx(
            tx_id="d1",
            order_id="od",
            symbol=old_sym,
            action="Sell to Open",
            amount=100,
            when=datetime(2026, 6, 20, 12, 0, 0),
        ),
        _tx(
            tx_id="d2",
            order_id="rd",
            symbol=old_sym,
            action="Buy to Close",
            amount=40,
            when=t_roll,
        ),
        _tx(
            tx_id="d3",
            order_id="rd",
            symbol=new_sym,
            action="Sell to Open",
            amount=80,
            when=t_roll,
        ),
    ]
    pos = [
        Position(
            symbol=new_sym,
            underlying_symbol="SPY",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=10,
        )
    ]
    chain = RollTracker().detect_rolls(txs, pos, "5WT00001")[0]
    assert chain.rolls[0].old_dte == 2
    assert chain.rolls[0].new_dte == 30
    assert chain.underlying == "SPY"


def test_unmatched_roll_legs_preserved_or_close_chain() -> None:
    """Unmatched opens become lots; unmatched closes terminate active chains."""
    t0 = datetime(2026, 6, 26, 15, 0, 0)
    t1 = datetime(2026, 7, 1, 15, 0, 0)
    t2 = datetime(2026, 7, 2, 15, 0, 0)
    short_a = "MU    260710P00800000"
    short_b = "MU    260717P00730000"
    extra_open = "MU    260731P00800000"
    later = "MU    260814P00750000"

    # Roll pairs short_a→short_b, but also has an unmatched extra open lot.
    txs = [
        _tx(
            tx_id="u1",
            order_id="o1",
            symbol=short_a,
            action="Sell to Open",
            amount=100,
            when=t0,
        ),
        _tx(
            tx_id="u2",
            order_id="r1",
            symbol=short_a,
            action="Buy to Close",
            amount=40,
            when=t1,
        ),
        _tx(
            tx_id="u3",
            order_id="r1",
            symbol=short_b,
            action="Sell to Open",
            amount=90,
            when=t1,
        ),
        # Unmatched open in the same roll order (no paired close of same type/side).
        # Use a long open so it does not pair with the short close.
        _tx(
            tx_id="u4",
            order_id="r1",
            symbol=extra_open,
            action="Buy to Open",
            amount=50,
            when=t1,
        ),
        # Later exact-symbol roll of the unmatched long lot.
        _tx(
            tx_id="u5",
            order_id="r2",
            symbol=extra_open,
            action="Sell to Close",
            amount=30,
            when=t2,
        ),
        _tx(
            tx_id="u6",
            order_id="r2",
            symbol=later,
            action="Buy to Open",
            amount=55,
            when=t2,
        ),
    ]
    positions = [
        Position(
            symbol=short_b,
            underlying_symbol="MU",
            quantity=-1,
            quantity_direction="Short",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=10,
        ),
        Position(
            symbol=later,
            underlying_symbol="MU",
            quantity=1,
            quantity_direction="Long",
            position_type=PositionType.EQUITY_OPTION,
            unrealized_pnl=5,
        ),
    ]
    chains = RollTracker().detect_rolls(txs, positions, "5WT00001")
    by_term = {c.resolved_terminal_symbol(): c for c in chains}
    assert normalize_occ_symbol(short_b) in by_term
    # Unmatched long open retained cost basis → later long roll realizes STC - BTO.
    long_chain = by_term[normalize_occ_symbol(later)]
    assert long_chain.history_complete is True
    assert long_chain.rolls[0].roll_pnl == -20  # STC +30 + BTO -50
    assert long_chain.underlying == "MU"

    # Unmatched close alone closes an active short chain.
    t3 = datetime(2026, 7, 3, 15, 0, 0)
    txs_close = [
        _tx(
            tx_id="v1",
            order_id="ov",
            symbol=short_a,
            action="Sell to Open",
            amount=100,
            when=t0,
        ),
        _tx(
            tx_id="v2",
            order_id="rv",
            symbol=short_a,
            action="Buy to Close",
            amount=40,
            when=t1,
        ),
        _tx(
            tx_id="v3",
            order_id="rv",
            symbol=short_b,
            action="Sell to Open",
            amount=90,
            when=t1,
        ),
        # Roll order with only a closing leg for short_b + unrelated open (no pair).
        _tx(
            tx_id="v4",
            order_id="rc",
            symbol=short_b,
            action="Buy to Close",
            amount=50,
            when=t3,
        ),
        _tx(
            tx_id="v5",
            order_id="rc",
            symbol=extra_open,
            action="Buy to Open",
            amount=10,
            when=t3,
        ),
    ]
    chains_c = RollTracker().detect_rolls(
        txs_close,
        [
            Position(
                symbol=extra_open,
                underlying_symbol="MU",
                quantity=1,
                quantity_direction="Long",
                position_type=PositionType.EQUITY_OPTION,
                unrealized_pnl=1,
            )
        ],
        "5WT00001",
        include_closed=True,
    )
    short_closed = [
        c
        for c in chains_c
        if c.root_symbol == normalize_occ_symbol(short_a)
        or c.rolls[0].old_symbol == normalize_occ_symbol(short_a)
    ]
    assert any(
        c.is_open is False or c.resolved_terminal_symbol() == normalize_occ_symbol(short_b)
        for c in short_closed
    )
    closed = next(
        c for c in chains_c if c.rolls and c.rolls[0].old_symbol == normalize_occ_symbol(short_a)
    )
    # After unmatched BTC of terminal short_b, chain is closed / not open.
    assert closed.is_open is False


def test_portfolio_multi_account_partial_does_not_downgrade_unavailable(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")

    class MultiSource:
        def get_accounts(self) -> list[Account]:
            return [
                Account(account_number="5WT00001", account_type="Individual"),
                Account(account_number="5WT00002", account_type="Individual"),
            ]

        def get_account_balances(self, account_number: str) -> dict:
            return {
                "net_liquidating_value": 50_000,
                "cash_balance": 10_000,
                "buying_power": 20_000,
            }

        def get_positions(self, account_number: str) -> list[Position]:
            return mu_positions()

    class MultiTx:
        def get_transactions(
            self,
            account_number: str,
            *,
            start_date=None,
            end_date=None,
            limit: int = 250,
            force_refresh: bool = False,
        ) -> list[Transaction]:
            if account_number.endswith("00001"):
                raise ConnectionError("down")
            # Incomplete only for second account.
            return [
                _tx(
                    tx_id="z1",
                    order_id="zr",
                    symbol=MU_800P_OLD,
                    action="Buy to Close",
                    amount=386,
                    when=datetime(2026, 1, 20, 15, 30, 0),
                ),
                _tx(
                    tx_id="z2",
                    order_id="zr",
                    symbol=MU_730P,
                    action="Sell to Open",
                    amount=693,
                    when=datetime(2026, 1, 20, 15, 30, 0),
                ),
            ]

    roll_service = RollService(
        database,
        transaction_source=MultiTx(),
        sync_ttl_seconds=0,
        clock=lambda: datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    portfolio = PortfolioService(
        database=database,
        source=MultiSource(),
        roll_service=roll_service,
        clock=lambda: datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    snapshot = portfolio.refresh(enrich=False)
    assert snapshot.freshness_by_panel["rolls"].state is FreshnessState.UNAVAILABLE


class _PortfolioSource:
    def get_accounts(self) -> list[Account]:
        return [Account(account_number="5WT00001", account_type="Individual")]

    def get_account_balances(self, account_number: str) -> dict:
        return {
            "net_liquidating_value": 100_000,
            "cash_balance": 50_000,
            "buying_power": 80_000,
        }

    def get_positions(self, account_number: str) -> list[Position]:
        return mu_positions()


def test_roll_freshness_as_of_uses_sync_timestamp_not_only_captured_at(tmp_path) -> None:
    """Older cached sync timestamps still surface as rolls.as_of when available."""
    database = PositionPilotDatabase(tmp_path / "freshness.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    sync_time = datetime(2026, 7, 10, 9, 0, tzinfo=UTC)
    capture_time = datetime(2026, 7, 13, 16, 0, tzinfo=UTC)
    # Persist a prior sync timestamp (TTL not expired relative to capture via force path).
    database.set_setting(f"rolls.sync.{identity.account_id}", sync_time.isoformat())

    # Inside TTL: sync returns cached with synced_at=sync_time (older than capture).
    roll_service = RollService(
        database,
        transaction_source=_TxSource(mu_transactions()),
        sync_ttl_seconds=30 * 24 * 3600,
        clock=lambda: capture_time,
    )
    # Seed broker chains so cached path has data.
    for chain in RollTracker().detect_rolls(mu_transactions(), mu_positions(), "5WT00001"):
        roll_service.save_chain(identity.account_id, chain, source=SOURCE_BROKER)

    portfolio = PortfolioService(
        database=database,
        source=_PortfolioSource(),
        roll_service=roll_service,
        clock=lambda: capture_time,
    )
    snapshot = portfolio.refresh(enrich=False)
    assert snapshot.captured_at == capture_time
    assert snapshot.freshness_by_panel["rolls"].as_of == sync_time
    assert snapshot.freshness_by_panel["rolls"].as_of != capture_time


def test_portfolio_refresh_applies_roll_pnl_without_changing_raw_totals(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT00001", "Individual")
    source = _TxSource(mu_transactions())
    roll_service = RollService(
        database,
        transaction_source=source,
        sync_ttl_seconds=0,
        clock=lambda: datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    portfolio = PortfolioService(
        database=database,
        source=_PortfolioSource(),
        roll_service=roll_service,
        clock=lambda: datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )

    snapshot = portfolio.refresh(enrich=False)
    assert snapshot.accounts[0].account_id == identity.account_id
    serialized = snapshot.model_dump_json()
    assert "5WT00001" not in serialized

    by_symbol = {p.symbol: p for p in snapshot.accounts[0].positions}
    assert by_symbol[MU_730P].unrealized_pnl == 450
    assert by_symbol[MU_730P].pnl_open == 549
    assert by_symbol[MU_1400C].pnl_open == 1157
    assert by_symbol[MU_800P].pnl_open == -212

    # Roll freshness as_of uses the sync timestamp (not portfolio captured_at alone).
    rolls_fresh = snapshot.freshness_by_panel["rolls"]
    assert rolls_fresh.as_of == datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    raw_sum = sum(p.unrealized_pnl for p in snapshot.accounts[0].positions)
    assert snapshot.totals.unrealized_pnl == raw_sum
    assert snapshot.totals.unrealized_pnl != sum(
        p.pnl_open or 0 for p in snapshot.accounts[0].positions
    )

    strangle = next(
        s for s in snapshot.strategies if {leg.symbol for leg in s.legs} == {MU_800P, MU_1400C}
    )
    assert strangle.pnl_open == 945
    assert strangle.pnl_open_basis == 2238 + 1962

    mu_chains = roll_service.chains(identity.account_id, symbol="MU")
    assert len(mu_chains) == 2
    assert all(c.underlying == "MU" for c in mu_chains)
