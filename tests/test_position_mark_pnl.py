"""Enriched mark updates must recompute market value and raw unrealized P/L."""

from __future__ import annotations

import math

import pytest

from position_pilot.models.position import Position, PositionType, is_valid_mark_price


def _option(
    *,
    direction: str,
    avg_open: float,
    mark: float | None,
    quantity: int = 1,
    multiplier: int = 100,
) -> Position:
    cost_basis = avg_open * abs(quantity) * multiplier
    market_value = (mark or 0.0) * abs(quantity) * multiplier
    if direction == "Short":
        unrealized = cost_basis - market_value
    else:
        unrealized = market_value - cost_basis
    percent = (unrealized / abs(cost_basis)) * 100 if cost_basis else None
    return Position(
        symbol="MU    260731C01400000",
        underlying_symbol="MU",
        quantity=quantity,
        quantity_direction=direction,
        position_type=PositionType.EQUITY_OPTION,
        strike_price=1400,
        option_type="C",
        average_open_price=avg_open,
        mark_price=mark,
        cost_basis=cost_basis,
        market_value=market_value,
        unrealized_pnl=unrealized,
        unrealized_pnl_percent=percent,
        multiplier=multiplier,
    )


def test_apply_mark_price_recomputes_long_option_pnl() -> None:
    position = _option(direction="Long", avg_open=2.0, mark=2.0)
    assert position.unrealized_pnl == 0.0

    position.apply_mark_price(3.5)

    assert position.mark_price == 3.5
    assert position.market_value == 350.0
    assert position.unrealized_pnl == 150.0
    assert position.unrealized_pnl_percent == 75.0


def test_apply_mark_price_recomputes_short_option_pnl() -> None:
    # Live MU short call evidence: STO 16.01 credit → cost basis 1601.
    position = _option(direction="Short", avg_open=16.01, mark=4.05)
    assert position.market_value == pytest.approx(405.0)
    assert position.unrealized_pnl == pytest.approx(1196.0)

    # Mark drifts lower (favorable for short) — raw P/L must move.
    position.apply_mark_price(3.725)

    assert position.mark_price == 3.725
    assert position.market_value == pytest.approx(372.5)
    assert position.unrealized_pnl == pytest.approx(1228.5)
    assert position.unrealized_pnl_percent == pytest.approx(pytest_approx_pct(1228.5, 1601.0))


def test_apply_mark_price_missing_does_not_corrupt_when_not_called() -> None:
    position = _option(direction="Short", avg_open=16.01, mark=4.05)
    # Enrichment only calls apply_mark_price when a mark is present.
    assert position.mark_price == 4.05
    assert position.market_value == pytest.approx(405.0)
    assert position.unrealized_pnl == pytest.approx(1196.0)
    assert position.unrealized_pnl_percent == pytest.approx(pytest_approx_pct(1196.0, 1601.0))


def test_apply_mark_price_zero_cost_basis_clears_percent() -> None:
    position = _option(direction="Long", avg_open=0.0, mark=1.0)
    position.cost_basis = 0.0
    position.apply_mark_price(2.0)
    assert position.market_value == 200.0
    assert position.unrealized_pnl == 200.0
    assert position.unrealized_pnl_percent is None


def test_enrich_batch_applies_mark_recompute(monkeypatch) -> None:
    from position_pilot.client.tastytrade import TastytradeClient

    client = TastytradeClient.__new__(TastytradeClient)
    short = _option(direction="Short", avg_open=16.01, mark=4.05)
    long_pos = _option(direction="Long", avg_open=2.0, mark=2.0)
    long_pos.symbol = "MU    260731P00800000"

    def fake_quotes(symbols: list[str], force_refresh: bool = False) -> dict:
        del force_refresh
        out: dict = {}
        for symbol in symbols:
            if "C01400000" in symbol.replace(" ", ""):
                out[symbol] = {"mark": 3.725, "delta": 0.1}
            elif "P00800000" in symbol.replace(" ", ""):
                out[symbol] = {"mark": 5.0, "delta": -0.2}
            else:
                out[symbol] = {"mark": 100.0}
        return out

    monkeypatch.setattr(client, "get_quotes_batch", fake_quotes)
    enriched = client.enrich_positions_greeks_batch([short, long_pos])

    assert enriched[0].mark_price == 3.725
    assert enriched[0].market_value == pytest.approx(372.5)
    assert enriched[0].unrealized_pnl == pytest.approx(1228.5)
    assert enriched[1].mark_price == 5.0
    assert enriched[1].market_value == pytest.approx(500.0)
    assert enriched[1].unrealized_pnl == pytest.approx(300.0)


def test_enrich_single_applies_mark_recompute(monkeypatch) -> None:
    from position_pilot.client.tastytrade import TastytradeClient

    client = TastytradeClient.__new__(TastytradeClient)
    position = _option(direction="Short", avg_open=10.0, mark=5.0)

    monkeypatch.setattr(
        client,
        "get_quote",
        lambda symbol, force_refresh=False: {"mark": 4.0, "delta": 0.2},
    )
    result = client.enrich_position_greeks(position)
    assert result.mark_price == 4.0
    assert result.market_value == 400.0
    assert result.unrealized_pnl == 600.0


def test_enrich_without_mark_leaves_accounting_untouched(monkeypatch) -> None:
    from position_pilot.client.tastytrade import TastytradeClient

    client = TastytradeClient.__new__(TastytradeClient)
    position = _option(direction="Short", avg_open=10.0, mark=5.0)
    before = (position.market_value, position.unrealized_pnl)

    monkeypatch.setattr(
        client,
        "get_quote",
        lambda symbol, force_refresh=False: {"delta": 0.2},  # no mark
    )
    result = client.enrich_position_greeks(position)
    assert (result.market_value, result.unrealized_pnl) == before
    assert result.greeks is not None
    assert result.greeks.delta == 0.2


def test_is_valid_mark_price_zero_and_rejects_invalid() -> None:
    assert is_valid_mark_price(0) is True
    assert is_valid_mark_price(0.0) is True
    assert is_valid_mark_price(3.725) is True
    assert is_valid_mark_price(None) is False
    assert is_valid_mark_price(math.nan) is False
    assert is_valid_mark_price(math.inf) is False
    assert is_valid_mark_price(-math.inf) is False
    assert is_valid_mark_price(-0.01) is False
    assert is_valid_mark_price(True) is False  # bool is not a price


def test_apply_mark_price_zero_recomputes_accounting() -> None:
    position = _option(direction="Short", avg_open=10.0, mark=5.0)
    assert position.apply_mark_price(0.0) is True
    assert position.mark_price == 0.0
    assert position.market_value == 0.0
    # Short credit fully open: raw P/L == cost basis.
    assert position.unrealized_pnl == pytest.approx(1000.0)
    assert position.unrealized_pnl_percent == pytest.approx(100.0)


def test_apply_mark_price_rejects_invalid_without_corruption() -> None:
    position = _option(direction="Short", avg_open=10.0, mark=5.0)
    before = (
        position.mark_price,
        position.market_value,
        position.unrealized_pnl,
        position.unrealized_pnl_percent,
    )
    for bad in (math.nan, math.inf, -math.inf, -1.0, float("nan")):
        assert position.apply_mark_price(bad) is False
        assert (
            position.mark_price,
            position.market_value,
            position.unrealized_pnl,
            position.unrealized_pnl_percent,
        ) == before


def test_enrich_applies_zero_mark_and_ignores_invalid(monkeypatch) -> None:
    from position_pilot.client.tastytrade import TastytradeClient

    client = TastytradeClient.__new__(TastytradeClient)
    zero_pos = _option(direction="Short", avg_open=10.0, mark=5.0)
    bad_pos = _option(direction="Long", avg_open=2.0, mark=2.0)
    bad_pos.symbol = "MU    260731P00800000"
    before_bad = (bad_pos.mark_price, bad_pos.market_value, bad_pos.unrealized_pnl)

    def fake_quotes(symbols: list[str], force_refresh: bool = False) -> dict:
        del force_refresh
        out: dict = {}
        for symbol in symbols:
            if "C01400000" in symbol.replace(" ", ""):
                out[symbol] = {"mark": 0.0, "delta": 0.1}
            elif "P00800000" in symbol.replace(" ", ""):
                out[symbol] = {"mark": float("nan"), "delta": -0.2}
            else:
                out[symbol] = {"mark": -1.0}
        return out

    monkeypatch.setattr(client, "get_quotes_batch", fake_quotes)
    enriched = client.enrich_positions_greeks_batch([zero_pos, bad_pos])

    assert enriched[0].mark_price == 0.0
    assert enriched[0].market_value == 0.0
    assert enriched[0].unrealized_pnl == pytest.approx(1000.0)
    assert (
        enriched[1].mark_price,
        enriched[1].market_value,
        enriched[1].unrealized_pnl,
    ) == before_bad


def test_enrich_single_applies_zero_and_skips_negative(monkeypatch) -> None:
    from position_pilot.client.tastytrade import TastytradeClient

    client = TastytradeClient.__new__(TastytradeClient)
    position = _option(direction="Long", avg_open=2.0, mark=2.0)
    monkeypatch.setattr(
        client,
        "get_quote",
        lambda symbol, force_refresh=False: {"mark": 0.0, "delta": 0.1},
    )
    result = client.enrich_position_greeks(position)
    assert result.mark_price == 0.0
    assert result.market_value == 0.0
    assert result.unrealized_pnl == pytest.approx(-200.0)

    position2 = _option(direction="Long", avg_open=2.0, mark=2.0)
    before = (position2.mark_price, position2.market_value, position2.unrealized_pnl)
    monkeypatch.setattr(
        client,
        "get_quote",
        lambda symbol, force_refresh=False: {"mark": -3.0, "delta": 0.1},
    )
    result2 = client.enrich_position_greeks(position2)
    assert (result2.mark_price, result2.market_value, result2.unrealized_pnl) == before


def pytest_approx_pct(pnl: float, basis: float) -> float:
    return (pnl / abs(basis)) * 100
