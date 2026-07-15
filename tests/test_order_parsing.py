"""Regression tests for Tastytrade order payload parsing (no live broker calls)."""

from position_pilot.client.tastytrade import TastytradeClient
from position_pilot.models.transaction import OrderStatus


def _client() -> TastytradeClient:
    return TastytradeClient.__new__(TastytradeClient)


def _base_order_payload(**overrides: object) -> dict:
    payload: dict = {
        "id": "order-1",
        "status": "Received",
        "type": "Limit",
        "created-at": "2026-07-10T14:00:00.000Z",
        "updated-at": "2026-07-10T14:05:00.000Z",
        "symbol": "SPY",
        "action": "Buy to Open",
        "underlying-symbol": "SPY",
        "quantity": 2,
        "filled-quantity": 0,
    }
    payload.update(overrides)
    return payload


def test_parse_order_null_quantity_and_filled_quantity_succeeds() -> None:
    """Explicit JSON null for both quantity fields must not drop the order."""
    client = _client()
    payload = _base_order_payload(quantity=None, **{"filled-quantity": None})

    order = client._parse_order(payload, "acct-test")

    assert order is not None
    assert order.quantity == 0.0
    assert order.filled_quantity == 0.0
    assert order.order_id == "order-1"
    assert order.status == OrderStatus.RECEIVED


def test_parse_order_null_quantity_falls_back_to_total_quantity() -> None:
    """When top-level quantity is null, prefer a usable total-quantity."""
    client = _client()
    payload = _base_order_payload(
        quantity=None,
        **{"total-quantity": 3, "filled-quantity": None},
    )

    order = client._parse_order(payload, "acct-test")

    assert order is not None
    assert order.quantity == 3.0
    assert order.filled_quantity == 0.0


def test_parse_order_absent_quantity_fields_use_safe_zeros() -> None:
    """Missing quantity fields must not pass None into required Order numerics."""
    client = _client()
    payload = _base_order_payload()
    del payload["quantity"]
    del payload["filled-quantity"]

    order = client._parse_order(payload, "acct-test")

    assert order is not None
    assert order.quantity == 0.0
    assert order.filled_quantity == 0.0


def test_parse_order_preserves_numeric_and_numeric_string_quantities() -> None:
    """Valid numbers and numeric strings must continue to parse as today."""
    client = _client()

    numeric = client._parse_order(
        _base_order_payload(quantity=5, **{"filled-quantity": 2}),
        "acct-test",
    )
    as_strings = client._parse_order(
        _base_order_payload(quantity="4", **{"filled-quantity": "1.5"}),
        "acct-test",
    )

    assert numeric is not None
    assert numeric.quantity == 5.0
    assert numeric.filled_quantity == 2.0
    assert as_strings is not None
    assert as_strings.quantity == 4.0
    assert as_strings.filled_quantity == 1.5


def test_parse_order_unusable_quantity_values_normalize_to_zero() -> None:
    """Non-numeric quantity values must not cause parse failure via None fields."""
    client = _client()
    payload = _base_order_payload(quantity="n/a", **{"filled-quantity": "bad"})

    order = client._parse_order(payload, "acct-test")

    assert order is not None
    assert order.quantity == 0.0
    assert order.filled_quantity == 0.0
