"""Regression tests for Tastytrade order timestamp parsing.

Uses synthetic fixtures only — no live broker calls or private payloads.
"""

from datetime import UTC, datetime

from position_pilot.client.tastytrade import TastytradeClient


def _client() -> TastytradeClient:
    return TastytradeClient.__new__(TastytradeClient)


def _order_item(**overrides: object) -> dict:
    item: dict = {
        "id": "order-1",
        "status": "Filled",
        "quantity": 1,
        "type": "Limit",
        "filled-quantity": 1,
        "legs": [{"symbol": "SPY", "action": "Buy to Open"}],
    }
    item.update(overrides)
    return item


def test_parse_order_accepts_iso8601_timestamp_strings() -> None:
    client = _client()
    order = client._parse_order(
        _order_item(
            **{
                "created-at": "2024-07-03T12:00:00.000Z",
                "updated-at": "2024-07-03T12:05:00Z",
            }
        ),
        "synthetic-account",
    )

    assert order is not None
    assert order.created_at == datetime(2024, 7, 3, 12, 0, 0, tzinfo=UTC)
    assert order.updated_at == datetime(2024, 7, 3, 12, 5, 0, tzinfo=UTC)


def test_parse_order_accepts_unix_seconds_timestamps() -> None:
    client = _client()
    created = datetime(2024, 7, 3, 12, 0, 0, tzinfo=UTC)
    updated = datetime(2024, 7, 3, 12, 5, 0, tzinfo=UTC)

    order = client._parse_order(
        _order_item(
            **{
                "created-at": int(created.timestamp()),
                "updated-at": int(updated.timestamp()),
            }
        ),
        "synthetic-account",
    )

    assert order is not None
    assert order.created_at == created
    assert order.updated_at == updated


def test_parse_order_accepts_numeric_id_with_integer_timestamps() -> None:
    """Live JSON uses integer order IDs plus numeric timestamps.

    Both must normalize together; either alone can still drop the order.
    """
    client = _client()
    created = datetime(2024, 7, 3, 12, 0, 0, tzinfo=UTC)
    updated = datetime(2024, 7, 3, 12, 5, 0, tzinfo=UTC)

    order = client._parse_order(
        _order_item(
            **{
                "id": 9876543210,
                "created-at": int(created.timestamp()),
                "updated-at": int(updated.timestamp() * 1000),
            }
        ),
        "synthetic-account",
    )

    assert order is not None
    assert order.order_id == "9876543210"
    assert isinstance(order.order_id, str)
    assert order.created_at == created
    assert order.updated_at == updated


def test_parse_order_accepts_unix_millisecond_timestamps() -> None:
    client = _client()
    created = datetime(2024, 7, 3, 12, 0, 0, tzinfo=UTC)
    updated = datetime(2024, 7, 3, 12, 5, 0, tzinfo=UTC)

    order = client._parse_order(
        _order_item(
            **{
                "created-at": int(created.timestamp() * 1000),
                "updated-at": int(updated.timestamp() * 1000),
            }
        ),
        "synthetic-account",
    )

    assert order is not None
    assert order.created_at == created
    assert order.updated_at == updated


def test_parse_order_accepts_float_unix_seconds() -> None:
    client = _client()
    created = datetime(2024, 7, 3, 12, 0, 0, tzinfo=UTC)

    order = client._parse_order(
        _order_item(
            **{
                "created-at": float(created.timestamp()),
                "updated-at": float(created.timestamp()) + 60.0,
            }
        ),
        "synthetic-account",
    )

    assert order is not None
    assert order.created_at == created
    assert order.updated_at == datetime(2024, 7, 3, 12, 1, 0, tzinfo=UTC)


def test_parse_order_boolean_timestamps_do_not_crash_or_decode() -> None:
    """bool subclasses int; must not be treated as Unix timestamps."""
    client = _client()
    before = datetime.now()
    order = client._parse_order(
        _order_item(**{"created-at": True, "updated-at": False}),
        "synthetic-account",
    )
    after = datetime.now()

    assert order is not None
    # Fallback matches historical behavior: datetime.now() (naive local).
    assert before <= order.created_at <= after
    assert before <= order.updated_at <= after


def test_parse_order_missing_timestamps_use_current_time_fallback() -> None:
    client = _client()
    before = datetime.now()
    order = client._parse_order(_order_item(), "synthetic-account")
    after = datetime.now()

    assert order is not None
    assert before <= order.created_at <= after
    assert before <= order.updated_at <= after


def test_parse_order_malformed_timestamps_use_current_time_fallback() -> None:
    client = _client()
    before = datetime.now()
    order = client._parse_order(
        _order_item(**{"created-at": "not-a-date", "updated-at": "also-bad"}),
        "synthetic-account",
    )
    after = datetime.now()

    assert order is not None
    assert before <= order.created_at <= after
    assert before <= order.updated_at <= after


def test_parse_timestamp_helper_rejects_unusable_values() -> None:
    client = _client()
    assert client._parse_timestamp(None) is None
    assert client._parse_timestamp("") is None
    assert client._parse_timestamp("   ") is None
    assert client._parse_timestamp(True) is None
    assert client._parse_timestamp(False) is None
    assert client._parse_timestamp([]) is None
    assert client._parse_timestamp({}) is None
    assert client._parse_timestamp("garbage") is None
