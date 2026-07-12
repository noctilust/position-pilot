"""Opt-in live Tastytrade read-only smoke test.

Skipped by default. Enable with:

    POSITION_PILOT_LIVE_SMOKE=1 uv run pytest tests/test_live_tastytrade_smoke.py -q

Performs only accounts/positions/quotes style reads. Never calls order endpoints
or any write/mutation APIs.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("POSITION_PILOT_LIVE_SMOKE", "").strip() not in {"1", "true", "yes"},
    reason="Set POSITION_PILOT_LIVE_SMOKE=1 to run live read-only Tastytrade smoke tests.",
)


def test_live_tastytrade_accounts_positions_and_quote_are_readable() -> None:
    from position_pilot.client import get_client

    client = get_client()
    assert client.is_enabled, "Tastytrade credentials must be configured for live smoke."

    accounts = client.get_accounts()
    assert accounts, "Expected at least one account from Tastytrade."

    account_number = accounts[0].account_number
    # Read-only position fetch.
    positions = client.get_positions(account_number)
    assert isinstance(positions, list)

    if positions:
        # Shared/batched enrichment path (preferred over per-leg loops).
        enriched = client.enrich_positions_greeks_batch(positions[: min(20, len(positions))])
        assert len(enriched) == min(20, len(positions))

    # Quote read for a liquid index ETF — never orders.
    quote = client.get_quote("SPY")
    assert quote is not None
    # Ensure we did not somehow receive order-shaped payloads.
    assert "order_id" not in (quote or {})
