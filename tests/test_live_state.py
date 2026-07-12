import asyncio
import json

from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.streaming.account import AccountStreamEvent
from position_pilot.streaming.dxlink import MarketStreamEvent
from position_pilot.streaming.hub import LiveStateHub


def test_account_events_publish_only_reconcile_signal(tmp_path) -> None:
    """Account SSE is event_type + opaque account_id + timestamp + reconcile only."""

    async def scenario() -> None:
        database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
        identity = database.account_identity("5WT12345", "Individual")
        hub = LiveStateHub(database)
        queue = hub.subscribe()

        await hub.publish_account(
            AccountStreamEvent(
                event_type="CurrentPosition",
                data={
                    "account-number": "5WT12345",
                    "symbol": "SPY",
                    "quantity": "100",
                    "mark-price": 550.25,
                    "delta": -0.12,
                    "username": "private-user",
                    "customerId": "customer-secret",
                    "order-id": "order-secret",
                    "transaction_id": "transaction-secret",
                    "first-name": "Private",
                    "nickname": "Primary Trading",
                    "email": "trader@example.com",
                    "address": "1 Secret Lane",
                    # Secrets under allowed-looking keys must still never leak.
                    "status": "Bearer secret-token-value",
                    "price": "SSN-000-00-0000",
                    "quantity-direction": "exfiltrate-me",
                    "cash-balance": "account-secret-balance",
                    "nested": {
                        "account-number": "5WT12345",
                        "complexOrderId": "complex-secret",
                        "quantity": 2,
                        "symbol": "QQQ",
                    },
                    "legs": [
                        {
                            "symbol": "SPY  260821P00500000",
                            "quantity": -1,
                            "secret-token": "tok-secret",
                            "account_number": "5WT12345",
                        }
                    ],
                },
                timestamp=123,
            )
        )
        event = await asyncio.wait_for(queue.get(), timeout=1)
        blob = event.model_dump_json()
        payload = event.payload

        assert event.event_type == "account.CurrentPosition"
        assert set(payload.keys()) == {"account_id", "timestamp", "reconcile"}
        assert payload["account_id"] == identity.account_id
        assert payload["timestamp"] == 123
        assert payload["reconcile"] is True
        assert "data" not in payload
        assert "status" not in payload

        for forbidden in (
            "5WT12345",
            "private-user",
            "customer-secret",
            "order-secret",
            "transaction-secret",
            "complex-secret",
            "Primary Trading",
            "trader@example.com",
            "1 Secret Lane",
            "tok-secret",
            "Bearer secret-token-value",
            "SSN-000-00-0000",
            "exfiltrate-me",
            "account-secret-balance",
            "SPY",
            "QQQ",
            "550.25",
            "Private",
        ):
            assert forbidden not in blob, f"leaked {forbidden!r}"

    asyncio.run(scenario())


def test_account_balance_events_never_forward_broker_payload(tmp_path) -> None:
    async def scenario() -> None:
        database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
        identity = database.account_identity("ACCT999", "Margin")
        hub = LiveStateHub(database)
        queue = hub.subscribe()

        await hub.publish_account(
            AccountStreamEvent(
                event_type="AccountBalance",
                data={
                    "account-number": "ACCT999",
                    "cash-balance": 12_500.5,
                    "net-liquidating-value": 98_000.0,
                    "equity-buying-power": 40_000.0,
                    "nickname": "Nest Egg",
                    "account-name": "Forrest Brokerage",
                    "status": "active",
                },
                timestamp=999,
            )
        )
        event = await asyncio.wait_for(queue.get(), timeout=1)
        blob = json.dumps(event.model_dump(mode="json"))

        assert event.payload == {
            "account_id": identity.account_id,
            "timestamp": 999,
            "reconcile": True,
        }
        assert "ACCT999" not in blob
        assert "Nest Egg" not in blob
        assert "12500" not in blob
        assert "98000" not in blob

    asyncio.run(scenario())


def test_secrets_under_allowed_looking_keys_cannot_leak(tmp_path) -> None:
    async def scenario() -> None:
        hub = LiveStateHub(PositionPilotDatabase(tmp_path / "position-pilot.sqlite3"))
        queue = hub.subscribe()
        await hub.publish_account(
            AccountStreamEvent(
                event_type="Order",
                data={
                    "quantity": "LEAK-QUANTITY-SECRET",
                    "symbol": "LEAK-SYMBOL",
                    "status": "LEAK-STATUS",
                    "mark-price": "LEAK-MARK",
                    "delta": "LEAK-DELTA",
                    "authorization": "Bearer leak",
                    "ssn": "000-00-0000",
                },
                timestamp=1,
            )
        )
        event = await asyncio.wait_for(queue.get(), timeout=1)
        blob = event.model_dump_json()
        assert event.payload.keys() == {"account_id", "timestamp", "reconcile"}
        for forbidden in (
            "LEAK-QUANTITY-SECRET",
            "LEAK-SYMBOL",
            "LEAK-STATUS",
            "LEAK-MARK",
            "LEAK-DELTA",
            "Bearer leak",
            "000-00-0000",
        ):
            assert forbidden not in blob

    asyncio.run(scenario())


def test_market_events_update_latest_state_and_fan_out(tmp_path) -> None:
    async def scenario() -> None:
        hub = LiveStateHub(PositionPilotDatabase(tmp_path / "position-pilot.sqlite3"))
        first = hub.subscribe()
        second = hub.subscribe()
        market = MarketStreamEvent(
            event_type="Quote",
            symbol="SPY",
            values={"eventSymbol": "SPY", "bidPrice": 550.0, "askPrice": 550.1},
        )

        await hub.publish_market(market)

        assert (await first.get()).event_type == "market.Quote"
        assert (await second.get()).payload["symbol"] == "SPY"
        assert hub.latest_market["SPY"]["Quote"]["askPrice"] == 550.1

    asyncio.run(scenario())
