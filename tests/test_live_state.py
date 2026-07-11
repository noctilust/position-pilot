import asyncio

from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.streaming.account import AccountStreamEvent
from position_pilot.streaming.dxlink import MarketStreamEvent
from position_pilot.streaming.hub import LiveStateHub


def test_account_events_are_redacted_and_scoped_before_publication(tmp_path) -> None:
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
                    "username": "private-user",
                    "customerId": "customer-secret",
                    "order-id": "order-secret",
                    "transaction_id": "transaction-secret",
                    "first-name": "Private",
                    "nested": {
                        "account-number": "5WT12345",
                        "complexOrderId": "complex-secret",
                        "safe": "kept",
                    },
                },
                timestamp=123,
            )
        )
        event = await asyncio.wait_for(queue.get(), timeout=1)

        assert event.payload["account_id"] == identity.account_id
        assert event.payload["data"] == {
            "symbol": "SPY",
            "quantity": "100",
            "nested": {"safe": "kept"},
        }
        assert "5WT12345" not in event.model_dump_json()
        assert "private-user" not in event.model_dump_json()
        assert "customer-secret" not in event.model_dump_json()
        assert "order-secret" not in event.model_dump_json()
        assert "transaction-secret" not in event.model_dump_json()
        assert "complex-secret" not in event.model_dump_json()

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
