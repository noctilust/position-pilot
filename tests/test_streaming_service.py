import asyncio
from typing import Any

from position_pilot.streaming import service as service_module
from position_pilot.streaming.service import TastytradeStreamingService


def test_streaming_service_refreshes_credentials_for_every_connection(monkeypatch) -> None:
    market_tokens: list[str] = []
    account_tokens: list[str] = []

    class FakeClient:
        account_streamer_url = "wss://accounts.example.test"

        def __init__(self) -> None:
            self.quote_calls = 0
            self.access_calls = 0

        def get_quote_streamer_credentials(self) -> dict[str, str]:
            self.quote_calls += 1
            return {
                "url": "wss://quotes.example.test",
                "token": f"quote-{self.quote_calls}",
            }

        def get_access_token(self) -> str:
            self.access_calls += 1
            return f"access-{self.access_calls}"

    class FakeReconciliation:
        def startup(self) -> None:
            pass

        def on_reconnect(self) -> None:
            pass

        def observe_sequence(self, sequence: int) -> None:
            pass

    class FakeSupervisor:
        def __init__(self, **_: Any) -> None:
            pass

        def connected(self) -> None:
            pass

        def activity(self) -> None:
            pass

        async def run_forever(self, runner, stop: asyncio.Event) -> None:
            await runner()
            await runner()
            stop.set()

    class FakeMarketClient:
        async def run(self, **kwargs: Any) -> None:
            market_tokens.append(kwargs["token"])

    class FakeAccountClient:
        async def run(self, **kwargs: Any) -> None:
            account_tokens.append(kwargs["access_token"])

    monkeypatch.setattr(service_module, "StreamingSupervisor", FakeSupervisor)
    monkeypatch.setattr(service_module, "DxLinkClient", FakeMarketClient)
    monkeypatch.setattr(service_module, "AccountStreamerClient", FakeAccountClient)

    client = FakeClient()
    streaming = TastytradeStreamingService(
        client=client,  # type: ignore[arg-type]
        reconciliation=FakeReconciliation(),  # type: ignore[arg-type]
    )
    asyncio.run(
        streaming.run(
            account_numbers=["5WT00000"],
            symbols=["SPY"],
            on_market_event=lambda _: None,
            on_account_event=lambda _: None,
        )
    )

    assert market_tokens == ["quote-1", "quote-2"]
    assert account_tokens == ["access-1", "access-2"]


def test_streaming_service_queues_only_changed_subscription_sets() -> None:
    streaming = TastytradeStreamingService(
        client=object(),  # type: ignore[arg-type]
        reconciliation=object(),  # type: ignore[arg-type]
    )
    streaming._symbols = {"SPY"}

    assert streaming.update_symbols(["SPY"]) is False
    assert streaming.update_symbols(["SPY", "SPY   260821P00500000"]) is True
    assert streaming._subscription_updates.get_nowait() == [
        "SPY",
        "SPY   260821P00500000",
    ]
