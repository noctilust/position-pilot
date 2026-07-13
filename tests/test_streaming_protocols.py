import asyncio

import pytest

from position_pilot.streaming.account import AccountStreamerProtocol, AccountStreamUnavailable
from position_pilot.streaming.clients import (
    StreamingSupervisor,
    StreamStaleError,
    _dxlink_handshake,
    _periodic_send,
    _receive,
)
from position_pilot.streaming.dxlink import (
    DxLinkProtocol,
    browser_event_symbol,
    from_dxlink_symbol,
    to_dxlink_symbol,
)


def test_dxlink_protocol_builds_official_handshake_and_decodes_compact_feed() -> None:
    assert DxLinkProtocol.setup() == {
        "type": "SETUP",
        "channel": 0,
        "version": "0.1-DXF-JS/0.3.0",
        "keepaliveTimeout": 60,
        "acceptKeepaliveTimeout": 60,
    }
    assert DxLinkProtocol.authorize("quote-token") == {
        "type": "AUTH",
        "channel": 0,
        "token": "quote-token",
    }
    subscription = DxLinkProtocol.subscription(["SPY", ".SPY260821C550"])
    assert subscription["type"] == "FEED_SUBSCRIPTION"
    assert {item["type"] for item in subscription["add"]} == {
        "Trade",
        "TradeETH",
        "Quote",
        "Greeks",
        "Summary",
    }

    events = DxLinkProtocol.decode(
        {
            "type": "FEED_DATA",
            "channel": 3,
            "data": [
                "Trade",
                ["Trade", "SPY", 559.36, 13_743_299.0, 100.0],
                "Quote",
                ["Quote", "SPY", 559.35, 559.37, 10.0, 12.0],
            ],
        }
    )
    assert events[0].event_type == "Trade"
    assert events[0].symbol == "SPY"
    assert events[0].values["price"] == 559.36
    assert events[1].values["askPrice"] == 559.37


def test_dxlink_subscription_normalizes_broker_occ_option_symbols() -> None:
    assert to_dxlink_symbol("SPY   260821P00500000") == ".SPY260821P500"
    assert to_dxlink_symbol("BRK.B 260821C00450500") == ".BRK.B260821C450.5"
    assert to_dxlink_symbol("SPY") == "SPY"

    subscription = DxLinkProtocol.subscription(["SPY", "SPY   260821P00500000"])
    subscribed_symbols = {item["symbol"] for item in subscription["add"]}
    assert subscribed_symbols == {"SPY", ".SPY260821P500"}


def test_dxlink_from_symbol_round_trips_options_and_equities() -> None:
    assert from_dxlink_symbol("SPY") == "SPY"
    assert from_dxlink_symbol("spy") == "SPY"
    assert from_dxlink_symbol(".SPY260821P500") == "SPY   260821P00500000"
    assert from_dxlink_symbol(".MU260731C1400") == "MU    260731C01400000"
    assert from_dxlink_symbol(".BRK.B260821C450.5") == "BRK.B 260821C00450500"

    # Round-trip representative OCC symbols (including decimal strike).
    samples = [
        "SPY   260821P00500000",
        "MU    260731C01400000",
        "BRK.B 260821C00450500",
        "SPY",
    ]
    for sample in samples:
        assert from_dxlink_symbol(to_dxlink_symbol(sample)) == (
            sample if " " in sample or sample == "SPY" else sample
        )
        # Equity and options round-trip to the same DXLink form.
        assert to_dxlink_symbol(from_dxlink_symbol(to_dxlink_symbol(sample))) == to_dxlink_symbol(
            sample
        )


def test_browser_event_symbol_is_stable_match_key() -> None:
    # Feed notation and padded OCC collapse to the same browser match key.
    assert browser_event_symbol(".MU260731C1400") == "MU 260731C01400000"
    assert browser_event_symbol("MU    260731C01400000") == "MU 260731C01400000"
    assert browser_event_symbol("MU 260731C01400000") == "MU 260731C01400000"
    assert browser_event_symbol("spy") == "SPY"


def test_account_streamer_protocol_connects_then_accepts_full_notifications() -> None:
    assert AccountStreamerProtocol.connect(["5WT00000"], "access-token") == {
        "action": "connect",
        "value": ["5WT00000"],
        "auth-token": "Bearer access-token",
        "request-id": 2,
    }
    assert AccountStreamerProtocol.heartbeat("access-token") == {
        "action": "heartbeat",
        "auth-token": "Bearer access-token",
        "request-id": 1,
    }
    event = AccountStreamerProtocol.decode(
        {
            "type": "CurrentPosition",
            "data": {"account-number": "5WT00000", "symbol": "SPY", "quantity": "100"},
            "timestamp": 1_688_595_114_405,
        }
    )
    assert event is not None
    assert event.event_type == "CurrentPosition"
    assert event.data["symbol"] == "SPY"

    try:
        AccountStreamerProtocol.validate_connect_response(
            {"status": "error", "action": "connect", "message": "Unknown domain"}
        )
    except AccountStreamUnavailable as error:
        assert str(error) == "Unknown domain"
    else:
        raise AssertionError("connect errors must not be treated as subscribed")


def test_supervisor_reconciles_only_after_a_successful_reconnection() -> None:
    reconnects: list[str] = []
    status = {"market": {"state": "stopped", "error": None}}
    supervisor = StreamingSupervisor(
        name="market",
        status=status,
        on_reconnect=lambda: reconnects.append("reconnected"),
    )

    supervisor.connected()
    supervisor.connected()

    assert reconnects == ["reconnected"]
    assert status["market"]["state"] == "live"
    assert status["market"]["error"] is None
    assert status["market"]["last_message_at"] is not None

    supervisor._delay = 30
    supervisor.connected()
    assert supervisor._delay == 1


def test_periodic_liveness_sender_does_not_depend_on_inbound_messages() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send(self, message: str) -> None:
            self.messages.append(message)

    async def scenario() -> list[str]:
        websocket = FakeWebSocket()
        stop = asyncio.Event()
        task = asyncio.create_task(
            _periodic_send(websocket, DxLinkProtocol.keepalive, stop, interval=0.001)
        )
        while len(websocket.messages) < 2:
            await asyncio.sleep(0.001)
        stop.set()
        await task
        return websocket.messages

    messages = asyncio.run(scenario())

    assert len(messages) >= 2
    assert all('"type": "KEEPALIVE"' in message for message in messages)


def test_receive_timeout_marks_a_silent_live_stream_stale() -> None:
    class SilentWebSocket:
        async def recv(self) -> str:
            await asyncio.Event().wait()
            return "unreachable"

    async def scenario() -> None:
        with pytest.raises(StreamStaleError):
            await _receive(SilentWebSocket(), timeout=0.001, stale=True)

    asyncio.run(scenario())


def test_dxlink_handshake_has_one_deadline_even_with_unrelated_frames() -> None:
    class NonProgressingWebSocket:
        async def send(self, message: str) -> None:
            pass

        async def recv(self) -> str:
            await asyncio.sleep(0)
            return '{"type":"KEEPALIVE","channel":0}'

    async def scenario() -> None:
        with pytest.raises(ConnectionError, match="handshake timed out"):
            await _dxlink_handshake(
                NonProgressingWebSocket(),
                token="token",
                symbols=["SPY"],
                timeout=0.001,
            )

    asyncio.run(scenario())
