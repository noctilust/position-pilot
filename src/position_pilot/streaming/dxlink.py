"""DXLink protocol messages and compact feed decoding."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

EVENT_FIELDS = {
    "Trade": ["eventType", "eventSymbol", "price", "dayVolume", "size"],
    "TradeETH": ["eventType", "eventSymbol", "price", "dayVolume", "size"],
    "Quote": [
        "eventType",
        "eventSymbol",
        "bidPrice",
        "askPrice",
        "bidSize",
        "askSize",
    ],
    "Greeks": [
        "eventType",
        "eventSymbol",
        "volatility",
        "delta",
        "gamma",
        "theta",
        "rho",
        "vega",
    ],
    "Summary": [
        "eventType",
        "eventSymbol",
        "openInterest",
        "dayOpenPrice",
        "dayHighPrice",
        "dayLowPrice",
        "prevDayClosePrice",
    ],
}
OCC_SYMBOL = re.compile(r"^([A-Z0-9.]+)\s+(\d{6})([CP])(\d{8})$")
# DXLink option: .<root><YYMMDD><C|P><strike> with optional decimal strike.
DXLINK_OPTION = re.compile(r"^\.([A-Z0-9.]+?)(\d{6})([CP])(\d+(?:\.\d+)?)$")


def to_dxlink_symbol(symbol: str) -> str:
    """Convert a broker OCC option symbol to the DXLink option notation."""

    normalized = symbol.strip().upper()
    if normalized.startswith("."):
        return normalized
    match = OCC_SYMBOL.fullmatch(normalized)
    if match is None:
        return normalized
    root, expiration, option_type, encoded_strike = match.groups()
    strike = int(encoded_strike) / 1000
    strike_text = f"{strike:.3f}".rstrip("0").rstrip(".")
    return f".{root}{expiration}{option_type}{strike_text}"


def from_dxlink_symbol(symbol: str) -> str:
    """Convert DXLink option notation to broker OCC, or pass through equities.

    Examples:
    - ``.MU260731C1400`` → ``MU    260731C01400000`` (6-char root field)
    - ``.BRK.B260821C450.5`` → ``BRK.B 260821C00450500``
    - ``SPY`` → ``SPY``
    """

    normalized = symbol.strip().upper()
    if not normalized.startswith("."):
        return normalized
    match = DXLINK_OPTION.fullmatch(normalized)
    if match is None:
        return normalized
    root, expiration, option_type, strike_text = match.groups()
    encoded_strike = int(round(float(strike_text) * 1000))
    root_field = f"{root:<6}"
    return f"{root_field}{expiration}{option_type}{encoded_strike:08d}"


def browser_event_symbol(symbol: str) -> str:
    """Stable browser-facing match key for market SSE events.

    Options are converted DXLink → OCC then whitespace-normalized so portfolio
    legs (padded or single-spaced OCC) match stream ticks without private ids.
    Equities pass through uppercased.
    """

    converted = from_dxlink_symbol(symbol)
    return " ".join(converted.split()).upper()


class MarketStreamEvent(BaseModel):
    event_type: str
    symbol: str
    values: dict[str, Any]


class DxLinkProtocol:
    @staticmethod
    def setup() -> dict:
        return {
            "type": "SETUP",
            "channel": 0,
            "version": "0.1-DXF-JS/0.3.0",
            "keepaliveTimeout": 60,
            "acceptKeepaliveTimeout": 60,
        }

    @staticmethod
    def authorize(token: str) -> dict:
        return {"type": "AUTH", "channel": 0, "token": token}

    @staticmethod
    def channel_request() -> dict:
        return {
            "type": "CHANNEL_REQUEST",
            "channel": 3,
            "service": "FEED",
            "parameters": {"contract": "AUTO"},
        }

    @staticmethod
    def feed_setup() -> dict:
        return {
            "type": "FEED_SETUP",
            "channel": 3,
            "acceptAggregationPeriod": 0.1,
            "acceptDataFormat": "COMPACT",
            "acceptEventFields": EVENT_FIELDS,
        }

    @staticmethod
    def subscription(symbols: list[str]) -> dict:
        normalized_symbols = list(dict.fromkeys(to_dxlink_symbol(symbol) for symbol in symbols))
        return {
            "type": "FEED_SUBSCRIPTION",
            "channel": 3,
            "reset": True,
            "add": [
                {"type": event_type, "symbol": symbol}
                for symbol in normalized_symbols
                for event_type in EVENT_FIELDS
            ],
        }

    @staticmethod
    def keepalive() -> dict:
        return {"type": "KEEPALIVE", "channel": 0}

    @staticmethod
    def decode(message: dict) -> list[MarketStreamEvent]:
        if message.get("type") != "FEED_DATA":
            return []
        data = message.get("data", [])
        events: list[MarketStreamEvent] = []
        for index in range(0, len(data) - 1, 2):
            event_type = data[index]
            rows = data[index + 1]
            if not isinstance(rows, list):
                continue
            rows = rows if rows and isinstance(rows[0], list) else [rows]
            fields = EVENT_FIELDS.get(event_type)
            if fields is None:
                continue
            for row in rows:
                values = dict(zip(fields, row, strict=False))
                symbol = str(values.get("eventSymbol", ""))
                events.append(
                    MarketStreamEvent(
                        event_type=event_type,
                        symbol=symbol,
                        values=values,
                    )
                )
        return events
