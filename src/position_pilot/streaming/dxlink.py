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
