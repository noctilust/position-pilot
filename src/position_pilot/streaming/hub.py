"""In-process fan-out for redacted live dashboard events."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from ..persistence.sqlite import PositionPilotDatabase
from .account import AccountStreamEvent
from .dxlink import MarketStreamEvent

PRIVATE_NORMALIZED_KEYS = {
    "accountid",
    "accountnumber",
    "complexorderid",
    "customerid",
    "externalid",
    "orderid",
    "transactionid",
    "userid",
    "username",
}


def _is_private_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    if normalized in PRIVATE_NORMALIZED_KEYS:
        return True
    return bool(
        re.search(r"(?:^|[-_])id$", key, flags=re.IGNORECASE)
        or re.search(r"(?:Id|ID)$", key)
        or normalized.endswith("username")
        or normalized.endswith("accountname")
        or normalized in {"firstname", "lastname", "fullname", "email"}
    )


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items() if not _is_private_key(str(key))}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class LiveEvent(BaseModel):
    event_type: str
    payload: dict[str, Any]
    received_at: datetime


class LiveStateHub:
    def __init__(self, database: PositionPilotDatabase) -> None:
        self.database = database
        self.subscribers: set[asyncio.Queue[LiveEvent]] = set()
        self.latest_market: dict[str, dict[str, dict[str, Any]]] = {}

    def subscribe(self) -> asyncio.Queue[LiveEvent]:
        queue: asyncio.Queue[LiveEvent] = asyncio.Queue(maxsize=100)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[LiveEvent]) -> None:
        self.subscribers.discard(queue)

    async def publish_market(self, event: MarketStreamEvent) -> None:
        self.latest_market.setdefault(event.symbol, {})[event.event_type] = event.values
        await self.publish(
            LiveEvent(
                event_type=f"market.{event.event_type}",
                payload={
                    "symbol": event.symbol,
                    "values": event.values,
                },
                received_at=datetime.now(UTC),
            )
        )

    async def publish_account(self, event: AccountStreamEvent) -> None:
        broker_number = event.data.get("account-number") or event.data.get("account_number")
        account_id = (
            self.database.account_id_for_broker_number(str(broker_number))
            if broker_number
            else None
        )
        data = _redact(event.data)
        await self.publish(
            LiveEvent(
                event_type=f"account.{event.event_type}",
                payload={"account_id": account_id, "data": data, "timestamp": event.timestamp},
                received_at=datetime.now(UTC),
            )
        )

    async def publish(self, event: LiveEvent) -> None:
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
