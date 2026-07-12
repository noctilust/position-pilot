"""In-process fan-out for redacted live dashboard events."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from ..persistence.sqlite import PositionPilotDatabase
from .account import AccountStreamEvent
from .dxlink import MarketStreamEvent


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
        """Publish a minimal reconcile signal for an account stream event.

        Browser-facing payload contains only:
        - event_type (on LiveEvent)
        - opaque account_id (resolved via local DB)
        - timestamp
        - reconcile=true

        Raw broker data, numeric fields, symbols, status strings, and any
        values under allowed-looking keys are never forwarded. The browser
        reconciles portfolio state via typed REST snapshots.
        """

        raw = event.data if isinstance(event.data, dict) else {}
        broker_number = raw.get("account-number") or raw.get("account_number")
        account_id = (
            self.database.account_id_for_broker_number(str(broker_number))
            if broker_number
            else None
        )
        await self.publish(
            LiveEvent(
                event_type=f"account.{event.event_type}",
                payload={
                    "account_id": account_id,
                    "timestamp": event.timestamp,
                    "reconcile": True,
                },
                received_at=datetime.now(UTC),
            )
        )

    async def publish(self, event: LiveEvent) -> None:
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
