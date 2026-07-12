"""Five-minute REST authority and stream gap/reconnect triggers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel


class ReconciliationStatus(BaseModel):
    last_reconciled_at: datetime | None = None
    last_sequence: int | None = None
    gap_count: int = 0
    reconnect_count: int = 0
    last_reason: str | None = None


class ReconciliationWorkQueue:
    """Coalesce bursts while preserving one trailing authoritative refresh."""

    def __init__(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self.callback = callback
        self._next_reason: str | None = None
        self._task: asyncio.Task[None] | None = None

    def submit(self, reason: str) -> None:
        self._next_reason = reason
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while self._next_reason is not None:
            reason = self._next_reason
            self._next_reason = None
            await self.callback(reason)

    async def wait_idle(self) -> None:
        if self._task is not None:
            await self._task

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class ReconciliationCoordinator:
    def __init__(
        self,
        *,
        reconcile: Callable[[str], None],
        interval: timedelta = timedelta(minutes=5),
        account_event_interval: timedelta = timedelta(seconds=5),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.reconcile = reconcile
        self.interval = interval
        self.account_event_interval = account_event_interval
        self.clock = clock or (lambda: datetime.now(UTC))
        self.status = ReconciliationStatus()
        self._last_account_event_reconcile: datetime | None = None

    def startup(self) -> None:
        self._run("startup", self.clock())

    def on_reconnect(self) -> None:
        self.status.reconnect_count += 1
        self._run("stream_reconnect", self.clock())

    def observe_sequence(self, sequence: int) -> None:
        previous = self.status.last_sequence
        if previous is not None and sequence > previous + 1:
            self.status.gap_count += 1
            self._run(f"sequence_gap:{previous}->{sequence}", self.clock())
        self.status.last_sequence = max(sequence, previous or sequence)

    def on_impossible_state(self, detail: str) -> None:
        self._run(f"impossible_state:{detail}", self.clock())

    def on_account_event(self, event_type: str) -> bool:
        """Bound account-stream bursts to at most one authoritative REST refresh."""

        now = self.clock()
        last = self._last_account_event_reconcile
        if last is not None and now - last < self.account_event_interval:
            return False
        self._last_account_event_reconcile = now
        self._run(f"account_event:{event_type}", now)
        return True

    def run_if_due(self, now: datetime | None = None) -> bool:
        current = now or self.clock()
        last = self.status.last_reconciled_at
        if last is None or current - last >= self.interval:
            self._run("scheduled_5_minute", current)
            return True
        return False

    def _run(self, reason: str, at: datetime) -> None:
        self.reconcile(reason)
        self.status.last_reconciled_at = at
        self.status.last_reason = reason
