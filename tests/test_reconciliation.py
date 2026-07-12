import asyncio
from datetime import UTC, datetime, timedelta

from position_pilot.streaming.reconciliation import (
    ReconciliationCoordinator,
    ReconciliationWorkQueue,
)


def test_reconciliation_runs_on_startup_interval_reconnect_gap_and_impossible_state() -> None:
    reasons: list[str] = []
    now = datetime(2026, 7, 11, 17, 0, tzinfo=UTC)
    coordinator = ReconciliationCoordinator(
        reconcile=reasons.append,
        interval=timedelta(minutes=5),
        clock=lambda: now,
    )

    coordinator.startup()
    coordinator.on_reconnect()
    coordinator.observe_sequence(10)
    coordinator.observe_sequence(12)
    coordinator.on_impossible_state("negative buying power invariant")
    coordinator.run_if_due(now + timedelta(minutes=5))

    assert reasons == [
        "startup",
        "stream_reconnect",
        "sequence_gap:10->12",
        "impossible_state:negative buying power invariant",
        "scheduled_5_minute",
    ]
    assert coordinator.status.gap_count == 1
    assert coordinator.status.last_reconciled_at == now + timedelta(minutes=5)


def test_account_events_trigger_a_bounded_authoritative_refresh() -> None:
    reasons: list[str] = []
    times = iter(
        [
            datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
            datetime(2026, 7, 11, 17, 0, 1, tzinfo=UTC),
            datetime(2026, 7, 11, 17, 0, 5, tzinfo=UTC),
        ]
    )
    coordinator = ReconciliationCoordinator(reconcile=reasons.append, clock=lambda: next(times))

    assert coordinator.on_account_event("CurrentPosition") is True
    assert coordinator.on_account_event("AccountBalance") is False
    assert coordinator.on_account_event("Order") is True
    assert reasons == ["account_event:CurrentPosition", "account_event:Order"]


def test_reconciliation_work_queue_preserves_one_trailing_refresh() -> None:
    async def scenario() -> list[str]:
        reasons: list[str] = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def callback(reason: str) -> None:
            reasons.append(reason)
            if len(reasons) == 1:
                first_started.set()
                await release_first.wait()

        queue = ReconciliationWorkQueue(callback)
        queue.submit("startup")
        await first_started.wait()
        queue.submit("account_event:CurrentPosition")
        queue.submit("sequence_gap:10->12")
        release_first.set()
        await queue.wait_idle()
        return reasons

    assert asyncio.run(scenario()) == ["startup", "sequence_gap:10->12"]
