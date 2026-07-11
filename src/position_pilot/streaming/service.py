"""Composition service for live market/account streaming and REST authority."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from ..client.tastytrade import TastytradeClient
from .account import AccountStreamEvent
from .clients import AccountStreamerClient, DxLinkClient, StreamingSupervisor
from .dxlink import MarketStreamEvent
from .reconciliation import ReconciliationCoordinator


class TastytradeStreamingService:
    def __init__(
        self,
        *,
        client: TastytradeClient,
        reconciliation: ReconciliationCoordinator,
    ) -> None:
        self.client = client
        self.reconciliation = reconciliation
        self.stop = asyncio.Event()
        self._symbols: set[str] = set()
        self._subscription_updates: asyncio.Queue[list[str]] = asyncio.Queue(maxsize=1)
        self.status: dict[str, dict[str, str | None]] = {
            "market": {"state": "stopped", "error": None},
            "account": {"state": "stopped", "error": None},
        }

    async def run(
        self,
        *,
        account_numbers: list[str],
        symbols: list[str],
        on_market_event: Callable[[MarketStreamEvent], None | Awaitable[None]],
        on_account_event: Callable[[AccountStreamEvent], None | Awaitable[None]],
    ) -> None:
        self._symbols = set(symbols)
        market_supervisor = StreamingSupervisor(
            name="market",
            status=self.status,
            on_reconnect=self.reconciliation.on_reconnect,
        )
        account_supervisor = StreamingSupervisor(
            name="account",
            status=self.status,
            on_reconnect=self.reconciliation.on_reconnect,
        )
        market_client = DxLinkClient()
        account_client = AccountStreamerClient()

        async def market_runner() -> None:
            credentials = await asyncio.to_thread(self.client.get_quote_streamer_credentials)
            if credentials is None:
                raise ConnectionError("Tastytrade quote streamer credentials are unavailable")
            await market_client.run(
                url=credentials["url"],
                token=credentials["token"],
                symbols=sorted(self._symbols),
                on_event=on_market_event,
                stop=self.stop,
                on_connected=market_supervisor.connected,
                on_message=market_supervisor.activity,
                subscription_updates=self._subscription_updates,
            )

        async def account_runner() -> None:
            access_token = await asyncio.to_thread(self.client.get_access_token)
            if access_token is None:
                raise ConnectionError("Tastytrade access token is unavailable")

            async def handle_account_event(event: AccountStreamEvent) -> None:
                if event.sequence is not None:
                    self.reconciliation.observe_sequence(event.sequence)
                result = on_account_event(event)
                if inspect.isawaitable(result):
                    await result

            await account_client.run(
                url=self.client.account_streamer_url,
                access_token=access_token,
                account_numbers=account_numbers,
                on_event=handle_account_event,
                stop=self.stop,
                on_connected=account_supervisor.connected,
                on_message=account_supervisor.activity,
            )

        self.reconciliation.startup()
        await asyncio.gather(
            market_supervisor.run_forever(market_runner, self.stop),
            account_supervisor.run_forever(account_runner, self.stop),
        )

    def close(self) -> None:
        self.stop.set()

    def update_symbols(self, symbols: list[str]) -> bool:
        """Reset DXLink subscriptions when an authoritative snapshot changes holdings."""

        updated = set(symbols)
        if updated == self._symbols:
            return False
        self._symbols = updated
        if self._subscription_updates.full():
            self._subscription_updates.get_nowait()
        self._subscription_updates.put_nowait(sorted(updated))
        return True
