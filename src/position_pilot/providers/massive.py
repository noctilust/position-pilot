"""Massive Stocks, Options, and News REST fallback adapter."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from .contracts import ProviderHealth, ProviderState, ProviderValue

MASSIVE_API_URL = "https://api.massive.com"


class MassiveProvider:
    def __init__(
        self,
        *,
        api_key: str,
        capability: str = "stocks",
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.api_key = api_key
        self.capability = capability
        self.name = f"massive-{capability}"
        self.client = client or httpx.Client(timeout=15)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._option_snapshot_cache: dict[str, ProviderValue] = {}
        self._option_negative_cache: dict[str, datetime] = {}
        self._option_snapshot_ttl = timedelta(minutes=10)
        self._option_circuit_open_until: datetime | None = None
        self._option_failure_cooldown = timedelta(seconds=30)
        self._health = ProviderHealth(
            provider=self.name,
            state=(ProviderState.DEGRADED if api_key else ProviderState.NOT_CONFIGURED),
        )

    def health(self) -> ProviderHealth:
        return self._health

    def fetch(self, field: str, symbol: str) -> ProviderValue | None:
        now = self.clock()
        if field == "stock.bars":
            return self.bars(symbol, now - timedelta(hours=72), now)
        if field == "stock.quote":
            return self.stock_snapshot(symbol)
        if field == "stock.news":
            return self.news(symbol, now - timedelta(hours=72), now)
        if field == "option.snapshot":
            return self.option_snapshot(symbol)
        if field in {"option.mark", "option.greeks"}:
            snapshot = self.option_snapshot(symbol)
            if snapshot is None or not isinstance(snapshot.value, dict):
                return None
            if field == "option.mark":
                quote = snapshot.value.get("last_quote", {})
                bid = quote.get("bid")
                ask = quote.get("ask")
                value = (bid + ask) / 2 if bid is not None and ask is not None else None
            else:
                value = dict(snapshot.value.get("greeks") or {})
                implied_volatility = snapshot.value.get("implied_volatility")
                if implied_volatility is not None:
                    value["implied_volatility"] = implied_volatility
            return (
                snapshot.model_copy(update={"field": field, "value": value})
                if value is not None
                else None
            )
        if field == "option.bars":
            return self.option_bars(symbol, now - timedelta(hours=72), now)
        if field == "option.quotes":
            return self.option_quotes(symbol, now - timedelta(hours=24), now)
        return None

    def stock_snapshot(self, symbol: str) -> ProviderValue | None:
        snapshot = self._request_value(
            "stock.quote",
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}",
            {},
        )
        if snapshot is None or not isinstance(snapshot.value, dict):
            return snapshot
        nested_ticker = snapshot.value.get("ticker")
        ticker = nested_ticker if isinstance(nested_ticker, dict) else snapshot.value
        quote = ticker.get("lastQuote", {})
        trade = ticker.get("lastTrade", {})
        bid = quote.get("p")
        ask = quote.get("P")
        value = (bid + ask) / 2 if bid is not None and ask is not None else trade.get("p")
        return snapshot.model_copy(update={"value": value}) if value is not None else None

    def bars(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None:
        path = (
            f"/v2/aggs/ticker/{symbol.upper()}/range/1/minute/"
            f"{start.date().isoformat()}/{end.date().isoformat()}"
        )
        return self._request_value("stock.bars", path, {"adjusted": "true", "sort": "asc"})

    def news(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None:
        return self._request_value(
            "stock.news",
            "/v2/reference/news",
            {
                "ticker": symbol.upper(),
                "published_utc.gte": start.isoformat(),
                "published_utc.lte": end.isoformat(),
                "order": "desc",
                "limit": 100,
            },
        )

    def option_snapshot(self, symbol: str) -> ProviderValue | None:
        normalized = symbol if symbol.startswith("O:") else f"O:{symbol.replace(' ', '')}"
        now = self.clock()
        cached = self._option_snapshot_cache.get(normalized)
        if cached is not None and now - cached.observed_at < self._option_snapshot_ttl:
            return cached
        negative_until = self._option_negative_cache.get(normalized)
        if negative_until is not None and now < negative_until:
            return None
        if self._option_circuit_open_until is not None and now < self._option_circuit_open_until:
            return None
        match = re.match(r"O:([A-Z.]+)", normalized)
        if match is None:
            return None
        underlying = match.group(1)
        snapshot = self._request_value(
            "option.snapshot",
            f"/v3/snapshot/options/{underlying}/{normalized}",
            {},
        )
        if snapshot is not None:
            self._option_snapshot_cache[normalized] = snapshot
            self._option_negative_cache.pop(normalized, None)
            self._option_circuit_open_until = None
        elif self._health.state is ProviderState.UNAVAILABLE:
            self._option_circuit_open_until = now + self._option_failure_cooldown
        else:
            self._option_negative_cache[normalized] = now + self._option_failure_cooldown
        return snapshot

    def option_bars(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None:
        normalized = symbol if symbol.startswith("O:") else f"O:{symbol.replace(' ', '')}"
        path = (
            f"/v2/aggs/ticker/{normalized}/range/1/minute/"
            f"{start.date().isoformat()}/{end.date().isoformat()}"
        )
        return self._request_value("option.bars", path, {"sort": "asc"})

    def option_quotes(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None:
        normalized = symbol if symbol.startswith("O:") else f"O:{symbol.replace(' ', '')}"
        return self._request_value(
            "option.quotes",
            f"/v3/quotes/{normalized}",
            {
                "timestamp.gte": start.isoformat(),
                "timestamp.lte": end.isoformat(),
                "sort": "timestamp",
                "order": "asc",
                "limit": 50000,
            },
        )

    def _request_value(
        self,
        field: str,
        path: str,
        parameters: dict[str, Any],
    ) -> ProviderValue | None:
        checked_at = self.clock()
        if not self.api_key:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.NOT_CONFIGURED,
                checked_at=checked_at,
            )
            return None
        try:
            response = self.client.get(
                f"{MASSIVE_API_URL}{path}",
                params={**parameters, "apiKey": self.api_key},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            provider_wide = status_code in {401, 403, 429} or status_code >= 500
            self._health = ProviderHealth(
                provider=self.name,
                state=(ProviderState.UNAVAILABLE if provider_wide else ProviderState.DEGRADED),
                checked_at=checked_at,
                error=type(error).__name__,
            )
            return None
        except (httpx.RequestError, ValueError) as error:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=checked_at,
                error=type(error).__name__,
            )
            return None
        self._health = ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY,
            checked_at=checked_at,
            last_success_at=checked_at,
        )
        value = payload.get("results")
        if value is None:
            value = payload.get("result")
        if value is None and field == "stock.quote":
            value = payload.get("ticker")
        if value is None:
            return None
        return ProviderValue(
            field=field,
            value=value,
            provider=self.name,
            observed_at=checked_at,
        )
