"""Configuration-gated Benzinga news adapter for premium catalyst coverage."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from .contracts import ProviderHealth, ProviderState, ProviderValue

BENZINGA_NEWS_URL = "https://api.benzinga.com/api/v2/news"


class BenzingaProvider:
    """Licensed news provider used only when an API key is configured."""

    name = "benzinga"

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
        base_url: str = BENZINGA_NEWS_URL,
    ) -> None:
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=15)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.base_url = base_url
        self._health = ProviderHealth(
            provider=self.name,
            state=ProviderState.NOT_CONFIGURED if not api_key else ProviderState.DEGRADED,
        )

    def health(self) -> ProviderHealth:
        return self._health

    def fetch(self, field: str, symbol: str) -> ProviderValue | None:
        if field != "stock.news":
            return None
        now = self.clock()
        return self.news(symbol, now.replace(hour=0, minute=0, second=0, microsecond=0), now)

    def news(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None:
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
                self.base_url,
                params={
                    "token": self.api_key,
                    "tickers": symbol.upper(),
                    "displayOutput": "full",
                    "pageSize": 50,
                    "dateFrom": start.astimezone(UTC).date().isoformat(),
                    "dateTo": end.astimezone(UTC).date().isoformat(),
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            provider_wide = status_code in {401, 403, 429} or status_code >= 500
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE if provider_wide else ProviderState.DEGRADED,
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

        rows = (
            payload
            if isinstance(payload, list)
            else payload.get("data") or payload.get("result") or []
        )
        normalized = [self._normalize_row(symbol, row) for row in rows if isinstance(row, dict)]
        normalized = [row for row in normalized if row is not None]
        self._health = ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY,
            checked_at=checked_at,
            last_success_at=checked_at,
        )
        return ProviderValue(
            field="stock.news",
            value=normalized,
            provider=self.name,
            observed_at=checked_at,
        )

    def _normalize_row(self, symbol: str, row: dict[str, Any]) -> dict[str, Any] | None:
        title = str(row.get("title") or "").strip()
        url = str(row.get("url") or row.get("benzinga_url") or "").strip()
        if not title or not url:
            return None
        stocks = row.get("stocks") or []
        tickers = {
            str(item.get("name") or item.get("ticker") or "").upper()
            for item in stocks
            if isinstance(item, dict)
        }
        if tickers and symbol.upper() not in tickers:
            return None
        return {
            "id": row.get("id"),
            "title": title,
            "url": url,
            "created": row.get("created") or row.get("updated"),
            "author": row.get("author") or "Benzinga",
            "teaser": row.get("teaser") or row.get("description"),
            "body": row.get("body"),
            "publisher": "Benzinga",
            "provider": self.name,
            "source_tier": "licensed",
        }
