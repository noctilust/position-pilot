"""Tastytrade-first field provider adapter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from ..client.tastytrade import TastytradeClient
from .contracts import ProviderHealth, ProviderState, ProviderValue


class TastytradeProvider:
    name = "tastytrade"

    def __init__(
        self,
        client: TastytradeClient,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.client = client
        self.clock = clock or (lambda: datetime.now(UTC))
        self._health = ProviderHealth(
            provider=self.name,
            state=(ProviderState.DEGRADED if client.is_enabled else ProviderState.NOT_CONFIGURED),
        )

    def health(self) -> ProviderHealth:
        return self._health

    def fetch(self, field: str, symbol: str) -> ProviderValue | None:
        checked_at = self.clock()
        if not self.client.is_enabled:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.NOT_CONFIGURED,
                checked_at=checked_at,
            )
            return None

        if field in {
            "stock.metrics",
            "stock.iv",
            "stock.iv_rank",
            "stock.iv_percentile",
            "stock.liquidity",
        }:
            return self._fetch_metrics(field, symbol, checked_at=checked_at)

        quote = self.client.get_quote(symbol)
        if not quote:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=checked_at,
                error="quote_unavailable",
            )
            return None
        if field not in {"stock.quote", "option.mark", "option.greeks"}:
            return None
        field_key = "mark" if field == "option.mark" else None
        value = (
            dict(quote)
            if field == "stock.quote"
            else {
                key: quote.get(key)
                for key in ("delta", "gamma", "theta", "vega", "implied_volatility")
                if quote.get(key) is not None
            }
            if field == "option.greeks"
            else quote.get(field_key)
        )
        if value is None:
            return None
        self._health = ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY,
            checked_at=checked_at,
            last_success_at=checked_at,
        )
        return ProviderValue(
            field=field,
            value=value,
            provider=self.name,
            observed_at=checked_at,
        )

    def _fetch_metrics(
        self, field: str, symbol: str, *, checked_at: datetime
    ) -> ProviderValue | None:
        try:
            metrics = self.client.get_market_metrics(symbol) or {}
        except Exception as error:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=checked_at,
                error=type(error).__name__,
            )
            return None
        if not metrics:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=checked_at,
                error="metrics_unavailable",
            )
            return None
        self._health = ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY,
            checked_at=checked_at,
            last_success_at=checked_at,
        )
        if field == "stock.metrics":
            value: object = {
                key: metrics.get(key)
                for key in (
                    "implied_volatility",
                    "iv_rank",
                    "iv_percentile",
                    "liquidity_rating",
                )
                if metrics.get(key) is not None
            }
            if not value:
                return None
        else:
            key_map = {
                "stock.iv": "implied_volatility",
                "stock.iv_rank": "iv_rank",
                "stock.iv_percentile": "iv_percentile",
                "stock.liquidity": "liquidity_rating",
            }
            value = metrics.get(key_map[field])
            if value is None:
                return None
        return ProviderValue(
            field=field,
            value=value,
            provider=self.name,
            observed_at=checked_at,
        )
