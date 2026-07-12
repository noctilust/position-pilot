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
        # Testable HTTP call counter for batch-bound assertions.
        self.http_call_count = 0
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
        if field in {
            "stock.metrics",
            "stock.iv",
            "stock.iv_rank",
            "stock.iv_percentile",
            "stock.liquidity",
        }:
            return self.stock_metrics(field, symbol)
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

    def stock_metrics(self, field: str, symbol: str) -> ProviderValue | None:
        """Derive IV/liquidity metrics for FieldRouter stock.* metric contracts."""

        batch = self.fetch_batch(field if field != "stock.metrics" else "stock.metrics", [symbol])
        return batch.get(str(symbol).upper())

    def fetch_batch(
        self,
        field: str,
        symbols: list[str],
        *,
        chunk_size: int = 50,
    ) -> dict[str, ProviderValue]:
        """Bounded multi-symbol fetch for quotes/metrics (chunked HTTP)."""

        if field not in {
            "stock.quote",
            "stock.metrics",
            "stock.iv",
            "stock.iv_rank",
            "stock.iv_percentile",
            "stock.liquidity",
        }:
            return {}
        normalized = [str(symbol).upper() for symbol in symbols if symbol]
        if not normalized:
            return {}
        if field == "stock.quote":
            if self.capability != "stocks":
                return {}
            return self._stocks_quotes_batch(normalized, chunk_size=chunk_size)
        if self.capability == "options":
            return self._options_metrics_batch(field, normalized, chunk_size=chunk_size)
        return self._stocks_metrics_batch(field, normalized, chunk_size=chunk_size)

    def _stocks_quotes_batch(
        self,
        symbols: list[str],
        *,
        chunk_size: int,
    ) -> dict[str, ProviderValue]:
        """Fetch stock quotes with one multi-ticker snapshot request per chunk."""

        checked_at = self.clock()
        out: dict[str, ProviderValue] = {}
        size = max(1, min(int(chunk_size), 100))
        for offset in range(0, len(symbols), size):
            chunk = symbols[offset : offset + size]
            snapshot = self._request_value(
                "stock.quote.batch",
                "/v2/snapshot/locale/us/markets/stocks/tickers",
                {"tickers": ",".join(chunk)},
            )
            rows = self._extract_ticker_rows(snapshot.value if snapshot else None)
            for symbol, row in rows:
                if symbol not in chunk:
                    continue
                quote = self._quote_from_stock_snapshot_row(row)
                if quote is None:
                    continue
                out[symbol] = ProviderValue(
                    field="stock.quote",
                    value=quote,
                    provider=self.name,
                    observed_at=checked_at,
                )
        return out

    @staticmethod
    def _quote_from_stock_snapshot_row(row: dict[str, Any]) -> dict[str, Any] | None:
        last_quote = row.get("lastQuote") if isinstance(row.get("lastQuote"), dict) else {}
        last_trade = row.get("lastTrade") if isinstance(row.get("lastTrade"), dict) else {}
        day = row.get("day") if isinstance(row.get("day"), dict) else {}
        bid = last_quote.get("p")
        ask = last_quote.get("P")
        last = last_trade.get("p")
        mark = (
            (bid + ask) / 2
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float))
            else last
        )
        if mark is None:
            close = day.get("c") or day.get("close")
            mark = close if isinstance(close, (int, float)) else None
        if mark is None:
            return None
        return {"mark": mark, "bid": bid, "ask": ask, "last": last}

    def _stocks_metrics_batch(
        self,
        field: str,
        symbols: list[str],
        *,
        chunk_size: int,
    ) -> dict[str, ProviderValue]:
        """Fetch stock snapshots in bounded chunks (one HTTP call per chunk)."""

        checked_at = self.clock()
        out: dict[str, ProviderValue] = {}
        size = max(1, min(int(chunk_size), 100))
        for offset in range(0, len(symbols), size):
            chunk = symbols[offset : offset + size]
            # Multi-ticker snapshot query — one HTTP request per chunk.
            snapshot = self._request_value(
                "stock.metrics.batch",
                "/v2/snapshot/locale/us/markets/stocks/tickers",
                {"tickers": ",".join(chunk)},
            )
            rows = self._extract_ticker_rows(snapshot.value if snapshot else None)
            by_symbol = {row_symbol: row for row_symbol, row in rows}
            for symbol in chunk:
                row = by_symbol.get(symbol)
                if row is None:
                    continue
                metrics = self._metrics_from_stock_snapshot_row(row)
                if not metrics:
                    continue
                value = self._metrics_field_value(field, metrics, checked_at=checked_at)
                if value is not None:
                    out[symbol] = value
        return out

    def _options_metrics_batch(
        self,
        field: str,
        symbols: list[str],
        *,
        chunk_size: int,
    ) -> dict[str, ProviderValue]:
        """Options batch: only concrete OCC symbols; never invent chain IV for underlyings."""

        checked_at = self.clock()
        out: dict[str, ProviderValue] = {}
        option_symbols = [s for s in symbols if s.startswith("O:")]
        if not option_symbols:
            return out
        # Chunk OCC lookups; each chunk is still multiple underlying paths but bounded.
        size = max(1, min(int(chunk_size), 50))
        for offset in range(0, len(option_symbols), size):
            chunk = option_symbols[offset : offset + size]
            for symbol in chunk:
                snap = self.option_snapshot(symbol)
                if snap is None or not isinstance(snap.value, dict):
                    continue
                metrics: dict[str, Any] = {}
                iv = snap.value.get("implied_volatility")
                if isinstance(iv, (int, float)):
                    metrics["implied_volatility"] = float(iv)
                if not metrics:
                    continue
                value = self._metrics_field_value(field, metrics, checked_at=checked_at)
                if value is not None:
                    out[symbol] = value
        return out

    @staticmethod
    def _extract_ticker_rows(payload: Any) -> list[tuple[str, dict[str, Any]]]:
        rows: list[Any]
        if payload is None:
            return []
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("tickers", "results", "ticker"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows = value
                    break
                if isinstance(value, dict) and key == "ticker":
                    rows = [value]
                    break
            else:
                rows = [payload]
        else:
            return []
        out: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or row.get("symbol") or row.get("T") or "").upper()
            nested = row.get("ticker")
            if isinstance(nested, dict) and not ticker:
                ticker = str(nested.get("ticker") or nested.get("symbol") or "").upper()
                row = nested
            if ticker:
                out.append((ticker, row))
        return out

    @staticmethod
    def _metrics_from_stock_snapshot_row(row: dict[str, Any]) -> dict[str, Any]:
        """Best-effort metrics from a Massive/Polygon stock snapshot row.

        Does not invent IV rank/percentile when the provider omits them.
        """

        metrics: dict[str, Any] = {}
        day = row.get("day") if isinstance(row.get("day"), dict) else {}
        prev = row.get("prevDay") if isinstance(row.get("prevDay"), dict) else {}
        volume = day.get("v") or day.get("volume")
        prev_volume = prev.get("v") or prev.get("volume")
        if isinstance(volume, (int, float)) and volume > 0:
            if volume >= 20_000_000:
                metrics["liquidity_rating"] = 5
            elif volume >= 5_000_000:
                metrics["liquidity_rating"] = 4
            elif volume >= 1_000_000:
                metrics["liquidity_rating"] = 3
            elif volume >= 250_000:
                metrics["liquidity_rating"] = 2
            else:
                metrics["liquidity_rating"] = 1
        elif isinstance(prev_volume, (int, float)) and prev_volume > 0:
            metrics["liquidity_rating"] = 2 if prev_volume >= 1_000_000 else 1
        # Pass through IV fields only when explicitly present (never invent).
        for src, dest in (
            ("implied_volatility", "implied_volatility"),
            ("iv_rank", "iv_rank"),
            ("iv_percentile", "iv_percentile"),
        ):
            value = row.get(src)
            if isinstance(value, (int, float)):
                metrics[dest] = float(value)
        return metrics

    def _metrics_field_value(
        self, field: str, metrics: dict[str, Any], *, checked_at: datetime
    ) -> ProviderValue | None:
        if field == "stock.metrics":
            value: object = {
                key: metrics[key]
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
            value = metrics.get(key_map.get(field, ""))
            if value is None:
                return None
        return ProviderValue(
            field=field,
            value=value,
            provider=self.name,
            observed_at=checked_at,
        )

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
            self.http_call_count += 1
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
        # Full/multi snapshot responses often use "tickers".
        if value is None and field in {
            "stock.quote.batch",
            "stock.metrics.batch",
            "stock.metrics",
        }:
            value = payload.get("tickers")
        if value is None and field in {"stock.quote.batch", "stock.metrics.batch"}:
            # Some responses nest under status/tickers; keep whole payload for row extract.
            if isinstance(payload.get("tickers"), list):
                value = payload.get("tickers")
            elif isinstance(payload, dict):
                value = payload
        if value is None:
            return None
        return ProviderValue(
            field=field,
            value=value,
            provider=self.name,
            observed_at=checked_at,
        )
