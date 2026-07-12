"""Provider-backed market snapshots with explicit provenance."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..providers.router import FieldRouter
from .snapshots import DataFreshness, FieldProvenance


class IVEnvironment(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MarketSnapshot(BaseModel):
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    iv: float | None = None
    iv_rank: float | None = None
    iv_percentile: float | None = None
    liquidity_rating: int | str | None = None
    iv_environment: IVEnvironment = IVEnvironment.NORMAL
    spread_percent: float | None = None
    freshness: DataFreshness
    provenance: dict[str, FieldProvenance] = Field(default_factory=dict)


class MarketSource(Protocol):
    def get_quote(self, symbol: str, force_refresh: bool = False) -> dict | None: ...

    def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None: ...


class ProviderRoutedMarketSource:
    """Adapt field-level routing to the existing coherent market service seam."""

    def __init__(self, *, primary: MarketSource, router: FieldRouter) -> None:
        self.primary = primary
        self.router = router

    def get_quote(self, symbol: str, force_refresh: bool = False) -> dict | None:
        value = self.router.resolve("stock.quote", symbol)
        if value is None:
            return None
        quote = dict(value.value) if isinstance(value.value, dict) else {"mark": value.value}
        quote["__provenance__"] = {
            "provider": value.provider,
            "observed_at": value.observed_at,
            "fallback_reason": value.fallback_reason,
        }
        return quote

    def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict[str, dict]:
        """Batch primary quotes first, then batch every fallback-provider miss.

        Healthy full-primary batches stay one bounded call. Missing/failed symbols
        use provider batch contracts so a 100-symbol watchlist remains O(chunks).
        """

        normalized = [str(symbol).upper() for symbol in symbols]
        out: dict[str, dict] = {}
        primary_reason = "tastytrade returned no value"
        batch = getattr(self.primary, "get_quotes_batch", None)
        if callable(batch) and normalized:
            try:
                raw = batch(list(normalized), force_refresh=force_refresh) or {}
            except Exception as error:
                raw = {}
                primary_reason = f"tastytrade failed: {type(error).__name__}"
            observed = datetime.now(UTC)
            for symbol, payload in raw.items():
                key = str(symbol).upper()
                quote = dict(payload) if isinstance(payload, dict) else {"mark": payload}
                # Unusable primary rows are treated as misses so router fallback runs.
                if quote.get("mark") is None and quote.get("last") is None:
                    continue
                quote.setdefault(
                    "__provenance__",
                    {"provider": "tastytrade", "observed_at": observed},
                )
                out[key] = quote
        missing = [symbol for symbol in normalized if symbol not in out]
        route = [
            name for name in self.router.routes.get("stock.quote", []) if name != "tastytrade"
        ]
        for provider_name in route:
            if not missing:
                break
            provider = self.router.providers.get(provider_name)
            fetch_batch = getattr(provider, "fetch_batch", None) if provider is not None else None
            if not callable(fetch_batch):
                continue
            try:
                values = fetch_batch("stock.quote", list(missing), chunk_size=50) or {}
            except Exception:
                continue
            for symbol, provider_value in values.items():
                key = str(symbol).upper()
                if key not in missing:
                    continue
                value = getattr(provider_value, "value", provider_value)
                quote = dict(value) if isinstance(value, dict) else {"mark": value}
                if quote.get("mark") is None and quote.get("last") is None:
                    continue
                quote["__provenance__"] = {
                    "provider": getattr(provider_value, "provider", provider_name),
                    "observed_at": getattr(provider_value, "observed_at", datetime.now(UTC)),
                    "fallback_reason": primary_reason,
                }
                out[key] = quote
            missing = [symbol for symbol in missing if symbol not in out]
        # Preserve compatibility with simple providers for small requests without
        # allowing a failed 100-symbol primary batch to fan out into 100 calls.
        if 0 < len(missing) <= 20:
            for symbol in list(missing):
                quote = self.get_quote(symbol, force_refresh=force_refresh)
                if quote is not None:
                    out[symbol] = quote
        return out

    # IV / liquidity / options-related metric keys that may fall back via FieldRouter.
    _METRIC_FIELDS = (
        "implied_volatility",
        "iv_rank",
        "iv_percentile",
        "liquidity_rating",
    )

    def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
        """Tastytrade primary metrics, with per-field Massive fallback via FieldRouter."""

        normalized = str(symbol).upper()
        primary: dict | None = None
        primary_error: str | None = None
        try:
            primary = self.primary.get_market_metrics(normalized, force_refresh=force_refresh)
        except Exception as error:
            primary_error = f"tastytrade failed: {type(error).__name__}"
            primary = None

        result: dict[str, Any] = {}
        field_provenance: dict[str, dict[str, Any]] = {}
        observed = datetime.now(UTC)

        if isinstance(primary, dict):
            for key, value in primary.items():
                if key.startswith("__"):
                    continue
                result[key] = value
                if key in self._METRIC_FIELDS and value is not None:
                    field_provenance[key] = {
                        "provider": "tastytrade",
                        "observed_at": observed,
                        "fallback_reason": None,
                    }

        missing = [field for field in self._METRIC_FIELDS if result.get(field) is None]
        if missing:
            # Primary already consulted once — fallback skips Tastytrade re-entry.
            routed = self._resolve_metrics_via_router(
                normalized,
                missing_fields=missing,
                primary_error=primary_error,
                had_primary=bool(primary),
                skip_providers={"tastytrade"},
            )
            for field, payload in routed.items():
                if result.get(field) is None and payload.get("value") is not None:
                    result[field] = payload["value"]
                    field_provenance[field] = {
                        "provider": payload["provider"],
                        "observed_at": payload["observed_at"],
                        "fallback_reason": payload.get("fallback_reason"),
                    }

        if not result:
            return None
        result["__field_provenance__"] = field_provenance
        # Aggregate freshness: prefer primary when it supplied any metric.
        if primary and any(result.get(f) is not None for f in self._METRIC_FIELDS):
            primary_provider = "tastytrade"
            primary_reason = None
        else:
            primary_provider = next(
                (
                    field_provenance[f]["provider"]
                    for f in self._METRIC_FIELDS
                    if f in field_provenance
                ),
                "tastytrade",
            )
            primary_reason = primary_error or (
                "tastytrade returned no value" if not primary else "tastytrade incomplete metrics"
            )
        result["__provenance__"] = {
            "provider": primary_provider,
            "observed_at": observed,
            "fallback_reason": primary_reason if primary_provider != "tastytrade" else None,
        }
        return result

    def get_market_metrics_batch(
        self, symbols: list[str], force_refresh: bool = False
    ) -> dict[str, dict]:
        """Batch primary metrics once, then fill missing fields via fallback only.

        Healthy full-primary batches stay one bounded call. Partial or failed
        symbols never re-enter Tastytrade per-symbol; missing fields route
        directly to Massive (and peers) with provenance preserved.
        """

        normalized = [str(symbol).upper() for symbol in symbols]
        out: dict[str, dict] = {}
        batch = getattr(self.primary, "get_market_metrics_batch", None)
        observed = datetime.now(UTC)
        primary_error: str | None = None
        batch_attempted = False
        if callable(batch) and normalized:
            batch_attempted = True
            try:
                raw = batch(list(normalized), force_refresh=force_refresh) or {}
            except Exception as error:
                primary_error = f"tastytrade failed: {type(error).__name__}"
                raw = {}
            for symbol, payload in raw.items():
                key = str(symbol).upper()
                if not isinstance(payload, dict) or not payload:
                    continue
                metrics = {
                    field: value
                    for field, value in payload.items()
                    if not str(field).startswith("__")
                }
                if not metrics:
                    continue
                field_provenance = {
                    field: {
                        "provider": "tastytrade",
                        "observed_at": observed,
                        "fallback_reason": None,
                    }
                    for field in self._METRIC_FIELDS
                    if metrics.get(field) is not None
                }
                metrics["__field_provenance__"] = field_provenance
                metrics["__provenance__"] = {
                    "provider": "tastytrade",
                    "observed_at": observed,
                    "fallback_reason": None,
                }
                out[key] = metrics
        elif normalized and not callable(batch):
            # No batch API: one primary call path still allowed via get_market_metrics,
            # but we never fan out N primary calls when batch exists.
            for symbol in normalized:
                single = self.get_market_metrics(symbol, force_refresh=force_refresh)
                if single is not None:
                    out[symbol] = single
            return out

        # Collect symbols still missing any metric fields after the one primary batch.
        missing_symbols: list[str] = []
        missing_fields_by_symbol: dict[str, list[str]] = {}
        for symbol in normalized:
            current = out.get(symbol)
            missing = [
                field
                for field in self._METRIC_FIELDS
                if current is None or current.get(field) is None
            ]
            if missing:
                missing_symbols.append(symbol)
                missing_fields_by_symbol[symbol] = missing

        if missing_symbols:
            base_reason = primary_error or (
                "tastytrade returned no value"
                if batch_attempted
                else "tastytrade incomplete metrics"
            )
            self._fill_metrics_batch_via_fallback(
                out,
                missing_symbols=missing_symbols,
                missing_fields_by_symbol=missing_fields_by_symbol,
                base_reason=base_reason,
                observed=observed,
            )
        return out

    def _fill_metrics_batch_via_fallback(
        self,
        out: dict[str, dict],
        *,
        missing_symbols: list[str],
        missing_fields_by_symbol: dict[str, list[str]],
        base_reason: str,
        observed: datetime,
    ) -> None:
        """Fill missing metrics using provider fetch_batch in bounded chunks.

        Skips Tastytrade entirely (primary batch already ran). Prefer providers that
        implement fetch_batch so HTTP volume is O(chunks), not O(symbols×fields).
        """

        # Route order without tastytrade.
        route = [
            name for name in self.router.routes.get("stock.metrics", []) if name != "tastytrade"
        ]
        still_missing = set(missing_symbols)
        for provider_name in route:
            if not still_missing:
                break
            provider = self.router.providers.get(provider_name)
            if provider is None:
                continue
            fetch_batch = getattr(provider, "fetch_batch", None)
            batch_values: dict[str, Any] = {}
            if callable(fetch_batch):
                try:
                    batch_values = (
                        fetch_batch("stock.metrics", sorted(still_missing), chunk_size=50) or {}
                    )
                except Exception:
                    batch_values = {}
            else:
                # No batch API: still avoid per-field fan-out; one fetch per symbol max.
                for symbol in list(still_missing):
                    try:
                        value = provider.fetch("stock.metrics", symbol)
                    except Exception:
                        value = None
                    if value is not None:
                        batch_values[symbol] = value

            for symbol, pval in batch_values.items():
                key = str(symbol).upper()
                if key not in still_missing:
                    continue
                metrics_payload = pval.value if hasattr(pval, "value") else pval
                if not isinstance(metrics_payload, dict):
                    # Single-field ProviderValue — map from field name when possible.
                    field_name = getattr(pval, "field", "stock.metrics")
                    metrics_payload = self._coerce_single_metric(field_name, metrics_payload)
                if not metrics_payload:
                    continue
                current = out.get(key)
                if current is None:
                    current = {
                        "__field_provenance__": {},
                        "__provenance__": {
                            "provider": provider_name,
                            "observed_at": getattr(pval, "observed_at", observed),
                            "fallback_reason": base_reason,
                        },
                    }
                    out[key] = current
                provenance = dict(current.get("__field_provenance__") or {})
                needed = missing_fields_by_symbol.get(key, list(self._METRIC_FIELDS))
                for field in needed:
                    if current.get(field) is not None:
                        continue
                    value = metrics_payload.get(field)
                    if value is None:
                        continue
                    current[field] = value
                    provenance[field] = {
                        "provider": getattr(pval, "provider", provider_name),
                        "observed_at": getattr(pval, "observed_at", observed),
                        "fallback_reason": base_reason
                        if field in needed
                        else getattr(pval, "fallback_reason", base_reason),
                    }
                current["__field_provenance__"] = provenance
                # Drop from still_missing only when all requested fields filled or
                # provider returned something (best-effort — leave IV gaps empty).
                remaining = [
                    field
                    for field in missing_fields_by_symbol.get(key, list(self._METRIC_FIELDS))
                    if current.get(field) is None
                ]
                missing_fields_by_symbol[key] = remaining
                if not remaining:
                    still_missing.discard(key)
                else:
                    # Keep symbol for next fallback provider for residual fields.
                    still_missing.add(key)

    @staticmethod
    def _coerce_single_metric(field_name: str, value: Any) -> dict[str, Any]:
        key_map = {
            "stock.iv": "implied_volatility",
            "stock.iv_rank": "iv_rank",
            "stock.iv_percentile": "iv_percentile",
            "stock.liquidity": "liquidity_rating",
            "implied_volatility": "implied_volatility",
            "iv_rank": "iv_rank",
            "iv_percentile": "iv_percentile",
            "liquidity_rating": "liquidity_rating",
        }
        mapped = key_map.get(field_name)
        if mapped is None or value is None:
            return {}
        return {mapped: value}

    def _resolve_metrics_via_router(
        self,
        symbol: str,
        *,
        missing_fields: list[str],
        primary_error: str | None,
        had_primary: bool,
        skip_providers: set[str] | frozenset[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Resolve missing metric fields through FieldRouter stock.metrics contracts."""

        base_reason_parts = []
        if primary_error:
            base_reason_parts.append(primary_error)
        elif not had_primary:
            base_reason_parts.append("tastytrade returned no value")
        else:
            base_reason_parts.append(
                "tastytrade missing metrics: " + ",".join(sorted(missing_fields))
            )
        base_reason = "; ".join(base_reason_parts)
        skip = set(skip_providers or ())

        resolved: dict[str, dict[str, Any]] = {}
        # Prefer a composite stock.metrics value when providers implement it.
        try:
            composite = self.router.resolve(
                "stock.metrics",
                symbol,
                required_keys=set(missing_fields),
                skip_providers=skip,
            )
        except Exception:
            composite = None
        if composite is not None and isinstance(composite.value, dict):
            for field in missing_fields:
                value = composite.value.get(field)
                if value is None:
                    continue
                reason = composite.fallback_reason or base_reason
                resolved[field] = {
                    "value": value,
                    "provider": composite.provider,
                    "observed_at": composite.observed_at,
                    "fallback_reason": reason,
                }
            if len(resolved) == len(missing_fields):
                return resolved

        # Per-field routes for finer Massive Stocks / Options fallback.
        field_routes = {
            "implied_volatility": "stock.iv",
            "iv_rank": "stock.iv_rank",
            "iv_percentile": "stock.iv_percentile",
            "liquidity_rating": "stock.liquidity",
        }
        for field in missing_fields:
            if field in resolved:
                continue
            route = field_routes.get(field)
            if not route:
                continue
            try:
                value = self.router.resolve(route, symbol, skip_providers=skip)
            except Exception:
                value = None
            if value is None or value.value is None:
                continue
            reason = value.fallback_reason or base_reason
            resolved[field] = {
                "value": value.value,
                "provider": value.provider,
                "observed_at": value.observed_at,
                "fallback_reason": reason,
            }
        return resolved


class MarketBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class ChartEventMarker(BaseModel):
    """Lightweight event marker overlaid on intraday charts."""

    catalyst_id: str
    timestamp: datetime
    headline: str
    confidence: str
    attribution: str


class ChartSnapshot(BaseModel):
    symbol: str
    bars: list[MarketBar] = Field(default_factory=list)
    source: str = "unavailable"
    notice: str | None = None
    prior_close: float | None = None
    include_extended_hours: bool = True
    extended_hours_truthful: bool = False
    window_start: datetime | None = None
    window_end: datetime | None = None
    event_markers: list[ChartEventMarker] = Field(default_factory=list)
    volume_series: list[float] = Field(default_factory=list)


class MarketOverview(BaseModel):
    captured_at: datetime
    quotes: list[MarketSnapshot] = Field(default_factory=list)
    iv_summary: dict[str, int] = Field(default_factory=dict)


DEFAULT_MARKET_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA", "VIX")


class MarketService:
    """Build coherent market measurements without inventing missing values."""

    def __init__(
        self,
        *,
        source: MarketSource,
        clock: Callable[[], datetime] | None = None,
        bar_source: Callable[[str], list[dict] | None] | None = None,
    ) -> None:
        self.source = source
        self.clock = clock or (lambda: datetime.now(UTC))
        self.bar_source = bar_source

    def snapshot(self, symbol: str, *, force_refresh: bool = False) -> MarketSnapshot | None:
        batch = self.snapshots_batch([symbol], force_refresh=force_refresh)
        return batch[0] if batch else None

    def snapshots_batch(
        self, symbols: list[str], *, force_refresh: bool = False
    ) -> list[MarketSnapshot]:
        """Shared enrichment for many symbols with batch provider calls when available."""

        normalized = [symbol.upper() for symbol in symbols]
        if not normalized:
            return []
        quotes_batch_fn = getattr(self.source, "get_quotes_batch", None)
        metrics_batch_fn = getattr(self.source, "get_market_metrics_batch", None)
        if callable(quotes_batch_fn):
            quotes_map = quotes_batch_fn(normalized, force_refresh=force_refresh) or {}
        else:
            quotes_map = {}
            for symbol in normalized:
                quote = self.source.get_quote(symbol, force_refresh=force_refresh)
                if quote is not None:
                    quotes_map[symbol] = quote
        if callable(metrics_batch_fn):
            metrics_map = metrics_batch_fn(normalized, force_refresh=force_refresh) or {}
        else:
            metrics_map = {}
            for symbol in normalized:
                metrics = self.source.get_market_metrics(symbol, force_refresh=force_refresh)
                if metrics is not None:
                    metrics_map[symbol] = metrics

        observed_at = self.clock()
        snapshots: list[MarketSnapshot] = []
        for symbol in normalized:
            quote = dict(quotes_map.get(symbol) or {})
            if not quote:
                continue
            metrics = dict(metrics_map.get(symbol) or {})
            quote_provenance = quote.pop("__provenance__", {}) if isinstance(quote, dict) else {}
            metrics_field_prov = (
                metrics.pop("__field_provenance__", {}) if isinstance(metrics, dict) else {}
            )
            metrics_provenance = (
                metrics.pop("__provenance__", {}) if isinstance(metrics, dict) else {}
            )
            price = quote.get("mark") or quote.get("last")
            if price is None:
                continue
            quote_observed_at = quote_provenance.get("observed_at", observed_at)
            quote_provider = quote_provenance.get("provider", "tastytrade")
            fallback_reason = quote_provenance.get("fallback_reason")
            iv_rank = metrics.get("iv_rank")
            environment = self._iv_environment(iv_rank)
            bid = quote.get("bid")
            ask = quote.get("ask")
            spread = ((ask - bid) / price) * 100 if bid is not None and ask is not None else None
            fields = {
                "price": price,
                "bid": bid,
                "ask": ask,
                "iv": metrics.get("implied_volatility"),
                "iv_rank": iv_rank,
                "iv_percentile": metrics.get("iv_percentile"),
                "liquidity_rating": metrics.get("liquidity_rating"),
            }
            # Map snapshot field names onto metrics source field names for provenance.
            metrics_field_aliases = {
                "iv": "implied_volatility",
                "iv_rank": "iv_rank",
                "iv_percentile": "iv_percentile",
                "liquidity_rating": "liquidity_rating",
            }
            provenance: dict[str, FieldProvenance] = {}
            for field, value in fields.items():
                if value is None:
                    continue
                if field in {"price", "bid", "ask"}:
                    provenance[field] = FieldProvenance(
                        provider=quote_provider,
                        observed_at=quote_observed_at,
                        field=field,
                        fallback_reason=fallback_reason,
                    )
                    continue
                source_key = metrics_field_aliases.get(field, field)
                meta = metrics_field_prov.get(source_key) or {}
                provenance[field] = FieldProvenance(
                    provider=meta.get("provider")
                    or metrics_provenance.get("provider")
                    or "tastytrade",
                    observed_at=meta.get("observed_at")
                    or metrics_provenance.get("observed_at")
                    or observed_at,
                    field=field,
                    fallback_reason=meta.get("fallback_reason"),
                )
            snapshots.append(
                MarketSnapshot(
                    symbol=symbol,
                    price=price,
                    bid=bid,
                    ask=ask,
                    iv=fields["iv"],
                    iv_rank=iv_rank,
                    iv_percentile=fields["iv_percentile"],
                    liquidity_rating=fields["liquidity_rating"],
                    iv_environment=environment,
                    spread_percent=spread,
                    freshness=DataFreshness(as_of=quote_observed_at, provider=quote_provider),
                    provenance=provenance,
                )
            )
        return snapshots

    def overview(
        self,
        symbols: list[str] | None = None,
        *,
        force_refresh: bool = False,
    ) -> MarketOverview:
        selected = [symbol.upper() for symbol in (symbols or list(DEFAULT_MARKET_SYMBOLS))]
        quotes = self.snapshots_batch(selected, force_refresh=force_refresh)
        iv_summary = {env.value: 0 for env in IVEnvironment}
        for snap in quotes:
            iv_summary[snap.iv_environment.value] = iv_summary.get(snap.iv_environment.value, 0) + 1
        return MarketOverview(
            captured_at=self.clock(),
            quotes=quotes,
            iv_summary=iv_summary,
        )

    def chart(
        self,
        symbol: str,
        *,
        prior_close: float | None = None,
        event_markers: list[ChartEventMarker] | list[dict] | None = None,
        include_extended_hours: bool = True,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> ChartSnapshot:
        """Return bars for the requested window with volume series and prior close."""

        normalized = symbol.upper()
        markers = self._normalize_markers(event_markers)
        resolved_prior = prior_close
        end = window_end or self.clock()
        start = window_start
        if self.bar_source is not None:
            try:
                raw_bars = self.bar_source(normalized) or []
            except Exception:
                raw_bars = []
            bars: list[MarketBar] = []
            for item in raw_bars:
                try:
                    timestamp = item.get("timestamp", item.get("t"))
                    if isinstance(timestamp, (int, float)):
                        # Massive aggregate bars use Unix milliseconds.
                        if timestamp > 10_000_000_000:
                            timestamp /= 1000
                        timestamp = datetime.fromtimestamp(timestamp, tz=UTC)
                    elif isinstance(timestamp, datetime):
                        timestamp = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
                    bar = MarketBar(
                        timestamp=timestamp,
                        open=float(item["open"] if "open" in item else item["o"]),
                        high=float(item["high"] if "high" in item else item["h"]),
                        low=float(item["low"] if "low" in item else item["l"]),
                        close=float(item["close"] if "close" in item else item["c"]),
                        volume=item.get("volume", item.get("v")),
                    )
                    if start is not None and bar.timestamp < start:
                        continue
                    if end is not None and bar.timestamp > end:
                        continue
                    bars.append(bar)
                except (KeyError, OSError, OverflowError, TypeError, ValueError):
                    continue
            if bars:
                if resolved_prior is None:
                    resolved_prior = self._prior_close_from_quote(normalized)
                # Truthful extended-hours flag: bars span outside regular 09:30–16:00 ET.
                truthful = (
                    self._bars_include_extended_hours(bars) if include_extended_hours else False
                )
                volume_series = [
                    float(bar.volume) if bar.volume is not None else 0.0 for bar in bars
                ]
                return ChartSnapshot(
                    symbol=normalized,
                    bars=bars,
                    source="provider",
                    prior_close=resolved_prior,
                    include_extended_hours=include_extended_hours and truthful,
                    extended_hours_truthful=truthful,
                    window_start=start or bars[0].timestamp,
                    window_end=end or bars[-1].timestamp,
                    event_markers=markers,
                    volume_series=volume_series,
                )
        quote = self.snapshot(normalized)
        if quote is None:
            return ChartSnapshot(
                symbol=normalized,
                bars=[],
                source="unavailable",
                notice="No chart data is available for this symbol.",
                prior_close=resolved_prior,
                include_extended_hours=False,
                extended_hours_truthful=False,
                window_start=start,
                window_end=end,
                event_markers=markers,
                volume_series=[],
            )
        if resolved_prior is None:
            resolved_prior = self._prior_close_from_quote(normalized)
        return ChartSnapshot(
            symbol=normalized,
            bars=[
                MarketBar(
                    timestamp=quote.freshness.as_of,
                    open=quote.price,
                    high=quote.price,
                    low=quote.price,
                    close=quote.price,
                )
            ],
            source=quote.freshness.provider,
            notice="Only the latest mark is available; historical bars are not configured.",
            prior_close=resolved_prior,
            include_extended_hours=False,
            extended_hours_truthful=False,
            window_start=start,
            window_end=end,
            event_markers=markers,
            volume_series=[],
        )

    def _prior_close_from_quote(self, symbol: str) -> float | None:
        try:
            quote = self.source.get_quote(symbol, force_refresh=False) or {}
        except Exception:
            return None
        # Never treat live "close" as official prior close.
        for key in ("prior_close", "previous_close", "prev_close", "prevClose", "previousClose"):
            value = quote.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _bars_include_extended_hours(bars: list[MarketBar]) -> bool:
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        for bar in bars:
            local = bar.timestamp.astimezone(et)
            minutes = local.hour * 60 + local.minute
            if minutes < 9 * 60 + 30 or minutes >= 16 * 60:
                return True
        return False

    @staticmethod
    def _normalize_markers(
        event_markers: list[ChartEventMarker] | list[dict] | list[Any] | None,
    ) -> list[ChartEventMarker]:
        if not event_markers:
            return []
        markers: list[ChartEventMarker] = []
        for item in event_markers:
            if isinstance(item, ChartEventMarker):
                markers.append(item)
                continue
            payload: dict | None
            if isinstance(item, dict):
                payload = item
            elif hasattr(item, "model_dump"):
                try:
                    payload = item.model_dump(mode="json")
                except Exception:
                    payload = None
            else:
                payload = None
            if payload is None:
                continue
            try:
                markers.append(ChartEventMarker.model_validate(payload))
            except Exception:
                continue
        return markers

    @staticmethod
    def _iv_environment(iv_rank: float | None) -> IVEnvironment:
        if iv_rank is None or 30 <= iv_rank < 50:
            return IVEnvironment.NORMAL
        if iv_rank < 15:
            return IVEnvironment.VERY_LOW
        if iv_rank < 30:
            return IVEnvironment.LOW
        if iv_rank < 70:
            return IVEnvironment.ELEVATED
        if iv_rank < 85:
            return IVEnvironment.HIGH
        return IVEnvironment.VERY_HIGH
