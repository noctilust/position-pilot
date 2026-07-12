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
        """Batch primary quotes first, then resolve every miss via FieldRouter/get_quote.

        Healthy full-primary batches stay one bounded call. Missing/failed symbols
        (absent keys, empty payloads, or unusable marks) fall through to the router
        so Massive provenance and fallback_reason are preserved.
        """

        normalized = [str(symbol).upper() for symbol in symbols]
        out: dict[str, dict] = {}
        batch = getattr(self.primary, "get_quotes_batch", None)
        if callable(batch) and normalized:
            try:
                raw = batch(list(normalized), force_refresh=force_refresh) or {}
            except Exception:
                raw = {}
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
        for symbol in missing:
            quote = self.get_quote(symbol, force_refresh=force_refresh)
            if quote is not None:
                out[symbol] = quote
        return out

    def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
        try:
            return self.primary.get_market_metrics(symbol, force_refresh=force_refresh)
        except Exception:
            return None

    def get_market_metrics_batch(
        self, symbols: list[str], force_refresh: bool = False
    ) -> dict[str, dict]:
        batch = getattr(self.primary, "get_market_metrics_batch", None)
        if callable(batch):
            try:
                return batch(symbols, force_refresh=force_refresh) or {}
            except Exception:
                return {}
        result: dict[str, dict] = {}
        for symbol in symbols:
            metrics = self.get_market_metrics(symbol, force_refresh=force_refresh)
            if metrics is not None:
                result[symbol] = metrics
        return result


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
            metrics = metrics_map.get(symbol) or {}
            quote_provenance = quote.pop("__provenance__", {}) if isinstance(quote, dict) else {}
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
                    provenance={
                        field: FieldProvenance(
                            provider=(
                                quote_provider if field in {"price", "bid", "ask"} else "tastytrade"
                            ),
                            observed_at=(
                                quote_observed_at
                                if field in {"price", "bid", "ask"}
                                else observed_at
                            ),
                            field=field,
                            fallback_reason=(
                                fallback_reason if field in {"price", "bid", "ask"} else None
                            ),
                        )
                        for field, value in fields.items()
                        if value is not None
                    },
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
