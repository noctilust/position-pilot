"""Provider-backed market snapshots with explicit provenance."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

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

    def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
        try:
            return self.primary.get_market_metrics(symbol, force_refresh=force_refresh)
        except Exception:
            return None


class MarketBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class ChartSnapshot(BaseModel):
    symbol: str
    bars: list[MarketBar] = Field(default_factory=list)
    source: str = "unavailable"
    notice: str | None = None


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
        normalized = symbol.upper()
        quote = self.source.get_quote(normalized, force_refresh=force_refresh)
        if not quote:
            return None
        metrics = self.source.get_market_metrics(normalized, force_refresh=force_refresh) or {}
        quote_provenance = quote.pop("__provenance__", {})
        price = quote.get("mark") or quote.get("last")
        if price is None:
            return None
        observed_at = self.clock()
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
        return MarketSnapshot(
            symbol=normalized,
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
                    provider=(quote_provider if field in {"price", "bid", "ask"} else "tastytrade"),
                    observed_at=(
                        quote_observed_at if field in {"price", "bid", "ask"} else observed_at
                    ),
                    field=field,
                    fallback_reason=(fallback_reason if field in {"price", "bid", "ask"} else None),
                )
                for field, value in fields.items()
                if value is not None
            },
        )

    def overview(
        self,
        symbols: list[str] | None = None,
        *,
        force_refresh: bool = False,
    ) -> MarketOverview:
        selected = [symbol.upper() for symbol in (symbols or list(DEFAULT_MARKET_SYMBOLS))]
        quotes: list[MarketSnapshot] = []
        iv_summary = {env.value: 0 for env in IVEnvironment}
        for symbol in selected:
            snap = self.snapshot(symbol, force_refresh=force_refresh)
            if snap is None:
                continue
            quotes.append(snap)
            iv_summary[snap.iv_environment.value] = iv_summary.get(snap.iv_environment.value, 0) + 1
        return MarketOverview(
            captured_at=self.clock(),
            quotes=quotes,
            iv_summary=iv_summary,
        )

    def chart(self, symbol: str) -> ChartSnapshot:
        """Return historical bars when a bar provider is configured; else a single mark."""

        normalized = symbol.upper()
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
                    bars.append(
                        MarketBar(
                            timestamp=timestamp,
                            open=float(item["open"] if "open" in item else item["o"]),
                            high=float(item["high"] if "high" in item else item["h"]),
                            low=float(item["low"] if "low" in item else item["l"]),
                            close=float(item["close"] if "close" in item else item["c"]),
                            volume=item.get("volume", item.get("v")),
                        )
                    )
                except (KeyError, OSError, OverflowError, TypeError, ValueError):
                    continue
            if bars:
                return ChartSnapshot(symbol=normalized, bars=bars, source="provider")
        quote = self.snapshot(normalized)
        if quote is None:
            return ChartSnapshot(
                symbol=normalized,
                bars=[],
                source="unavailable",
                notice="No chart data is available for this symbol.",
            )
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
        )

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
