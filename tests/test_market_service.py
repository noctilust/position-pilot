from datetime import UTC, datetime

from position_pilot.domain.market import MarketService, ProviderRoutedMarketSource
from position_pilot.providers.contracts import ProviderHealth, ProviderState, ProviderValue
from position_pilot.providers.router import FieldRouter


class FakeMarketSource:
    def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
        return {"bid": 549.9, "ask": 550.1, "mark": 550.0, "last": 549.95}

    def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict:
        return {
            "iv_rank": 62.0,
            "iv_percentile": 71.0,
            "implied_volatility": 0.22,
            "liquidity_rating": 5,
        }


def test_market_snapshot_retains_field_provenance_and_freshness() -> None:
    service = MarketService(
        source=FakeMarketSource(),
        clock=lambda: datetime(2026, 7, 11, 16, 45, tzinfo=UTC),
    )

    snapshot = service.snapshot("spy")

    assert snapshot.symbol == "SPY"
    assert snapshot.price == 550.0
    assert snapshot.spread_percent == 0.03636363636364463
    assert snapshot.iv_environment == "elevated"
    assert snapshot.freshness.provider == "tastytrade"
    assert snapshot.provenance["price"].field == "price"
    assert snapshot.provenance["iv_rank"].provider == "tastytrade"


def test_unavailable_quote_abstains_instead_of_building_zero_price_snapshot() -> None:
    class MissingMarketSource(FakeMarketSource):
        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict | None:
            return None

    assert MarketService(source=MissingMarketSource()).snapshot("SPY") is None


def test_market_service_uses_configured_fallback_with_field_provenance() -> None:
    class EmptyProvider:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            return None

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    class FallbackProvider(EmptyProvider):
        name = "massive-stocks"

        def fetch(self, field: str, symbol: str) -> ProviderValue:
            return ProviderValue(
                field=field,
                value=551.25,
                provider=self.name,
                observed_at=datetime(2026, 7, 11, 16, 44, tzinfo=UTC),
            )

    providers = [EmptyProvider(), FallbackProvider()]
    router = FieldRouter(
        providers={provider.name: provider for provider in providers},
        routes={"stock.quote": ["tastytrade", "massive-stocks"]},
    )
    source = ProviderRoutedMarketSource(primary=FakeMarketSource(), router=router)

    snapshot = MarketService(source=source).snapshot("SPY")

    assert snapshot is not None
    assert snapshot.price == 551.25
    assert snapshot.freshness.provider == "massive-stocks"
    assert snapshot.provenance["price"].provider == "massive-stocks"
    assert snapshot.provenance["price"].fallback_reason == "tastytrade returned no value"


def test_quotes_batch_empty_primary_falls_back_via_router_with_provenance() -> None:
    """Primary batch {} must not skip Massive/router fallback for missing symbols."""

    class EmptyBatchPrimary:
        batch_calls = 0

        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict | None:
            return None

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            self.batch_calls += 1
            return {}

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
            return {
                "iv_rank": 40.0,
                "implied_volatility": 0.2,
                "iv_percentile": 50.0,
                "liquidity_rating": 4,
            }

    class EmptyTasty:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            return None

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    class MassiveFallback(EmptyTasty):
        name = "massive-stocks"

        def fetch(self, field: str, symbol: str) -> ProviderValue:
            return ProviderValue(
                field=field,
                value={"mark": 412.5, "bid": 412.4, "ask": 412.6, "last": 412.5},
                provider=self.name,
                observed_at=datetime(2026, 7, 11, 16, 50, tzinfo=UTC),
            )

    primary = EmptyBatchPrimary()
    router = FieldRouter(
        providers={
            "tastytrade": EmptyTasty(),
            "massive-stocks": MassiveFallback(),
        },
        routes={"stock.quote": ["tastytrade", "massive-stocks"]},
    )
    source = ProviderRoutedMarketSource(primary=primary, router=router)
    snapshots = MarketService(source=source).snapshots_batch(["SPY", "QQQ"])

    assert primary.batch_calls == 1
    assert len(snapshots) == 2
    assert all(snap.freshness.provider == "massive-stocks" for snap in snapshots)
    assert all(snap.price == 412.5 for snap in snapshots)
    assert snapshots[0].provenance["price"].fallback_reason == "tastytrade returned no value"


def test_quotes_batch_healthy_primary_stays_bounded_for_100_symbols() -> None:
    class CountingPrimary:
        batch_calls = 0
        quote_calls = 0

        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict | None:
            self.quote_calls += 1
            return {"mark": 1.0}

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            self.batch_calls += 1
            return {
                symbol: {"mark": 100.0 + index, "bid": 99.0, "ask": 101.0, "last": 100.0 + index}
                for index, symbol in enumerate(symbols)
            }

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
            return {"iv_rank": 30.0}

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            return {symbol: {"iv_rank": 30.0} for symbol in symbols}

    class OkProvider:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            raise AssertionError("router should not be needed for healthy primary batch")

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    primary = CountingPrimary()
    router = FieldRouter(
        providers={"tastytrade": OkProvider()},
        routes={"stock.quote": ["tastytrade"]},
    )
    source = ProviderRoutedMarketSource(primary=primary, router=router)
    symbols = [f"S{index:03d}" for index in range(100)]
    snapshots = MarketService(source=source).snapshots_batch(symbols)

    assert len(snapshots) == 100
    assert primary.batch_calls == 1
    assert primary.quote_calls == 0


def test_chart_normalizes_massive_compact_aggregate_bars() -> None:
    service = MarketService(
        source=FakeMarketSource(),
        bar_source=lambda symbol: [
            {
                "t": 1_783_785_600_000,
                "o": 548.0,
                "h": 552.0,
                "l": 547.5,
                "c": 550.0,
                "v": 1_250_000,
            }
        ],
    )

    chart = service.chart("spy")

    assert chart.source == "provider"
    assert chart.symbol == "SPY"
    assert len(chart.bars) == 1
    assert chart.bars[0].timestamp == datetime.fromtimestamp(1_783_785_600, tz=UTC)
    assert chart.bars[0].open == 548.0
    assert chart.bars[0].volume == 1_250_000


def test_market_metrics_partial_primary_fills_missing_fields_via_router() -> None:
    """Tastytrade partial metrics must fall back field-by-field with provenance."""

    class PartialPrimary:
        batch_calls = 0

        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
            return {"bid": 100.0, "ask": 100.2, "mark": 100.1, "last": 100.1}

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {
                symbol: {"bid": 100.0, "ask": 100.2, "mark": 100.1, "last": 100.1}
                for symbol in symbols
            }

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict:
            # Primary only supplies IV rank; IV and liquidity missing.
            return {"iv_rank": 55.0}

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            self.batch_calls += 1
            return {symbol: {"iv_rank": 55.0} for symbol in symbols}

    class TastyMetrics:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            return None

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.DEGRADED)

    class MassiveMetrics(TastyMetrics):
        name = "massive-stocks"

        def fetch(self, field: str, symbol: str) -> ProviderValue | None:
            if field in {"stock.liquidity", "stock.metrics"}:
                value = (
                    {"liquidity_rating": 4, "implied_volatility": 0.31}
                    if field == "stock.metrics"
                    else 4
                )
                return ProviderValue(
                    field=field,
                    value=value,
                    provider=self.name,
                    observed_at=datetime(2026, 7, 11, 16, 55, tzinfo=UTC),
                )
            if field == "stock.iv":
                return ProviderValue(
                    field=field,
                    value=0.31,
                    provider=self.name,
                    observed_at=datetime(2026, 7, 11, 16, 55, tzinfo=UTC),
                )
            return None

    primary = PartialPrimary()
    router = FieldRouter(
        providers={
            "tastytrade": TastyMetrics(),
            "massive-stocks": MassiveMetrics(),
        },
        routes={
            "stock.metrics": ["tastytrade", "massive-stocks"],
            "stock.iv": ["tastytrade", "massive-stocks"],
            "stock.iv_rank": ["tastytrade", "massive-stocks"],
            "stock.iv_percentile": ["tastytrade", "massive-stocks"],
            "stock.liquidity": ["tastytrade", "massive-stocks"],
            "stock.quote": ["tastytrade"],
        },
    )
    source = ProviderRoutedMarketSource(primary=primary, router=router)
    snapshots = MarketService(source=source).snapshots_batch(["SPY"])

    assert primary.batch_calls == 1
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.iv_rank == 55.0
    assert snap.provenance["iv_rank"].provider == "tastytrade"
    assert snap.iv == 0.31
    assert snap.provenance["iv"].provider == "massive-stocks"
    assert snap.provenance["iv"].fallback_reason
    assert snap.liquidity_rating == 4
    assert snap.provenance["liquidity_rating"].provider == "massive-stocks"


def test_market_metrics_failing_primary_uses_massive_with_reason() -> None:
    class FailingPrimary:
        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
            return {"mark": 10.0, "bid": 9.9, "ask": 10.1, "last": 10.0}

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {
                symbol: {"mark": 10.0, "bid": 9.9, "ask": 10.1, "last": 10.0} for symbol in symbols
            }

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict:
            raise ConnectionError("metrics down")

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            raise ConnectionError("metrics down")

    class EmptyTasty:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            return None

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    class MassiveOk(EmptyTasty):
        name = "massive-stocks"

        def fetch(self, field: str, symbol: str) -> ProviderValue:
            return ProviderValue(
                field=field,
                value={
                    "implied_volatility": 0.4,
                    "iv_rank": 70.0,
                    "iv_percentile": 80.0,
                    "liquidity_rating": 3,
                }
                if field == "stock.metrics"
                else 0.4,
                provider=self.name,
                observed_at=datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
            )

    router = FieldRouter(
        providers={"tastytrade": EmptyTasty(), "massive-stocks": MassiveOk()},
        routes={
            "stock.metrics": ["tastytrade", "massive-stocks"],
            "stock.iv": ["tastytrade", "massive-stocks"],
            "stock.iv_rank": ["tastytrade", "massive-stocks"],
            "stock.iv_percentile": ["tastytrade", "massive-stocks"],
            "stock.liquidity": ["tastytrade", "massive-stocks"],
        },
    )
    source = ProviderRoutedMarketSource(primary=FailingPrimary(), router=router)
    snap = MarketService(source=source).snapshot("QQQ")
    assert snap is not None
    assert snap.iv_rank == 70.0
    assert snap.provenance["iv_rank"].provider == "massive-stocks"
    assert "tastytrade" in (snap.provenance["iv_rank"].fallback_reason or "")


def test_market_metrics_healthy_batch_stays_bounded() -> None:
    class CountingPrimary:
        batch_calls = 0
        single_calls = 0

        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
            return {"mark": 1.0, "bid": 1.0, "ask": 1.1, "last": 1.0}

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {
                symbol: {"mark": 1.0, "bid": 1.0, "ask": 1.1, "last": 1.0} for symbol in symbols
            }

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict:
            self.single_calls += 1
            return {
                "iv_rank": 30.0,
                "iv_percentile": 40.0,
                "implied_volatility": 0.2,
                "liquidity_rating": 5,
            }

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            self.batch_calls += 1
            return {
                symbol: {
                    "iv_rank": 30.0,
                    "iv_percentile": 40.0,
                    "implied_volatility": 0.2,
                    "liquidity_rating": 5,
                }
                for symbol in symbols
            }

    class OkProvider:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            raise AssertionError("router must not run for complete primary metrics")

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    primary = CountingPrimary()
    router = FieldRouter(
        providers={"tastytrade": OkProvider()},
        routes={
            "stock.metrics": ["tastytrade"],
            "stock.iv": ["tastytrade"],
            "stock.iv_rank": ["tastytrade"],
            "stock.iv_percentile": ["tastytrade"],
            "stock.liquidity": ["tastytrade"],
            "stock.quote": ["tastytrade"],
        },
    )
    source = ProviderRoutedMarketSource(primary=primary, router=router)
    symbols = [f"S{index:03d}" for index in range(100)]
    snapshots = MarketService(source=source).snapshots_batch(symbols)
    assert len(snapshots) == 100
    assert primary.batch_calls == 1
    assert primary.single_calls == 0


def test_partial_metrics_batch_never_recalls_tastytrade_per_symbol() -> None:
    """100-symbol partial primary: one batch only; fallback uses fetch_batch chunks."""

    class PartialBatchPrimary:
        batch_calls = 0
        single_calls = 0

        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
            return {"mark": 10.0, "bid": 9.9, "ask": 10.1, "last": 10.0}

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {
                symbol: {"mark": 10.0, "bid": 9.9, "ask": 10.1, "last": 10.0} for symbol in symbols
            }

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict:
            self.single_calls += 1
            raise AssertionError("per-symbol primary metrics must not run after batch")

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            self.batch_calls += 1
            out = {}
            for index, symbol in enumerate(symbols):
                if index % 2 == 0:
                    out[symbol] = {"iv_rank": 40.0}
            return out

    class CountingTasty:
        name = "tastytrade"
        calls = 0

        def fetch(self, field: str, symbol: str):
            self.calls += 1
            raise AssertionError("tastytrade must be skipped in metrics fallback")

        def fetch_batch(self, field: str, symbols: list[str], *, chunk_size: int = 50):
            self.calls += 1
            raise AssertionError("tastytrade batch must be skipped in fallback")

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    class CountingMassive:
        name = "massive-stocks"
        fetch_calls = 0
        batch_calls = 0
        symbols_seen: list[str] = []

        def fetch(self, field: str, symbol: str) -> ProviderValue:
            self.fetch_calls += 1
            raise AssertionError("per-symbol fetch must not run when fetch_batch exists")

        def fetch_batch(
            self, field: str, symbols: list[str], *, chunk_size: int = 50
        ) -> dict[str, ProviderValue]:
            self.batch_calls += 1
            self.symbols_seen = list(symbols)
            # Simulate one HTTP chunk covering the whole missing set.
            out: dict[str, ProviderValue] = {}
            for symbol in symbols:
                out[symbol] = ProviderValue(
                    field="stock.metrics",
                    value={
                        "implied_volatility": 0.22,
                        "iv_rank": 41.0,
                        "iv_percentile": 50.0,
                        "liquidity_rating": 3,
                    },
                    provider=self.name,
                    observed_at=datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
                )
            return out

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    primary = PartialBatchPrimary()
    tasty = CountingTasty()
    massive = CountingMassive()
    router = FieldRouter(
        providers={"tastytrade": tasty, "massive-stocks": massive},
        routes={
            "stock.metrics": ["tastytrade", "massive-stocks"],
            "stock.iv": ["tastytrade", "massive-stocks"],
            "stock.iv_rank": ["tastytrade", "massive-stocks"],
            "stock.iv_percentile": ["tastytrade", "massive-stocks"],
            "stock.liquidity": ["tastytrade", "massive-stocks"],
            "stock.quote": ["tastytrade"],
        },
    )
    source = ProviderRoutedMarketSource(primary=primary, router=router)
    symbols = [f"S{index:03d}" for index in range(100)]
    snapshots = MarketService(source=source).snapshots_batch(symbols)

    assert primary.batch_calls == 1
    assert primary.single_calls == 0
    assert tasty.calls == 0
    assert massive.fetch_calls == 0
    # One fetch_batch invocation (provider may chunk internally; here one call).
    assert massive.batch_calls == 1
    assert len(snapshots) == 100
    even = next(s for s in snapshots if s.symbol == "S000")
    assert even.iv_rank == 40.0
    assert even.provenance["iv_rank"].provider == "tastytrade"
    assert even.provenance["iv"].provider == "massive-stocks"
    assert even.provenance["iv"].fallback_reason
    odd = next(s for s in snapshots if s.symbol == "S001")
    assert odd.iv_rank == 41.0
    assert odd.provenance["iv_rank"].provider == "massive-stocks"


def test_failing_metrics_batch_uses_fallback_without_per_symbol_primary() -> None:
    class FailingBatchPrimary:
        batch_calls = 0
        single_calls = 0

        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
            return {"mark": 1.0, "bid": 1.0, "ask": 1.1, "last": 1.0}

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {
                symbol: {"mark": 1.0, "bid": 1.0, "ask": 1.1, "last": 1.0} for symbol in symbols
            }

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict:
            self.single_calls += 1
            raise AssertionError("no per-symbol primary after failed batch")

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            self.batch_calls += 1
            raise ConnectionError("batch down")

    class EmptyTasty:
        name = "tastytrade"
        calls = 0

        def fetch(self, field: str, symbol: str):
            self.calls += 1
            raise AssertionError("tastytrade skipped after batch failure")

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    class MassiveOk(EmptyTasty):
        name = "massive-stocks"
        batch_calls = 0

        def fetch_batch(
            self, field: str, symbols: list[str], *, chunk_size: int = 50
        ) -> dict[str, ProviderValue]:
            self.batch_calls += 1
            return {
                symbol: ProviderValue(
                    field=field,
                    value={
                        "implied_volatility": 0.3,
                        "iv_rank": 60.0,
                        "iv_percentile": 70.0,
                        "liquidity_rating": 2,
                    },
                    provider=self.name,
                    observed_at=datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
                )
                for symbol in symbols
            }

    primary = FailingBatchPrimary()
    tasty = EmptyTasty()
    massive = MassiveOk()
    router = FieldRouter(
        providers={"tastytrade": tasty, "massive-stocks": massive},
        routes={
            "stock.metrics": ["tastytrade", "massive-stocks"],
            "stock.iv": ["tastytrade", "massive-stocks"],
            "stock.iv_rank": ["tastytrade", "massive-stocks"],
            "stock.iv_percentile": ["tastytrade", "massive-stocks"],
            "stock.liquidity": ["tastytrade", "massive-stocks"],
            "stock.quote": ["tastytrade"],
        },
    )
    source = ProviderRoutedMarketSource(primary=primary, router=router)
    symbols = [f"T{index:03d}" for index in range(100)]
    snapshots = MarketService(source=source).snapshots_batch(symbols)
    assert primary.batch_calls == 1
    assert primary.single_calls == 0
    assert tasty.calls == 0
    assert massive.batch_calls == 1
    assert len(snapshots) == 100
    assert all(s.provenance["iv_rank"].provider == "massive-stocks" for s in snapshots)
    assert all(s.provenance["iv_rank"].fallback_reason for s in snapshots)


def test_metrics_fallback_http_bounded_by_chunk_count_not_symbols_times_fields() -> None:
    """Massive fetch_batch must be O(chunks), not O(symbols × fields)."""

    class EmptyPrimary:
        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {
                symbol: {"mark": 1.0, "bid": 1.0, "ask": 1.1, "last": 1.0} for symbol in symbols
            }

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            return {}

        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
            return {"mark": 1.0}

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
            return None

    class ChunkCountingMassive:
        name = "massive-stocks"
        http_chunks = 0
        per_symbol_fetches = 0

        def fetch(self, field: str, symbol: str):
            self.per_symbol_fetches += 1
            raise AssertionError("must use fetch_batch")

        def fetch_batch(
            self, field: str, symbols: list[str], *, chunk_size: int = 50
        ) -> dict[str, ProviderValue]:
            # Simulate real provider: one HTTP per chunk of 50.
            out: dict[str, ProviderValue] = {}
            size = chunk_size
            for offset in range(0, len(symbols), size):
                self.http_chunks += 1
                chunk = symbols[offset : offset + size]
                for symbol in chunk:
                    out[symbol] = ProviderValue(
                        field="stock.metrics",
                        value={"liquidity_rating": 3},  # IV intentionally absent
                        provider=self.name,
                        observed_at=datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
                    )
            return out

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    class NoopTasty:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            raise AssertionError("skip tastytrade")

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    massive = ChunkCountingMassive()
    router = FieldRouter(
        providers={"tastytrade": NoopTasty(), "massive-stocks": massive},
        routes={
            "stock.metrics": ["tastytrade", "massive-stocks"],
            "stock.quote": ["tastytrade"],
        },
    )
    source = ProviderRoutedMarketSource(primary=EmptyPrimary(), router=router)
    symbols = [f"U{index:03d}" for index in range(100)]
    snapshots = MarketService(source=source).snapshots_batch(symbols)
    assert len(snapshots) == 100
    assert massive.per_symbol_fetches == 0
    # 100 symbols / chunk 50 => 2 HTTP-equivalent chunks (not 100 or 400).
    assert massive.http_chunks == 2
    assert all(s.liquidity_rating == 3 for s in snapshots)
    # Missing IV fields stay missing — no invention.
    assert all(s.iv is None for s in snapshots)
    assert all(s.iv_rank is None for s in snapshots)


def test_quote_fallback_http_bounded_by_chunk_count_for_100_symbols() -> None:
    class EmptyPrimary:
        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {}

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            return {}

    class NoopTasty:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            raise AssertionError("primary must not be retried per symbol")

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    class ChunkedMassive:
        name = "massive-stocks"
        http_chunks = 0
        per_symbol_calls = 0

        def fetch(self, field: str, symbol: str):
            self.per_symbol_calls += 1
            raise AssertionError("quote fallback must use fetch_batch")

        def fetch_batch(
            self, field: str, symbols: list[str], *, chunk_size: int = 50
        ) -> dict[str, ProviderValue]:
            assert field == "stock.quote"
            out: dict[str, ProviderValue] = {}
            for offset in range(0, len(symbols), chunk_size):
                self.http_chunks += 1
                for symbol in symbols[offset : offset + chunk_size]:
                    out[symbol] = ProviderValue(
                        field=field,
                        value={"mark": 12.5, "bid": 12.4, "ask": 12.6, "last": 12.5},
                        provider=self.name,
                        observed_at=datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
                    )
            return out

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    massive = ChunkedMassive()
    router = FieldRouter(
        providers={"tastytrade": NoopTasty(), "massive-stocks": massive},
        routes={
            "stock.quote": ["tastytrade", "massive-stocks"],
            "stock.metrics": ["tastytrade"],
        },
    )
    source = ProviderRoutedMarketSource(primary=EmptyPrimary(), router=router)
    symbols = [f"Q{index:03d}" for index in range(100)]
    snapshots = MarketService(source=source).snapshots_batch(symbols)
    assert len(snapshots) == 100
    assert massive.per_symbol_calls == 0
    assert massive.http_chunks == 2
    assert all(item.freshness.provider == "massive-stocks" for item in snapshots)
    assert all(item.provenance["price"].fallback_reason for item in snapshots)
