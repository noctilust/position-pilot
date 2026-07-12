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
