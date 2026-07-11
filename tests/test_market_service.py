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
