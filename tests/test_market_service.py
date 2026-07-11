from datetime import UTC, datetime

from position_pilot.domain.market import MarketService


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
