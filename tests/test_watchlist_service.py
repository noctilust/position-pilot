import pytest

from position_pilot.domain.watchlist import WatchlistService
from position_pilot.persistence.sqlite import PositionPilotDatabase


class EmptyMarketService:
    def snapshot(self, symbol: str, *, force_refresh: bool = False):
        return None


def test_watchlist_normalizes_whitespace_and_rejects_empty_symbols(tmp_path) -> None:
    service = WatchlistService(
        PositionPilotDatabase(tmp_path / "position-pilot.sqlite3"),
        EmptyMarketService(),
    )

    assert service.set_symbols([" spy ", "SPY", " aapl"]) == ["SPY", "AAPL"]
    with pytest.raises(ValueError, match="cannot be empty"):
        service.set_symbols(["SPY", "  "])


def test_watchlist_enforces_one_hundred_symbol_limit(tmp_path) -> None:
    service = WatchlistService(
        PositionPilotDatabase(tmp_path / "position-pilot.sqlite3"),
        EmptyMarketService(),
    )

    assert len(service.set_symbols([f"S{index}" for index in range(100)])) == 100
    with pytest.raises(ValueError, match="at most 100"):
        service.set_symbols([f"S{index}" for index in range(101)])


def test_watchlist_surfaces_fallback_metrics_provenance(tmp_path) -> None:
    """Watchlist snapshots retain FieldRouter fallback provenance for metrics."""

    from datetime import UTC, datetime

    from position_pilot.domain.market import (
        MarketService,
        ProviderRoutedMarketSource,
    )
    from position_pilot.providers.contracts import ProviderHealth, ProviderState, ProviderValue
    from position_pilot.providers.router import FieldRouter

    class PartialPrimary:
        def get_quote(self, symbol: str, force_refresh: bool = False) -> dict:
            return {"mark": 50.0, "bid": 49.9, "ask": 50.1, "last": 50.0}

        def get_quotes_batch(self, symbols: list[str], force_refresh: bool = False) -> dict:
            return {s: self.get_quote(s) for s in symbols}

        def get_market_metrics(self, symbol: str, force_refresh: bool = False) -> dict | None:
            return None

        def get_market_metrics_batch(
            self, symbols: list[str], force_refresh: bool = False
        ) -> dict[str, dict]:
            return {}

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
                    "implied_volatility": 0.25,
                    "iv_rank": 40.0,
                    "iv_percentile": 45.0,
                    "liquidity_rating": 4,
                }
                if field == "stock.metrics"
                else 40.0,
                provider=self.name,
                observed_at=datetime(2026, 7, 11, 18, 0, tzinfo=UTC),
            )

    router = FieldRouter(
        providers={"tastytrade": EmptyTasty(), "massive-stocks": MassiveOk()},
        routes={
            "stock.quote": ["tastytrade", "massive-stocks"],
            "stock.metrics": ["tastytrade", "massive-stocks"],
            "stock.iv": ["tastytrade", "massive-stocks"],
            "stock.iv_rank": ["tastytrade", "massive-stocks"],
            "stock.iv_percentile": ["tastytrade", "massive-stocks"],
            "stock.liquidity": ["tastytrade", "massive-stocks"],
        },
    )
    market = MarketService(
        source=ProviderRoutedMarketSource(primary=PartialPrimary(), router=router)
    )
    service = WatchlistService(
        PositionPilotDatabase(tmp_path / "position-pilot.sqlite3"),
        market,
    )
    service.set_symbols(["TSLA"])
    snap = service.snapshot()
    assert len(snap.quotes) == 1
    assert snap.quotes[0].iv_rank == 40.0
    assert snap.quotes[0].provenance["iv_rank"].provider == "massive-stocks"
    assert snap.quotes[0].provenance["iv_rank"].fallback_reason
