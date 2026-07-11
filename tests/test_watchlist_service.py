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
