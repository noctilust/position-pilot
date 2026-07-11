"""Watchlist application service backed by SQLite settings."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..persistence.sqlite import PositionPilotDatabase
from .market import MarketService, MarketSnapshot

DEFAULT_WATCHLIST = ["SPY", "QQQ", "IWM", "VIX"]
MAX_WATCHLIST = 100


class WatchlistSnapshot(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    quotes: list[MarketSnapshot] = Field(default_factory=list)


class WatchlistService:
    """Manage watched symbols and assemble quote snapshots."""

    def __init__(self, database: PositionPilotDatabase, market_service: MarketService) -> None:
        self.database = database
        self.market = market_service

    def list_symbols(self) -> list[str]:
        value = self.database.get_setting("watchlist", DEFAULT_WATCHLIST)
        if not isinstance(value, list):
            return list(DEFAULT_WATCHLIST)
        return [str(symbol).upper() for symbol in value]

    def set_symbols(self, symbols: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
        if any(not symbol for symbol in normalized):
            raise ValueError("Watchlist symbols cannot be empty")
        if len(normalized) > MAX_WATCHLIST:
            raise ValueError(f"Watchlist supports at most {MAX_WATCHLIST} symbols")
        self.database.set_setting("watchlist", normalized)
        return normalized

    def add(self, symbol: str) -> list[str]:
        normalized = symbol.upper()
        current = self.list_symbols()
        if normalized in current:
            return current
        return self.set_symbols([*current, normalized])

    def remove(self, symbol: str) -> list[str]:
        normalized = symbol.upper()
        return self.set_symbols([item for item in self.list_symbols() if item != normalized])

    def snapshot(self, *, force_refresh: bool = False) -> WatchlistSnapshot:
        symbols = self.list_symbols()
        quotes: list[MarketSnapshot] = []
        for symbol in symbols:
            quote = self.market.snapshot(symbol, force_refresh=force_refresh)
            if quote is not None:
                quotes.append(quote)
        return WatchlistSnapshot(symbols=symbols, quotes=quotes)
