"""Application-service composition root."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..client import get_client
from ..models import Account, Position
from ..models.transaction import Order, Transaction
from ..persistence.sqlite import PositionPilotDatabase
from ..providers.massive import MassiveProvider
from ..providers.router import FieldRouter
from ..providers.tastytrade import TastytradeProvider
from .market import MarketService, ProviderRoutedMarketSource
from .orders import OrderService
from .plans import PlansService
from .portfolio import PortfolioService
from .risk import RiskService
from .rolls import RollService
from .watchlist import WatchlistService


class TastytradePortfolioSource:
    """Checked adapter that turns swallowed transport failures into domain failures."""

    def __init__(self) -> None:
        self.client = get_client()

    def get_accounts(self) -> list[Account]:
        accounts = self.client.get_accounts()
        if not accounts:
            raise ConnectionError("Tastytrade account request failed")
        return accounts

    def get_account_balances(self, account_number: str) -> dict:
        balances = self.client.get_account_balances(account_number)
        if balances is None:
            raise ConnectionError("Tastytrade balance request failed")
        return balances

    def get_positions(self, account_number: str) -> list[Position]:
        succeeded, positions = self.client.get_positions_checked(account_number)
        if not succeeded:
            raise ConnectionError("Tastytrade position request failed")
        return positions

    def enrich_positions_greeks_batch(self, positions: list[Position]) -> list[Position]:
        return self.client.enrich_positions_greeks_batch(positions)

    def get_orders(self, account_number: str, *, limit: int = 100) -> list[Order]:
        return self.client.get_orders(account_number, limit=limit)

    def get_transactions(
        self,
        account_number: str,
        *,
        start_date=None,
        end_date=None,
    ) -> list[Transaction]:
        return self.client.get_transactions(
            account_number,
            start_date=start_date,
            end_date=end_date,
        )


def data_directory() -> Path:
    """Return the configurable local state directory."""

    return Path(
        os.getenv(
            "POSITION_PILOT_DATA_DIR",
            Path.home() / ".local" / "share" / "position-pilot",
        )
    )


@lru_cache(maxsize=1)
def get_database() -> PositionPilotDatabase:
    database = PositionPilotDatabase(
        data_directory() / "position-pilot.sqlite3",
        legacy_config_path=Path.home() / ".config" / "position-pilot" / "config.json",
        legacy_cache_directory=Path.home() / ".cache" / "position-pilot",
    )
    database.ensure_daily_backup()
    return database


@lru_cache(maxsize=1)
def get_portfolio_service() -> PortfolioService:
    return PortfolioService(
        database=get_database(),
        source=TastytradePortfolioSource(),
        field_router=get_field_router(),
    )


@lru_cache(maxsize=1)
def get_market_service() -> MarketService:
    router = get_field_router()

    def bar_source(symbol: str) -> list[dict] | None:
        value = router.resolve("stock.bars", symbol)
        if value is None:
            return None
        if isinstance(value.value, list):
            return value.value
        return None

    return MarketService(
        source=ProviderRoutedMarketSource(
            primary=get_client(),
            router=router,
        ),
        bar_source=bar_source,
    )


@lru_cache(maxsize=1)
def get_roll_service() -> RollService:
    service = RollService(
        get_database(),
        legacy_history_path=Path.home() / ".cache" / "position-pilot" / "roll_history.json",
    )
    service.migrate_legacy_cache()
    return service


@lru_cache(maxsize=1)
def get_risk_service() -> RiskService:
    return RiskService()


@lru_cache(maxsize=1)
def get_order_service() -> OrderService:
    return OrderService(database=get_database(), source=TastytradePortfolioSource())


@lru_cache(maxsize=1)
def get_plans_service() -> PlansService:
    return PlansService(get_database())


@lru_cache(maxsize=1)
def get_watchlist_service() -> WatchlistService:
    return WatchlistService(get_database(), get_market_service())


@lru_cache(maxsize=1)
def get_field_router() -> FieldRouter:
    tastytrade = TastytradeProvider(get_client())
    massive_stocks = MassiveProvider(
        api_key=os.getenv("MASSIVE_API_KEY", ""),
        capability="stocks",
    )
    massive_options = MassiveProvider(
        api_key=os.getenv("MASSIVE_API_KEY", ""),
        capability="options",
    )
    providers = {
        provider.name: provider for provider in (tastytrade, massive_stocks, massive_options)
    }
    return FieldRouter(
        providers=providers,
        routes={
            "stock.quote": ["tastytrade", "massive-stocks"],
            "stock.bars": ["massive-stocks"],
            "stock.news": ["massive-stocks"],
            "option.mark": ["tastytrade", "massive-options"],
            "option.greeks": ["tastytrade", "massive-options"],
            "option.snapshot": ["massive-options"],
            "option.bars": ["massive-options"],
            "option.quotes": ["massive-options"],
        },
    )
