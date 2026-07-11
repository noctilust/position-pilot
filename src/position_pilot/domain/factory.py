"""Application-service composition root."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..client import get_client
from ..models import Account, Position
from ..persistence.sqlite import PositionPilotDatabase
from .portfolio import PortfolioService


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
    return PortfolioService(database=get_database(), source=TastytradePortfolioSource())
