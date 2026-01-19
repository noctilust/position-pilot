"""Persistent storage for roll history."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models.roll import RollChain, RollEvent

logger = logging.getLogger(__name__)


class RollHistory:
    """Persistent cache for roll history data."""

    VERSION = 1

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the roll history storage.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cache/position-pilot/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "position-pilot"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "roll_history.json"
        self._data: dict = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)

                # Check version
                if data.get("version") != self.VERSION:
                    logger.warning(f"Roll history version mismatch, starting fresh")
                    self._data = {"version": self.VERSION, "accounts": {}}
                    return

                self._data = data
                logger.debug(f"Loaded roll history from {self.cache_file}")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid roll history cache: {e}")
                self._data = {"version": self.VERSION, "accounts": {}}
        else:
            self._data = {"version": self.VERSION, "accounts": {}}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._data, f, indent=2)
            logger.debug(f"Saved roll history to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save roll history: {e}")

    def _get_account_data(self, account_number: str) -> dict:
        """Get account data section."""
        if "accounts" not in self._data:
            self._data["accounts"] = {}

        if account_number not in self._data["accounts"]:
            self._data["accounts"][account_number] = {
                "chains": {},
                "last_updated": None,
            }

        return self._data["accounts"][account_number]

    def _generate_chain_key(self, underlying: str, strategy_type: str) -> str:
        """Generate a unique key for a roll chain."""
        return f"{underlying}:{strategy_type}"

    def add_roll(self, roll: RollEvent, underlying: str, strategy_type: str, account_number: str) -> None:
        """Add a roll event to history.

        Args:
            roll: The roll event to add
            underlying: Underlying symbol
            strategy_type: Strategy type
            account_number: Account number
        """
        account_data = self._get_account_data(account_number)
        chain_key = self._generate_chain_key(underlying, strategy_type)

        if chain_key not in account_data["chains"]:
            # Create new chain
            account_data["chains"][chain_key] = {
                "underlying": underlying,
                "strategy_type": strategy_type,
                "original_open_date": roll.timestamp.isoformat(),
                "rolls": [],
            }

        # Add roll to chain
        account_data["chains"][chain_key]["rolls"].append(roll.to_dict())
        account_data["last_updated"] = datetime.now().isoformat()

        self._save_cache()
        logger.debug(f"Added roll {roll.roll_id} to chain {chain_key}")

    def get_chain(
        self,
        underlying: str,
        strategy_type: str,
        account_number: str
    ) -> Optional[RollChain]:
        """Get a roll chain by underlying and strategy type.

        Args:
            underlying: Underlying symbol
            strategy_type: Strategy type
            account_number: Account number

        Returns:
            RollChain if found, None otherwise
        """
        account_data = self._get_account_data(account_number)
        chain_key = self._generate_chain_key(underlying, strategy_type)

        if chain_key not in account_data["chains"]:
            return None

        chain_data = account_data["chains"][chain_key]
        return RollChain.from_dict(chain_data)

    def get_all_chains(
        self,
        account_number: str,
        symbol: Optional[str] = None
    ) -> list[RollChain]:
        """Get all roll chains for an account.

        Args:
            account_number: Account number
            symbol: Optional symbol filter

        Returns:
            List of RollChain objects
        """
        account_data = self._get_account_data(account_number)
        chains = []

        for chain_key, chain_data in account_data.get("chains", {}).items():
            chain = RollChain.from_dict(chain_data)

            # Filter by symbol if specified
            if symbol is None or chain.underlying == symbol:
                chains.append(chain)

        return chains

    def get_recent_rolls(
        self,
        account_number: str,
        days: int = 30
    ) -> list[RollEvent]:
        """Get recent rolls across all chains.

        Args:
            account_number: Account number
            days: Number of days to look back

        Returns:
            List of RollEvent objects
        """
        cutoff_date = datetime.now() - datetime.timedelta(days=days)
        recent_rolls = []

        chains = self.get_all_chains(account_number)

        for chain in chains:
            for roll in chain.rolls:
                if roll.timestamp >= cutoff_date:
                    recent_rolls.append(roll)

        # Sort by timestamp descending
        recent_rolls.sort(key=lambda r: r.timestamp, reverse=True)

        return recent_rolls

    def add_chain(self, chain: RollChain, account_number: str) -> None:
        """Add an entire roll chain to history.

        Args:
            chain: The RollChain to add
            account_number: Account number
        """
        account_data = self._get_account_data(account_number)
        chain_key = self._generate_chain_key(chain.underlying, chain.strategy_type)

        account_data["chains"][chain_key] = chain.to_dict()
        account_data["last_updated"] = datetime.now().isoformat()

        self._save_cache()
        logger.debug(f"Added chain {chain_key} with {len(chain.rolls)} rolls")

    def clear(self, account_number: Optional[str] = None) -> None:
        """Clear roll history.

        Args:
            account_number: If specified, only clear that account. Otherwise clear all.
        """
        if account_number:
            if account_number in self._data.get("accounts", {}):
                del self._data["accounts"][account_number]
                logger.info(f"Cleared roll history for account {account_number}")
        else:
            self._data["accounts"] = {}
            logger.info("Cleared all roll history")

        self._save_cache()

    def get_cache_info(self) -> dict:
        """Get information about the cache.

        Returns:
            Dictionary with cache information
        """
        accounts = self._data.get("accounts", {})

        total_chains = sum(len(acc.get("chains", {})) for acc in accounts.values())
        total_rolls = sum(
            len(chain.get("rolls", []))
            for acc in accounts.values()
            for chain in acc.get("chains", {}).values()
        )

        last_updated = None
        for acc in accounts.values():
            acc_updated = acc.get("last_updated")
            if acc_updated:
                acc_updated = datetime.fromisoformat(acc_updated)
                if last_updated is None or acc_updated > last_updated:
                    last_updated = acc_updated

        return {
            "total_accounts": len(accounts),
            "total_chains": total_chains,
            "total_rolls": total_rolls,
            "last_updated": last_updated.isoformat() if last_updated else None,
        }


# Global singleton
_roll_history: Optional[RollHistory] = None


def get_roll_history() -> RollHistory:
    """Get or create the global roll history instance.

    Returns:
        RollHistory instance
    """
    global _roll_history
    if _roll_history is None:
        _roll_history = RollHistory()
    return _roll_history
