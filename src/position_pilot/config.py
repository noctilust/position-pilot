"""SQLite-backed configuration management for Position Pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .domain.factory import get_database

CONFIG_DIR = Path.home() / ".config" / "position-pilot"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "watchlist": ["SPY", "QQQ", "IWM", "VIX"],
    "default_account": None,
    "refresh_interval": 60,
    "alerts": {
        "dte_warning": 21,
        "dte_critical": 7,
        "loss_warning_pct": -25,
        "loss_critical_pct": -50,
        "profit_target_pct": 50,
    },
}


def get_settings_database():
    """Return the process-wide settings repository."""

    return get_database()


def ensure_config_dir() -> None:
    """Ensure the legacy configuration directory exists for compatibility."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    """Load all supported settings from SQLite with durable defaults."""

    database = get_settings_database()
    return {
        key: database.get_setting(key, default)
        for key, default in DEFAULT_CONFIG.items()
    }


def save_config(config: dict[str, Any]) -> None:
    """Persist supported settings atomically per key."""

    database = get_settings_database()
    for key, value in config.items():
        database.set_setting(key, value)


def get_watchlist() -> list[str]:
    """Get the current watchlist."""

    return load_config()["watchlist"]


def set_watchlist(symbols: list[str]) -> None:
    """Set the watchlist, capped at the confirmed 100-symbol scale target."""

    normalized = list(dict.fromkeys(symbol.upper() for symbol in symbols))
    if len(normalized) > 100:
        raise ValueError("Watchlist supports at most 100 symbols")
    get_settings_database().set_setting("watchlist", normalized)


def add_to_watchlist(symbol: str) -> bool:
    """Add a symbol to the watchlist if capacity permits."""

    normalized = symbol.upper()
    watchlist = get_watchlist()
    if normalized in watchlist:
        return False
    set_watchlist([*watchlist, normalized])
    return True


def remove_from_watchlist(symbol: str) -> bool:
    """Remove a symbol from the watchlist."""

    normalized = symbol.upper()
    watchlist = get_watchlist()
    if normalized not in watchlist:
        return False
    set_watchlist([item for item in watchlist if item != normalized])
    return True


def get_default_account() -> str | None:
    """Get the broker account selected for CLI compatibility."""

    return get_settings_database().get_setting("default_account")


def set_default_account(account_number: str | None) -> None:
    """Set the broker account selected for CLI compatibility."""

    get_settings_database().set_setting("default_account", account_number)
