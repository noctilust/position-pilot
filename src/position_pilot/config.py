"""Configuration management for Position Pilot."""

import json
from pathlib import Path
from typing import Any

# Default config location
CONFIG_DIR = Path.home() / ".config" / "position-pilot"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Default configuration
DEFAULT_CONFIG = {
    "watchlist": ["SPY", "QQQ", "IWM", "VIX"],
    "default_account": None,
    "refresh_interval": 60,  # seconds
    "alerts": {
        "dte_warning": 21,
        "dte_critical": 7,
        "loss_warning_pct": -25,
        "loss_critical_pct": -50,
        "profit_target_pct": 50,
    },
}


def ensure_config_dir() -> None:
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    """Load configuration from file, creating defaults if needed."""
    ensure_config_dir()

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            # Merge with defaults to ensure all keys exist
            return {**DEFAULT_CONFIG, **config}
        except (json.JSONDecodeError, IOError):
            return DEFAULT_CONFIG.copy()

    # Create default config file
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file."""
    ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_watchlist() -> list[str]:
    """Get the current watchlist."""
    config = load_config()
    return config.get("watchlist", DEFAULT_CONFIG["watchlist"])


def set_watchlist(symbols: list[str]) -> None:
    """Set the watchlist."""
    config = load_config()
    config["watchlist"] = [s.upper() for s in symbols]
    save_config(config)


def add_to_watchlist(symbol: str) -> bool:
    """Add a symbol to watchlist. Returns True if added, False if already exists."""
    config = load_config()
    symbol = symbol.upper()
    watchlist = config.get("watchlist", [])

    if symbol in watchlist:
        return False

    watchlist.append(symbol)
    config["watchlist"] = watchlist
    save_config(config)
    return True


def remove_from_watchlist(symbol: str) -> bool:
    """Remove a symbol from watchlist. Returns True if removed, False if not found."""
    config = load_config()
    symbol = symbol.upper()
    watchlist = config.get("watchlist", [])

    if symbol not in watchlist:
        return False

    watchlist.remove(symbol)
    config["watchlist"] = watchlist
    save_config(config)
    return True


def get_default_account() -> str | None:
    """Get the default account number."""
    config = load_config()
    return config.get("default_account")


def set_default_account(account_number: str | None) -> None:
    """Set the default account number."""
    config = load_config()
    config["default_account"] = account_number
    save_config(config)
