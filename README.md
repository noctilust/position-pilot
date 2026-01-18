# Position Pilot

CLI tool and TUI dashboard for options traders using Tastytrade.

## Features

- **Strategy Detection** – Auto-groups positions (Iron Condors, Vertical Spreads, Calendars, etc.)
- **Health Monitoring** – Alerts on DTE and profit/loss thresholds
- **Interactive Dashboard** – Terminal UI with real-time data
- **Market Data** – Quotes with IV rank

## Installation

```bash
uv install
```

## Configuration

Set Tastytrade credentials:

```bash
export TASTYTRADE_USERNAME=your_username
export TASTYTRADE_PASSWORD=your_password
```

Config file: `~/.config/position-pilot/config.json`

## Commands

```bash
# Interactive dashboard (recommended)
pilot dashboard

# View positions (grouped by strategy)
pilot positions

# View positions (ungrouped)
pilot positions --raw

# Get a quote
pilot quote SPY

# Run health analysis
pilot analyze

# Market overview
pilot market

# Watchlist
pilot watchlist show
pilot watchlist add TSLA
pilot watchlist remove SPY

# Account management
pilot account list
pilot account set 12345678
```

## Dashboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Refresh |
| `Enter` | Expand/collapse strategy |
| `e` | Collapse all strategies |
| `g` | Toggle strategies/raw view |

## Requirements

- Python 3.12+
- Tastytrade account

## License

MIT
