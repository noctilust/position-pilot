# Position Pilot

CLI tool and TUI dashboard for options traders using Tastytrade with **AI-powered recommendations** via Claude.

## Features

- **🤖 AI-Powered Analysis** – Claude 3.5 Sonnet generates trading recommendations (no rule-based fallback)
- **📊 Strategy Detection** – Auto-groups positions (Iron Condors, Vertical Spreads, Synthetic Longs, etc.)
- **💰 Extrinsic & Intrinsic Value** – Shows time value and real value per contract (key metrics for option traders)
- **⚠️ Health Monitoring** – AI assesses risks based on Greeks, DTE, P/L, and market context
- **🖥️ Interactive Dashboard** – Terminal UI with real-time data
- **📈 Market Data** – Quotes with IV rank and Greeks

## Installation

```bash
uv install
```

## Configuration

Set required credentials:

```bash
# Tastytrade OAuth (get from your Tastytrade account settings)
export TASTYTRADE_CLIENT_SECRET=your_client_secret
export TASTYTRADE_REFRESH_TOKEN=your_refresh_token

# Anthropic Claude API (get from https://console.anthropic.com/)
export ANTHROPIC_API_KEY=your_api_key
```

Config file: `~/.config/position-pilot/config.json`

**Market Data Cache**: Market data (quotes, Greeks, IV metrics) is cached for 10 minutes in `~/.cache/position-pilot/` to reduce API calls and improve performance. Use `r` key in dashboard to refresh with fresh market data.

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
| `r` | Refresh all data |
| `s` | Hide/show stock positions |
| `f` | Hide/show financial numbers (account balances & P/L) |
| `Enter` | Expand/collapse strategy |
| `c` | Collapse all strategies |
| `g` | Toggle strategies/raw view |

## Requirements

- Python 3.12+
- Tastytrade account
- Anthropic API key (for Claude-powered recommendations)

## License

MIT
