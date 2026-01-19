# Position Pilot

Position Pilot is a sophisticated Python CLI/TUI tool for options traders using Tastytrade, featuring AI-powered analysis through Claude Sonnet 4.5.

## Installation

```bash
# Install dependencies
uv install
```

## Quick Start

```bash
# Quickest way to start - launch the interactive dashboard
uv run pilot dashboard
```

This will launch the interactive TUI dashboard with your LLM-powered recommendations.

### Alternative: Activate the virtual environment

If you prefer to run commands without `uv run`, activate the virtual environment first:

```bash
# Activate venv (add this to your shell config for auto-activation)
source .venv/bin/activate

# Then run pilot directly
pilot dashboard
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
| `g` | Toggle strategies/raw view |
| `s` | Hide/show stock positions |
| `f` | Hide/show financial numbers |
| `c` | Collapse all strategies |
| `Enter` | Expand/collapse strategy |
| `a` | Generate AI recommendation for selected strategy |

**Note:** AI recommendations are generated on-demand. Press `a` on any strategy row to get a personalized recommendation with timestamp.

## Architecture

### Core Structure

- **cli.py** - Typer CLI entry point with commands like `pilot dashboard`, `pilot positions`, `pilot analyze`
- **client/tastytrade.py** - OAuth API client with 10-minute caching for market data
- **models/position.py** - Pydantic models for positions, accounts, recommendations
- **analysis/strategies.py** - Detects 25+ multi-leg option strategies
- **analysis/llm_signals.py** - Claude-powered trading recommendations
- **analysis/market.py** - IV rank and market environment analysis
- **dashboard/app.py** - Interactive Textual TUI
- **recommendation_cache.py** - Disk-based caching at `~/.cache/position-pilot/`

### Key Design Decisions

1. **LLM-First Analysis**: All recommendations use Claude Sonnet 4.5 with no rule-based fallback
2. **Singleton Pattern**: Global instances for client, analyzer, LLM analyzer
3. **Reactive UI**: Textual framework for automatic dashboard updates
4. **Dual Caching**: Memory + disk cache for market data (10-min TTL), persistent cache for AI recommendations
5. **Cost Optimization**: On-demand AI recommendations (press 'a' in dashboard) with persistent caching

### Strategy Detection

The StrategyDetector can identify complex multi-leg strategies including Iron Condors, Iron Butterflies, Verticals, Straddles, Strangles, Calendars, Diagonals, Covered Calls, Protective Puts, Collars, Jade Lizards, Synthetic positions, and Risk Reversals.

### Data Flow

```
Tastytrade API → Raw Positions
                  ↓
enrich_positions_greeks_batch() → Adds Greeks & Underlying Prices
                  ↓
detect_strategies() → Groups into multi-leg strategies
                  ↓
LLMPositionAnalyzer → Health assessments & recommendations
                  ↓
Dashboard/CLI Display
```

## Technology Stack

- Python 3.12+, Typer, Textual, Pydantic, Rich, httpx
- Anthropic Claude SDK for AI analysis
- Tastytrade OAuth for market data & positions

## Requirements

- Python 3.12+
- Tastytrade account
- Anthropic API key (for Claude-powered recommendations)

## License

Apache 2.0
