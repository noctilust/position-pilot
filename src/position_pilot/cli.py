"""CLI entry point for Position Pilot."""

from datetime import datetime, timedelta
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .client import get_client
from .config import get_default_account, set_default_account
from .models import Position

app = typer.Typer(
    name="pilot",
    help="Position Pilot - Market analysis and position guidance",
    no_args_is_help=True,
)
console = Console()


def resolve_account(account_flag: str | None = None) -> tuple[str | None, str]:
    """
    Resolve which account to use.

    Priority: CLI flag > config default > first account
    Returns: (account_number, source) where source describes how it was resolved
    """
    if account_flag:
        return account_flag, "flag"

    default = get_default_account()
    if default:
        return default, "config"

    return None, "first"


def format_pnl(value: float, percent: float | None = None) -> Text:
    """Format P/L with color."""
    color = "green" if value >= 0 else "red"
    sign = "+" if value >= 0 else ""
    text = f"{sign}${value:,.2f}"
    if percent is not None:
        text += f" ({sign}{percent:.1f}%)"
    return Text(text, style=color)


def format_position_row(pos: Position) -> list:
    """Format a position as a table row."""
    # Symbol display
    if pos.is_option:
        exp = pos.expiration_date.strftime("%m/%d") if pos.expiration_date else "?"
        opt_type = "C" if pos.option_type == "C" else "P"
        strike = f"${pos.strike_price:.0f}" if pos.strike_price else "?"
        symbol = f"{pos.underlying_symbol} {exp} {strike}{opt_type}"
        dte = f"{pos.days_to_expiration}d" if pos.days_to_expiration is not None else "-"
    else:
        symbol = pos.symbol
        dte = "-"

    # Quantity with direction
    qty_color = "red" if pos.is_short else "green"
    qty = Text(pos.display_quantity, style=qty_color)

    # Price
    price = pos.mark_price or pos.close_price
    price_str = f"${price:.2f}" if price else "-"

    # P/L
    pnl = format_pnl(pos.unrealized_pnl, pos.unrealized_pnl_percent)

    # Greeks (if option)
    delta = f"{pos.greeks.delta:.2f}" if pos.greeks and pos.greeks.delta else "-"
    if pos.greeks and pos.greeks.theta:
        # Adjust theta for short positions (positive theta = gains from time decay)
        adjusted_theta = -pos.greeks.theta if pos.is_short else pos.greeks.theta
        theta = f"${adjusted_theta * pos.multiplier * abs(pos.quantity):.0f}"
    else:
        theta = "-"

    return [symbol, qty, dte, price_str, pnl, delta, theta]


def format_strategy_row(strategy) -> list:
    """Format a strategy as a table row."""
    from .analysis import StrategyGroup

    # Strategy name with underlying and expiration
    exp_str = ""
    if strategy.expiration:
        exp_str = strategy.expiration.strftime("%m/%d")
    elif strategy.days_to_expiration is not None:
        exp_str = f"{strategy.days_to_expiration}d"

    name = f"{strategy.underlying} {strategy.strategy_type.value}"

    # Strikes
    strikes = strategy.strikes_display

    # Quantity
    qty = str(strategy.total_quantity)

    # DTE
    dte = f"{strategy.days_to_expiration}d" if strategy.days_to_expiration is not None else "-"

    # P/L
    pnl = format_pnl(strategy.unrealized_pnl, strategy.unrealized_pnl_percent)

    # Delta
    delta = f"{strategy.total_delta:.0f}" if strategy.total_delta else "-"

    # Theta
    theta = f"${strategy.total_theta:.0f}" if strategy.total_theta else "-"

    return [name, strikes, qty, dte, pnl, delta, theta]


@app.command()
def positions(
    account: str = typer.Option(None, "--account", "-a", help="Account number (uses default if not specified)"),
    enrich: bool = typer.Option(False, "--enrich", "-e", help="Fetch live Greeks (slower)"),
    group: bool = typer.Option(False, "--group", "-g", help="Group positions by strategy"),
):
    """Show current positions."""
    from .analysis import detect_strategies

    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        console.print("Set TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN in .env")
        raise typer.Exit(1)

    with console.status("Fetching accounts..."):
        accts = client.get_accounts()

    if not accts:
        console.print("[yellow]No accounts found[/yellow]")
        raise typer.Exit(0)

    # Resolve account: CLI flag > config default > first account
    account_num, source = resolve_account(account)

    if account_num:
        selected = next((a for a in accts if a.account_number == account_num), None)
        if not selected:
            console.print(f"[red]Account {account_num} not found[/red]")
            raise typer.Exit(1)
    else:
        selected = accts[0]

    # Fetch balances and positions
    with console.status(f"Fetching positions for {selected.display_name}..."):
        balances = client.get_account_balances(selected.account_number)
        positions = client.get_positions(selected.account_number)

        if enrich:
            for i, pos in enumerate(positions):
                if pos.is_option:
                    console.status.update(f"Enriching {i+1}/{len(positions)}...")
                    positions[i] = client.enrich_position_greeks(pos)

    if balances:
        selected.net_liquidating_value = balances.get("net_liquidating_value") or 0
        selected.cash_balance = balances.get("cash_balance") or 0
        selected.buying_power = balances.get("buying_power") or 0

    selected.positions = positions

    # Display account summary
    summary = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    summary.add_column("Label", style="dim")
    summary.add_column("Value")

    summary.add_row("Net Liq", f"${selected.net_liquidating_value:,.2f}")
    summary.add_row("Cash", f"${selected.cash_balance:,.2f}")
    summary.add_row("Buying Power", f"${selected.buying_power:,.2f}")
    summary.add_row("Positions", str(len(positions)))

    console.print(Panel(summary, title=f"[bold]{selected.display_name}[/bold]", border_style="blue"))

    if not positions:
        console.print("[dim]No open positions[/dim]")
        return

    total_pnl = 0.0

    if group:
        # Detect and display strategies
        strategies = detect_strategies(positions)

        table = Table(title="Strategies", box=box.ROUNDED)
        table.add_column("Strategy", style="cyan")
        table.add_column("Strikes", style="dim")
        table.add_column("Qty", justify="right")
        table.add_column("DTE", justify="right", style="dim")
        table.add_column("P/L", justify="right")
        table.add_column("Delta", justify="right", style="dim")
        table.add_column("Theta", justify="right", style="dim")

        for strat in strategies:
            row = format_strategy_row(strat)
            table.add_row(*row)
            total_pnl += strat.unrealized_pnl

        console.print(table)
        console.print(f"\n[dim]{len(strategies)} strategies from {len(positions)} positions[/dim]")
    else:
        # Display individual positions
        table = Table(title="Open Positions", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("DTE", justify="right", style="dim")
        table.add_column("Price", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("Delta", justify="right", style="dim")
        table.add_column("Theta", justify="right", style="dim")

        # Sort: options first, then by underlying
        sorted_positions = sorted(
            positions,
            key=lambda p: (not p.is_option, p.underlying_symbol, p.expiration_date or ""),
        )

        for pos in sorted_positions:
            row = format_position_row(pos)
            table.add_row(*row)
            total_pnl += pos.unrealized_pnl

        console.print(table)

    # Total P/L
    total_text = format_pnl(total_pnl)
    console.print(f"\n[bold]Total Unrealized P/L:[/bold] ", end="")
    console.print(total_text)


# Account subcommand group
account_app = typer.Typer(
    help="Manage accounts (list, select, set, show)",
    invoke_without_command=True,
)
app.add_typer(account_app, name="account")


@account_app.callback()
def account_callback(ctx: typer.Context):
    """Manage accounts. Run without subcommand to list accounts."""
    if ctx.invoked_subcommand is None:
        account_list()


@account_app.command("list")
def account_list():
    """List all accounts."""
    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    with console.status("Fetching accounts..."):
        accts = client.get_accounts()

    if not accts:
        console.print("[yellow]No accounts found[/yellow]")
        return

    default = get_default_account()

    table = Table(title="Accounts", box=box.ROUNDED)
    table.add_column("", width=2)  # Default indicator
    table.add_column("Account", style="cyan")
    table.add_column("Type")
    table.add_column("Nickname")

    for acc in accts:
        is_default = acc.account_number == default
        indicator = "[green]✓[/green]" if is_default else ""
        table.add_row(
            indicator,
            acc.account_number,
            acc.account_type,
            acc.nickname or "-",
        )

    console.print(table)

    if default:
        console.print(f"\n[dim]Default account: {default}[/dim]")
    else:
        console.print("\n[dim]No default account set. Use 'pilot account select' to set one.[/dim]")


@account_app.command("select")
def account_select():
    """Interactively select a default account."""
    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    with console.status("Fetching accounts..."):
        accts = client.get_accounts()

    if not accts:
        console.print("[yellow]No accounts found[/yellow]")
        raise typer.Exit(1)

    current_default = get_default_account()

    console.print("\n[bold]Select an account:[/bold]\n")

    for i, acc in enumerate(accts, 1):
        is_default = acc.account_number == current_default
        marker = " [green](current)[/green]" if is_default else ""
        name = acc.nickname or acc.account_type
        console.print(f"  {i}. {acc.account_number} - {name}{marker}")

    console.print(f"  0. Clear default\n")

    choice = typer.prompt("Enter number", type=int)

    if choice == 0:
        set_default_account(None)
        console.print("[green]Default account cleared[/green]")
    elif 1 <= choice <= len(accts):
        selected = accts[choice - 1]
        set_default_account(selected.account_number)
        name = selected.nickname or selected.account_type
        console.print(f"[green]Default account set to {selected.account_number} ({name})[/green]")
    else:
        console.print("[red]Invalid selection[/red]")
        raise typer.Exit(1)


@account_app.command("set")
def account_set(account_number: str = typer.Argument(..., help="Account number to set as default")):
    """Set the default account directly."""
    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    with console.status("Validating account..."):
        accts = client.get_accounts()

    # Validate account exists
    account = next((a for a in accts if a.account_number == account_number), None)

    if not account:
        console.print(f"[red]Account {account_number} not found[/red]")
        console.print("\nAvailable accounts:")
        for acc in accts:
            console.print(f"  • {acc.account_number} ({acc.nickname or acc.account_type})")
        raise typer.Exit(1)

    set_default_account(account_number)
    name = account.nickname or account.account_type
    console.print(f"[green]Default account set to {account_number} ({name})[/green]")


@account_app.command("show")
def account_show():
    """Show the current default account."""
    default = get_default_account()

    if not default:
        console.print("[dim]No default account set[/dim]")
        console.print("Use 'pilot account select' or 'pilot account set <number>' to set one.")
        return

    client = get_client()

    if client.is_enabled:
        with console.status("Fetching account info..."):
            accts = client.get_accounts()
            account = next((a for a in accts if a.account_number == default), None)

        if account:
            name = account.nickname or account.account_type
            console.print(f"Default account: [cyan]{default}[/cyan] ({name})")
        else:
            console.print(f"Default account: [cyan]{default}[/cyan] [yellow](not found in API)[/yellow]")
    else:
        console.print(f"Default account: [cyan]{default}[/cyan]")


@app.command()
def quote(symbol: str = typer.Argument(..., help="Symbol to quote")):
    """Get quote and metrics for a symbol."""
    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    symbol = symbol.upper()

    with console.status(f"Fetching {symbol}..."):
        quote_data = client.get_quote(symbol)
        metrics = client.get_market_metrics(symbol.split()[0])  # Use underlying for metrics

    if not quote_data:
        console.print(f"[yellow]No quote data for {symbol}[/yellow]")
        return

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Bid", f"${quote_data['bid']:.2f}" if quote_data.get("bid") else "-")
    table.add_row("Ask", f"${quote_data['ask']:.2f}" if quote_data.get("ask") else "-")
    table.add_row("Mark", f"${quote_data['mark']:.2f}" if quote_data.get("mark") else "-")
    table.add_row("Last", f"${quote_data['last']:.2f}" if quote_data.get("last") else "-")

    if quote_data.get("delta"):
        table.add_row("Delta", f"{quote_data['delta']:.3f}")
    if quote_data.get("theta"):
        table.add_row("Theta", f"{quote_data['theta']:.3f}")
    if quote_data.get("implied_volatility"):
        table.add_row("IV", f"{quote_data['implied_volatility']:.1%}")

    if metrics:
        if metrics.get("iv_rank"):
            table.add_row("IV Rank", f"{metrics['iv_rank']:.1f}")
        if metrics.get("iv_percentile"):
            table.add_row("IV %ile", f"{metrics['iv_percentile']:.1f}")

    console.print(Panel(table, title=f"[bold cyan]{symbol}[/bold cyan]", border_style="blue"))


@app.command()
def dashboard():
    """Launch interactive TUI dashboard."""
    from .dashboard import run_dashboard
    run_dashboard()


@app.command()
def analyze(
    account: str = typer.Option(None, "--account", "-a", help="Account number (uses default if not specified)"),
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Force refresh AI recommendations (bypass cache)"),
):
    """Analyze portfolio health and metrics (AI recommendations are on-demand per position)."""
    from .analysis import get_llm_analyzer, get_analyzer, RiskLevel, get_recommendation_cache

    client = get_client()
    analyzer = get_llm_analyzer()  # Use LLM analyzer
    market = get_analyzer()
    cache = get_recommendation_cache()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    with console.status("Fetching data..."):
        accts = client.get_accounts()
        if not accts:
            console.print("[yellow]No accounts found[/yellow]")
            raise typer.Exit(0)

        # Resolve account: CLI flag > config default > first account
        account_num, source = resolve_account(account)

        if account_num:
            selected = next((a for a in accts if a.account_number == account_num), None)
            if not selected:
                console.print(f"[red]Account {account_num} not found[/red]")
                raise typer.Exit(1)
        else:
            selected = accts[0]

        positions = client.get_positions(selected.account_number)

        # Enrich all options with Greeks
        for i, pos in enumerate(positions):
            if pos.is_option:
                positions[i] = client.enrich_position_greeks(pos)

    if not positions:
        console.print("[dim]No open positions to analyze[/dim]")
        return

    # Run analysis (metrics only)
    analysis = analyzer.analyze_portfolio(positions)

    # Show cache info
    cache_info = analysis.get("cache_info", {})
    if cache_info:
        total = cache_info.get("total", 0)
        console.print(f"[dim]Cached AI Recommendations: {total} positions[/dim]")
        console.print()

    # Risk summary
    risk_table = Table(show_header=False, box=box.SIMPLE)
    risk_table.add_column("Risk Level", style="bold")
    risk_table.add_column("Count", justify="right")

    risk_colors = {
        RiskLevel.CRITICAL: "red",
        RiskLevel.HIGH: "yellow",
        RiskLevel.MODERATE: "cyan",
        RiskLevel.LOW: "green",
    }

    for level in RiskLevel:
        count = analysis["risk_summary"].get(level, 0)
        if count > 0:
            color = risk_colors[level]
            risk_table.add_row(f"[{color}]{level.value.title()}[/{color}]", str(count))

    console.print(Panel(risk_table, title="Risk Summary", border_style="blue"))

    # Portfolio metrics
    metrics_table = Table(show_header=False, box=box.SIMPLE)
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value", justify="right")

    total_pnl = sum(p.unrealized_pnl for p in positions)
    pnl_color = "green" if total_pnl >= 0 else "red"
    metrics_table.add_row("Total P/L", f"[{pnl_color}]${total_pnl:+,.2f}[/{pnl_color}]")

    theta = analysis["total_theta"]
    theta_color = "green" if theta < 0 else "yellow"
    metrics_table.add_row("Daily Theta", f"[{theta_color}]${theta:+,.2f}[/{theta_color}]")

    metrics_table.add_row("Net Delta", f"{analysis['total_delta']:+,.0f}")

    console.print(Panel(metrics_table, title="Portfolio Metrics", border_style="cyan"))

    console.print("\n[dim]💡 Tip: Use the dashboard and press 'a' on a strategy row to generate AI recommendations[/dim]")


@app.command()
def market(
    symbols: list[str] = typer.Argument(None, help="Symbols to check (default: SPY QQQ IWM)"),
):
    """Check market conditions and IV environment."""
    from .analysis import get_analyzer, IVEnvironment

    client = get_client()
    market_analyzer = get_analyzer()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    if not symbols:
        symbols = ["SPY", "QQQ", "IWM", "VIX"]

    table = Table(title="Market Overview", box=box.ROUNDED)
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("IV Rank", justify="right")
    table.add_column("IV %ile", justify="right")
    table.add_column("Environment")

    env_styles = {
        IVEnvironment.VERY_LOW: ("green", "Very Low"),
        IVEnvironment.LOW: ("green", "Low"),
        IVEnvironment.NORMAL: ("yellow", "Normal"),
        IVEnvironment.ELEVATED: ("yellow", "Elevated"),
        IVEnvironment.HIGH: ("red", "High"),
        IVEnvironment.VERY_HIGH: ("red", "Very High"),
    }

    with console.status("Fetching market data..."):
        for symbol in symbols:
            symbol = symbol.upper()
            snapshot = market_analyzer.get_snapshot(symbol)

            if snapshot:
                price = f"${snapshot.price:.2f}"
                iv_rank = f"{snapshot.iv_rank:.1f}" if snapshot.iv_rank else "-"
                iv_pct = f"{snapshot.iv_percentile:.1f}" if snapshot.iv_percentile else "-"

                style, label = env_styles.get(snapshot.iv_environment, ("dim", "Unknown"))
                env_text = f"[{style}]{label}[/{style}]"

                table.add_row(symbol, price, iv_rank, iv_pct, env_text)
            else:
                table.add_row(symbol, "-", "-", "-", "[dim]No data[/dim]")

    console.print(table)

    # Strategy suggestions
    console.print("\n[bold]Strategy Guidance:[/bold]")
    console.print("  [green]Low IV (< 30)[/green]: Favor buying premium (long calls/puts, debit spreads)")
    console.print("  [yellow]Normal IV (30-50)[/yellow]: Neutral strategies or directional plays")
    console.print("  [red]High IV (> 50)[/red]: Favor selling premium (short puts, credit spreads, iron condors)")


# Watchlist subcommand group
watchlist_app = typer.Typer(help="Manage your watchlist (show, add, remove, clear, set)")
app.add_typer(watchlist_app, name="watchlist")


@watchlist_app.command("show")
def watchlist_show():
    """Show current watchlist."""
    from .config import get_watchlist
    from .analysis import get_analyzer, IVEnvironment

    watchlist = get_watchlist()

    if not watchlist:
        console.print("[dim]Watchlist is empty. Add symbols with: pilot watchlist add SYMBOL[/dim]")
        return

    client = get_client()
    market = get_analyzer()

    table = Table(title="Watchlist", box=box.ROUNDED)
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("IV Rank", justify="right")
    table.add_column("Environment")

    env_styles = {
        IVEnvironment.VERY_LOW: ("green", "Very Low"),
        IVEnvironment.LOW: ("green", "Low"),
        IVEnvironment.NORMAL: ("yellow", "Normal"),
        IVEnvironment.ELEVATED: ("yellow", "Elevated"),
        IVEnvironment.HIGH: ("red", "High"),
        IVEnvironment.VERY_HIGH: ("red", "Very High"),
    }

    with console.status("Fetching watchlist data..."):
        for symbol in watchlist:
            if client.is_enabled:
                snapshot = market.get_snapshot(symbol)
                if snapshot:
                    price = f"${snapshot.price:.2f}"
                    iv_rank = f"{snapshot.iv_rank:.1f}" if snapshot.iv_rank else "-"
                    style, label = env_styles.get(snapshot.iv_environment, ("dim", "Unknown"))
                    env_text = f"[{style}]{label}[/{style}]"
                    table.add_row(symbol, price, iv_rank, env_text)
                else:
                    table.add_row(symbol, "-", "-", "[dim]No data[/dim]")
            else:
                table.add_row(symbol, "-", "-", "[dim]No API[/dim]")

    console.print(table)


@watchlist_app.command("add")
def watchlist_add(symbol: str = typer.Argument(..., help="Symbol to add")):
    """Add a symbol to watchlist."""
    from .config import add_to_watchlist

    symbol = symbol.upper()
    if add_to_watchlist(symbol):
        console.print(f"[green]Added {symbol} to watchlist[/green]")
    else:
        console.print(f"[yellow]{symbol} is already in watchlist[/yellow]")


@watchlist_app.command("remove")
def watchlist_remove(symbol: str = typer.Argument(..., help="Symbol to remove")):
    """Remove a symbol from watchlist."""
    from .config import remove_from_watchlist

    symbol = symbol.upper()
    if remove_from_watchlist(symbol):
        console.print(f"[green]Removed {symbol} from watchlist[/green]")
    else:
        console.print(f"[yellow]{symbol} is not in watchlist[/yellow]")


@watchlist_app.command("clear")
def watchlist_clear():
    """Clear all symbols from watchlist."""
    from .config import set_watchlist

    set_watchlist([])
    console.print("[green]Watchlist cleared[/green]")


@watchlist_app.command("set")
def watchlist_set(symbols: list[str] = typer.Argument(..., help="Symbols to set")):
    """Set watchlist to specific symbols."""
    from .config import set_watchlist

    symbols = [s.upper() for s in symbols]
    set_watchlist(symbols)
    console.print(f"[green]Watchlist set to: {', '.join(symbols)}[/green]")


# Roll tracking commands group
rolls_app = typer.Typer(help="Manage roll tracking and analyze rolling patterns")
app.add_typer(rolls_app, name="rolls")


@rolls_app.command("history")
def rolls_history(
    symbol: str = typer.Argument(..., help="Underlying symbol (e.g., SPY)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Filter by strategy type"),
    days: int = typer.Option(365, "--days", "-d", help="Days of history to analyze (default: 1 year)"),
    account: str = typer.Option(None, "--account", "-a", help="Account number (uses default if not specified)"),
):
    """Show roll history for a symbol."""
    from .client import get_client
    from .analysis.roll_tracker import RollTracker
    from .analysis.roll_history import get_roll_history
    from .analysis.roll_analytics import format_roll_summary
    from .config import get_default_account

    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    # Resolve account
    default_account = get_default_account()
    accounts = client.get_accounts()

    if not accounts:
        console.print("[yellow]No accounts found[/yellow]")
        raise typer.Exit(0)

    if account:
        selected = next((a for a in accounts if a.account_number == account), None)
        if not selected:
            console.print(f"[red]Account {account} not found[/red]")
            raise typer.Exit(1)
    else:
        selected = accounts[0] if default_account is None else next(
            (a for a in accounts if a.account_number == default_account), accounts[0]
        )

    # Fetch transactions
    start_date = datetime.now() - timedelta(days=days)

    with console.status(f"[dim]Fetching transactions since {start_date.strftime('%Y-%m-%d')}...[/dim]"):
        transactions = client.get_transactions(
            selected.account_number,
            start_date=start_date,
            limit=1000,
            force_refresh=False
        )

    if not transactions:
        console.print("[yellow]No transactions found[/yellow]")
        raise typer.Exit(0)

    # Get current positions
    positions = client.get_positions(selected.account_number)

    # Detect rolls
    tracker = RollTracker()
    roll_chains = tracker.detect_rolls(transactions, positions, selected.account_number)

    # Filter by strategy if specified
    if strategy:
        roll_chains = [c for c in roll_chains if strategy.lower() in c.strategy_type.lower()]

    # Filter by symbol
    symbol_upper = symbol.upper()
    roll_chains = [c for c in roll_chains if c.underlying == symbol_upper]

    if not roll_chains:
        console.print(f"[yellow]No roll history found for {symbol_upper}[/yellow]")
        console.print("[dim]Tip: Rolls will appear here once you roll positions[/dim]")
        raise typer.Exit(0)

    # Display roll chains
    for chain in roll_chains:
        console.print(f"\n[bold cyan]{format_roll_summary(chain)}[/bold cyan]\n")


@rolls_app.command("chain")
def rolls_chain(
    symbol: str = typer.Argument(..., help="Underlying symbol (e.g., SPY)"),
    days: int = typer.Option(365, "--days", "-d", help="Days of history to analyze (default: 1 year)"),
    account: str = typer.Option(None, "--account", "-a", help="Account number (uses default if not specified)"),
):
    """Display option chain with roll activity heatmap."""
    from .client import get_client
    from .analysis.roll_tracker import RollTracker
    from .analysis.roll_history import get_roll_history
    from .config import get_default_account

    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    # Resolve account (same logic as above)
    default_account = get_default_account()
    accounts = client.get_accounts()

    if not accounts:
        console.print("[yellow]No accounts found[/yellow]")
        raise typer.Exit(0)

    if account:
        selected = next((a for a in accounts if a.account_number == account), None)
        if not selected:
            console.print(f"[red]Account {account} not found[/red]")
            raise typer.Exit(1)
    else:
        selected = accounts[0] if default_account is None else next(
            (a for a in accounts if a.account_number == default_account), accounts[0]
        )

    symbol_upper = symbol.upper()

    # Get roll history
    history = get_roll_history()
    chains = history.get_all_chains(selected.account_number, symbol=symbol_upper)

    if not chains:
        console.print(f"[yellow]No roll history found for {symbol_upper}[/yellow]")
        console.print("[dim]Tip: Use 'pilot rolls history' to view roll details[/dim]")
        raise typer.Exit(0)

    # Build heatmap data
    heatmap_data = {}
    for chain in chains:
        for roll in chain.rolls:
            # Group by old strike and old DTE
            strike = roll.old_strike
            dte = roll.old_dte

            # Determine DTE bucket
            if dte <= 7:
                dte_bucket = "0-7"
            elif dte <= 14:
                dte_bucket = "8-14"
            elif dte <= 21:
                dte_bucket = "15-21"
            elif dte <= 35:
                dte_bucket = "22-35"
            else:
                dte_bucket = "36+"

            key = (strike, dte_bucket)
            if key not in heatmap_data:
                heatmap_data[key] = 0
            heatmap_data[key] += 1

    # Display heatmap
    console.print(f"\n[bold cyan]{symbol_upper} Option Chain - Roll Activity ({days} days)[/bold cyan]\n")

    # Group strikes
    strikes = sorted(set(k[0] for k in heatmap_data.keys()))

    # Create table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Strike", style="cyan")
    table.add_column("0-7DTE", justify="right")
    table.add_column("8-14DTE", justify="right")
    table.add_column("15-21DTE", justify="right")
    table.add_column("22-35DTE", justify="right")
    table.add_column("36+DTE", justify="right")
    table.add_column("Total", justify="right")

    dte_buckets = ["0-7", "8-14", "15-21", "22-35", "36+"]

    for strike in strikes:
        row = [f"${strike:.0f}"]
        row_totals = []

        for bucket in dte_buckets:
            count = heatmap_data.get((strike, bucket), 0)
            # Format with roll count bars
            if count == 0:
                cell = "[dim]-[/dim]"
            elif count <= 2:
                cell = f"[yellow]{count}[/yellow]"
            else:
                cell = f"[red bold]{count}[/red bold]"
            row.append(cell)
            row_totals.append(count)

        total = sum(row_totals)
        row.append(str(total))
        table.add_row(*row)

    console.print(table)
    console.print("\n[dim]Legend: [yellow]1-2[/yellow] [red bold]3+[/red bold] rolls at that strike/DTE[/dim]")


@rolls_app.command("patterns")
def rolls_patterns(
    symbol: str = typer.Argument(..., help="Underlying symbol (e.g., SPY)"),
    days: int = typer.Option(365, "--days", "-d", help="Days of history to analyze (default: 1 year)"),
    account: str = typer.Option(None, "--account", "-a", help="Account number (uses default if not specified)"),
):
    """Show learned rolling patterns and statistics."""
    from .client import get_client
    from .analysis.roll_tracker import RollTracker
    from .analysis.roll_analytics import analyze_patterns, format_patterns_summary
    from .config import get_default_account

    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    # Resolve account (same logic as above)
    default_account = get_default_account()
    accounts = client.get_accounts()

    if not accounts:
        console.print("[yellow]No accounts found[/yellow]")
        raise typer.Exit(0)

    if account:
        selected = next((a for a in accounts if a.account_number == account), None)
        if not selected:
            console.print(f"[red]Account {account} not found[/red]")
            raise typer.Exit(1)
    else:
        selected = accounts[0] if default_account is None else next(
            (a for a in accounts if a.account_number == default_account), accounts[0]
        )

    symbol_upper = symbol.upper()

    # Fetch transactions and detect rolls
    start_date = datetime.now() - timedelta(days=days)

    with console.status(f"[dim]Fetching transactions since {start_date.strftime('%Y-%m-%d')}...[/dim]"):
        transactions = client.get_transactions(
            selected.account_number,
            start_date=start_date,
            limit=1000,
            force_refresh=False
        )

    if not transactions:
        console.print("[yellow]No transactions found[/yellow]")
        raise typer.Exit(0)

    positions = client.get_positions(selected.account_number)

    # Detect rolls
    tracker = RollTracker()
    roll_chains = tracker.detect_rolls(transactions, positions, selected.account_number)

    # Filter by symbol
    roll_chains = [c for c in roll_chains if c.underlying == symbol_upper]

    if not roll_chains:
        console.print(f"[yellow]No roll history found for {symbol_upper}[/yellow]")
        console.print("[dim]Tip: Patterns will appear here once you have rolled positions[/dim]")
        raise typer.Exit(0)

    # Analyze patterns
    patterns = analyze_patterns(roll_chains)

    # Display patterns
    console.print(f"\n[bold cyan]{symbol_upper} Rolling Patterns (last {days} days)[/bold cyan]\n")
    console.print(Panel(format_patterns_summary(patterns), border_style="blue"))

    # Actionable insights
    console.print("\n[bold yellow]💡 Key Insights:[/bold yellow]")
    if patterns.typical_roll_days:
        console.print(f"  • You typically roll to {patterns.typical_roll_days[0]} or {patterns.typical_roll_days[1] if len(patterns.typical_roll_days) > 1 else ''} DTE")
    console.print(f"  • Average roll occurs at {patterns.avg_dte_at_roll:.1f} DTE")
    console.print(f"  • Best DTE window: {patterns.best_dte_window[0]}-{patterns.best_dte_window[1]} days")

    if patterns.win_rate >= 0.7:
        console.print(f"  • [green]Strong win rate: {patterns.win_rate:.1%}[/green]")
    elif patterns.win_rate >= 0.5:
        console.print(f"  • [yellow]Moderate win rate: {patterns.win_rate:.1%}[/yellow]")
    else:
        console.print(f"  • [red]Low win rate: {patterns.win_rate:.1%}[/red]")


@rolls_app.command("fetch")
def rolls_fetch(
    days: int = typer.Option(365, "--days", "-d", help="Days of history to fetch (default: 1 year)"),
    account: str = typer.Option(None, "--account", "-a", help="Account number (uses default if not specified)"),
    force_refresh: bool = typer.Option(False, "--force", "-f", help="Force refresh from API (bypass cache)"),
):
    """Fetch and store roll history from transactions.

    This command fetches your transaction history, detects roll operations,
    and stores them in the roll history cache for faster access later.
    """
    from .client import get_client
    from .analysis.roll_tracker import RollTracker
    from .analysis.roll_history import get_roll_history
    from .config import get_default_account

    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    # Resolve account
    default_account = get_default_account()
    accounts = client.get_accounts()

    if not accounts:
        console.print("[yellow]No accounts found[/yellow]")
        raise typer.Exit(0)

    if account:
        selected = next((a for a in accounts if a.account_number == account), None)
        if not selected:
            console.print(f"[red]Account {account} not found[/red]")
            raise typer.Exit(1)
    else:
        selected = accounts[0] if default_account is None else next(
            (a for a in accounts if a.account_number == default_account), accounts[0]
        )

    # Fetch transactions
    start_date = datetime.now() - timedelta(days=days)

    with console.status(f"[dim]Fetching transactions since {start_date.strftime('%Y-%m-%d')}...[/dim]"):
        transactions = client.get_transactions(
            selected.account_number,
            start_date=start_date,
            limit=1000,
            force_refresh=force_refresh
        )

    if not transactions:
        console.print("[yellow]No transactions found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]✓[/green] Fetched {len(transactions)} transactions")

    # Detect rolls
    positions = client.get_positions(selected.account_number)
    tracker = RollTracker()
    roll_chains = tracker.detect_rolls(transactions, positions, selected.account_number)

    console.print(f"[green]✓[/green] Detected {len(roll_chains)} roll chains")

    if len(roll_chains) == 0:
        console.print("[yellow]No rolls found in transaction history[/yellow]")
        console.print("[dim]Rolls will be detected once you close and reopen positions[/dim]")
        raise typer.Exit(0)

    # Store in roll history
    roll_history = get_roll_history()
    total_rolls = 0

    for chain in roll_chains:
        roll_history.add_chain(chain, selected.account_number)
        total_rolls += chain.roll_count

    console.print(f"[green]✓[/green] Stored {total_rolls} rolls from {len(roll_chains)} roll chains")

    # Summary
    console.print("\n[bold]Roll History Summary:[/bold]")
    for chain in roll_chains:
        pnl_style = "green" if chain.net_pnl >= 0 else "red"
        console.print(
            f"  {chain.underlying:6} {chain.strategy_type:15}: "
            f"[{pnl_style}]${chain.net_pnl:+,.2f}[/{pnl_style}] "
            f"({chain.roll_count} rolls)"
        )

    console.print(f"\n[dim]Roll history cached at: ~/.cache/position-pilot/roll_history.json[/dim]")


if __name__ == "__main__":
    app()
