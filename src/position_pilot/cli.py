"""CLI entry point for Position Pilot."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .client import get_client
from .models import Position

app = typer.Typer(
    name="pilot",
    help="Position Pilot - Market analysis and position guidance",
    no_args_is_help=True,
)
console = Console()


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
        symbol = f"{pos.underlying_symbol} {exp} ${pos.strike_price:.0f}{opt_type}"
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
    theta = f"${pos.greeks.theta * pos.multiplier * abs(pos.quantity):.0f}" if pos.greeks and pos.greeks.theta else "-"

    return [symbol, qty, dte, price_str, pnl, delta, theta]


@app.command()
def positions(
    account: str = typer.Option(None, "--account", "-a", help="Account number (uses first if not specified)"),
    enrich: bool = typer.Option(False, "--enrich", "-e", help="Fetch live Greeks (slower)"),
):
    """Show current positions."""
    client = get_client()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        console.print("Set TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN in .env")
        raise typer.Exit(1)

    with console.status("Fetching accounts..."):
        accounts = client.get_accounts()

    if not accounts:
        console.print("[yellow]No accounts found[/yellow]")
        raise typer.Exit(0)

    # Select account
    if account:
        selected = next((a for a in accounts if a.account_number == account), None)
        if not selected:
            console.print(f"[red]Account {account} not found[/red]")
            raise typer.Exit(1)
    else:
        selected = accounts[0]

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

    # Positions table
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

    total_pnl = 0.0
    for pos in sorted_positions:
        row = format_position_row(pos)
        table.add_row(*row)
        total_pnl += pos.unrealized_pnl

    console.print(table)

    # Total P/L
    total_text = format_pnl(total_pnl)
    console.print(f"\n[bold]Total Unrealized P/L:[/bold] ", end="")
    console.print(total_text)


@app.command()
def accounts():
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

    table = Table(title="Accounts", box=box.ROUNDED)
    table.add_column("Account", style="cyan")
    table.add_column("Type")
    table.add_column("Nickname")

    for acc in accts:
        table.add_row(
            acc.account_number,
            acc.account_type,
            acc.nickname or "-",
        )

    console.print(table)


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
    account: str = typer.Option(None, "--account", "-a", help="Account number"),
):
    """Analyze positions and show recommendations."""
    from .analysis import get_position_analyzer, get_analyzer, RiskLevel

    client = get_client()
    analyzer = get_position_analyzer()
    market = get_analyzer()

    if not client.is_enabled:
        console.print("[red]Error:[/red] Tastytrade credentials not configured")
        raise typer.Exit(1)

    with console.status("Fetching data..."):
        accounts = client.get_accounts()
        if not accounts:
            console.print("[yellow]No accounts found[/yellow]")
            raise typer.Exit(0)

        selected = accounts[0] if not account else next(
            (a for a in accounts if a.account_number == account), accounts[0]
        )

        positions = client.get_positions(selected.account_number)

        # Enrich all options with Greeks
        for i, pos in enumerate(positions):
            if pos.is_option:
                positions[i] = client.enrich_position_greeks(pos)

    if not positions:
        console.print("[dim]No open positions to analyze[/dim]")
        return

    # Run analysis
    analysis = analyzer.analyze_portfolio(positions)

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

    # Recommendations
    recs = analysis["recommendations"]
    if recs:
        console.print("\n[bold]Recommendations:[/bold]\n")

        signal_icons = {
            "strong_buy": "[green]⬆⬆[/green]",
            "buy": "[green]⬆[/green]",
            "hold": "[yellow]▬[/yellow]",
            "sell": "[red]⬇[/red]",
            "strong_sell": "[red]⬇⬇[/red]",
            "roll": "[cyan]↻[/cyan]",
            "close": "[red]✕[/red]",
        }

        for rec in recs:
            pos = rec.position
            if pos.is_option:
                exp = pos.expiration_date.strftime("%m/%d") if pos.expiration_date else "?"
                name = f"{pos.underlying_symbol} {exp} ${pos.strike_price:.0f}"
            else:
                name = pos.symbol

            icon = signal_icons.get(rec.signal.value, "?")
            urgency = "!" * min(rec.urgency, 3) if rec.urgency > 1 else ""

            console.print(f"{icon} [bold]{name}[/bold] {urgency}")
            console.print(f"   {rec.reason}")
            if rec.suggested_action:
                console.print(f"   [dim]→ {rec.suggested_action}[/dim]")
            console.print()
    else:
        console.print("\n[green]All positions healthy - no action needed[/green]")


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
watchlist_app = typer.Typer(help="Manage your watchlist")
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


if __name__ == "__main__":
    app()
