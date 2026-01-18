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
    """Launch interactive dashboard (coming soon)."""
    console.print("[yellow]Interactive dashboard coming soon![/yellow]")
    console.print("For now, use [cyan]pilot positions[/cyan] to view positions")


if __name__ == "__main__":
    app()
