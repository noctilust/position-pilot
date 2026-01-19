#!/usr/bin/env python3
"""Test script for roll tracking functionality."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from position_pilot.client import get_client
from position_pilot.analysis.roll_tracker import RollTracker
from position_pilot.analysis.roll_history import get_roll_history
from position_pilot.config import get_default_account

console = Console()


def test_transaction_fetching():
    """Test fetching transactions from Tastytrade."""
    console.print("\n[bold cyan]Test 1: Fetching Transactions[/bold cyan]")

    client = get_client()

    if not client.is_enabled:
        console.print("[red]❌ Tastytrade credentials not configured[/red]")
        console.print("Set TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN in .env")
        return None

    # Resolve account
    default_account = get_default_account()
    accounts = client.get_accounts()

    if not accounts:
        console.print("[red]❌ No accounts found[/red]")
        return None

    account = accounts[0]
    if default_account:
        account = next((a for a in accounts if a.account_number == default_account), account)

    console.print(f"Account: [cyan]{account.display_name}[/cyan]")

    # Fetch transactions from last 90 days
    start_date = datetime.now() - timedelta(days=90)
    console.print(f"Fetching transactions since {start_date.strftime('%Y-%m-%d')}...")

    transactions = client.get_transactions(
        account.account_number,
        start_date=start_date,
        limit=500,
        force_refresh=True  # Bypass cache to get fresh data
    )

    if not transactions:
        console.print("[yellow]⚠ No transactions found[/yellow]")
        return None

    console.print(f"[green]✓ Found {len(transactions)} transactions[/green]")

    # Show sample transactions
    order_fills = [t for t in transactions if t.transaction_type.value == "order-fill"]
    console.print(f"  Order fills: {len(order_fills)}")

    # Show first few
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="dim")
    table.add_column("Type")
    table.add_column("Symbol")
    table.add_column("Description")
    table.add_column("Amount", justify="right")

    for tx in transactions[:5]:
        table.add_row(
            tx.transaction_date.strftime("%Y-%m-%d"),
            tx.transaction_type.value,
            tx.symbol or "-",
            tx.description[:40] + "..." if tx.description and len(tx.description) > 40 else (tx.description or "-"),
            f"${tx.amount:+,.2f}" if tx.amount else "-"
        )

    console.print(table)
    console.print(f"[dim]... and {len(transactions) - 5} more[/dim]")

    return account


def test_roll_detection(account):
    """Test roll detection from transactions."""
    console.print("\n[bold cyan]Test 2: Detecting Rolls[/bold cyan]")

    client = get_client()

    # Fetch 1 year of transactions
    start_date = datetime.now() - timedelta(days=365)
    transactions = client.get_transactions(
        account.account_number,
        start_date=start_date,
        limit=1000,
        force_refresh=True  # Bypass cache
    )

    if not transactions:
        console.print("[yellow]⚠ No transactions to analyze[/yellow]")
        return

    # Get current positions for context
    positions = client.get_positions(account.account_number)

    # Detect rolls
    tracker = RollTracker(time_window_hours=48)
    roll_chains = tracker.detect_rolls(transactions, positions, account.account_number)

    console.print(f"[green]✓ Detected {len(roll_chains)} roll chains[/green]")

    if not roll_chains:
        console.print("[yellow]No rolls detected in transaction history[/yellow]")
        console.print("[dim]This is normal if you haven't rolled any positions yet[/dim]")
        return

    # Show roll chains
    for chain in roll_chains:
        console.print(f"\n[bold]{chain.underlying} {chain.strategy_type}[/bold]")
        console.print(f"  Rolls: {chain.roll_count}")
        console.print(f"  Total P/L: ${chain.total_roll_pnl:+,.2f}")

        if chain.rolls:
            console.print(f"  Roll History:")
            for roll in chain.rolls:
                console.print(f"    {roll.timestamp.strftime('%Y-%m-%d')}: "
                           f"${roll.old_strike} → ${roll.new_strike} "
                           f"({roll.old_dte} → {roll.new_dte} DTE) "
                           f"P/L: ${roll.roll_pnl:+,.2f}")


def test_roll_history_storage(account):
    """Test persistent roll history storage."""
    console.print("\n[bold cyan]Test 3: Roll History Storage[/bold cyan]")

    # Get roll history instance
    history = get_roll_history()

    # Get cache info
    info = history.get_cache_info()
    console.print(f"Cache info:")
    console.print(f"  Accounts: {info['total_accounts']}")
    console.print(f"  Chains: {info['total_chains']}")
    console.print(f"  Total rolls: {info['total_rolls']}")
    console.print(f"  Last updated: {info['last_updated'] or 'Never'}")

    # Get all chains for this account
    chains = history.get_all_chains(account.account_number)
    console.print(f"\n[green]✓ Stored {len(chains)} roll chains[/green]")

    # Get recent rolls
    recent = history.get_recent_rolls(account.account_number, days=90)
    console.print(f"[green]✓ Found {len(recent)} rolls in last 90 days[/green]")

    if recent:
        console.print("\nRecent rolls:")
        for roll in recent[:5]:
            console.print(f"  {roll.timestamp.strftime('%Y-%m-%d')}: "
                       f"{roll.underlying} ${roll.old_strike} → ${roll.new_strike}")


def main():
    """Run all tests."""
    console.print("\n[bold blue]Roll Tracking Test Suite[/bold blue]")
    console.print("=" * 50)

    # Test 1: Transaction fetching
    account = test_transaction_fetching()
    if not account:
        console.print("\n[red]❌ Cannot continue without account data[/red]")
        return

    # Test 2: Roll detection
    test_roll_detection(account)

    # Test 3: Roll history storage
    test_roll_history_storage(account)

    console.print("\n[bold green]✓ All tests completed[/bold green]")
    console.print("\n[dim]Check ~/.cache/position-pilot/roll_history.json for stored data[/dim]")


if __name__ == "__main__":
    main()
