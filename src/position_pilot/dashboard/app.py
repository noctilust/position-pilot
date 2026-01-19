"""Interactive Textual dashboard for Position Pilot."""

from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widget import Widget
from textual.widgets import (
    Header,
    Footer,
    Static,
    DataTable,
    Button,
    Label,
    LoadingIndicator,
)
from textual.binding import Binding
from textual.reactive import reactive
from textual import work, events

from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from ..client import get_client
from ..config import get_default_account, get_watchlist
from ..models import Account, Position
from ..analysis import get_analyzer, RiskLevel, detect_strategies, StrategyGroup, get_llm_analyzer


class AccountPanel(Static):
    """Displays account summary."""

    account: reactive[Account | None] = reactive(None)
    show_financials: reactive[bool] = reactive(True)

    def render(self) -> Panel:
        if not self.account:
            return Panel("Loading...", title="Account")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value", justify="right")

        if self.show_financials:
            table.add_row("Net Liq", f"${self.account.net_liquidating_value:,.2f}")
            table.add_row("Buying Power", f"${self.account.buying_power:,.2f}")
            table.add_row("Cash", f"${self.account.cash_balance:,.2f}")
        else:
            # Match asterisk length to actual value length
            net_liq_str = f"${self.account.net_liquidating_value:,.2f}"
            buying_power_str = f"${self.account.buying_power:,.2f}"
            cash_str = f"${self.account.cash_balance:,.2f}"

            table.add_row("Net Liq", "$" + "*" * (len(net_liq_str) - 1))
            table.add_row("Buying Power", "$" + "*" * (len(buying_power_str) - 1))
            table.add_row("Cash", "$" + "*" * (len(cash_str) - 1))

        table.add_row("Positions", str(len(self.account.positions)))

        return Panel(table, title=f"[bold]{self.account.display_name}[/bold]", border_style="blue")


class PositionsTable(DataTable):
    """Displays positions or strategies in a table."""

    positions: reactive[list[Position]] = reactive(list)
    show_strategies: reactive[bool] = reactive(True)

    BINDINGS = []

    def __init__(self, **kwargs):
        """Initialize the positions table with empty state."""
        super().__init__(**kwargs)
        self._initialized: bool = False
        # Track expanded strategies: key = (symbol, strategy_type, strikes_tuple), value = bool
        self._expanded_strategies: set[tuple] = set()
        # Map row indices to strategy data for enter key handling
        self._row_to_strategy: dict[int, tuple[str, StrategyGroup]] = {}

    def _handle_toggle_expand(self) -> None:
        """Toggle expansion of the currently selected strategy row."""
        if not self.show_strategies:
            return

        cursor_row = self.cursor_row

        if cursor_row in self._row_to_strategy:
            symbol, strategy = self._row_to_strategy[cursor_row]
            # Create a unique key for this strategy
            strikes = tuple(sorted([
                p.strike_price for p in strategy.positions
                if p.strike_price is not None
            ]))
            strategy_key = (symbol, strategy.strategy_type, strikes)

            # Toggle expansion state
            if strategy_key in self._expanded_strategies:
                self._expanded_strategies.remove(strategy_key)
            else:
                self._expanded_strategies.add(strategy_key)

            # Repopulate table
            self._populate_table()

            # Restore cursor to the same strategy row
            self._restore_cursor_to_strategy(strategy_key)

    def _restore_cursor_to_strategy(self, strategy_key: tuple) -> None:
        """Restore cursor to the row containing the given strategy key."""
        for row_idx, (symbol, strategy) in self._row_to_strategy.items():
            strikes = tuple(sorted([
                p.strike_price for p in strategy.positions
                if p.strike_price is not None
            ]))
            row_key = (symbol, strategy.strategy_type, strikes)
            if row_key == strategy_key:
                self.move_cursor(row=row_idx)
                return

    def _toggle_all_strategies(self) -> None:
        """Toggle expand/collapse state of all strategy rows."""
        if not self.show_strategies:
            return

        # Save current cursor position's strategy key
        cursor_row = self.cursor_row
        saved_strategy_key = None
        if cursor_row in self._row_to_strategy:
            symbol, strategy = self._row_to_strategy[cursor_row]
            strikes = tuple(sorted([
                p.strike_price for p in strategy.positions
                if p.strike_price is not None
            ]))
            saved_strategy_key = (symbol, strategy.strategy_type, strikes)

        # Get all strategies
        strategies = detect_strategies(self.positions)

        # Group strategies by underlying symbol
        by_symbol: dict[str, list[StrategyGroup]] = {}
        for strat in strategies:
            if strat.underlying not in by_symbol:
                by_symbol[strat.underlying] = []
            by_symbol[strat.underlying].append(strat)

        # Check if all are currently expanded
        all_expanded = True
        all_strategy_keys = []
        for symbol, symbol_strategies in by_symbol.items():
            for strat in symbol_strategies:
                strikes = tuple(sorted([
                    p.strike_price for p in strat.positions
                    if p.strike_price is not None
                ]))
                strategy_key = (symbol, strat.strategy_type, strikes)
                all_strategy_keys.append(strategy_key)
                if strategy_key not in self._expanded_strategies:
                    all_expanded = False

        if all_expanded:
            # Collapse all
            self._expanded_strategies.clear()
        else:
            # Expand all
            for strategy_key in all_strategy_keys:
                self._expanded_strategies.add(strategy_key)

        self._populate_table()

        # Restore cursor to the same strategy row
        if saved_strategy_key:
            self._restore_cursor_to_strategy(saved_strategy_key)

    def on_mount(self) -> None:
        """Set up table columns when widget is mounted."""
        # Initialize state BEFORE marking as initialized to avoid race conditions
        self._expanded_strategies = set()
        self._row_to_strategy = {}

        self._initialized = True
        self.cursor_type = "row"
        self.show_header = True
        self.fixed_header = False
        self.z_stripes = True

        # Set up initial columns
        self.add_columns("Position", "Price", "Qty", "Strikes", "DTE", "P/L", "P/L %", "Delta", "Extrinsic", "Intrinsic")

    def set_positions(self, positions: list[Position]) -> None:
        """Set positions and trigger table update via reactive system."""
        self.positions = positions

    def watch_show_strategies(self, old_value: bool, new_value: bool) -> None:
        """React to view mode change."""
        if not self._initialized:
            return

        # Clear and rebuild columns
        self.clear(columns=True)

        if self.show_strategies:
            self.add_columns("Position", "Price", "Qty", "Strikes", "DTE", "P/L", "P/L %", "Delta", "Extrinsic", "Intrinsic")
        else:
            self.add_columns("Symbol", "Price", "Qty", "DTE", "P/L", "P/L %", "Delta", "Extrinsic", "Intrinsic")

        self._populate_table()

    def watch_positions(self, old_value: list[Position], new_value: list[Position]) -> None:
        """React to positions change."""
        if not self._initialized:
            return
        self._populate_table()

    def _populate_table(self) -> None:
        """Populate table with current data."""
        self.clear()
        self._row_to_strategy.clear()

        if not self.positions:
            return

        try:
            if self.show_strategies:
                self._populate_strategies()
            else:
                self._populate_positions()
        except Exception as e:
            import sys
            import traceback
            sys.stderr.write(f"ERROR in _populate_table: {e}\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

    def _populate_strategies(self) -> None:
        """Populate table with strategies, grouped by symbol."""
        strategies = detect_strategies(self.positions)

        # Group strategies by underlying symbol
        by_symbol: dict[str, list[StrategyGroup]] = {}
        for strat in strategies:
            if strat.underlying not in by_symbol:
                by_symbol[strat.underlying] = []
            by_symbol[strat.underlying].append(strat)

        # Sort symbols
        sorted_symbols = sorted(by_symbol.keys())
        for i, symbol in enumerate(sorted_symbols):
            symbol_strategies = by_symbol[symbol]
            total_pnl = sum(s.unrealized_pnl for s in symbol_strategies)
            pnl_style = "green" if total_pnl >= 0 else "red"

            # Add symbol summary row
            self.add_row(
                Text(f" {symbol}", style="bold magenta"),
                "",
                "",
                Text(f"{len(symbol_strategies)} strategies", style="dim"),
                "",
                Text(f"${total_pnl:+,.2f}", style=f"bold {pnl_style}"),
                "",
                "",
                "",
                "",
            )

            # Add strategies for this symbol
            for strat in symbol_strategies:
                # Create unique key for this strategy
                strikes = tuple(sorted([
                    p.strike_price for p in strat.positions
                    if p.strike_price is not None
                ]))
                strategy_key = (symbol, strat.strategy_type, strikes)
                is_expanded = strategy_key in self._expanded_strategies

                # Expand/collapse indicator
                indicator = "▼" if is_expanded else "▶"
                name = Text(f"  {indicator} {strat.strategy_type.value}", style="")

                # For multi-leg strategies, don't show a single price (each leg has its own price)
                price_str = "-"

                strikes_display = strat.strikes_display or "-"
                qty = str(strat.total_quantity)
                # Don't show DTE on collapsed strategy row - it's shown on each leg instead
                dte = ""

                pnl_style = "green" if strat.unrealized_pnl >= 0 else "red"
                pnl = Text(f"${strat.unrealized_pnl:+,.2f}", style=pnl_style)

                pnl_pct = "-"
                if strat.unrealized_pnl_percent is not None:
                    pnl_pct = Text(f"{strat.unrealized_pnl_percent:+.1f}%", style=pnl_style)

                delta = f"{strat.total_delta:.0f}" if strat.total_delta else "-"

                # Calculate average extrinsic value per contract for strategy
                # For multi-leg strategies, show weighted average based on quantity
                extrinsic_values = [
                    (pos.extrinsic_value, abs(pos.quantity))
                    for pos in strat.positions
                    if pos.extrinsic_value is not None and pos.is_option
                ]
                if extrinsic_values:
                    weighted_sum = sum(ext * qty for ext, qty in extrinsic_values)
                    total_qty = sum(qty for _, qty in extrinsic_values)
                    avg_extrinsic = weighted_sum / total_qty if total_qty > 0 else 0
                    extrinsic_str = f"${avg_extrinsic:.2f}" if avg_extrinsic > 0 else "-"
                else:
                    extrinsic_str = "-"

                # Calculate average intrinsic value per contract for strategy
                intrinsic_values = [
                    (pos.intrinsic_value, abs(pos.quantity))
                    for pos in strat.positions
                    if pos.intrinsic_value is not None and pos.is_option
                ]
                if intrinsic_values:
                    weighted_sum = sum(intrin * qty for intrin, qty in intrinsic_values)
                    total_qty = sum(qty for _, qty in intrinsic_values)
                    avg_intrinsic = weighted_sum / total_qty if total_qty > 0 else 0
                    intrinsic_str = f"${avg_intrinsic:.2f}" if avg_intrinsic > 0 else "-"
                else:
                    intrinsic_str = "-"

                self.add_row(name, price_str, qty, strikes_display, dte, pnl, pnl_pct, delta, extrinsic_str, intrinsic_str)

                # Track row index for this strategy (current row count after adding)
                row_index = len(self.rows) - 1
                self._row_to_strategy[row_index] = (symbol, strat)

                # Show legs if expanded
                if is_expanded:
                    for pos in strat.positions:
                        # Format position as a leg
                        if pos.is_option:
                            exp = pos.expiration_date.strftime("%m/%d") if pos.expiration_date else "?"
                            opt = "C" if pos.option_type == "C" else "P"
                            strike = f"${pos.strike_price:.0f}" if pos.strike_price else "?"
                            pos_symbol = f"    └─ {exp} {strike}{opt}"
                        else:
                            pos_symbol = f"    └─ {pos.symbol}"

                        # Quantity with style
                        qty_style = "red" if pos.is_short else "green"
                        pos_qty = Text(pos.display_quantity, style=qty_style)

                        pos_dte = str(pos.days_to_expiration) if pos.days_to_expiration is not None else "-"

                        pos_price = pos.mark_price or pos.close_price
                        pos_price_str = f"${pos_price:.2f}" if pos_price else "-"

                        pos_pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
                        pos_pnl = Text(f"${pos.unrealized_pnl:+,.2f}", style=pos_pnl_style)

                        pos_pnl_pct = "-"
                        if pos.unrealized_pnl_percent is not None:
                            pos_pnl_pct = Text(f"{pos.unrealized_pnl_percent:+.1f}%", style=pos_pnl_style)

                        pos_delta = "-"
                        if pos.greeks and pos.greeks.delta is not None:
                            pos_delta = f"{pos.greeks.delta:.2f}"

                        # Extrinsic value per contract
                        pos_extrinsic = "-"
                        if pos.extrinsic_value is not None:
                            pos_extrinsic = f"${pos.extrinsic_value:.2f}"

                        # Intrinsic value per contract
                        pos_intrinsic = "-"
                        if pos.intrinsic_value is not None:
                            pos_intrinsic = f"${pos.intrinsic_value:.2f}"

                        self.add_row(
                            Text(pos_symbol, style="dim"),  # Strategy
                            pos_price_str,  # Price
                            pos_qty,  # Qty
                            "",  # Empty for strikes column
                            pos_dte,  # DTE
                            pos_pnl,  # P/L
                            pos_pnl_pct,  # P/L %
                            Text(pos_delta if pos_delta != "-" else "", style="dim"),  # Delta
                            Text(pos_extrinsic if pos_extrinsic != "-" else "", style="dim"),  # Extrinsic
                            Text(pos_intrinsic if pos_intrinsic != "-" else "", style="dim"),  # Intrinsic
                        )

            # Add blank row after each symbol group (except last)
            if i < len(sorted_symbols) - 1:
                self.add_row("", "", "", "", "", "", "", "", "", "")

    def _populate_positions(self) -> None:
        """Populate table with individual positions."""
        for pos in sorted(self.positions, key=lambda p: (not p.is_option, p.underlying_symbol)):
            if pos.is_option:
                exp = pos.expiration_date.strftime("%m/%d") if pos.expiration_date else "?"
                opt = "C" if pos.option_type == "C" else "P"
                strike = f"${pos.strike_price:.0f}" if pos.strike_price else "?"
                symbol = f"{pos.underlying_symbol} {exp} {strike}{opt}"
            else:
                symbol = pos.symbol

            qty_style = "red" if pos.is_short else "green"
            qty = Text(pos.display_quantity, style=qty_style)

            dte = str(pos.days_to_expiration) if pos.days_to_expiration is not None else "-"

            price = pos.mark_price or pos.close_price
            price_str = f"${price:.2f}" if price else "-"

            pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
            pnl = Text(f"${pos.unrealized_pnl:+,.2f}", style=pnl_style)

            pnl_pct = "-"
            if pos.unrealized_pnl_percent is not None:
                pnl_pct = Text(f"{pos.unrealized_pnl_percent:+.1f}%", style=pnl_style)

            delta = "-"
            if pos.greeks and pos.greeks.delta is not None:
                delta = f"{pos.greeks.delta:.2f}"

            # Extrinsic value per contract
            extrinsic = "-"
            if pos.extrinsic_value is not None:
                extrinsic = f"${pos.extrinsic_value:.2f}"

            # Intrinsic value per contract
            intrinsic = "-"
            if pos.intrinsic_value is not None:
                intrinsic = f"${pos.intrinsic_value:.2f}"

            self.add_row(symbol, price_str, qty, dte, pnl, pnl_pct, delta, extrinsic, intrinsic)


class RecommendationsPanel(Static):
    """Displays trading recommendations (on-demand with timestamps)."""

    recommendation: reactive[tuple[object, datetime]] = reactive((None, None))  # (Recommendation, timestamp)

    def render(self) -> Panel:
        rec, generated_at = self.recommendation

        if not rec:
            return Panel(
                "[dim]Press 'a' on a strategy row to generate AI recommendation[/dim]",
                title="AI Recommendation",
                border_style="yellow"
            )

        # Signal emoji
        signal_icons = {
            "strong_buy": "🟢",
            "buy": "🟢",
            "hold": "🟡",
            "sell": "🟠",
            "strong_sell": "🔴",
            "roll": "🔄",
            "close": "❌",
        }
        icon = signal_icons.get(rec.signal.value, "⚪")

        # Format position name
        pos = rec.position
        if pos.is_option:
            exp = pos.expiration_date.strftime("%m/%d") if pos.expiration_date else "?"
            strike = f"${pos.strike_price:.0f}" if pos.strike_price else "?"
            name = f"{pos.underlying_symbol} {exp} {strike}"
        else:
            name = pos.symbol

        # Urgency indicator
        urgency = "!" * min(rec.urgency, 3)

        # Format timestamp
        if generated_at:
            time_str = generated_at.strftime("%H:%M:%S")
            date_str = generated_at.strftime("%Y-%m-%d") if generated_at.date() != datetime.now().date() else "Today"
            timestamp_str = f"[dim]Generated: {date_str} at {time_str}[/dim]"
        else:
            timestamp_str = "[dim]Recently generated[/dim]"

        lines = [
            f"{icon} [bold]{name}[/bold] {urgency}",
            f"   {rec.reason}",
            timestamp_str,
        ]

        if rec.suggested_action:
            lines.append(f"   [dim]→ {rec.suggested_action}[/dim]")

        content = "\n".join(lines)
        return Panel(content, title="AI Recommendation", border_style="yellow")


class MarketPanel(Static):
    """Displays market overview."""

    symbols: reactive[list[str]] = reactive(list)
    data: reactive[dict] = reactive(dict)

    def render(self) -> Panel:
        if not self.data:
            return Panel("[dim]Loading market data...[/dim]", title="Market", border_style="green")

        lines = []
        for symbol, info in self.data.items():
            if info:
                price = info.get("price", 0)
                iv_rank = info.get("iv_rank")
                iv_str = f"IVR: {iv_rank:.0f}" if iv_rank else ""

                # IV rank color
                if iv_rank:
                    if iv_rank >= 50:
                        iv_style = "red"
                    elif iv_rank >= 30:
                        iv_style = "yellow"
                    else:
                        iv_style = "green"
                    iv_str = f"[{iv_style}]IVR: {iv_rank:.0f}[/{iv_style}]"

                lines.append(f"[bold]{symbol}[/bold] ${price:.2f} {iv_str}")

        content = "\n".join(lines) if lines else "[dim]No data[/dim]"
        return Panel(content, title="Market", border_style="green")


class PortfolioSummary(Static):
    """Portfolio-level summary metrics."""

    total_pnl: reactive[float] = reactive(0.0)
    total_theta: reactive[float] = reactive(0.0)
    total_delta: reactive[float] = reactive(0.0)
    risk_summary: reactive[dict] = reactive(dict)
    show_financials: reactive[bool] = reactive(True)

    def render(self) -> Panel:
        # P/L styling and masking
        pnl_style = "green" if self.total_pnl >= 0 else "red"
        if self.show_financials:
            pnl_text = f"[{pnl_style}]${self.total_pnl:+,.2f}[/{pnl_style}]"
        else:
            # Match asterisk length to actual value
            actual_pnl_text = f"${self.total_pnl:+,.2f}"
            pnl_text = f"[{pnl_style}]$" + "*" * (len(actual_pnl_text) - 1) + f"[/{pnl_style}]"

        # Theta styling (always visible)
        theta_style = "green" if self.total_theta < 0 else "red"
        theta_text = f"[{theta_style}]${self.total_theta:+,.2f}[/{theta_style}]"

        # Risk counts
        critical = self.risk_summary.get(RiskLevel.CRITICAL, 0)
        high = self.risk_summary.get(RiskLevel.HIGH, 0)

        risk_text = ""
        if critical > 0:
            risk_text += f"[red]{critical} critical[/red] "
        if high > 0:
            risk_text += f"[yellow]{high} high risk[/yellow]"
        if not risk_text:
            risk_text = "[green]All positions healthy[/green]"

        content = f"""[bold]Total P/L:[/bold] {pnl_text}
[bold]Daily Theta:[/bold] {theta_text}/day
[bold]Net Delta:[/bold] {self.total_delta:+,.0f}
[bold]Risk:[/bold] {risk_text}"""

        return Panel(content, title="Portfolio", border_style="cyan")


class StatusBar(Static):
    """Status bar showing last update time."""

    last_update: reactive[datetime | None] = reactive(None)
    loading: reactive[bool] = reactive(False)

    def render(self) -> str:
        if self.loading:
            return "[yellow]Refreshing...[/yellow]"
        if self.last_update:
            return f"[dim]Last updated: {self.last_update.strftime('%H:%M:%S')}[/dim]"
        return ""


class PilotDashboard(App):
    """Position Pilot interactive dashboard."""

    TITLE = "Position Pilot"
    SUB_TITLE = "Market Analysis & Position Guidance"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-columns: 2fr 1fr;
        grid-rows: auto 1fr auto;
    }

    #top-row {
        column-span: 2;
        height: auto;
        layout: horizontal;
    }

    #account-panel {
        width: 1fr;
    }

    #portfolio-panel {
        width: 1fr;
    }

    #positions-container {
        height: 1fr;
    }

    PositionsTable {
        height: 1fr;
    }

    #side-panels {
        height: 100%;
        layout: vertical;
    }

    #recommendations-panel {
        height: 1fr;
    }

    #market-panel {
        height: auto;
        max-height: 12;
    }

    #status-bar {
        column-span: 2;
        height: 1;
        padding: 0 1;
    }

    #positions-table-widget {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("g", "toggle_group", "Group"),
        Binding("s", "toggle_stocks", "Stocks"),
        Binding("f", "toggle_financials", "Financials"),
        Binding("c", "toggle_all_strategies", "Collapse All"),
        Binding("enter", "toggle_expand", "Expand", show=False),
        Binding("a", "analyze", "AI Rec"),
    ]

    def __init__(self):
        super().__init__()
        self.client = get_client()
        self.market_analyzer = get_analyzer()
        self.account: Account | None = None
        self.watchlist = get_watchlist()
        self.show_stocks = True  # Toggle for showing/hiding stock positions
        # Lazy initialize LLM analyzer (only when 'a' is pressed)
        self._analyzer = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="top-row"):
            yield AccountPanel(id="account-panel")
            yield PortfolioSummary(id="portfolio-panel")

        with Container(id="positions-container"):
            yield PositionsTable(id="positions-table-widget")

        with Vertical(id="side-panels"):
            yield RecommendationsPanel(id="recommendations-panel")
            yield MarketPanel(id="market-panel")

        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Load data when app starts."""
        self.load_data()

    @work(exclusive=True)
    async def load_data(self, force_refresh: bool = False) -> None:
        """Load all data asynchronously."""
        status = self.query_one(StatusBar)
        status.loading = True

        try:
            # Fetch accounts
            accounts = self.client.get_accounts()

            if accounts:
                # Use default account from config, or fall back to first
                default_account_num = get_default_account()
                if default_account_num:
                    self.account = next(
                        (a for a in accounts if a.account_number == default_account_num),
                        accounts[0]
                    )
                else:
                    self.account = accounts[0]

                # Fetch balances
                balances = self.client.get_account_balances(self.account.account_number)
                if balances:
                    self.account.net_liquidating_value = balances.get("net_liquidating_value") or 0
                    self.account.cash_balance = balances.get("cash_balance") or 0
                    self.account.buying_power = balances.get("buying_power") or 0

                # Fetch positions
                positions = self.client.get_positions(self.account.account_number)

                # Enrich with Greeks (batch - much faster!)
                positions = self.client.enrich_positions_greeks_batch(positions, force_refresh=force_refresh)

                self.account.positions = positions

                # Update account panel
                account_panel = self.query_one(AccountPanel)
                account_panel.account = self.account

                # Update positions table (filter stocks if hidden)
                positions_widget = self.query_one(PositionsTable)
                filtered_positions = positions if self.show_stocks else [p for p in positions if p.is_option]
                positions_widget.set_positions(filtered_positions)

                # Calculate portfolio metrics directly (no LLM calls on initial load)
                total_theta = 0.0
                total_delta = 0.0
                for pos in positions:
                    if pos.greeks:
                        if pos.greeks.theta:
                            total_theta += pos.greeks.theta * pos.multiplier * abs(pos.quantity)
                        if pos.greeks.delta:
                            qty = pos.quantity if not pos.is_short else -pos.quantity
                            total_delta += pos.greeks.delta * pos.multiplier * qty

                # Update portfolio summary
                portfolio = self.query_one(PortfolioSummary)
                portfolio.total_pnl = sum(p.unrealized_pnl for p in positions)
                portfolio.total_theta = total_theta
                portfolio.total_delta = total_delta
                portfolio.risk_summary = {}  # Empty until user requests analysis

                # Note: AI recommendations are now on-demand (press 'a' on a strategy row)
                # Initialize empty recommendations panel
                recs_panel = self.query_one(RecommendationsPanel)
                recs_panel.recommendation = (None, None)

            # Fetch market data
            market_data = {}
            for symbol in self.watchlist:
                snapshot = self.market_analyzer.get_snapshot(symbol, force_refresh=force_refresh)
                if snapshot:
                    market_data[symbol] = {
                        "price": snapshot.price,
                        "iv_rank": snapshot.iv_rank,
                    }

            market_panel = self.query_one(MarketPanel)
            market_panel.data = market_data

        except Exception as e:
            # Log error but don't crash the dashboard
            import sys
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            status.loading = False
            status.last_update = datetime.now()

    def action_refresh(self) -> None:
        """Refresh all data with fresh market data (bypass cache)."""
        self.load_data(force_refresh=True)

    def action_analyze(self) -> None:
        """Generate AI recommendation for selected strategy/position."""
        positions_widget = self.query_one(PositionsTable)
        cursor_row = positions_widget.cursor_row

        # Get recommendations panel
        recs_panel = self.query_one(RecommendationsPanel)

        if cursor_row < 0:
            # No row selected
            recs_panel.recommendation = (None, None)
            return

        # Check if a strategy row is selected
        if cursor_row in positions_widget._row_to_strategy:
            symbol, strategy = positions_widget._row_to_strategy[cursor_row]

            # Generate recommendation for this strategy (use first position)
            if not strategy.positions:
                recs_panel.recommendation = (None, None)
                return

            # Lazy initialize LLM analyzer if needed
            if self._analyzer is None:
                self._analyzer = get_llm_analyzer()

            # Show loading status
            status = self.query_one(StatusBar)
            status.loading = True

            try:
                rec, generated_at = self._analyzer.generate_recommendation(strategy.positions[0])
                # Update recommendations panel
                recs_panel.recommendation = (rec, generated_at)
                # Explicitly refresh the panel
                recs_panel.refresh()
            except Exception as e:
                import sys
                import traceback
                traceback.print_exc(file=sys.stderr)
                # On error, show empty panel
                recs_panel.recommendation = (None, None)
                recs_panel.refresh()
            finally:
                status.loading = False
                status.last_update = datetime.now()
        else:
            # Individual position row - not implemented yet
            recs_panel.recommendation = (None, None)
            recs_panel.refresh()

    def action_toggle_group(self) -> None:
        """Toggle between strategies and positions view."""
        positions_widget = self.query_one(PositionsTable)
        positions_widget.show_strategies = not positions_widget.show_strategies

    def action_toggle_financials(self) -> None:
        """Toggle display of account financial numbers and portfolio P/L."""
        account_panel = self.query_one(AccountPanel)
        portfolio_panel = self.query_one(PortfolioSummary)

        # Toggle both panels together
        new_state = not account_panel.show_financials
        account_panel.show_financials = new_state
        portfolio_panel.show_financials = new_state

    def action_toggle_stocks(self) -> None:
        """Toggle display of stock positions."""
        self.show_stocks = not self.show_stocks
        # Re-filter and update positions table
        if self.account and self.account.positions:
            positions_widget = self.query_one(PositionsTable)
            filtered_positions = self.account.positions if self.show_stocks else [p for p in self.account.positions if p.is_option]
            positions_widget.set_positions(filtered_positions)

    def action_toggle_expand(self) -> None:
        """Toggle expansion of the currently selected strategy row."""
        positions_widget = self.query_one(PositionsTable)
        positions_widget._handle_toggle_expand()

    def action_toggle_all_strategies(self) -> None:
        """Toggle expand/collapse state of all strategy rows."""
        positions_widget = self.query_one(PositionsTable)
        positions_widget._toggle_all_strategies()

    async def on_event(self, event: events.Event) -> None:
        """Handle all events at the app level."""
        if isinstance(event, events.Key):
            if event.key == "enter":
                positions_widget = self.query_one(PositionsTable)
                if positions_widget.show_strategies:
                    positions_widget._handle_toggle_expand()
                    event.stop()
                    return

        await super().on_event(event)


def run_dashboard():
    """Run the dashboard app."""
    app = PilotDashboard()
    app.run()


if __name__ == "__main__":
    run_dashboard()
