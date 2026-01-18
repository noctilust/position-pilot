"""Interactive Textual dashboard for Position Pilot."""

from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
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
from textual import work

from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from ..client import get_client
from ..models import Account, Position
from ..analysis import get_position_analyzer, get_analyzer, RiskLevel


class AccountPanel(Static):
    """Displays account summary."""

    account: reactive[Account | None] = reactive(None)

    def render(self) -> Panel:
        if not self.account:
            return Panel("Loading...", title="Account")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Net Liq", f"${self.account.net_liquidating_value:,.2f}")
        table.add_row("Buying Power", f"${self.account.buying_power:,.2f}")
        table.add_row("Cash", f"${self.account.cash_balance:,.2f}")
        table.add_row("Positions", str(len(self.account.positions)))

        return Panel(table, title=f"[bold]{self.account.display_name}[/bold]", border_style="blue")


class PositionsTable(Static):
    """Displays positions in a table."""

    positions: reactive[list[Position]] = reactive(list)

    def compose(self) -> ComposeResult:
        yield DataTable(id="positions-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Qty", "DTE", "Price", "P/L", "P/L %", "Delta")
        table.cursor_type = "row"

    def watch_positions(self, positions: list[Position]) -> None:
        table = self.query_one(DataTable)
        table.clear()

        for pos in sorted(positions, key=lambda p: (not p.is_option, p.underlying_symbol)):
            # Format symbol
            if pos.is_option:
                exp = pos.expiration_date.strftime("%m/%d") if pos.expiration_date else "?"
                opt = "C" if pos.option_type == "C" else "P"
                symbol = f"{pos.underlying_symbol} {exp} ${pos.strike_price:.0f}{opt}"
            else:
                symbol = pos.symbol

            # Format quantity
            qty_style = "red" if pos.is_short else "green"
            qty = Text(pos.display_quantity, style=qty_style)

            # DTE
            dte = str(pos.days_to_expiration) if pos.days_to_expiration is not None else "-"

            # Price
            price = pos.mark_price or pos.close_price
            price_str = f"${price:.2f}" if price else "-"

            # P/L
            pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
            pnl = Text(f"${pos.unrealized_pnl:+,.2f}", style=pnl_style)

            # P/L %
            pnl_pct = "-"
            if pos.unrealized_pnl_percent is not None:
                pnl_pct = Text(f"{pos.unrealized_pnl_percent:+.1f}%", style=pnl_style)

            # Delta
            delta = "-"
            if pos.greeks and pos.greeks.delta is not None:
                delta = f"{pos.greeks.delta:.2f}"

            table.add_row(symbol, qty, dte, price_str, pnl, pnl_pct, delta)


class RecommendationsPanel(Static):
    """Displays trading recommendations."""

    recommendations: reactive[list] = reactive(list)

    def render(self) -> Panel:
        if not self.recommendations:
            return Panel("[dim]No recommendations[/dim]", title="Recommendations", border_style="yellow")

        lines = []
        for rec in self.recommendations[:5]:  # Show top 5
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
                name = f"{pos.underlying_symbol} {exp} ${pos.strike_price:.0f}"
            else:
                name = pos.symbol

            # Urgency indicator
            urgency = "!" * min(rec.urgency, 3)

            lines.append(f"{icon} [bold]{name}[/bold] {urgency}")
            lines.append(f"   {rec.reason}")
            if rec.suggested_action:
                lines.append(f"   [dim]→ {rec.suggested_action}[/dim]")
            lines.append("")

        content = "\n".join(lines) if lines else "[dim]All positions healthy[/dim]"
        return Panel(content, title="Recommendations", border_style="yellow")


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

    def render(self) -> Panel:
        # P/L styling
        pnl_style = "green" if self.total_pnl >= 0 else "red"
        pnl_text = f"[{pnl_style}]${self.total_pnl:+,.2f}[/{pnl_style}]"

        # Theta styling (positive theta is good for premium sellers)
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
        height: 100%;
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
        dock: bottom;
        padding: 0 1;
    }

    DataTable {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("a", "analyze", "Analyze"),
    ]

    def __init__(self):
        super().__init__()
        self.client = get_client()
        self.analyzer = get_position_analyzer()
        self.market_analyzer = get_analyzer()
        self.account: Account | None = None

        # Load watchlist from config
        from ..config import get_watchlist
        self.watchlist = get_watchlist()

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="top-row"):
            yield AccountPanel(id="account-panel")
            yield PortfolioSummary(id="portfolio-panel")

        with ScrollableContainer(id="positions-container"):
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
    async def load_data(self) -> None:
        """Load all data asynchronously."""
        status = self.query_one(StatusBar)
        status.loading = True

        try:
            # Fetch accounts
            accounts = self.client.get_accounts()
            if accounts:
                self.account = accounts[0]

                # Fetch balances
                balances = self.client.get_account_balances(self.account.account_number)
                if balances:
                    self.account.net_liquidating_value = balances.get("net_liquidating_value") or 0
                    self.account.cash_balance = balances.get("cash_balance") or 0
                    self.account.buying_power = balances.get("buying_power") or 0

                # Fetch positions
                positions = self.client.get_positions(self.account.account_number)

                # Enrich with Greeks
                for i, pos in enumerate(positions):
                    if pos.is_option:
                        positions[i] = self.client.enrich_position_greeks(pos)

                self.account.positions = positions

                # Update account panel
                account_panel = self.query_one(AccountPanel)
                account_panel.account = self.account

                # Update positions table
                positions_widget = self.query_one(PositionsTable)
                positions_widget.positions = positions

                # Analyze portfolio
                analysis = self.analyzer.analyze_portfolio(positions)

                # Update portfolio summary
                portfolio = self.query_one(PortfolioSummary)
                portfolio.total_pnl = sum(p.unrealized_pnl for p in positions)
                portfolio.total_theta = analysis["total_theta"]
                portfolio.total_delta = analysis["total_delta"]
                portfolio.risk_summary = analysis["risk_summary"]

                # Update recommendations
                recs_panel = self.query_one(RecommendationsPanel)
                recs_panel.recommendations = analysis["recommendations"]

            # Fetch market data
            market_data = {}
            for symbol in self.watchlist:
                snapshot = self.market_analyzer.get_snapshot(symbol)
                if snapshot:
                    market_data[symbol] = {
                        "price": snapshot.price,
                        "iv_rank": snapshot.iv_rank,
                    }

            market_panel = self.query_one(MarketPanel)
            market_panel.data = market_data

        finally:
            status.loading = False
            status.last_update = datetime.now()

    def action_refresh(self) -> None:
        """Refresh all data."""
        self.load_data()

    def action_analyze(self) -> None:
        """Run full analysis."""
        self.load_data()


def run_dashboard():
    """Run the dashboard app."""
    app = PilotDashboard()
    app.run()


if __name__ == "__main__":
    run_dashboard()
