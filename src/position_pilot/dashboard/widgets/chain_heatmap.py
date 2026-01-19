"""Option chain heatmap widget for visualizing roll activity."""

from typing import Optional
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static
from textual.reactive import reactive


class OptionChainHeatmap(Static):
    """Displays roll activity as a heatmap across strikes and DTE buckets."""

    roll_data: reactive[dict] = reactive(dict)  # {(strike, dte_bucket): count}
    strikes: reactive[list[float]] = reactive(list)
    dte_buckets: reactive[list[tuple[int, int]]] = reactive(list)  # [(min, max), ...]
    current_strike: reactive[Optional[float]] = reactive(None)
    current_dte: reactive[Optional[int]] = reactive(None)
    best_strike: reactive[Optional[float]] = reactive(None)
    best_dte: reactive[Optional[tuple[int, int]]] = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Default DTE buckets
        self.dte_buckets = [(0, 7), (8, 14), (15, 21), (22, 35), (36, 999)]

    def render(self) -> Panel:
        """Render the heatmap as a panel."""
        if not self.strikes:
            return Panel(
                "[dim]No roll data available[/dim]\n[dim]Press 'h' on a strategy to load roll history[/dim]",
                title="Option Chain - Roll Activity",
                border_style="blue"
            )

        # Create heatmap table
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Strike", style="cyan", width=8)

        # Add DTE bucket headers
        bucket_labels = []
        for min_dte, max_dte in self.dte_buckets:
            if max_dte == 999:
                label = f"{min_dte}+DTE"
            else:
                label = f"{min_dte}-{max_dte}DTE"
            bucket_labels.append(label)
            table.add_column(label, justify="center", width=10)

        table.add_column("Total", justify="center", style="bold yellow")

        # Sort strikes
        sorted_strikes = sorted(self.strikes, reverse=True)

        # Find max count for scaling
        max_count = max(
            (self.roll_data.get((strike, bucket), 0)
             for strike in sorted_strikes
             for bucket in self.dte_buckets),
            default=0
        )

        # Add rows for each strike
        for strike in sorted_strikes:
            # Highlight current position
            is_current = self.current_strike and abs(strike - self.current_strike) < 0.5
            is_best = self.best_strike and abs(strike - self.best_strike) < 0.5

            if is_current:
                strike_text = Text(f"${strike:.0f} ←", style="bold green")
            elif is_best:
                strike_text = Text(f"${strike:.0f} ★", style="bold yellow")
            else:
                strike_text = Text(f"${strike:.0f}", style="cyan")

            row = [strike_text]

            # Add heatmap cells
            total_rolls = 0
            for bucket in self.dte_buckets:
                count = self.roll_data.get((strike, bucket), 0)
                total_rolls += count

                # Determine cell style based on count
                if count == 0:
                    cell = "·"
                    style = "dim"
                elif max_count > 0:
                    ratio = count / max_count
                    if ratio >= 0.75:
                        cell = "██"
                        style = "red" if count > 0 else "dim"
                    elif ratio >= 0.5:
                        cell = "▓▓"
                        style = "yellow"
                    elif ratio >= 0.25:
                        cell = "▒▒"
                        style = "green"
                    else:
                        cell = "░░"
                        style = "cyan"
                else:
                    cell = "·"
                    style = "dim"

                # Highlight current DTE position
                if self.current_dte and bucket[0] <= self.current_dte <= bucket[1]:
                    cell = f"[{style} on blue]{cell}[/{style} on blue]"

                cell_text = Text(cell, style=style)
                row.append(cell_text)

            # Add total
            total_style = "bold yellow" if total_rolls > 0 else "dim"
            row.append(Text(str(total_rolls) if total_rolls > 0 else "-", style=total_style))

            table.add_row(*row)

        # Add legend
        legend = "\n"
        legend += "[dim]Legend: [/dim]"
        legend += "[dim]· = 0 rolls[/dim] "
        legend += "[cyan]░░[/cyan] [dim]= 1-2[/dim] "
        legend += "[green]▒▒[/green] [dim]= 3-4[/dim] "
        legend += "[yellow]▓▓[/yellow] [dim]= 5-6[/dim] "
        legend += "[red]██[/red] [dim]= 7+[/dim] "
        legend += "[dim]← = Current[/dim] "
        legend += "[yellow]★ = Best[/dim]"

        content = Text.assemble(table, legend)
        return Panel(content, title="Option Chain - Roll Activity", border_style="blue")

    def load_from_chain(self, chain, current_position=None):
        """Load roll data from a RollChain.

        Args:
            chain: RollChain with roll history
            current_position: Current Position (optional, for highlighting)
        """
        # Reset data
        self.roll_data = {}
        self.strikes = set()

        # Extract strikes and DTE from rolls
        for roll in chain.rolls:
            old_strike = roll.old_strike
            new_strike = roll.new_strike
            old_dte = roll.old_dte
            new_dte = roll.new_dte

            # Add old strike and DTE bucket
            bucket = self._get_dte_bucket(old_dte)
            if bucket:
                self.strikes.add(old_strike)
                key = (old_strike, bucket)
                self.roll_data[key] = self.roll_data.get(key, 0) + 1

            # Add new strike and DTE bucket
            bucket = self._get_dte_bucket(new_dte)
            if bucket:
                self.strikes.add(new_strike)
                key = (new_strike, bucket)
                self.roll_data[key] = self.roll_data.get(key, 0) + 1

        # Update current position if provided
        if current_position:
            self.current_strike = current_position.strike_price
            self.current_dte = current_position.days_to_expiration

        # Find best strike/DTE combination (most rolls)
        self._find_best_combination()

        self.strikes = list(self.strikes)

    def _get_dte_bucket(self, dte: int) -> Optional[tuple[int, int]]:
        """Get the DTE bucket for a given DTE value.

        Args:
            dte: Days to expiration

        Returns:
            Tuple of (min_dte, max_dte) or None
        """
        for min_dte, max_dte in self.dte_buckets:
            if min_dte <= dte <= max_dte:
                return (min_dte, max_dte)
        return None

    def _find_best_combination(self) -> None:
        """Find the strike/DTE combination with the most rolls."""
        max_count = 0
        best_key = None

        for key, count in self.roll_data.items():
            if count > max_count:
                max_count = count
                best_key = key

        if best_key:
            self.best_strike = best_key[0]
            self.best_dte = best_key[1]
