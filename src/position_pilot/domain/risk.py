"""Deterministic portfolio and strategy risk measurements.

These calculations never depend on AI. Missing inputs produce partial results
with explicit null fields rather than invented values.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .snapshots import PortfolioSnapshot, PositionSnapshot, StrategySnapshot


class CombinedGreeks(BaseModel):
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    average_iv: float | None = None
    nearest_dte: int | None = None


class ConcentrationSlice(BaseModel):
    underlying: str
    market_value: float
    share_of_portfolio: float
    strategy_count: int
    net_delta: float


class StressScenario(BaseModel):
    name: str
    label: str
    estimated_pnl_change: float
    description: str


class StrategyRisk(BaseModel):
    strategy_id: str
    underlying: str
    strategy_type: str
    combined: CombinedGreeks
    max_profit: float | None = None
    max_loss: float | None = None
    breakevens: list[float] = Field(default_factory=list)
    distance_to_nearest_strike: float | None = None
    underlying_price: float | None = None
    current_pnl: float = 0
    defined_risk: bool = False
    valuation_basis: str = "current_mark"
    stress: list[StressScenario] = Field(default_factory=list)


class PortfolioRisk(BaseModel):
    total_delta: float = 0
    total_gamma: float = 0
    total_theta: float = 0
    total_vega: float = 0
    unrealized_pnl: float = 0
    net_liquidating_value: float = 0
    account_count: int = 0
    strategy_count: int = 0
    position_count: int = 0
    concentration: list[ConcentrationSlice] = Field(default_factory=list)
    stress: list[StressScenario] = Field(default_factory=list)


class RiskService:
    """Objective risk measurements shared by CLI and web consumers."""

    def strategy_risk(
        self,
        strategy: StrategySnapshot,
        *,
        underlying_price: float | None = None,
    ) -> StrategyRisk:
        combined = self.combined_greeks(strategy.legs)
        price = underlying_price or self._implied_underlying(strategy.legs)
        max_profit, max_loss, breakevens, defined = self._structure_bounds(strategy, price)
        nearest = self._distance_to_nearest_strike(strategy.legs, price)
        return StrategyRisk(
            strategy_id=strategy.strategy_id,
            underlying=strategy.underlying,
            strategy_type=strategy.strategy_type,
            combined=combined,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            distance_to_nearest_strike=nearest,
            underlying_price=price,
            current_pnl=strategy.unrealized_pnl,
            defined_risk=defined,
            stress=self.stress_scenarios(strategy, underlying_price=price),
        )

    def portfolio_risk(self, snapshot: PortfolioSnapshot) -> PortfolioRisk:
        legs = [leg for account in snapshot.accounts for leg in account.positions]
        combined = self.combined_greeks(legs)
        by_underlying: dict[str, list[PositionSnapshot]] = {}
        for leg in legs:
            by_underlying.setdefault(leg.underlying_symbol, []).append(leg)
        nlv = snapshot.totals.net_liquidating_value or 1.0
        strategy_counts: dict[str, int] = {}
        for strategy in snapshot.strategies:
            strategy_counts[strategy.underlying] = strategy_counts.get(strategy.underlying, 0) + 1
        concentration = sorted(
            (
                ConcentrationSlice(
                    underlying=symbol,
                    market_value=sum(abs(leg.market_value) for leg in items),
                    share_of_portfolio=(
                        sum(abs(leg.market_value) for leg in items) / nlv if nlv else 0
                    ),
                    strategy_count=strategy_counts.get(symbol, 0),
                    net_delta=self.combined_greeks(items).delta or 0,
                )
                for symbol, items in by_underlying.items()
            ),
            key=lambda item: item.market_value,
            reverse=True,
        )
        # Portfolio stress uses aggregated Greeks only (no single underlying path).
        stress = self._portfolio_stress(combined)
        return PortfolioRisk(
            total_delta=combined.delta or 0,
            total_gamma=combined.gamma or 0,
            total_theta=combined.theta or 0,
            total_vega=combined.vega or 0,
            unrealized_pnl=snapshot.totals.unrealized_pnl,
            net_liquidating_value=snapshot.totals.net_liquidating_value,
            account_count=len(snapshot.accounts),
            strategy_count=len(snapshot.strategies),
            position_count=len(legs),
            concentration=concentration[:15],
            stress=stress,
        )

    def stress_scenarios(
        self,
        strategy: StrategySnapshot,
        *,
        underlying_price: float | None = None,
    ) -> list[StressScenario]:
        price = underlying_price or self._implied_underlying(strategy.legs) or 0
        greeks = self.combined_greeks(strategy.legs)
        delta = greeks.delta or 0
        gamma = greeks.gamma or 0
        theta = greeks.theta or 0
        vega = greeks.vega or 0

        def price_shock(pct: float) -> float:
            if not price:
                return 0.0
            move = price * pct
            # Second-order Taylor: delta*dS + 0.5*gamma*dS^2 (dollar Greeks already scaled)
            return delta * move + 0.5 * gamma * (move**2)

        scenarios = [
            StressScenario(
                name="price_down_10",
                label="Price −10%",
                estimated_pnl_change=price_shock(-0.10),
                description="Approximate P/L from a 10% decline using delta/gamma.",
            ),
            StressScenario(
                name="price_up_10",
                label="Price +10%",
                estimated_pnl_change=price_shock(0.10),
                description="Approximate P/L from a 10% rally using delta/gamma.",
            ),
            StressScenario(
                name="price_down_5",
                label="Price −5%",
                estimated_pnl_change=price_shock(-0.05),
                description="Approximate P/L from a 5% decline using delta/gamma.",
            ),
            StressScenario(
                name="price_up_5",
                label="Price +5%",
                estimated_pnl_change=price_shock(0.05),
                description="Approximate P/L from a 5% rally using delta/gamma.",
            ),
            StressScenario(
                name="iv_up_25",
                label="IV +25%",
                estimated_pnl_change=vega * 25,
                description="Approximate P/L from a 25-point IV increase (vega dollars).",
            ),
            StressScenario(
                name="iv_down_25",
                label="IV −25%",
                estimated_pnl_change=vega * -25,
                description="Approximate P/L from a 25-point IV decrease (vega dollars).",
            ),
            StressScenario(
                name="theta_1d",
                label="1-day theta",
                estimated_pnl_change=theta,
                description="Estimated one trading day of time decay.",
            ),
            StressScenario(
                name="theta_7d",
                label="7-day theta",
                estimated_pnl_change=theta * 7,
                description="Estimated seven calendar days of time decay (linear).",
            ),
            StressScenario(
                name="expiration",
                label="Expiration (extrinsic gone)",
                estimated_pnl_change=self._expiration_pnl(strategy.legs, price),
                description="Mark options to intrinsic only; equities unchanged.",
            ),
        ]
        return scenarios

    def _portfolio_stress(self, greeks: CombinedGreeks) -> list[StressScenario]:
        delta = greeks.delta or 0
        gamma = greeks.gamma or 0
        theta = greeks.theta or 0
        vega = greeks.vega or 0
        return [
            StressScenario(
                name="portfolio_delta_down_1pt",
                label="Portfolio −1 point",
                estimated_pnl_change=delta * -1 + 0.5 * gamma,
                description="Approximate portfolio impact of a 1-point underlying move down.",
            ),
            StressScenario(
                name="portfolio_delta_up_1pt",
                label="Portfolio +1 point",
                estimated_pnl_change=delta * 1 + 0.5 * gamma,
                description="Approximate portfolio impact of a 1-point underlying move up.",
            ),
            StressScenario(
                name="portfolio_iv_up_10",
                label="Portfolio IV +10",
                estimated_pnl_change=vega * 10,
                description="Approximate portfolio impact of a 10-point IV rise.",
            ),
            StressScenario(
                name="portfolio_theta_1d",
                label="Portfolio 1-day theta",
                estimated_pnl_change=theta,
                description="Estimated portfolio theta for one day.",
            ),
        ]

    def combined_greeks(self, legs: list[PositionSnapshot]) -> CombinedGreeks:
        """Return share-equivalent delta and dollar gamma/theta/vega for legs."""
        delta = gamma = theta = vega = 0.0
        has_delta = has_gamma = has_theta = has_vega = False
        ivs: list[float] = []
        dtes: list[int] = []
        for leg in legs:
            qty = abs(leg.quantity)
            short = leg.quantity_direction.value == "Short"
            sign = -1 if short else 1
            mult = leg.multiplier
            if leg.delta is not None:
                # Equity delta is 1.0 per share; option delta is per share of underlying.
                if leg.position_type.value == "Equity":
                    delta += sign * qty * leg.delta
                else:
                    delta += sign * qty * mult * leg.delta
                has_delta = True
            if leg.gamma is not None:
                gamma += sign * qty * mult * leg.gamma
                has_gamma = True
            if leg.theta is not None:
                # Short options: negative broker theta becomes positive income.
                adjusted = -leg.theta if short else leg.theta
                theta += adjusted * qty * mult
                has_theta = True
            if leg.vega is not None:
                adjusted_vega = -leg.vega if short else leg.vega
                vega += adjusted_vega * qty * mult
                has_vega = True
            if leg.implied_volatility is not None:
                ivs.append(leg.implied_volatility)
            if leg.days_to_expiration is not None:
                dtes.append(leg.days_to_expiration)
        return CombinedGreeks(
            delta=delta if has_delta else None,
            gamma=gamma if has_gamma else None,
            theta=theta if has_theta else None,
            vega=vega if has_vega else None,
            average_iv=(sum(ivs) / len(ivs)) if ivs else None,
            nearest_dte=min(dtes) if dtes else None,
        )

    @staticmethod
    def _implied_underlying(legs: list[PositionSnapshot]) -> float | None:
        for leg in legs:
            if leg.position_type.value == "Equity" and leg.mark_price is not None:
                return leg.mark_price
        return None

    @staticmethod
    def _distance_to_nearest_strike(
        legs: list[PositionSnapshot],
        price: float | None,
    ) -> float | None:
        if price is None:
            return None
        strikes = [leg.strike_price for leg in legs if leg.strike_price is not None]
        if not strikes:
            return None
        return min(abs(price - strike) for strike in strikes)

    def _structure_bounds(
        self,
        strategy: StrategySnapshot,
        price: float | None,
    ) -> tuple[float | None, float | None, list[float], bool]:
        """Return mark-relative expiration bounds for same-expiration option structures."""

        option_legs = [
            leg
            for leg in strategy.legs
            if leg.strike_price is not None
            and leg.option_type in {"C", "P"}
            and leg.mark_price is not None
        ]
        if not option_legs or len(option_legs) != len(strategy.legs):
            return None, None, [], False
        expirations = {leg.expiration_date for leg in option_legs}
        if len(expirations) != 1:
            # Calendar and diagonal expiration value depends on future volatility.
            return None, None, [], False

        def expiration_change(underlying: float) -> float:
            total = 0.0
            for leg in option_legs:
                strike = leg.strike_price or 0.0
                intrinsic = (
                    max(0.0, underlying - strike)
                    if leg.option_type == "C"
                    else max(0.0, strike - underlying)
                )
                direction = -1 if leg.quantity_direction.value == "Short" else 1
                total += (
                    direction
                    * abs(leg.quantity)
                    * leg.multiplier
                    * (intrinsic - (leg.mark_price or 0.0))
                )
            return total

        points = sorted({0.0, *(float(leg.strike_price or 0) for leg in option_legs)})
        values = [expiration_change(point) for point in points]
        tail_slope = sum(
            (-1 if leg.quantity_direction.value == "Short" else 1)
            * abs(leg.quantity)
            * leg.multiplier
            for leg in option_legs
            if leg.option_type == "C"
        )
        max_profit = None if tail_slope > 0 else max(0.0, max(values))
        max_loss = None if tail_slope < 0 else max(0.0, -min(values))

        roots: list[float] = []
        for left, right, left_value, right_value in zip(
            points,
            points[1:],
            values,
            values[1:],
        ):
            if abs(left_value) < 1e-9:
                roots.append(left)
            if left_value * right_value < 0:
                roots.append(left + (right - left) * (-left_value) / (right_value - left_value))
        if abs(values[-1]) < 1e-9:
            roots.append(points[-1])
        elif tail_slope:
            tail_root = points[-1] - values[-1] / tail_slope
            if tail_root > points[-1]:
                roots.append(tail_root)
        breakevens = sorted({round(root, 8) for root in roots if root >= 0})
        return max_profit, max_loss, breakevens, max_loss is not None

    def _expiration_pnl(self, legs: list[PositionSnapshot], price: float | None) -> float:
        if price is None:
            return 0.0
        total = 0.0
        for leg in legs:
            if leg.strike_price is None or leg.mark_price is None:
                continue
            if leg.option_type == "C":
                intrinsic = max(0.0, price - leg.strike_price)
            elif leg.option_type == "P":
                intrinsic = max(0.0, leg.strike_price - price)
            else:
                continue
            extrinsic = max(0.0, leg.mark_price - intrinsic)
            # Long loses extrinsic; short gains extrinsic at expiration.
            sign = 1 if leg.quantity_direction.value == "Short" else -1
            total += sign * extrinsic * leg.multiplier * abs(leg.quantity)
        return total
