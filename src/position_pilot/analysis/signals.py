"""Position analysis and trading signals."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ..models.position import Position, Signal, Recommendation
from .market import get_analyzer, IVEnvironment


class RiskLevel(str, Enum):
    """Risk level assessment."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionHealth:
    """Health assessment of a position."""
    position: Position
    risk_level: RiskLevel = RiskLevel.MODERATE
    issues: list[str] = field(default_factory=list)
    opportunities: list[str] = field(default_factory=list)

    # Key metrics
    dte_risk: bool = False
    pnl_risk: bool = False
    delta_risk: bool = False
    theta_status: str = ""

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)


class PositionAnalyzer:
    """Analyzes positions and generates recommendations."""

    # Thresholds
    DTE_WARNING = 21  # Days to expiration warning
    DTE_CRITICAL = 7  # Days to expiration critical
    LOSS_WARNING_PCT = -25  # P/L % warning
    LOSS_CRITICAL_PCT = -50  # P/L % critical
    PROFIT_TARGET_PCT = 50  # Take profit suggestion
    DELTA_HIGH = 0.70  # High delta warning

    def __init__(self):
        self.market = get_analyzer()

    def assess_health(self, position: Position) -> PositionHealth:
        """Assess the health of a position."""
        health = PositionHealth(position=position)

        if not position.is_option:
            # Stock position - simpler analysis
            self._assess_stock_health(position, health)
        else:
            # Option position - full analysis
            self._assess_option_health(position, health)

        # Determine overall risk level
        health.risk_level = self._calculate_risk_level(health)

        return health

    def _assess_stock_health(self, position: Position, health: PositionHealth):
        """Assess stock position health."""
        # P/L assessment
        if position.unrealized_pnl_percent is not None:
            if position.unrealized_pnl_percent <= self.LOSS_CRITICAL_PCT:
                health.issues.append(f"Large unrealized loss ({position.unrealized_pnl_percent:.1f}%)")
                health.pnl_risk = True
            elif position.unrealized_pnl_percent <= self.LOSS_WARNING_PCT:
                health.issues.append(f"Significant loss ({position.unrealized_pnl_percent:.1f}%)")
                health.pnl_risk = True
            elif position.unrealized_pnl_percent >= self.PROFIT_TARGET_PCT:
                health.opportunities.append(f"Consider taking profits ({position.unrealized_pnl_percent:.1f}% gain)")

    def _assess_option_health(self, position: Position, health: PositionHealth):
        """Assess option position health."""
        # DTE assessment
        if position.days_to_expiration is not None:
            if position.days_to_expiration <= self.DTE_CRITICAL:
                health.issues.append(f"Expiring soon ({position.days_to_expiration} DTE)")
                health.dte_risk = True
            elif position.days_to_expiration <= self.DTE_WARNING:
                health.issues.append(f"Approaching expiration ({position.days_to_expiration} DTE)")
                health.dte_risk = True

        # P/L assessment
        if position.unrealized_pnl_percent is not None:
            if position.unrealized_pnl_percent <= self.LOSS_CRITICAL_PCT:
                health.issues.append(f"Large loss ({position.unrealized_pnl_percent:.1f}%)")
                health.pnl_risk = True
            elif position.unrealized_pnl_percent <= self.LOSS_WARNING_PCT:
                health.issues.append(f"Significant loss ({position.unrealized_pnl_percent:.1f}%)")
                health.pnl_risk = True
            elif position.unrealized_pnl_percent >= self.PROFIT_TARGET_PCT:
                health.opportunities.append(f"At profit target ({position.unrealized_pnl_percent:.1f}%)")

        # Greeks assessment
        if position.greeks:
            # Delta risk (for short options)
            if position.greeks.delta is not None:
                abs_delta = abs(position.greeks.delta)
                if position.is_short and abs_delta >= self.DELTA_HIGH:
                    health.issues.append(f"High delta risk ({position.greeks.delta:.2f})")
                    health.delta_risk = True

            # Theta status
            if position.greeks.theta is not None:
                daily_theta = position.greeks.theta * position.multiplier * abs(position.quantity)
                if position.is_short:
                    # Short options benefit from theta decay
                    if daily_theta < 0:  # Theta is typically negative
                        health.theta_status = f"Earning ${abs(daily_theta):.2f}/day from theta"
                        health.opportunities.append(health.theta_status)
                else:
                    # Long options lose to theta decay
                    if daily_theta < 0:
                        health.theta_status = f"Losing ${abs(daily_theta):.2f}/day to theta"
                        if abs(daily_theta) > 10:  # Significant daily loss
                            health.issues.append(health.theta_status)

    def _calculate_risk_level(self, health: PositionHealth) -> RiskLevel:
        """Calculate overall risk level."""
        critical_factors = sum([
            health.dte_risk and health.position.days_to_expiration <= self.DTE_CRITICAL,
            health.pnl_risk and (health.position.unrealized_pnl_percent or 0) <= self.LOSS_CRITICAL_PCT,
        ])

        warning_factors = sum([
            health.dte_risk,
            health.pnl_risk,
            health.delta_risk,
        ])

        if critical_factors >= 1:
            return RiskLevel.CRITICAL
        elif warning_factors >= 2:
            return RiskLevel.HIGH
        elif warning_factors >= 1:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def generate_recommendation(self, position: Position) -> Recommendation:
        """Generate a trading recommendation for a position."""
        health = self.assess_health(position)

        # Determine signal and reason
        signal, reason, action, urgency = self._determine_signal(position, health)

        return Recommendation(
            position=position,
            signal=signal,
            reason=reason,
            urgency=urgency,
            suggested_action=action,
            risk_notes="; ".join(health.issues) if health.issues else None,
        )

    def _determine_signal(self, position: Position, health: PositionHealth) -> tuple[Signal, str, str, int]:
        """Determine the appropriate signal for a position."""
        # Critical situations
        if health.risk_level == RiskLevel.CRITICAL:
            if health.dte_risk and position.days_to_expiration <= self.DTE_CRITICAL:
                return (
                    Signal.CLOSE,
                    f"Position expiring in {position.days_to_expiration} days",
                    "Close position to avoid expiration risk",
                    5,
                )
            if health.pnl_risk:
                return (
                    Signal.CLOSE,
                    f"Large loss of {position.unrealized_pnl_percent:.1f}%",
                    "Consider closing to limit further losses",
                    4,
                )

        # Roll opportunities
        if position.is_option and health.dte_risk and not health.pnl_risk:
            if position.unrealized_pnl_percent and position.unrealized_pnl_percent > 0:
                return (
                    Signal.ROLL,
                    f"Profitable position ({position.unrealized_pnl_percent:.1f}%) approaching expiration",
                    "Roll to later expiration to maintain position",
                    3,
                )

        # Profit taking
        if health.opportunities and position.unrealized_pnl_percent:
            if position.unrealized_pnl_percent >= self.PROFIT_TARGET_PCT:
                return (
                    Signal.SELL,
                    f"At {position.unrealized_pnl_percent:.1f}% profit target",
                    "Consider taking profits",
                    2,
                )

        # High risk situations
        if health.risk_level == RiskLevel.HIGH:
            issues = "; ".join(health.issues[:2])
            return (
                Signal.SELL,
                f"Multiple risk factors: {issues}",
                "Review position and consider reducing exposure",
                3,
            )

        # Default: hold
        return (
            Signal.HOLD,
            "Position within normal parameters",
            "Continue monitoring",
            1,
        )

    def analyze_portfolio(self, positions: list[Position]) -> dict:
        """Analyze entire portfolio and return summary."""
        if not positions:
            return {
                "total_positions": 0,
                "recommendations": [],
                "risk_summary": {},
                "total_theta": 0,
                "total_delta": 0,
            }

        recommendations = []
        risk_counts = {level: 0 for level in RiskLevel}
        total_theta = 0.0
        total_delta = 0.0

        for pos in positions:
            rec = self.generate_recommendation(pos)
            recommendations.append(rec)

            health = self.assess_health(pos)
            risk_counts[health.risk_level] += 1

            if pos.greeks:
                if pos.greeks.theta:
                    total_theta += pos.greeks.theta * pos.multiplier * abs(pos.quantity)
                if pos.greeks.delta:
                    qty = pos.quantity if not pos.is_short else -pos.quantity
                    total_delta += pos.greeks.delta * pos.multiplier * qty

        # Sort recommendations by urgency
        recommendations.sort(key=lambda r: r.urgency, reverse=True)

        return {
            "total_positions": len(positions),
            "recommendations": recommendations,
            "risk_summary": risk_counts,
            "total_theta": total_theta,
            "total_delta": total_delta,
            "critical_count": risk_counts[RiskLevel.CRITICAL],
            "high_risk_count": risk_counts[RiskLevel.HIGH],
        }


# Global singleton
_position_analyzer: Optional[PositionAnalyzer] = None


def get_position_analyzer() -> PositionAnalyzer:
    """Get or create the global position analyzer."""
    global _position_analyzer
    if _position_analyzer is None:
        _position_analyzer = PositionAnalyzer()
    return _position_analyzer
