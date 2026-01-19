"""LLM-driven position analysis and trading signals using Claude."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from anthropic import Anthropic

from ..models.position import Position, Signal, Recommendation
from .market import get_analyzer


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


class LLMPositionAnalyzer:
    """Analyzes positions and generates recommendations using Claude LLM."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM analyzer with Anthropic API key.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.market = get_analyzer()

        # Model configuration
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens = 1024
        self.timeout = 10  # seconds

    def assess_health(self, position: Position) -> PositionHealth:
        """Assess the health of a position using LLM analysis.

        This uses the LLM to evaluate position health, risks, and opportunities
        rather than using fixed rule-based thresholds.
        """
        prompt = self._build_health_assessment_prompt(position)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse the LLM response into PositionHealth
            return self._parse_health_response(position, response.content[0].text)

        except Exception as e:
            # Since we don't have rule-based fallback, we create a neutral assessment
            # with a note about the analysis failure
            return PositionHealth(
                position=position,
                risk_level=RiskLevel.MODERATE,
                issues=[f"Analysis error: {str(e)}"],
            )

    def generate_recommendation(self, position: Position) -> Recommendation:
        """Generate a trading recommendation using LLM analysis.

        Args:
            position: The position to analyze

        Returns:
            Recommendation with signal, reasoning, and suggested action

        Raises:
            Exception: If LLM analysis fails (no fallback to rules)
        """
        # Gather market context
        market_context = self._gather_market_context(position)

        # Build comprehensive prompt
        prompt = self._build_recommendation_prompt(position, market_context)

        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse structured response
        return self._parse_recommendation_response(position, response.content[0].text)

    def analyze_portfolio(self, positions: list[Position]) -> dict:
        """Analyze entire portfolio and return summary using LLM insights.

        Args:
            positions: List of positions to analyze

        Returns:
            Dictionary with recommendations, risk summary, and portfolio metrics
        """
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
            try:
                rec = self.generate_recommendation(pos)
                recommendations.append(rec)

                health = self.assess_health(pos)
                risk_counts[health.risk_level] += 1

                # Calculate portfolio Greeks
                if pos.greeks:
                    if pos.greeks.theta:
                        total_theta += pos.greeks.theta * pos.multiplier * abs(pos.quantity)
                    if pos.greeks.delta:
                        qty = pos.quantity if not pos.is_short else -pos.quantity
                        total_delta += pos.greeks.delta * pos.multiplier * qty

            except Exception as e:
                # Log error but continue processing other positions
                # Create a placeholder recommendation indicating analysis failure
                recommendations.append(
                    Recommendation(
                        position=pos,
                        signal=Signal.HOLD,
                        reason=f"Analysis unavailable: {str(e)}",
                        urgency=1,
                        suggested_action="Review position manually",
                    )
                )

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

    def _gather_market_context(self, position: Position) -> dict:
        """Gather market context for the position."""
        context = {
            "underlying_price": position.underlying_price or 0,
            "iv_rank": None,
            "iv_percentile": None,
            "distance_to_strike": None,
        }

        try:
            # Get market snapshot for IV rank
            snapshot = self.market.get_snapshot(position.underlying_symbol)
            if snapshot:
                context["iv_rank"] = snapshot.iv_rank
                context["iv_percentile"] = snapshot.iv_percentile

            # Calculate distance to strike
            if position.is_option and position.strike_price and position.underlying_price:
                distance = ((position.strike_price - position.underlying_price) /
                           position.underlying_price * 100)
                context["distance_to_strike"] = distance

        except Exception:
            pass  # Market context is optional

        return context

    def _build_health_assessment_prompt(self, position: Position) -> str:
        """Build prompt for health assessment."""
        return f"""You are an expert options trading analyst. Assess the health of this position:

## Position Details
- Symbol: {position.symbol}
- Underlying: {position.underlying_symbol}
- Type: {"Short " if position.is_short else "Long"} {position.option_type if position.is_option else position.position_type}
- Quantity: {position.display_quantity}

{"## Option Details" if position.is_option else "## Equity Details"}
{"- Strike: ${position.strike_price}" if position.is_option else ""}
{"- Expiration: {position.expiration_date}" if position.is_option else ""}
{"- Days to Expiration: {position.days_to_expiration}" if position.is_option else ""}

## Pricing & P/L
- Open Price: ${position.average_open_price:.2f}
- Current Price: ${position.mark_price or position.close_price:.2f}
- Unrealized P/L: ${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_percent:+.1f}%)

{"## Greeks" if position.greeks else ""}
{"- Delta: {position.greeks.delta:.2f}" if position.greeks and position.greeks.delta else ""}
{"- Theta: {position.greeks.theta:.3f}" if position.greeks and position.greeks.theta else ""}
{"- Vega: {position.greeks.vega:.3f}" if position.greeks and position.greeks.vega else ""}
{"- Implied Volatility: {position.greeks.implied_volatility:.1%}" if position.greeks and position.greeks.implied_volatility else ""}

Assess the position health and respond in this exact format:

RISK_LEVEL: [LOW/MODERATE/HIGH/CRITICAL]
ISSUES: [comma-separated list of issues, or "None"]
OPPORTUNITIES: [comma-separated list of opportunities, or "None"]
DTE_RISK: [true/false]
PNL_RISK: [true/false]
DELTA_RISK: [true/false]
THETA_STATUS: [brief description of theta impact, or "N/A"]

Consider:
- Time decay (theta) and its daily impact on P/L
- Delta exposure and directional risk
- Gamma risk for short options
- Expiration proximity and assignment risk
- P/L trajectory and trend
- Intrinsic vs extrinsic value composition"""

    def _build_recommendation_prompt(self, position: Position, market_context: dict) -> str:
        """Build comprehensive prompt for trading recommendation."""
        # Format market context
        iv_info = ""
        if market_context.get("iv_rank"):
            iv_info = f"- IV Rank: {market_context['iv_rank']:.0f} "
            if market_context.get("iv_percentile"):
                iv_info += f"({market_context['iv_percentile']:.0f}th percentile)"

        distance_info = ""
        if market_context.get("distance_to_strike") is not None:
            distance_info = f"- Distance to Strike: {market_context['distance_to_strike']:+.1f}%"

        # Calculate theta/vega dollar impact
        theta_daily = ""
        if position.greeks and position.greeks.theta and position.is_option:
            theta_dollars = position.greeks.theta * position.multiplier * abs(position.quantity)
            theta_daily = f" (${theta_dollars:+.2f}/day)"

        return f"""You are an expert options trader and risk manager. Provide a trading recommendation for this position:

## Position Details
- Symbol: {position.symbol}
- Underlying: {position.underlying_symbol}
- Type: {"Short " if position.is_short else "Long"} {position.option_type if position.is_option else position.position_type}
- Quantity: {position.display_quantity}

{"## Option Details" if position.is_option else "## Equity Details"}
{"- Strike: ${position.strike_price}" if position.is_option else ""}
{"- Expiration: {position.expiration_date} ({position.days_to_expiration} DTE)" if position.is_option else ""}

## Pricing & P/L
- Cost Basis: ${position.cost_basis:,.2f}
- Market Value: ${position.market_value:,.2f}
- Unrealized P/L: ${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_percent:+.1f}%)

{"## Greeks" if position.greeks else ""}
{"- Delta: {position.greeks.delta:.2f}" if position.greeks and position.greeks.delta else ""}
{"- Gamma: {position.greeks.gamma:.3f}" if position.greeks and position.greeks.gamma else ""}
{"- Theta: {position.greeks.theta:.3f}{theta_daily}" if position.greeks and position.greeks.theta else ""}
{"- Vega: {position.greeks.vega:.3f}" if position.greeks and position.greeks.vega else ""}
{"- Implied Volatility: {position.greeks.implied_volatility:.1%}" if position.greeks and position.greeks.implied_volatility else ""}

## Value Breakdown
{"- Intrinsic Value: ${position.intrinsic_value:.2f}" if position.intrinsic_value else ""}
{"- Extrinsic Value: ${position.extrinsic_value:.2f}" if position.extrinsic_value else ""}

## Market Context
- Underlying Price: ${market_context['underlying_price']:.2f}
{iv_info}
{distance_info}

Provide your recommendation in this exact format:

SIGNAL: [HOLD/ROLL/SELL/CLOSE/BUY/STRONG_BUY/STRONG_SELL]
REASON: [2-3 sentence explanation of the rationale]
URGENCY: [1-5, where 1=low priority, 5=critical/immediate action needed]
SUGGESTED_ACTION: [specific actionable step, e.g., "Roll to 45 DTE", "Close 50% of position"]
ALTERNATIVE_ACTIONS: [2-3 alternative strategies to consider]

Consider:
- Theta decay and time value erosion
- Gamma risk and delta acceleration
- IV exposure and vega risk
- Expiration proximity and assignment probability
- P/L momentum and trend
- Portfolio correlation (if part of spread)
- Market volatility regime (IV rank context)
- Risk/reward ratio for holding vs closing/rolling

Be decisive but prudent. Prioritize capital preservation over profit maximization."""

    def _parse_health_response(self, position: Position, response_text: str) -> PositionHealth:
        """Parse LLM response into PositionHealth object."""
        health = PositionHealth(position=position)

        try:
            # Parse the structured response
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('RISK_LEVEL:'):
                    level_str = line.split(':', 1)[1].strip().upper()
                    health.risk_level = RiskLevel[level_str]
                elif line.startswith('ISSUES:'):
                    issues_str = line.split(':', 1)[1].strip()
                    if issues_str and issues_str.upper() != 'NONE':
                        health.issues = [i.strip() for i in issues_str.split(',')]
                elif line.startswith('OPPORTUNITIES:'):
                    opps_str = line.split(':', 1)[1].strip()
                    if opps_str and opps_str.upper() != 'NONE':
                        health.opportunities = [o.strip() for o in opps_str.split(',')]
                elif line.startswith('DTE_RISK:'):
                    health.dte_risk = line.split(':', 1)[1].strip().lower() == 'true'
                elif line.startswith('PNL_RISK:'):
                    health.pnl_risk = line.split(':', 1)[1].strip().lower() == 'true'
                elif line.startswith('DELTA_RISK:'):
                    health.delta_risk = line.split(':', 1)[1].strip().lower() == 'true'
                elif line.startswith('THETA_STATUS:'):
                    health.theta_status = line.split(':', 1)[1].strip()

        except Exception as e:
            # If parsing fails, return neutral assessment with error note
            health.issues = [f"Failed to parse LLM response: {str(e)}"]
            health.risk_level = RiskLevel.MODERATE

        return health

    def _parse_recommendation_response(self, position: Position, response_text: str) -> Recommendation:
        """Parse LLM response into Recommendation object."""
        signal = Signal.HOLD
        reason = "Analysis complete"
        urgency = 3
        suggested_action = None
        alternative_actions = None

        try:
            # Parse the structured response
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('SIGNAL:'):
                    signal_str = line.split(':', 1)[1].strip().upper()
                    # Map signal string to Signal enum
                    signal_map = {
                        'HOLD': Signal.HOLD,
                        'ROLL': Signal.ROLL,
                        'SELL': Signal.SELL,
                        'CLOSE': Signal.CLOSE,
                        'BUY': Signal.BUY,
                        'STRONG_BUY': Signal.STRONG_BUY,
                        'STRONG_SELL': Signal.STRONG_SELL,
                    }
                    signal = signal_map.get(signal_str, Signal.HOLD)
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()
                elif line.startswith('URGENCY:'):
                    urgency_str = line.split(':', 1)[1].strip()
                    urgency = int(urgency_str)
                    # Clamp urgency to 1-5 range
                    urgency = max(1, min(5, urgency))
                elif line.startswith('SUGGESTED_ACTION:'):
                    suggested_action = line.split(':', 1)[1].strip()
                elif line.startswith('ALTERNATIVE_ACTIONS:'):
                    alt_str = line.split(':', 1)[1].strip()
                    if alt_str and alt_str.upper() != 'NONE':
                        alternative_actions = [a.strip() for a in alt_str.split(',')]

        except Exception as e:
            # If parsing fails, return conservative recommendation
            return Recommendation(
                position=position,
                signal=Signal.HOLD,
                reason=f"Parsing error: {str(e)}",
                urgency=1,
                suggested_action="Review position manually",
            )

        return Recommendation(
            position=position,
            signal=signal,
            reason=reason,
            urgency=urgency,
            suggested_action=suggested_action,
            risk_notes=alternative_actions[0] if alternative_actions else None,
        )


# Global singleton
_llm_analyzer: Optional[LLMPositionAnalyzer] = None


def get_llm_analyzer(api_key: Optional[str] = None) -> LLMPositionAnalyzer:
    """Get or create the global LLM position analyzer.

    Args:
        api_key: Optional Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        LLMPositionAnalyzer instance

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set in environment or passed as parameter.
    """
    global _llm_analyzer
    if _llm_analyzer is None:
        _llm_analyzer = LLMPositionAnalyzer(api_key)
    return _llm_analyzer
