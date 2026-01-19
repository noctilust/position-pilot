"""LLM-driven position analysis and trading signals using Claude Sonnet 4.5."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from anthropic import Anthropic

from ..models.position import Position, Signal, Recommendation
from .market import get_analyzer
from .recommendation_cache import get_recommendation_cache


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
    """Analyzes positions and generates recommendations using Claude Sonnet 4.5."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM analyzer with Anthropic API key.

        Uses Claude Sonnet 4.5 for trading recommendations, offering enhanced
        reasoning capabilities for complex options strategies.

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
        self.model = "claude-sonnet-4-5-20250929"  # Claude Sonnet 4.5
        self.max_tokens = 2048  # Increased for response parsing
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
            error_msg = str(e)
            # Handle specific error types
            if "content-length" in error_msg.lower() or "too large" in error_msg.lower():
                return PositionHealth(
                    position=position,
                    risk_level=RiskLevel.MODERATE,
                    issues=["Request too large - unable to analyze"],
                )
            else:
                return PositionHealth(
                    position=position,
                    risk_level=RiskLevel.MODERATE,
                    issues=[f"Analysis error: {error_msg}"],
                )

    def generate_recommendation(self, position: Position, force_refresh: bool = False) -> tuple[Recommendation, datetime]:
        """Generate a trading recommendation using LLM analysis.

        Args:
            position: The position to analyze
            force_refresh: If True, bypass cache and generate new recommendation

        Returns:
            Tuple of (Recommendation, generated_at timestamp)

        Raises:
            Exception: If LLM analysis fails (no fallback to rules)
        """
        # Check cache first (unless force_refresh)
        cache = get_recommendation_cache()
        if not force_refresh:
            cached_result = cache.get(position)
            if cached_result:
                return cached_result

        # Gather market context
        market_context = self._gather_market_context(position)

        # Build comprehensive prompt
        prompt = self._build_recommendation_prompt(position, market_context)

        # Call Claude API with error handling
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # Parse structured response
            recommendation = self._parse_recommendation_response(position, response.content[0].text)

            # Cache the recommendation
            cache.set(position, recommendation)
            generated_at = datetime.now()

            return (recommendation, generated_at)
        except Exception as e:
            error_msg = str(e)
            # Handle specific error types
            if "content-length" in error_msg.lower() or "too large" in error_msg.lower():
                rec = Recommendation(
                    position=position,
                    signal=Signal.HOLD,
                    reason="Unable to analyze - request size exceeded",
                    urgency=1,
                    suggested_action="Review position manually",
                )
                return (rec, datetime.now())
            else:
                # Re-raise other exceptions
                raise

    def analyze_portfolio(self, positions: list[Position]) -> dict:
        """Analyze entire portfolio and return summary (without generating recommendations).

        Use generate_recommendation() explicitly to get AI recommendations for specific positions.

        Args:
            positions: List of positions to analyze

        Returns:
            Dictionary with portfolio metrics (no recommendations)
        """
        if not positions:
            return {
                "total_positions": 0,
                "total_theta": 0,
                "total_delta": 0,
            }

        risk_counts = {level: 0 for level in RiskLevel}
        total_theta = 0.0
        total_delta = 0.0

        for pos in positions:
            # Only assess health (no LLM recommendation)
            health = self.assess_health(pos)
            risk_counts[health.risk_level] += 1

            # Calculate portfolio Greeks
            if pos.greeks:
                if pos.greeks.theta:
                    # Adjust theta for short positions (positive theta = gains from time decay)
                    adjusted_theta = -pos.greeks.theta if pos.is_short else pos.greeks.theta
                    total_theta += adjusted_theta * pos.multiplier * abs(pos.quantity)
                if pos.greeks.delta:
                    qty = pos.quantity if not pos.is_short else -pos.quantity
                    total_delta += pos.greeks.delta * pos.multiplier * qty

        cache = get_recommendation_cache()
        cache_info = cache.get_cache_info()

        return {
            "total_positions": len(positions),
            "risk_summary": risk_counts,
            "total_theta": total_theta,
            "total_delta": total_delta,
            "critical_count": risk_counts[RiskLevel.CRITICAL],
            "high_risk_count": risk_counts[RiskLevel.HIGH],
            "cache_info": cache_info,
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
        """Build compact prompt for health assessment."""
        # Build compact position summary
        pos_type = f"{'Short ' if position.is_short else 'Long'}{position.option_type if position.is_option else position.position_type}"

        details = [
            f"Symbol: {position.symbol}",
            f"Type: {pos_type}",
            f"Qty: {position.display_quantity}",
        ]

        if position.is_option:
            details.extend([
                f"Strike: ${position.strike_price}",
                f"DTE: {position.days_to_expiration}",
            ])

        details.extend([
            f"P/L: ${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_percent:+.1f}%)",
        ])

        if position.greeks:
            # Adjust delta for short positions to show actual portfolio exposure
            if position.greeks.delta:
                adjusted_delta = -position.greeks.delta if position.is_short else position.greeks.delta
                details.append(f"Delta: {adjusted_delta:.2f}")
            # Adjust theta for short positions (positive theta = gains from time decay)
            if position.greeks.theta:
                adjusted_theta = -position.greeks.theta if position.is_short else position.greeks.theta
                details.append(f"Theta: {adjusted_theta:.3f}")

        prompt = """Analyze this options position:

{details}

Respond in this format:
RISK_LEVEL: [LOW/MODERATE/HIGH/CRITICAL]
ISSUES: [list or "None"]
OPPORTUNITIES: [list or "None"]
DTE_RISK: [true/false]
PNL_RISK: [true/false]
DELTA_RISK: [true/false]
THETA_STATUS: [description or "N/A"]

Focus on: theta decay, delta exposure, expiration risk, P/L trend.""".format(details="\n".join(details))

        return prompt

    def _build_recommendation_prompt(self, position: Position, market_context: dict) -> str:
        """Build compact prompt for trading recommendation."""
        # Build compact position summary
        pos_type = f"{'Short ' if position.is_short else 'Long'}{position.option_type if position.is_option else position.position_type}"

        details = [
            f"Symbol: {position.symbol}",
            f"Type: {pos_type}",
            f"Qty: {position.display_quantity}",
        ]

        if position.is_option:
            details.extend([
                f"Strike: ${position.strike_price}",
                f"DTE: {position.days_to_expiration}",
            ])

        details.extend([
            f"P/L: ${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_percent:+.1f}%)",
        ])

        if position.greeks:
            # Adjust delta for short positions to show actual portfolio exposure
            if position.greeks.delta:
                adjusted_delta = -position.greeks.delta if position.is_short else position.greeks.delta
                details.append(f"Delta: {adjusted_delta:.2f}")
            # Adjust theta for short positions (positive theta = gains from time decay)
            if position.greeks.theta:
                adjusted_theta = -position.greeks.theta if position.is_short else position.greeks.theta
                theta_daily = adjusted_theta * position.multiplier * abs(position.quantity)
                details.append(f"Theta: {adjusted_theta:.3f} (${theta_daily:+.2f}/day)")

        # Market context
        if market_context.get("iv_rank"):
            details.append(f"IV Rank: {market_context['iv_rank']:.0f}")
        if market_context.get("distance_to_strike") is not None:
            details.append(f"Distance to strike: {market_context['distance_to_strike']:+.1f}%")

        prompt = """Provide trading recommendation:

{details}

Respond in this format:
SIGNAL: [HOLD/ROLL/SELL/CLOSE/BUY]
REASON: [2-3 sentence rationale]
URGENCY: [1-5]
SUGGESTED_ACTION: [specific step]
ALTERNATIVE_ACTIONS: [2-3 alternatives]

Consider: theta decay, gamma risk, IV exposure, expiration proximity, P/L trend.
Prioritize capital preservation.""".format(details="\n".join(details))

        return prompt

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
