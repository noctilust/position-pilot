"""Market analysis and metrics."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from ..client import get_client


class Trend(str, Enum):
    """Market trend direction."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class IVEnvironment(str, Enum):
    """Implied volatility environment."""
    VERY_LOW = "very_low"      # IV Rank < 15
    LOW = "low"                # IV Rank 15-30
    NORMAL = "normal"          # IV Rank 30-50
    ELEVATED = "elevated"      # IV Rank 50-70
    HIGH = "high"              # IV Rank 70-85
    VERY_HIGH = "very_high"    # IV Rank > 85


@dataclass
class MarketSnapshot:
    """Current market data for a symbol."""
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None

    # Volatility metrics
    iv: Optional[float] = None
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None

    # Derived
    iv_environment: IVEnvironment = IVEnvironment.NORMAL
    spread_percent: Optional[float] = None

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Determine IV environment
        if self.iv_rank is not None:
            if self.iv_rank < 15:
                self.iv_environment = IVEnvironment.VERY_LOW
            elif self.iv_rank < 30:
                self.iv_environment = IVEnvironment.LOW
            elif self.iv_rank < 50:
                self.iv_environment = IVEnvironment.NORMAL
            elif self.iv_rank < 70:
                self.iv_environment = IVEnvironment.ELEVATED
            elif self.iv_rank < 85:
                self.iv_environment = IVEnvironment.HIGH
            else:
                self.iv_environment = IVEnvironment.VERY_HIGH

        # Calculate spread
        if self.bid and self.ask and self.price:
            self.spread_percent = ((self.ask - self.bid) / self.price) * 100


@dataclass
class VolatilityAnalysis:
    """Analysis of volatility conditions."""
    symbol: str
    iv_rank: Optional[float]
    iv_percentile: Optional[float]
    environment: IVEnvironment

    # Strategy suggestions based on IV
    favor_selling: bool = False
    favor_buying: bool = False
    suggestion: str = ""

    def __post_init__(self):
        if self.iv_rank is not None:
            if self.iv_rank >= 50:
                self.favor_selling = True
                self.suggestion = "Elevated IV favors premium selling strategies (short puts, credit spreads, iron condors)"
            elif self.iv_rank <= 25:
                self.favor_buying = True
                self.suggestion = "Low IV favors premium buying strategies (long calls/puts, debit spreads, calendars)"
            else:
                self.suggestion = "Neutral IV environment - consider direction-neutral strategies or wait for better setup"


class MarketAnalyzer:
    """Analyzes market conditions for trading decisions."""

    def __init__(self):
        self.client = get_client()
        self._cache: dict[str, MarketSnapshot] = {}

    def get_snapshot(self, symbol: str, force_refresh: bool = False) -> Optional[MarketSnapshot]:
        """Get current market snapshot for a symbol."""
        symbol = symbol.upper()

        # Check cache (valid for 60 seconds)
        if not force_refresh and symbol in self._cache:
            cached = self._cache[symbol]
            if (datetime.now() - cached.timestamp).seconds < 60:
                return cached

        # Fetch fresh data
        quote = self.client.get_quote(symbol)
        metrics = self.client.get_market_metrics(symbol)

        if not quote:
            return None

        snapshot = MarketSnapshot(
            symbol=symbol,
            price=quote.get("mark") or quote.get("last") or 0,
            bid=quote.get("bid"),
            ask=quote.get("ask"),
            iv=metrics.get("implied_volatility") if metrics else None,
            iv_rank=metrics.get("iv_rank") if metrics else None,
            iv_percentile=metrics.get("iv_percentile") if metrics else None,
        )

        self._cache[symbol] = snapshot
        return snapshot

    def analyze_volatility(self, symbol: str) -> Optional[VolatilityAnalysis]:
        """Analyze volatility conditions for a symbol."""
        snapshot = self.get_snapshot(symbol)
        if not snapshot:
            return None

        return VolatilityAnalysis(
            symbol=symbol,
            iv_rank=snapshot.iv_rank,
            iv_percentile=snapshot.iv_percentile,
            environment=snapshot.iv_environment,
        )

    def get_iv_environment_emoji(self, env: IVEnvironment) -> str:
        """Get emoji representation of IV environment."""
        return {
            IVEnvironment.VERY_LOW: "🟢",
            IVEnvironment.LOW: "🟢",
            IVEnvironment.NORMAL: "🟡",
            IVEnvironment.ELEVATED: "🟠",
            IVEnvironment.HIGH: "🔴",
            IVEnvironment.VERY_HIGH: "🔴",
        }.get(env, "⚪")


# Global singleton
_analyzer: Optional[MarketAnalyzer] = None


def get_analyzer() -> MarketAnalyzer:
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = MarketAnalyzer()
    return _analyzer
