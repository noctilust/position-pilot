"""Cache management for AI recommendations with persistent storage."""

import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..models.position import Position, Recommendation


class RecommendationCache:
    """Persistent cache for AI recommendations (no expiration)."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the recommendation cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cache/position-pilot/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "position-pilot"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "recommendations.json"
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime
                    for key, value in data.items():
                        if "generated_at" in value:
                            value["generated_at"] = datetime.fromisoformat(value["generated_at"])
                    self._cache = data
            except (json.JSONDecodeError, ValueError, KeyError):
                # Invalid cache, start fresh
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        # Convert datetime objects to strings for JSON serialization
        serializable_cache = {}
        for key, value in self._cache.items():
            serializable_cache[key] = value.copy()
            if "generated_at" in serializable_cache[key]:
                serializable_cache[key]["generated_at"] = serializable_cache[key]["generated_at"].isoformat()

        with open(self.cache_file, "w") as f:
            json.dump(serializable_cache, f, indent=2)

    def generate_cache_key(self, position: Position) -> str:
        """Generate a unique cache key for a position.

        The key includes factors that would change the recommendation:
        - Symbol
        - Quantity
        - Strike price (for options)
        - Days to expiration
        - Option type (call/put)

        Args:
            position: The position to generate a key for

        Returns:
            String cache key
        """
        key_parts = [
            position.symbol,
            str(position.quantity),
        ]

        if position.is_option:
            key_parts.extend([
                str(position.strike_price),
                str(position.days_to_expiration or 0),
            ])

        return ":".join(key_parts)

    def get(self, position: Position) -> Optional[tuple[Recommendation, datetime]]:
        """Get cached recommendation if available.

        Args:
            position: The position to get cached recommendation for

        Returns:
            Tuple of (Recommendation, generated_at timestamp) if cached, None otherwise
        """
        cache_key = self.generate_cache_key(position)

        if cache_key not in self._cache:
            return None

        cached_item = self._cache[cache_key]

        # Reconstruct Recommendation object from cached data
        rec_data = cached_item["recommendation"]
        recommendation = Recommendation(
            position=position,
            signal=rec_data["signal"],
            reason=rec_data["reason"],
            urgency=rec_data["urgency"],
            suggested_action=rec_data.get("suggested_action"),
            risk_notes=rec_data.get("risk_notes"),
        )

        generated_at = cached_item["generated_at"]

        return (recommendation, generated_at)

    def set(self, position: Position, recommendation: Recommendation) -> None:
        """Cache a recommendation with timestamp.

        Args:
            position: The position the recommendation is for
            recommendation: The recommendation to cache
        """
        cache_key = self.generate_cache_key(position)
        generated_at = datetime.now()

        self._cache[cache_key] = {
            "recommendation": {
                "signal": recommendation.signal.value,
                "reason": recommendation.reason,
                "urgency": recommendation.urgency,
                "suggested_action": recommendation.suggested_action,
                "risk_notes": recommendation.risk_notes,
            },
            "generated_at": generated_at,
            "position_snapshot": {
                "symbol": position.symbol,
                "quantity": position.quantity,
                "strike_price": position.strike_price,
                "days_to_expiration": position.days_to_expiration,
                "unrealized_pnl_percent": position.unrealized_pnl_percent,
            }
        }

        self._save_cache()

    def clear(self) -> None:
        """Clear all cached recommendations."""
        self._cache.clear()
        self._save_cache()

    def get_cache_info(self) -> dict:
        """Get information about the cache.

        Returns:
            Dictionary with cache information
        """
        # Calculate age of oldest and newest recommendations
        timestamps = [
            value["generated_at"]
            for value in self._cache.values()
            if "generated_at" in value
        ]

        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None

        return {
            "total": len(self._cache),
            "oldest": oldest.isoformat() if oldest else None,
            "newest": newest.isoformat() if newest else None,
        }


# Global cache instance
_recommendation_cache: Optional[RecommendationCache] = None


def get_recommendation_cache() -> RecommendationCache:
    """Get or create the global recommendation cache.

    Returns:
        RecommendationCache instance
    """
    global _recommendation_cache
    if _recommendation_cache is None:
        _recommendation_cache = RecommendationCache()
    return _recommendation_cache
