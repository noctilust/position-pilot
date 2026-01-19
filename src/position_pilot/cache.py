"""Simple cache with TTL for API responses."""

import json
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta


class Cache:
    """In-memory cache with TTL and optional disk persistence."""

    def __init__(self, ttl_seconds: int = 600, cache_dir: Optional[Path] = None):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time to live in seconds (default: 600 = 10 minutes)
            cache_dir: Optional directory for persistent cache
        """
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir
        self._memory_cache: dict[str, tuple[Any, float]] = {}

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.

        Returns None if key not found or expired.
        """
        # Try memory cache first
        if key in self._memory_cache:
            value, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                # Expired, remove it
                del self._memory_cache[key]

        # Try disk cache if enabled
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self._sanitize_key(key)}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        cached = json.load(f)
                    timestamp = cached.get("timestamp", 0)
                    if time.time() - timestamp < self.ttl_seconds:
                        value = cached.get("value")
                        # Store in memory for faster access
                        self._memory_cache[key] = (value, timestamp)
                        return value
                    else:
                        # Expired, delete file
                        cache_file.unlink()
                except (json.JSONDecodeError, IOError):
                    pass

        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp."""
        timestamp = time.time()
        self._memory_cache[key] = (value, timestamp)

        # Persist to disk if enabled
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self._sanitize_key(key)}.json"
            try:
                with open(cache_file, "w") as f:
                    json.dump({"value": value, "timestamp": timestamp}, f)
            except (IOError, TypeError):
                # Ignore errors (some objects may not be JSON serializable)
                pass

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()

        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except IOError:
                    pass

    def _sanitize_key(self, key: str) -> str:
        """Sanitize key for use as filename."""
        # Replace special characters with underscores
        return "".join(c if c.isalnum() or c in ".-_" else "_" for c in key)

    def get_age(self, key: str) -> Optional[timedelta]:
        """Get age of cached value, or None if not cached."""
        if key in self._memory_cache:
            _, timestamp = self._memory_cache[key]
            return timedelta(seconds=time.time() - timestamp)

        if self.cache_dir:
            cache_file = self.cache_dir / f"{self._sanitize_key(key)}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        cached = json.load(f)
                    timestamp = cached.get("timestamp", 0)
                    return timedelta(seconds=time.time() - timestamp)
                except (json.JSONDecodeError, IOError):
                    pass

        return None
