"""Provider contracts with independent health and field provenance."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ProviderState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


class ProviderHealth(BaseModel):
    provider: str
    state: ProviderState
    checked_at: datetime | None = None
    last_success_at: datetime | None = None
    error: str | None = None


class ProviderDiscrepancy(BaseModel):
    provider: str
    selected_value: Any
    other_value: Any


class ProviderValue(BaseModel):
    field: str
    value: Any
    provider: str
    observed_at: datetime
    fallback_reason: str | None = None
    discrepancies: list[ProviderDiscrepancy] = Field(default_factory=list)


class FieldProvider(Protocol):
    name: str

    def health(self) -> ProviderHealth: ...

    def fetch(self, field: str, symbol: str) -> ProviderValue | None: ...


class MarketDataProvider(Protocol):
    name: str

    def quote(self, symbol: str) -> ProviderValue | None: ...

    def bars(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None: ...

    def health(self) -> ProviderHealth: ...


class OptionsDataProvider(Protocol):
    name: str

    def option_snapshot(self, symbol: str) -> ProviderValue | None: ...

    def option_bars(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None: ...

    def health(self) -> ProviderHealth: ...


class NewsProvider(Protocol):
    name: str

    def news(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None: ...

    def health(self) -> ProviderHealth: ...
