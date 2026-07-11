"""Field-specific provider routing without silent aggregation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .contracts import (
    FieldProvider,
    ProviderDiscrepancy,
    ProviderHealth,
    ProviderState,
    ProviderValue,
)


class FieldRouter:
    def __init__(
        self,
        *,
        providers: dict[str, FieldProvider],
        routes: dict[str, list[str]],
    ) -> None:
        self.providers = providers
        self.routes = routes
        self._health_overrides: dict[str, ProviderHealth] = {}

    def resolve(
        self,
        field: str,
        symbol: str,
        *,
        diagnostics: bool = False,
        required_keys: set[str] | None = None,
    ) -> ProviderValue | None:
        route = self.routes.get(field, [])
        selected: ProviderValue | None = None
        failed: list[str] = []
        comparisons: list[ProviderValue] = []
        for route_index, provider_name in enumerate(route):
            provider = self.providers[provider_name]
            try:
                value = provider.fetch(field, symbol)
            except Exception as error:
                failed.append(f"{provider_name} failed: {type(error).__name__}")
                self._health_overrides[provider_name] = ProviderHealth(
                    provider=provider_name,
                    state=ProviderState.UNAVAILABLE,
                    checked_at=datetime.now(UTC),
                    error=type(error).__name__,
                )
                continue
            self._health_overrides.pop(provider_name, None)
            if value is None:
                failed.append(f"{provider_name} returned no value")
                continue
            if required_keys is not None and (
                not isinstance(value.value, dict)
                or not required_keys.issubset(
                    {key for key, item in value.value.items() if item is not None}
                )
            ):
                missing = sorted(
                    required_keys
                    - (
                        {key for key, item in value.value.items() if item is not None}
                        if isinstance(value.value, dict)
                        else set()
                    )
                )
                has_partial_value = isinstance(value.value, dict) and any(
                    value.value.get(key) is not None for key in required_keys
                )
                if route_index < len(route) - 1 or not has_partial_value:
                    failed.append(f"{provider_name} missing required keys: {','.join(missing)}")
                    continue
            if selected is None:
                selected = value.model_copy(update={"fallback_reason": "; ".join(failed) or None})
                if not diagnostics:
                    break
            else:
                comparisons.append(value)
        if selected and diagnostics:
            selected.discrepancies.extend(
                ProviderDiscrepancy(
                    provider=value.provider,
                    selected_value=selected.value,
                    other_value=value.value,
                )
                for value in comparisons
                if self._materially_different(selected.value, value.value)
            )
        return selected

    def health(self) -> dict[str, ProviderHealth]:
        return {
            name: self._health_overrides.get(name, provider.health())
            for name, provider in self.providers.items()
        }

    @staticmethod
    def _materially_different(selected: Any, other: Any) -> bool:
        if isinstance(selected, (int, float)) and isinstance(other, (int, float)):
            denominator = max(abs(float(selected)), 0.01)
            return abs(float(selected) - float(other)) / denominator >= 0.01
        return selected != other
