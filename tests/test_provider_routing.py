from datetime import UTC, datetime

from position_pilot.providers.contracts import ProviderHealth, ProviderState, ProviderValue
from position_pilot.providers.router import FieldRouter
from position_pilot.providers.tastytrade import TastytradeProvider


class FakeProvider:
    def __init__(self, name: str, values: dict[str, object]) -> None:
        self.name = name
        self.values = values

    def health(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    def fetch(self, field: str, symbol: str) -> ProviderValue | None:
        value = self.values.get(field)
        if value is None:
            return None
        return ProviderValue(
            field=field,
            value=value,
            provider=self.name,
            observed_at=datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
        )


class FailingProvider(FakeProvider):
    def fetch(self, field: str, symbol: str) -> ProviderValue | None:
        raise ConnectionError("provider unavailable")


def test_field_router_uses_tastytrade_first_without_averaging_values() -> None:
    tastytrade = FakeProvider("tastytrade", {"option.mark": 2.15})
    massive = FakeProvider("massive-options", {"option.mark": 2.25})
    router = FieldRouter(
        providers={provider.name: provider for provider in (tastytrade, massive)},
        routes={"option.mark": ["tastytrade", "massive-options"]},
    )

    result = router.resolve("option.mark", "SPY-option", diagnostics=True)

    assert result is not None
    assert result.value == 2.15
    assert result.provider == "tastytrade"
    assert result.fallback_reason is None
    assert result.discrepancies[0].other_value == 2.25


def test_field_router_records_explicit_fallback_and_independent_health() -> None:
    tastytrade = FakeProvider("tastytrade", {})
    massive = FakeProvider("massive-stocks", {"stock.bars": [{"close": 550}]})
    router = FieldRouter(
        providers={provider.name: provider for provider in (tastytrade, massive)},
        routes={"stock.bars": ["tastytrade", "massive-stocks"]},
    )

    result = router.resolve("stock.bars", "SPY")

    assert result is not None
    assert result.provider == "massive-stocks"
    assert result.fallback_reason == "tastytrade returned no value"
    assert router.health()["tastytrade"].state is ProviderState.HEALTHY


def test_field_router_isolates_provider_exceptions_and_uses_fallback() -> None:
    tastytrade = FailingProvider("tastytrade", {})
    massive = FakeProvider("massive-stocks", {"stock.quote": 550.25})
    router = FieldRouter(
        providers={provider.name: provider for provider in (tastytrade, massive)},
        routes={"stock.quote": ["tastytrade", "massive-stocks"]},
    )

    result = router.resolve("stock.quote", "SPY")

    assert result is not None
    assert result.provider == "massive-stocks"
    assert result.fallback_reason == "tastytrade failed: ConnectionError"
    assert router.health()["tastytrade"].state is ProviderState.UNAVAILABLE
    assert router.health()["tastytrade"].error == "ConnectionError"


def test_partial_tastytrade_greeks_do_not_block_missing_field_fallback() -> None:
    class PartialQuoteClient:
        is_enabled = True

        def get_quote(self, symbol: str) -> dict:
            return {"mark": 4.25, "delta": 0.4, "theta": None}

    tastytrade = TastytradeProvider(PartialQuoteClient())  # type: ignore[arg-type]
    massive = FakeProvider(
        "massive-options",
        {"option.greeks": {"theta": -0.08, "implied_volatility": 0.24}},
    )
    router = FieldRouter(
        providers={provider.name: provider for provider in (tastytrade, massive)},
        routes={"option.greeks": ["tastytrade", "massive-options"]},
    )

    result = router.resolve(
        "option.greeks",
        "SPY   260821C00550000",
        required_keys={"theta", "implied_volatility"},
    )

    assert result is not None
    assert result.provider == "massive-options"
    assert result.value == {"theta": -0.08, "implied_volatility": 0.24}
    assert result.fallback_reason == ("tastytrade missing required keys: implied_volatility,theta")
