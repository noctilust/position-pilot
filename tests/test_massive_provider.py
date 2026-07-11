from datetime import UTC, datetime

import httpx

from position_pilot.providers.contracts import ProviderState
from position_pilot.providers.massive import MassiveProvider


def test_massive_adapter_uses_current_api_and_preserves_source_payload() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"results": [{"t": 1_720_000_000_000, "o": 548, "c": 550, "v": 1000}]},
        )

    provider = MassiveProvider(
        api_key="massive-secret",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
        clock=lambda: datetime(2026, 7, 11, 17, 0, tzinfo=UTC),
    )

    result = provider.bars(
        "SPY",
        datetime(2026, 7, 10, tzinfo=UTC),
        datetime(2026, 7, 11, tzinfo=UTC),
    )

    assert result is not None
    assert result.provider == "massive-stocks"
    assert result.value[0]["c"] == 550
    assert requests[0].url.host == "api.massive.com"
    assert requests[0].url.path == "/v2/aggs/ticker/SPY/range/1/minute/2026-07-10/2026-07-11"
    assert provider.health().state is ProviderState.HEALTHY


def test_unconfigured_massive_provider_degrades_independently() -> None:
    provider = MassiveProvider(api_key="")

    assert provider.option_snapshot("O:SPY260821C00550000") is None
    assert provider.health().state is ProviderState.NOT_CONFIGURED


def test_massive_stock_snapshot_accepts_top_level_ticker_payload() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "ticker": {
                    "ticker": "SPY",
                    "lastQuote": {"p": 550.0, "P": 550.2},
                },
            },
        )

    provider = MassiveProvider(
        api_key="massive-secret",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    result = provider.stock_snapshot("SPY")

    assert result is not None
    assert result.value == 550.1


def test_massive_option_fields_share_snapshot_cache_and_include_iv() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": {
                    "last_quote": {"bid": 4.2, "ask": 4.3},
                    "greeks": {"delta": 0.52, "theta": -0.08},
                    "implied_volatility": 0.24,
                }
            },
        )

    provider = MassiveProvider(
        api_key="massive-secret",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    mark = provider.fetch("option.mark", "SPY   260821C00550000")
    greeks = provider.fetch("option.greeks", "SPY   260821C00550000")

    assert mark is not None and mark.value == 4.25
    assert greeks is not None and greeks.value["implied_volatility"] == 0.24
    assert len(requests) == 1


def test_massive_option_failure_opens_short_provider_circuit() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, json={"status": "unavailable"})

    provider = MassiveProvider(
        api_key="massive-secret",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    assert provider.fetch("option.mark", "SPY   260821C00550000") is None
    assert provider.fetch("option.greeks", "SPY   260821C00550000") is None
    assert provider.fetch("option.mark", "QQQ   260821C00500000") is None
    assert len(requests) == 1


def test_massive_option_circuit_serves_cache_and_no_data_is_symbol_scoped() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append(path)
        if "SPY" in path:
            return httpx.Response(
                200,
                json={
                    "results": {
                        "last_quote": {"bid": 4.2, "ask": 4.3},
                        "greeks": {"delta": 0.5},
                    }
                },
            )
        if "QQQ" in path:
            return httpx.Response(503, json={"status": "unavailable"})
        return httpx.Response(200, json={"status": "OK"})

    provider = MassiveProvider(
        api_key="massive-secret",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    assert provider.option_snapshot("SPY   260821C00550000") is not None
    assert provider.option_snapshot("IWM   260821C00300000") is None
    assert provider.option_snapshot("DIA   260821C00400000") is None
    assert provider.option_snapshot("QQQ   260821C00500000") is None
    assert provider.option_snapshot("SPY   260821C00550000") is not None

    assert sum("IWM" in path for path in requests) == 1
    assert sum("DIA" in path for path in requests) == 1
    assert sum("QQQ" in path for path in requests) == 1
    assert sum("SPY" in path for path in requests) == 1
