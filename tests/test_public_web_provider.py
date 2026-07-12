"""Public web news provider — SSRF, robots fail-closed, trust floor, mocked HTTP."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from position_pilot.domain.catalysts import (
    CatalystConfidence,
    CatalystService,
    CatalystSettings,
    confidence_for_item,
    is_causal_candidate,
    parse_raw_news_payload,
)
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.providers.contracts import ProviderHealth, ProviderState, ProviderValue
from position_pilot.providers.public_web import (
    PublicWebNewsProvider,
    PublicWebSourceConfig,
    is_blocked_ip,
    is_public_https_url,
)

FIXED_NOW = datetime(2026, 7, 10, 16, 0, tzinfo=UTC)


def _client(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, timeout=5.0, follow_redirects=False)


def _public_resolver(hostname: str) -> list[str]:
    # Controlled test resolver — never hits real DNS.
    # Use a non-reserved public address (TEST-NET ranges are is_reserved).
    if hostname.endswith(".example") or hostname in {
        "news.example",
        "json.example",
        "huge.example",
        "blocked.example",
    }:
        return ["8.8.8.8"]
    if hostname in {"private.internal"}:
        return ["10.0.0.5"]
    if hostname in {"meta.local"}:
        return ["169.254.169.254"]
    return ["8.8.8.8"]


def test_public_web_off_when_no_sources_configured() -> None:
    provider = PublicWebNewsProvider(sources=[], clock=lambda: FIXED_NOW)
    assert provider.news("AAPL", FIXED_NOW.replace(hour=0), FIXED_NOW) is None
    assert provider.health().state.value == "not_configured"


def test_rejects_http_credentials_localhost_and_private_literals() -> None:
    assert is_public_https_url("https://news.example/feed") is True
    assert is_public_https_url("http://news.example/feed") is False
    assert is_public_https_url("https://user:pass@news.example/feed") is False
    assert is_public_https_url("https://localhost/feed") is False
    assert is_public_https_url("https://127.0.0.1/feed") is False
    assert is_public_https_url("https://[::1]/feed") is False
    assert is_public_https_url("https://10.0.0.1/feed") is False
    assert is_public_https_url("https://192.168.1.1/feed") is False
    assert is_public_https_url("https://169.254.169.254/latest") is False
    assert is_blocked_ip("127.0.0.1") is True
    assert is_blocked_ip("10.1.2.3") is True
    assert is_blocked_ip("169.254.169.254") is True
    assert is_blocked_ip("8.8.8.8") is False


def test_public_web_parses_rss_with_attribution_and_dedupe() -> None:
    rss = """<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <title>Public Feed</title>
      <item>
        <title>AAPL announces product</title>
        <link>https://news.example/aapl-1</link>
        <pubDate>Thu, 10 Jul 2026 12:00:00 GMT</pubDate>
        <description>Short excerpt</description>
      </item>
      <item>
        <title>AAPL announces product</title>
        <link>https://news.example/aapl-1</link>
        <pubDate>Thu, 10 Jul 2026 12:05:00 GMT</pubDate>
      </item>
      <item>
        <title>Other story</title>
        <link>https://news.example/other</link>
        <pubDate>Thu, 10 Jul 2026 13:00:00 GMT</pubDate>
      </item>
    </channel></rss>"""
    robots = "User-agent: *\nAllow: /\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(200, text=robots)
        return httpx.Response(200, text=rss, headers={"content-type": "application/rss+xml"})

    provider = PublicWebNewsProvider(
        sources=[
            PublicWebSourceConfig(
                name="Example RSS",
                url_template="https://news.example/feed/{symbol}.xml",
                format="rss",
                source_tier="company",  # client claim ignored
            )
        ],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
        respect_robots=True,
    )
    result = provider.news("AAPL", FIXED_NOW.replace(hour=0), FIXED_NOW)
    assert result is not None
    items = result.value
    assert isinstance(items, list)
    urls = [item["url"] for item in items]
    assert urls.count("https://news.example/aapl-1") == 1
    assert all(item["provider"] == "public-web" for item in items)
    assert all(item.get("source_tier") == "public" for item in items)
    assert all(item.get("full_text") is None for item in items)
    assert all("Public source" in (item.get("attribution") or "") for item in items)


def test_direct_private_ip_url_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"must not fetch {request.url}")

    provider = PublicWebNewsProvider(
        sources=[{"name": "Bad", "url_template": "https://127.0.0.1/feed", "format": "rss"}],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
    )
    assert provider.news("SPY", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


def test_localhost_and_private_dns_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"must not fetch {request.url}")

    for template in (
        "https://localhost/feed",
        "https://private.internal/feed",
        "https://meta.local/latest",
        "https://user:secret@news.example/feed",
    ):
        provider = PublicWebNewsProvider(
            sources=[{"name": "Bad", "url_template": template, "format": "rss"}],
            client=_client(handler),
            clock=lambda: FIXED_NOW,
            resolve_host=_public_resolver,
        )
        assert provider.news("SPY", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


def test_redirect_to_private_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Requests are pinned to validated IPs; match by path / Host header.
        path = request.url.path
        host_header = request.headers.get("host", "")
        if path.endswith("robots.txt"):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if path.endswith("/start") or host_header.startswith("news.example"):
            if path.endswith("robots.txt"):
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            return httpx.Response(302, headers={"location": "https://10.0.0.8/secret"})
        raise AssertionError(f"unexpected fetch {request.url} host={host_header}")

    provider = PublicWebNewsProvider(
        sources=[{"name": "Redir", "url_template": "https://news.example/start", "format": "rss"}],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
    )
    assert provider.news("IBM", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


def test_redirect_to_new_origin_requires_robots() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path
        if path.endswith("robots.txt"):
            if host == "news.example":
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            # New origin missing robots → fail closed.
            return httpx.Response(404, text="missing")
        if str(request.url).startswith("https://news.example/start"):
            return httpx.Response(302, headers={"location": "https://json.example/next"})
        raise AssertionError(f"must not fetch content without robots: {request.url}")

    provider = PublicWebNewsProvider(
        sources=[{"name": "Redir", "url_template": "https://news.example/start", "format": "rss"}],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
    )
    assert provider.news("IBM", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


def test_missing_robots_fail_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(404, text="missing")
        return httpx.Response(200, text="<rss></rss>")

    provider = PublicWebNewsProvider(
        sources=[
            {
                "name": "No robots",
                "url_template": "https://news.example/feed.xml",
                "format": "rss",
            }
        ],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
    )
    assert provider.news("SPY", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


def test_robots_disallow_blocks_fetch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        raise AssertionError("content fetch must not occur when robots disallows")

    provider = PublicWebNewsProvider(
        sources=[
            {
                "name": "Blocked",
                "url_template": "https://blocked.example/feed.xml",
                "format": "rss",
            }
        ],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
    )
    assert provider.news("SPY", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


def test_public_web_abstains_on_login_wall() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(
            200,
            text=(
                "<html><body>Please sign in to continue reading "
                "premium subscribers only</body></html>"
            ),
            headers={"content-type": "text/html"},
        )

    provider = PublicWebNewsProvider(
        sources=[
            {
                "name": "Paywall",
                "url_template": "https://news.example/paywall/{symbol}",
                "format": "html",
            }
        ],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
    )
    assert provider.news("MSFT", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


def test_public_web_parses_json_feed() -> None:
    payload = {
        "items": [
            {
                "title": "Regulator filing",
                "url": "https://json.example/1",
                "published_at": "2026-07-10T10:00:00Z",
                "excerpt": "SEC form",
                "source_tier": "regulator",  # self-assertion ignored
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

    provider = PublicWebNewsProvider(
        sources=[
            {
                "name": "JSON Wire",
                "url_template": "https://json.example/news?q={symbol}",
                "format": "json",
                "source_tier": "government",
            }
        ],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        resolve_host=_public_resolver,
    )
    result = provider.news("XOM", FIXED_NOW.replace(hour=0), FIXED_NOW)
    assert result is not None
    assert result.value[0]["title"] == "Regulator filing"
    assert result.value[0]["source_tier"] == "public"
    assert result.value[0]["source_name"] == "JSON Wire"


def test_public_web_cannot_claim_confirmed_causality() -> None:
    items = parse_raw_news_payload(
        "AAPL",
        "public-web",
        [
            {
                "title": "AAPL earnings beat expectations",
                "url": "https://news.example/1",
                "published_at": "2026-07-10T10:00:00Z",
                "source_tier": "company",
                "taxonomy": "earnings",
                "source_name": "Fake Company PR",
            }
        ],
    )
    assert items
    assert items[0].source_tier == "public"
    assert is_causal_candidate(items[0]) is False
    assert confidence_for_item(items[0]) is CatalystConfidence.SUPPORTING


def test_public_web_enforces_content_byte_limit() -> None:
    huge = "x" * 600_000

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(
            200,
            content=huge.encode(),
            headers={"content-type": "application/rss+xml", "content-length": str(len(huge))},
        )

    provider = PublicWebNewsProvider(
        sources=[{"name": "Huge", "url_template": "https://huge.example/feed", "format": "rss"}],
        client=_client(handler),
        clock=lambda: FIXED_NOW,
        max_bytes=512_000,
        resolve_host=_public_resolver,
    )
    assert provider.news("IBM", FIXED_NOW.replace(hour=0), FIXED_NOW) is None


class _CountingNews:
    def __init__(self, name: str, items: list[dict] | None, *, state=ProviderState.HEALTHY) -> None:
        self.name = name
        self.items = items
        self.state = state
        self.calls = 0

    def health(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, state=self.state)

    def news(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None:
        self.calls += 1
        if self.items is None:
            return None
        return ProviderValue(
            field="stock.news",
            value=self.items,
            provider=self.name,
            observed_at=FIXED_NOW,
        )


def test_public_web_gap_gated_not_called_when_primary_has_causal_results(tmp_path) -> None:
    massive = _CountingNews(
        "massive-stocks",
        [
            {
                "title": "AAPL reports quarterly earnings beat",
                "url": "https://massive.example/1",
                "published_utc": "2026-07-10T12:00:00Z",
                "publisher": {"name": "Wall Street Journal"},
                "description": "Earnings coverage",
                "source_tier": "established",
            }
        ],
    )
    public = _CountingNews("public-web", [{"title": "Should not run", "url": "https://x/1"}])

    service = CatalystService(
        database=PositionPilotDatabase(tmp_path / "db.sqlite3"),
        news_providers=[massive, public],
        settings=CatalystSettings(),
        clock=lambda: FIXED_NOW,
    )
    result = service.analyze_symbol("AAPL")
    assert massive.calls == 1
    assert public.calls == 0
    assert any("public-web skipped" in note for note in result.coverage_notes)


def test_public_web_gap_gated_called_when_primary_items_irrelevant(tmp_path) -> None:
    """Unrelated/unknown primary articles must not suppress public-web."""

    massive = _CountingNews(
        "massive-stocks",
        [
            {
                "title": "Random lifestyle blog post",
                "url": "https://massive.example/noise",
                "published_utc": "2026-07-10T12:00:00Z",
                "publisher": {"name": "Unknown Blog"},
                "description": "Not market evidence",
                "source_tier": "unknown",
            }
        ],
    )
    public = _CountingNews(
        "public-web",
        [
            {
                "title": "Public wire note on AAPL product launch event",
                "url": "https://news.example/public-1",
                "published_at": "2026-07-10T12:00:00Z",
                "source_name": "Public Wire",
                "source_tier": "company",
                "excerpt": "A product note",
            }
        ],
    )

    service = CatalystService(
        database=PositionPilotDatabase(tmp_path / "db.sqlite3"),
        news_providers=[massive, public],
        settings=CatalystSettings(),
        clock=lambda: FIXED_NOW,
    )
    service.analyze_symbol("AAPL")
    assert massive.calls == 1
    assert public.calls == 1


def test_public_web_gap_gated_called_when_primary_empty(tmp_path) -> None:
    massive = _CountingNews("massive-stocks", [])
    public_items = [
        {
            "title": "Public wire note on AAPL product launch event",
            "url": "https://news.example/public-1",
            "published_at": "2026-07-10T12:00:00Z",
            "source_name": "Public Wire",
            "source_tier": "company",
            "excerpt": "A product note",
        }
    ]
    public = _CountingNews("public-web", public_items)

    service = CatalystService(
        database=PositionPilotDatabase(tmp_path / "db.sqlite3"),
        news_providers=[massive, public],
        settings=CatalystSettings(),
        clock=lambda: FIXED_NOW,
    )
    result = service.analyze_symbol("AAPL")
    assert massive.calls == 1
    assert public.calls == 1
    assert result.catalysts
    assert result.catalysts[0].confidence is CatalystConfidence.SUPPORTING
    assert result.confidence is CatalystConfidence.SUPPORTING
    assert all(source.tier == "public" for source in result.catalysts[0].sources)


def test_dns_pinning_uses_validated_ip_not_hostname_rebinding() -> None:
    """Connector must receive only the validated public IP (TOCTOU/rebinding safe)."""

    pins: list[dict[str, str]] = []
    resolve_calls = {"n": 0}

    def flapping_resolver(hostname: str) -> list[str]:
        resolve_calls["n"] += 1
        # First answer is public; a later re-resolve would return private (attack).
        if resolve_calls["n"] == 1:
            return ["8.8.8.8"]
        return ["10.0.0.9"]

    def request_fn(method, url, pinned_ip, hostname, headers):
        pins.append(
            {
                "method": method,
                "url": url,
                "pinned_ip": pinned_ip,
                "hostname": hostname,
                "host": headers.get("Host", ""),
            }
        )
        # Connection always uses the first validated public pin — never private.
        assert pinned_ip == "8.8.8.8"
        assert hostname == "news.example"
        assert headers.get("Host") == "news.example"
        if url.rstrip("/").endswith("robots.txt"):
            return "User-agent: *\nAllow: /\n", 200, "text/plain", None
        rss = (
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            "<item><title>Pinned</title>"
            "<link>https://news.example/a</link>"
            "<pubDate>Thu, 10 Jul 2026 12:00:00 GMT</pubDate>"
            "</item></channel></rss>"
        )
        return rss, 200, "application/rss+xml", None

    provider = PublicWebNewsProvider(
        sources=[{"name": "Pin", "url_template": "https://news.example/feed", "format": "rss"}],
        clock=lambda: FIXED_NOW,
        resolve_host=flapping_resolver,
        request_fn=request_fn,
    )
    result = provider.news("AAPL", FIXED_NOW.replace(hour=0), FIXED_NOW)
    assert result is not None
    assert pins
    assert all(p["pinned_ip"] == "8.8.8.8" for p in pins)
    assert all(p["hostname"] == "news.example" for p in pins)
    # Host header preserves original name for virtual hosting / TLS SNI name.
    assert all(p["host"] == "news.example" for p in pins)
    # Only one DNS resolution for the hostname (pin cache); flapping private answer unused.
    assert resolve_calls["n"] == 1
    assert "10.0.0.9" not in {p["pinned_ip"] for p in pins}


def test_ipv6_url_formatting_and_host_sni_headers() -> None:
    assert PublicWebNewsProvider.format_ip_for_url("2001:db8::1") == "[2001:db8::1]"
    assert PublicWebNewsProvider.format_ip_for_url("2001:db8::1", 8443) == "[2001:db8::1]:8443"
    assert PublicWebNewsProvider.format_ip_for_url("8.8.8.8") == "8.8.8.8"
    assert PublicWebNewsProvider.host_header_value("news.example", None) == "news.example"
    assert PublicWebNewsProvider.host_header_value("news.example", 8443) == "news.example:8443"

    seen: dict[str, str] = {}

    def request_fn(method, url, pinned_ip, hostname, headers):
        seen.update(
            {
                "pinned_ip": pinned_ip,
                "hostname": hostname,
                "host": headers["Host"],
            }
        )
        if "robots.txt" in url:
            return "User-agent: *\nAllow: /\n", 200, "text/plain", None
        return "[]", 200, "application/json", None

    def resolve_v6(hostname: str) -> list[str]:
        return ["2001:4860:4860::8888"]  # public v6

    provider = PublicWebNewsProvider(
        sources=[
            {
                "name": "V6",
                "url_template": "https://news.example/json",
                "format": "json",
            }
        ],
        clock=lambda: FIXED_NOW,
        resolve_host=resolve_v6,
        request_fn=request_fn,
    )
    # May return empty items but pin must be IPv6.
    provider.news("AAPL", FIXED_NOW.replace(hour=0), FIXED_NOW)
    assert seen.get("pinned_ip") == "2001:4860:4860::8888"
    assert seen.get("hostname") == "news.example"
    assert seen.get("host") == "news.example"
