"""Compliant public web/news supplementation after licensed providers.

Fetches only explicitly configured public HTTPS RSS/JSON/HTML sources.
Rejects SSRF targets (credentials, localhost, private/reserved IPs, metadata).
Respects robots.txt fail-closed, timeouts, content limits, attribution, and
dedupe. Never bypasses login walls, paywalls, CAPTCHAs, or robots restrictions.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

from .contracts import ProviderHealth, ProviderState, ProviderValue

DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_BYTES = 512_000
DEFAULT_MAX_ITEMS = 25
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_USER_AGENT = "PositionPilot/1.0 (+local; research; respects robots.txt)"

# Injected request seam: (method, original_url, pinned_ip, hostname, headers) -> response meta.
PinnedRequestFn = Callable[
    [str, str, str, str, dict[str, str]],
    tuple[str | None, int | None, str, str | None],
]

# Cloud metadata / link-local ranges that must never be fetched.
_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),  # link-local + AWS metadata
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT / some cloud
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::/128"),  # unspecified
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # unique local
    ipaddress.ip_network("fe80::/10"),  # link-local v6
    ipaddress.ip_network("ff00::/8"),  # multicast v6
)

_LOGIN_HINTS = re.compile(
    r"(sign[\s-]?in|log[\s-]?in|create an account|subscribe to continue|"
    r"paywall|premium subscribers only|captcha|cloudflare|access denied|"
    r"403 forbidden|please enable cookies)",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_LOCAL_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
    }
)


class PublicWebSourceConfig:
    """One configured public feed or page template.

    source_tier is always forced to ``public`` server-side regardless of client
    configuration — public-web can never claim trusted evidence tiers.
    """

    def __init__(
        self,
        *,
        name: str,
        url_template: str,
        source_tier: str = "public",
        format: str = "auto",
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.url_template = url_template
        # Trust floor: ignore client-supplied elevated tiers.
        self.source_tier = "public"
        self.format = format  # auto | rss | json | html
        self.enabled = enabled
        _ = source_tier  # accepted for API compat; never trusted

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> PublicWebSourceConfig | None:
        name = str(payload.get("name") or "").strip()
        template = str(payload.get("url_template") or payload.get("url") or "").strip()
        if not name or not template:
            return None
        return cls(
            name=name,
            url_template=template,
            source_tier="public",
            format=str(payload.get("format") or "auto").lower(),
            enabled=bool(payload.get("enabled", True)),
        )


def is_blocked_ip(address: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True when an IP is loopback/private/link-local/multicast/reserved/unspecified.

    Non-IP strings return False — callers must classify hostnames separately.
    """

    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    for network in _BLOCKED_NETWORKS:
        if ip in network:
            return True
    # Explicit AWS/GCP/Azure metadata classic address.
    if str(ip) in {"169.254.169.254", "169.254.169.253", "fd00:ec2::254"}:
        return True
    return False


def is_public_https_url(url: str) -> bool:
    """Syntactic HTTPS public URL check (no DNS). Rejects credentials and local names."""

    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    # urlparse may leave userinfo in netloc as user:pass@host
    if "@" in (parsed.netloc or ""):
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in _LOCAL_HOSTNAMES or host.endswith(".localhost") or host.endswith(".local"):
        return False
    # IP literal hosts — only public addresses allowed.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return True  # hostname; DNS checked at fetch time
    return not is_blocked_ip(host)


class PublicWebNewsProvider:
    """Configuration-gated public web/news provider (off when no sources configured)."""

    name = "public-web"

    def __init__(
        self,
        *,
        sources: list[PublicWebSourceConfig] | list[dict[str, Any]] | None = None,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots: bool = True,
        resolve_host: Callable[[str], Sequence[str]] | None = None,
        request_fn: PinnedRequestFn | None = None,
    ) -> None:
        parsed: list[PublicWebSourceConfig] = []
        for item in sources or []:
            if isinstance(item, PublicWebSourceConfig):
                if item.enabled:
                    # Re-wrap to force source_tier=public.
                    parsed.append(
                        PublicWebSourceConfig(
                            name=item.name,
                            url_template=item.url_template,
                            format=item.format,
                            enabled=True,
                        )
                    )
            elif isinstance(item, dict):
                config = PublicWebSourceConfig.from_mapping(item)
                if config is not None and config.enabled:
                    parsed.append(config)
        self.sources = parsed
        # Never auto-follow redirects — each hop is validated and re-pinned.
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": user_agent, "Accept": "*/*"},
        )
        try:
            self.client.follow_redirects = False
        except Exception:
            pass
        self.clock = clock or (lambda: datetime.now(UTC))
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_items = max_items
        self.max_redirects = max_redirects
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        # Injectable resolver for tests (hostname -> list of IP strings).
        self.resolve_host = resolve_host or self._default_resolve_host
        # Injectable request seam (MockTransport / rebinding tests).
        self.request_fn = request_fn
        # robots cache: origin -> parser | False(disallow/unknown fail-closed)
        self._robots_cache: dict[str, RobotFileParser | bool] = {}
        # Last pin observed (for tests / diagnostics).
        self.last_pin: dict[str, str] | None = None
        # Per-operation pin cache: resolve each hostname once; reuse for robots+content
        # on that hop so TOCTOU rebinding cannot swap a second private answer.
        self._pin_cache: dict[str, str] = {}
        self._health = ProviderHealth(
            provider=self.name,
            state=(ProviderState.NOT_CONFIGURED if not self.sources else ProviderState.DEGRADED),
        )

    def health(self) -> ProviderHealth:
        return self._health

    def fetch(self, field: str, symbol: str) -> ProviderValue | None:
        if field != "stock.news":
            return None
        now = self.clock()
        return self.news(symbol, now - timedelta(hours=72), now)

    def news(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None:
        checked_at = self.clock()
        if not self.sources:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.NOT_CONFIGURED,
                checked_at=checked_at,
            )
            return None

        # Fresh pin cache per news() invocation.
        self._pin_cache = {}
        items: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        any_success = False
        any_failure = False

        for source in self.sources:
            url = source.url_template.replace("{symbol}", symbol.upper()).replace(
                "{SYMBOL}", symbol.upper()
            )
            body, content_type, final_url = self._fetch_validated(url)
            if body is None:
                any_failure = True
                continue
            if self._looks_blocked(body, status=200):
                any_failure = True
                continue
            any_success = True
            fmt = (
                source.format
                if source.format != "auto"
                else self._detect_format(content_type, body)
            )
            try:
                if fmt == "rss":
                    parsed_items = self._parse_rss(body, source=source, base_url=final_url or url)
                elif fmt == "json":
                    parsed_items = self._parse_json(body, source=source, symbol=symbol)
                elif fmt == "html":
                    parsed_items = self._parse_html_list(
                        body, source=source, base_url=final_url or url
                    )
                else:
                    continue
            except Exception:
                any_failure = True
                continue

            for item in parsed_items:
                item_url = str(item.get("url") or "").strip()
                if not item_url or item_url in seen_urls:
                    continue
                # Article URLs must also be public HTTPS (no private targets in feed links).
                if not self._url_is_safe_for_fetch(item_url, require_robots=False):
                    continue
                published = item.get("published_at")
                if isinstance(published, datetime):
                    pub = published if published.tzinfo else published.replace(tzinfo=UTC)
                    if pub < start or pub > end:
                        continue
                # Force public tier regardless of parser/config.
                item["source_tier"] = "public"
                item["provider"] = self.name
                seen_urls.add(item_url)
                items.append(item)
                if len(items) >= self.max_items:
                    break
            if len(items) >= self.max_items:
                break

        if not any_success and not items:
            self._health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE if any_failure else ProviderState.DEGRADED,
                checked_at=checked_at,
                error="no_public_sources_available",
            )
            return None

        self._health = ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY if items or any_success else ProviderState.DEGRADED,
            checked_at=checked_at,
            last_success_at=checked_at if any_success else None,
        )
        return ProviderValue(
            field="stock.news",
            value=items,
            provider=self.name,
            observed_at=checked_at,
        )

    # ------------------------------------------------------------------ SSRF / fetch

    @staticmethod
    def _default_resolve_host(hostname: str) -> list[str]:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except OSError:
            return []
        addresses: list[str] = []
        for info in infos:
            addr = info[4][0]
            if addr and addr not in addresses:
                addresses.append(addr)
        return addresses

    @staticmethod
    def format_ip_for_url(ip: str, port: int | None = None) -> str:
        """Format IPv4/IPv6 for URL netloc (bracket IPv6)."""

        try:
            parsed_ip = ipaddress.ip_address(ip)
            host = f"[{parsed_ip}]" if parsed_ip.version == 6 else str(parsed_ip)
        except ValueError:
            host = ip
        if port is not None and port not in (80, 443):
            return f"{host}:{port}"
        return host

    @staticmethod
    def host_header_value(hostname: str, port: int | None) -> str:
        if port is not None and port not in (80, 443):
            return f"{hostname}:{port}"
        return hostname

    def _pin_public_ip(self, url: str) -> tuple[str, str] | None:
        """Validate URL and return (hostname, single pinned public IP).

        Fail closed if any resolved address is non-public (DNS rebinding risk).
        Exactly one validated public IP is chosen and cached per hostname so the
        connection reuses that IP without a second resolve (TOCTOU-safe).
        httpx must not re-resolve the original hostname.
        """

        if not is_public_https_url(url):
            return None
        parsed = urlparse(url)
        host = (parsed.hostname or "").strip().lower()
        if not host:
            return None
        try:
            ipaddress.ip_address(host)
            if is_blocked_ip(host):
                return None
            self._pin_cache[host] = host
            return host, host
        except ValueError:
            pass
        if host in self._pin_cache:
            return host, self._pin_cache[host]
        resolved = list(self.resolve_host(host) or [])
        if not resolved:
            return None
        public: list[str] = []
        for addr in resolved:
            if is_blocked_ip(addr):
                # Any private/link-local answer → fail closed (TOCTOU/rebinding).
                return None
            public.append(str(addr))
        if not public:
            return None
        pinned = public[0]
        self._pin_cache[host] = pinned
        return host, pinned

    def _url_is_safe_for_fetch(self, url: str, *, require_robots: bool) -> bool:
        pin = self._pin_public_ip(url)
        if pin is None:
            return False
        if require_robots and self.respect_robots:
            return self._robots_allows(url)
        return True

    def _robots_allows(self, url: str) -> bool:
        """Fail closed: missing/unparseable robots or Disallow → False."""

        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cached = self._robots_cache.get(origin)
        if cached is False:
            return False
        if isinstance(cached, RobotFileParser):
            try:
                return bool(cached.can_fetch(self.user_agent, url))
            except Exception:
                return False

        robots_url = urljoin(origin, "/robots.txt")
        if self._pin_public_ip(robots_url) is None:
            self._robots_cache[origin] = False
            return False

        body, status, _, _ = self._raw_get_pinned(robots_url)
        if body is None or status is None or status >= 400:
            self._robots_cache[origin] = False
            return False
        try:
            parser = RobotFileParser()
            parser.parse(body.splitlines())
            _ = parser.can_fetch(self.user_agent, url)
        except Exception:
            self._robots_cache[origin] = False
            return False
        self._robots_cache[origin] = parser
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:
            return False

    def _fetch_validated(self, url: str) -> tuple[str | None, str, str | None]:
        """GET with manual redirects; validate + pin every hop for SSRF + robots."""

        current = url
        for _ in range(self.max_redirects + 1):
            if not self._url_is_safe_for_fetch(current, require_robots=True):
                return None, "", None
            body, status, content_type, location = self._raw_get_pinned(current)
            if status is None:
                return None, "", None
            if status in {301, 302, 303, 307, 308}:
                if not location:
                    return None, "", None
                next_url = urljoin(current, location)
                # Redirect hop revalidated and pinned independently (new origin).
                if not self._url_is_safe_for_fetch(next_url, require_robots=True):
                    return None, "", None
                current = next_url
                continue
            if status in {401, 403, 407, 429, 451} or status >= 400:
                return None, "", None
            if body is None:
                return None, "", None
            return body, content_type, current
        return None, "", None

    def _raw_get_pinned(self, url: str) -> tuple[str | None, int | None, str, str | None]:
        """Connect only to a validated public IP; preserve Host + TLS SNI hostname."""

        pin = self._pin_public_ip(url)
        if pin is None:
            return None, None, "", None
        hostname, pinned_ip = pin
        parsed = urlparse(url)
        port = parsed.port
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
            "Host": self.host_header_value(hostname, port),
        }
        self.last_pin = {
            "hostname": hostname,
            "pinned_ip": pinned_ip,
            "host_header": headers["Host"],
            "url": url,
        }
        if self.request_fn is not None:
            return self.request_fn("GET", url, pinned_ip, hostname, headers)
        return self._default_pinned_httpx_request(url, pinned_ip, hostname, headers)

    def _default_pinned_httpx_request(
        self,
        url: str,
        pinned_ip: str,
        hostname: str,
        headers: dict[str, str],
    ) -> tuple[str | None, int | None, str, str | None]:
        """Production path: rewrite URL to pinned IP; Host + sni_hostname = original."""

        parsed = urlparse(url)
        netloc = self.format_ip_for_url(pinned_ip, parsed.port)
        pinned_url = urlunparse(
            (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )
        try:
            with self.client.stream(
                "GET",
                pinned_url,
                headers=headers,
                timeout=self.timeout_seconds,
                follow_redirects=False,
                # Pin TLS SNI + cert name to original hostname (not the IP).
                extensions={"sni_hostname": hostname},
            ) as response:
                status = response.status_code
                content_type = response.headers.get("content-type", "")
                location = response.headers.get("location")
                if status in {301, 302, 303, 307, 308}:
                    return None, status, content_type, location
                if status >= 400:
                    return None, status, content_type, None
                length = response.headers.get("content-length")
                if length and length.isdigit() and int(length) > self.max_bytes:
                    return None, status, content_type, None
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > self.max_bytes:
                        return None, status, content_type, None
                    chunks.append(chunk)
                text = b"".join(chunks).decode("utf-8", errors="replace")
                return text, status, content_type, None
        except (httpx.HTTPError, OSError, UnicodeError):
            return None, None, "", None

    # ------------------------------------------------------------------ parsers

    @staticmethod
    def _detect_format(content_type: str, body: str) -> str:
        lowered = (content_type or "").lower()
        sample = body.lstrip()[:200].lower()
        if "json" in lowered or sample.startswith("{") or sample.startswith("["):
            return "json"
        if "xml" in lowered or "rss" in lowered or "atom" in lowered:
            return "rss"
        if sample.startswith("<?xml") or "<rss" in sample or "<feed" in sample:
            return "rss"
        if "html" in lowered or sample.startswith("<!doctype") or sample.startswith("<html"):
            return "html"
        return "unknown"

    @staticmethod
    def _looks_blocked(body: str, *, status: int | None) -> bool:
        if status in {401, 403, 407, 429, 451}:
            return True
        sample = body[:4_000]
        return bool(_LOGIN_HINTS.search(sample))

    def _parse_rss(
        self, body: str, *, source: PublicWebSourceConfig, base_url: str
    ) -> list[dict[str, Any]]:
        root = ET.fromstring(body)
        items: list[dict[str, Any]] = []
        candidates = list(root.findall(".//item")) + list(
            root.findall(".//{http://www.w3.org/2005/Atom}entry")
        )
        for node in candidates[: self.max_items]:
            title = self._text(node, "title") or self._text(
                node, "{http://www.w3.org/2005/Atom}title"
            )
            link = self._text(node, "link")
            if not link:
                atom_link = node.find("{http://www.w3.org/2005/Atom}link")
                if atom_link is not None:
                    link = atom_link.attrib.get("href")
            if not title or not link:
                continue
            link = urljoin(base_url, link)
            published_raw = (
                self._text(node, "pubDate")
                or self._text(node, "published")
                or self._text(node, "{http://www.w3.org/2005/Atom}published")
                or self._text(node, "{http://www.w3.org/2005/Atom}updated")
            )
            published_at = self._parse_date(published_raw) or self.clock()
            excerpt = self._text(node, "description") or self._text(
                node, "{http://www.w3.org/2005/Atom}summary"
            )
            items.append(
                {
                    "title": _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", title)).strip()[:500],
                    "url": link,
                    "published_at": published_at,
                    "provider": self.name,
                    "source_name": source.name,
                    "source_tier": "public",
                    "excerpt": (
                        _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", excerpt or "")).strip()[:500]
                        or None
                    ),
                    "full_text": None,
                    "attribution": f"Public source: {source.name}",
                }
            )
        return items

    def _parse_json(
        self, body: str, *, source: PublicWebSourceConfig, symbol: str
    ) -> list[dict[str, Any]]:
        import json

        payload = json.loads(body)
        rows: list[Any]
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("items", "results", "articles", "data", "entries"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            else:
                rows = [payload]
        else:
            return []

        items: list[dict[str, Any]] = []
        for row in rows[: self.max_items]:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("headline") or "").strip()
            link = str(row.get("url") or row.get("link") or row.get("href") or "").strip()
            if not title or not link:
                continue
            published_raw = (
                row.get("published_at")
                or row.get("published")
                or row.get("pubDate")
                or row.get("date")
            )
            published_at = (
                self._parse_date(str(published_raw) if published_raw else None) or self.clock()
            )
            excerpt = row.get("excerpt") or row.get("summary") or row.get("description")
            items.append(
                {
                    "title": title[:500],
                    "url": link,
                    "published_at": published_at,
                    "provider": self.name,
                    "source_name": source.name,
                    "source_tier": "public",
                    "excerpt": str(excerpt)[:500] if excerpt else None,
                    "full_text": None,
                    "symbol": symbol.upper(),
                    "attribution": f"Public source: {source.name}",
                }
            )
        return items

    def _parse_html_list(
        self, body: str, *, source: PublicWebSourceConfig, base_url: str
    ) -> list[dict[str, Any]]:
        if self._looks_blocked(body, status=200):
            return []
        pattern = re.compile(
            r'<a[^>]+href=["\'](https?://[^"\']+|/?[^"\']+)["\'][^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for match in pattern.finditer(body):
            href = match.group(1).strip()
            text = _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", match.group(2))).strip()
            if len(text) < 12 or len(text) > 300:
                continue
            if text.lower() in {"read more", "home", "login", "sign in", "subscribe"}:
                continue
            url = urljoin(base_url, href)
            if url in seen:
                continue
            seen.add(url)
            items.append(
                {
                    "title": text[:500],
                    "url": url,
                    "published_at": self.clock(),
                    "provider": self.name,
                    "source_name": source.name,
                    "source_tier": "public",
                    "excerpt": None,
                    "full_text": None,
                    "attribution": f"Public source: {source.name}",
                }
            )
            if len(items) >= self.max_items:
                break
        return items

    @staticmethod
    def _text(node: ET.Element, tag: str) -> str | None:
        child = node.find(tag)
        if child is None or child.text is None:
            return None
        return child.text.strip()

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        raw = value.strip()
        try:
            if raw.endswith("Z"):
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
        except (TypeError, ValueError, IndexError):
            return None
