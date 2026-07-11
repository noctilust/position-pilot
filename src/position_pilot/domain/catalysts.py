"""Evidence-backed catalyst intelligence for held underlyings."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from ..persistence.sqlite import PositionPilotDatabase
from ..providers.contracts import ProviderValue
from .snapshots import DataFreshness, FreshnessState

ET = ZoneInfo("America/New_York")

BROAD_MARKET_ETFS = frozenset(
    {
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "IVV",
        "RSP",
        "MDY",
        "IWB",
        "ITOT",
        "SCHB",
        "SPLG",
    }
)

SOURCE_TIER_PRIORITY = {
    "company": 100,
    "regulator": 95,
    "government": 95,
    "established": 80,
    "specialist": 65,
    "licensed": 70,
    "unknown": 40,
    "social": 5,
}


class CatalystConfidence(StrEnum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    NO_CONFIRMED_CATALYST = "no_confirmed_catalyst_found"


class AttributionLevel(StrEnum):
    COMPANY = "company"
    PEER = "peer"
    MACRO = "macro"
    OPTIONS_MARKET = "options_market"
    NONE = "none"


class EventTaxonomy(StrEnum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    CORPORATE_ACTION = "corporate_action"
    PRODUCT = "product"
    REGULATORY = "regulatory"
    MACRO = "macro"
    ANALYST = "analyst"
    DIVIDEND = "dividend"
    M_AND_A = "m_and_a"
    LEGAL = "legal"
    SUPPLY_CHAIN = "supply_chain"
    MANAGEMENT = "management"
    PEER = "peer"
    SOCIAL = "social"
    OTHER = "other"


class CatalystEvidenceKind(StrEnum):
    NEWS = "news"
    FILING = "filing"
    MECHANISM = "mechanism"
    SOCIAL = "social"


class CatalystFeedbackKind(StrEnum):
    RELEVANT = "relevant"
    NOT_RELATED = "not_related"
    MISSING_CATALYST = "missing_catalyst"


class CoverageState(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    OFFLINE = "offline"
    UNAVAILABLE = "unavailable"


class OptionMechanismKind(StrEnum):
    IV_EXPANSION = "iv_expansion"
    IV_CRUSH = "iv_crush"
    SKEW = "skew"
    UNUSUAL_VOLUME = "unusual_volume"
    OPEN_INTEREST = "open_interest"
    LIQUIDITY = "liquidity"
    DIVIDEND = "dividend"
    EARNINGS_PROXIMITY = "earnings_proximity"
    EXPIRATION = "expiration"
    GAMMA = "gamma"


class CatalystSettings(BaseModel):
    stock_move_threshold_pct: float = 2.0
    etf_move_threshold_pct: float = 1.0
    abnormal_range_multiple: float = 1.5
    default_window_hours: int = 24
    scheduled_window_hours: int = 72
    news_cadence_seconds: int = 300
    benzinga_enabled: bool = True
    store_full_text_providers: set[str] = Field(default_factory=lambda: {"benzinga"})
    catalyst_retention_days: int = 365
    article_metadata_retention_days: int = 90


class RawNewsItem(BaseModel):
    symbol: str
    title: str
    url: str
    published_at: datetime
    provider: str
    source_name: str = "Unknown"
    source_tier: str = "unknown"
    excerpt: str | None = None
    full_text: str | None = None
    taxonomy: EventTaxonomy = EventTaxonomy.OTHER
    external_id: str | None = None


class CatalystSource(BaseModel):
    source_id: str
    name: str
    tier: str
    url: str
    provider: str
    published_at: datetime
    excerpt: str | None = None


class MechanismAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class OptionMechanism(BaseModel):
    kind: OptionMechanismKind
    label: str
    summary: str
    evidence_kind: CatalystEvidenceKind = CatalystEvidenceKind.MECHANISM
    magnitude: float | None = None
    observed_at: datetime | None = None
    availability: MechanismAvailability = MechanismAvailability.AVAILABLE


class OptionMechanismCoverage(BaseModel):
    """Explicit per-mechanism coverage so unavailable fields are not implied."""

    kind: OptionMechanismKind
    label: str
    availability: MechanismAvailability
    detail: str


class SocialSideNote(BaseModel):
    note_id: str
    headline: str
    summary: str
    evidence_kind: CatalystEvidenceKind = CatalystEvidenceKind.SOCIAL
    source_name: str
    url: str | None = None
    published_at: datetime | None = None
    confidence_note: str = "Social sentiment is not catalyst evidence."


class CatalystEvent(BaseModel):
    catalyst_id: str
    symbol: str
    headline: str
    summary: str
    taxonomy: EventTaxonomy
    confidence: CatalystConfidence
    attribution: AttributionLevel
    evidence_kind: CatalystEvidenceKind = CatalystEvidenceKind.NEWS
    event_at: datetime
    sources: list[CatalystSource] = Field(default_factory=list)
    rank_score: float = 0.0
    high_impact: bool = False
    fingerprint: str


class ChartEventMarker(BaseModel):
    catalyst_id: str
    timestamp: datetime
    headline: str
    confidence: CatalystConfidence
    attribution: AttributionLevel


class SymbolMoveInput(BaseModel):
    symbol: str
    last_price: float | None = None
    prior_close: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    is_broad_etf: bool | None = None


class SymbolCatalystResult(BaseModel):
    symbol: str
    confidence: CatalystConfidence
    attribution: AttributionLevel
    summary: str
    catalysts: list[CatalystEvent] = Field(default_factory=list)
    option_mechanisms: list[OptionMechanism] = Field(default_factory=list)
    option_mechanism_coverage: list[OptionMechanismCoverage] = Field(default_factory=list)
    social_side_notes: list[SocialSideNote] = Field(default_factory=list)
    move_percent: float | None = None
    prior_close: float | None = None
    last_price: float | None = None
    meaningful_move: bool = False
    promoted: bool = False
    lookback_start: datetime | None = None
    lookback_end: datetime | None = None
    coverage: CoverageState = CoverageState.COMPLETE
    coverage_notes: list[str] = Field(default_factory=list)
    freshness: DataFreshness
    quiet: bool = True
    cached: bool = False


class CatalystScanSnapshot(BaseModel):
    captured_at: datetime
    results: list[SymbolCatalystResult] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    freshness: DataFreshness
    coverage: CoverageState = CoverageState.COMPLETE
    coverage_notes: list[str] = Field(default_factory=list)


class CatalystFeedbackEvent(BaseModel):
    feedback_id: str
    kind: CatalystFeedbackKind
    catalyst_id: str | None = None
    symbol: str | None = None
    note: str = ""
    recorded_at: datetime


class NewsProvider(Protocol):
    name: str

    def news(self, symbol: str, start: datetime, end: datetime) -> ProviderValue | None: ...

    def health(self) -> Any: ...


def previous_regular_close(now: datetime) -> datetime:
    """Return the most recent *completed* regular-session close (16:00 America/New_York).

    Session rules (exchange holidays out of scope):
    - Premarket / regular session (before 16:00 ET): prior close is the previous weekday 16:00.
    - After-hours (at/after 16:00 ET on a weekday): prior close is *today's* 16:00.
    - Weekend: prior close is the preceding Friday 16:00.

    This never subtracts an extra day before 16:00 beyond the single step to the previous
    trading session.
    """

    local = now.astimezone(ET)
    day = local.date()
    close_today = datetime(day.year, day.month, day.day, 16, 0, 0, tzinfo=ET)
    # Before today's regular close, the official prior close is the previous session.
    if local < close_today:
        day = day - timedelta(days=1)
    # Walk back across weekends only (not an extra pre-close day).
    while day.weekday() >= 5:
        day = day - timedelta(days=1)
    return datetime(day.year, day.month, day.day, 16, 0, 0, tzinfo=ET).astimezone(UTC)


RELIABLE_SOURCE_TIERS = frozenset({"company", "regulator", "government", "established", "licensed"})
HIGH_IMPACT_TAXONOMIES = frozenset(
    {
        EventTaxonomy.EARNINGS,
        EventTaxonomy.GUIDANCE,
        EventTaxonomy.M_AND_A,
        EventTaxonomy.REGULATORY,
        EventTaxonomy.MACRO,
        EventTaxonomy.CORPORATE_ACTION,
        EventTaxonomy.MANAGEMENT,
    }
)
RELEVANT_TAXONOMIES = frozenset(
    {
        EventTaxonomy.EARNINGS,
        EventTaxonomy.GUIDANCE,
        EventTaxonomy.CORPORATE_ACTION,
        EventTaxonomy.PRODUCT,
        EventTaxonomy.REGULATORY,
        EventTaxonomy.MACRO,
        EventTaxonomy.ANALYST,
        EventTaxonomy.DIVIDEND,
        EventTaxonomy.M_AND_A,
        EventTaxonomy.LEGAL,
        EventTaxonomy.SUPPLY_CHAIN,
        EventTaxonomy.MANAGEMENT,
        EventTaxonomy.PEER,
    }
)


def stable_source_id(*, catalyst_id: str, provider: str, url: str) -> str:
    raw = f"{catalyst_id}|{provider}|{url}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def derive_move_from_bars(
    bars: list[dict[str, Any]],
    *,
    prior_close_at: datetime,
    now: datetime,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Derive official prior close and session high/low/last from extended-hours bars.

    Prior close = last bar close at or before ``prior_close_at``.
    Session range uses bars strictly after ``prior_close_at`` through ``now``.
    Never treats a live 'close' field from a quote as prior close.
    """

    parsed: list[tuple[datetime, float, float, float, float]] = []
    for item in bars:
        try:
            timestamp = item.get("timestamp", item.get("t"))
            if isinstance(timestamp, (int, float)):
                if timestamp > 10_000_000_000:
                    timestamp /= 1000
                ts = datetime.fromtimestamp(float(timestamp), tz=UTC)
            elif isinstance(timestamp, datetime):
                ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
            else:
                continue
            o = float(item["open"] if "open" in item else item["o"])
            h = float(item["high"] if "high" in item else item["h"])
            low = float(item["low"] if "low" in item else item["l"])
            c = float(item["close"] if "close" in item else item["c"])
            parsed.append((ts, o, h, low, c))
        except (KeyError, OSError, OverflowError, TypeError, ValueError):
            continue
    if not parsed:
        return None, None, None, None
    parsed.sort(key=lambda row: row[0])
    prior_close: float | None = None
    for ts, _o, _h, _l, c in parsed:
        if ts <= prior_close_at:
            prior_close = c
    session = [row for row in parsed if prior_close_at < row[0] <= now]
    if not session:
        last = parsed[-1][4] if parsed else None
        return prior_close, last, None, None
    last_price = session[-1][4]
    session_high = max(row[2] for row in session)
    session_low = min(row[3] for row in session)
    return prior_close, last_price, session_high, session_low


def is_broad_etf(symbol: str) -> bool:
    return symbol.upper() in BROAD_MARKET_ETFS


def normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", "", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def content_fingerprint(symbol: str, title: str, published_at: datetime) -> str:
    day = published_at.astimezone(UTC).date().isoformat()
    raw = f"{symbol.upper()}|{normalize_title(title)}|{day}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def classify_source_tier(source_name: str, provider: str) -> str:
    name = source_name.lower()
    if provider == "benzinga":
        return "licensed"
    if any(token in name for token in ("sec", "edgar", "federal reserve", "cftc", "doj", "fda")):
        return "regulator"
    if any(
        token in name
        for token in (
            "ir",
            "investor relations",
            "press release",
            "business wire",
            "globe newswire",
            "pr newswire",
        )
    ):
        return "company"
    if any(
        token in name
        for token in (
            "reuters",
            "bloomberg",
            "wsj",
            "wall street journal",
            "ft",
            "financial times",
            "associated press",
            "ap news",
            "cnbc",
            "barron",
            "marketwatch",
        )
    ):
        return "established"
    if any(token in name for token in ("twitter", "reddit", "stocktwits", "social", "rumor")):
        return "social"
    if any(token in name for token in ("benzinga", "thefly", "tipranks", "seeking alpha")):
        return "specialist"
    return "unknown"


def classify_taxonomy(title: str, body: str | None = None) -> EventTaxonomy:
    text = f"{title} {body or ''}".lower()
    rules: list[tuple[EventTaxonomy, tuple[str, ...]]] = [
        (
            EventTaxonomy.EARNINGS,
            ("earnings", "eps", "quarterly results", "beats estimates", "misses estimates"),
        ),
        (EventTaxonomy.GUIDANCE, ("guidance", "outlook", "raises forecast", "cuts forecast")),
        (EventTaxonomy.M_AND_A, ("acquire", "acquisition", "merger", "takeover", "buyout")),
        (EventTaxonomy.DIVIDEND, ("dividend", "buyback", "repurchase")),
        (
            EventTaxonomy.REGULATORY,
            ("sec ", "doj", "antitrust", "regulator", "lawsuit", "investigation"),
        ),
        (EventTaxonomy.LEGAL, ("lawsuit", "settlement", "court", "litigation")),
        (EventTaxonomy.MANAGEMENT, ("ceo", "cfo", "resign", "appoint", "executive")),
        (EventTaxonomy.PRODUCT, ("launch", "product", "unveil", "release")),
        (
            EventTaxonomy.MACRO,
            ("fed ", "fomc", "rate cut", "rate hike", "cpi", "payrolls", "inflation"),
        ),
        (EventTaxonomy.PEER, ("peer", "sector", "industry-wide", "rivals")),
        (EventTaxonomy.SUPPLY_CHAIN, ("supplier", "supply chain", "factory", "shortage")),
        (EventTaxonomy.ANALYST, ("upgrade", "downgrade", "price target", "initiates coverage")),
        (EventTaxonomy.SOCIAL, ("reddit", "stocktwits", "viral", "meme", "retail chatter")),
        (EventTaxonomy.CORPORATE_ACTION, ("8-k", "10-k", "10-q", "spin-off", "split")),
    ]
    for taxonomy, tokens in rules:
        if any(token in text for token in tokens):
            return taxonomy
    return EventTaxonomy.OTHER


def attribution_for_taxonomy(taxonomy: EventTaxonomy) -> AttributionLevel:
    if taxonomy is EventTaxonomy.OTHER or taxonomy is EventTaxonomy.SOCIAL:
        return AttributionLevel.NONE
    if taxonomy in {
        EventTaxonomy.EARNINGS,
        EventTaxonomy.GUIDANCE,
        EventTaxonomy.CORPORATE_ACTION,
        EventTaxonomy.PRODUCT,
        EventTaxonomy.DIVIDEND,
        EventTaxonomy.M_AND_A,
        EventTaxonomy.LEGAL,
        EventTaxonomy.SUPPLY_CHAIN,
        EventTaxonomy.MANAGEMENT,
        EventTaxonomy.REGULATORY,
        EventTaxonomy.ANALYST,
    }:
        return AttributionLevel.COMPANY
    if taxonomy is EventTaxonomy.PEER:
        return AttributionLevel.PEER
    if taxonomy is EventTaxonomy.MACRO:
        return AttributionLevel.MACRO
    return AttributionLevel.NONE


def is_causal_candidate(item: RawNewsItem) -> bool:
    """Unknown/weak/unclassified items cannot become causal catalyst evidence."""

    if item.source_tier in {"unknown", "social"}:
        return False
    if item.taxonomy is EventTaxonomy.OTHER or item.taxonomy is EventTaxonomy.SOCIAL:
        return False
    if item.taxonomy not in RELEVANT_TAXONOMIES:
        return False
    if item.source_tier not in RELIABLE_SOURCE_TIERS and item.source_tier != "specialist":
        return False
    return True


def confidence_for_item(item: RawNewsItem) -> CatalystConfidence | None:
    """Require reliable source plus relevant taxonomy for Likely/Confirmed."""

    if not is_causal_candidate(item):
        return None
    if (
        item.source_tier in {"company", "regulator", "government"}
        and item.taxonomy in RELEVANT_TAXONOMIES
    ):
        return CatalystConfidence.CONFIRMED
    if item.source_tier in {"established", "licensed"}:
        if item.taxonomy in HIGH_IMPACT_TAXONOMIES or item.taxonomy in {
            EventTaxonomy.DIVIDEND,
            EventTaxonomy.PRODUCT,
            EventTaxonomy.LEGAL,
            EventTaxonomy.SUPPLY_CHAIN,
            EventTaxonomy.ANALYST,
            EventTaxonomy.PEER,
        }:
            if item.taxonomy in HIGH_IMPACT_TAXONOMIES:
                return CatalystConfidence.CONFIRMED
            return CatalystConfidence.LIKELY
        return None
    if item.source_tier == "specialist" and item.taxonomy in RELEVANT_TAXONOMIES:
        return CatalystConfidence.LIKELY
    return None


def is_high_impact(item: RawNewsItem, confidence: CatalystConfidence) -> bool:
    """Restrict high-impact promotion to confirmed high-impact taxonomies only."""

    if confidence is not CatalystConfidence.CONFIRMED:
        return False
    return item.taxonomy in HIGH_IMPACT_TAXONOMIES


def parse_raw_news_payload(symbol: str, provider: str, value: Any) -> list[RawNewsItem]:
    """Normalize provider payloads into RawNewsItem records."""

    items: list[RawNewsItem] = []
    if value is None:
        return items
    rows = value if isinstance(value, list) else [value]
    for row in rows:
        if isinstance(row, RawNewsItem):
            items.append(row.model_copy(update={"symbol": symbol.upper()}))
            continue
        if not isinstance(row, dict):
            continue
        # Already-normalized dict from stub providers.
        if "title" in row and "published_at" in row and "url" in row and "provider" in row:
            try:
                items.append(RawNewsItem.model_validate({**row, "symbol": symbol.upper()}))
                continue
            except Exception:
                pass
        title = str(row.get("title") or row.get("headline") or "").strip()
        url = str(row.get("url") or row.get("article_url") or row.get("amp_url") or "").strip()
        if not title or not url:
            continue
        published_raw = (
            row.get("published_utc")
            or row.get("published_at")
            or row.get("created")
            or row.get("updated")
        )
        published_at = _parse_datetime(published_raw)
        if published_at is None:
            # Drop undated items — never invent "now" as a publication time.
            continue
        source_name = str(
            (row.get("publisher") or {}).get("name")
            if isinstance(row.get("publisher"), dict)
            else row.get("publisher") or row.get("author") or row.get("source") or provider
        )
        excerpt = row.get("description") or row.get("teaser") or row.get("excerpt")
        full_text = row.get("body") or row.get("full_text") or row.get("text")
        tier = str(row.get("source_tier") or classify_source_tier(source_name, provider))
        taxonomy = row.get("taxonomy")
        if isinstance(taxonomy, EventTaxonomy):
            tax = taxonomy
        elif isinstance(taxonomy, str):
            try:
                tax = EventTaxonomy(taxonomy)
            except ValueError:
                tax = classify_taxonomy(title, excerpt if isinstance(excerpt, str) else None)
        else:
            tax = classify_taxonomy(title, excerpt if isinstance(excerpt, str) else None)
        items.append(
            RawNewsItem(
                symbol=symbol.upper(),
                title=title,
                url=url,
                published_at=published_at,
                provider=str(row.get("provider") or provider),
                source_name=source_name,
                source_tier=tier,
                excerpt=str(excerpt) if excerpt else None,
                full_text=str(full_text) if full_text else None,
                taxonomy=tax,
                external_id=str(row.get("id") or row.get("external_id") or "") or None,
            )
        )
    return items


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


class CatalystService:
    """Scan held underlyings and assemble evidence-backed catalyst results."""

    def __init__(
        self,
        *,
        database: PositionPilotDatabase,
        news_providers: Sequence[NewsProvider] | None = None,
        news_provider_factory: Callable[[], Sequence[NewsProvider]] | None = None,
        quote_source: Callable[[str], SymbolMoveInput | None] | None = None,
        option_metrics_source: Callable[[str], dict[str, Any] | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        settings: CatalystSettings | None = None,
        benzinga_api_key_present: bool | None = None,
    ) -> None:
        self.database = database
        self._static_news_providers = list(news_providers or [])
        self.news_provider_factory = news_provider_factory
        self.quote_source = quote_source
        self.option_metrics_source = option_metrics_source
        self.clock = clock or (lambda: datetime.now(UTC))
        self.settings = settings or CatalystSettings()
        self.benzinga_api_key_present = benzinga_api_key_present

    def active_news_providers(self) -> list[NewsProvider]:
        """Resolve providers on each call so Benzinga toggle applies without restart."""

        if self.news_provider_factory is not None:
            return list(self.news_provider_factory())
        return list(self._static_news_providers)

    def scan_held(self, symbols: Iterable[str]) -> CatalystScanSnapshot:
        unique = list(dict.fromkeys(symbol.upper() for symbol in symbols if symbol))
        results = [self.analyze_symbol(symbol) for symbol in unique]
        coverage = CoverageState.COMPLETE
        notes: list[str] = []
        if any(item.coverage is CoverageState.OFFLINE for item in results):
            coverage = CoverageState.OFFLINE
        elif any(item.coverage is CoverageState.INCOMPLETE for item in results):
            coverage = CoverageState.INCOMPLETE
        elif any(item.coverage is CoverageState.UNAVAILABLE for item in results):
            coverage = CoverageState.UNAVAILABLE
        for item in results:
            notes.extend(item.coverage_notes)
        now = self.clock()
        freshness_state = FreshnessState.FRESH
        if coverage is CoverageState.OFFLINE:
            freshness_state = (
                FreshnessState.STALE
                if any(r.cached for r in results)
                else FreshnessState.UNAVAILABLE
            )
        return CatalystScanSnapshot(
            captured_at=now,
            results=sorted(results, key=lambda row: (not row.promoted, row.symbol)),
            settings=self.public_settings(),
            freshness=DataFreshness(as_of=now, provider="catalyst-service", state=freshness_state),
            coverage=coverage,
            coverage_notes=list(dict.fromkeys(notes)),
        )

    def analyze_symbol(self, symbol: str) -> SymbolCatalystResult:
        normalized = symbol.upper()
        now = self.clock()
        prior_close_at = previous_regular_close(now)
        lookback_start = prior_close_at
        scheduled_start = now - timedelta(hours=self.settings.scheduled_window_hours)

        move = self._move_input(normalized)
        move_percent = self._move_percent(move)
        threshold = (
            self.settings.etf_move_threshold_pct
            if (
                move.is_broad_etf
                if move and move.is_broad_etf is not None
                else is_broad_etf(normalized)
            )
            else self.settings.stock_move_threshold_pct
        )
        meaningful_move = move_percent is not None and abs(move_percent) >= threshold
        abnormal = self._abnormal_intraday(move)

        providers = self.active_news_providers()
        raw_items: list[RawNewsItem] = []
        coverage_notes: list[str] = []
        configured_count = 0
        providers_ok = 0
        providers_failed = 0

        if not providers:
            coverage = CoverageState.UNAVAILABLE
            coverage_notes.append("No news providers configured")
            mechanisms, mechanism_coverage = self._option_mechanisms(normalized, now)
            has_mechanism = bool(mechanisms)
            result = SymbolCatalystResult(
                symbol=normalized,
                confidence=(
                    CatalystConfidence.LIKELY
                    if has_mechanism
                    else CatalystConfidence.NO_CONFIRMED_CATALYST
                ),
                attribution=(
                    AttributionLevel.OPTIONS_MARKET if has_mechanism else AttributionLevel.NONE
                ),
                summary=mechanisms[0].summary if has_mechanism else "No confirmed catalyst found",
                option_mechanisms=mechanisms,
                option_mechanism_coverage=mechanism_coverage,
                move_percent=move_percent,
                prior_close=move.prior_close if move else None,
                last_price=move.last_price if move else None,
                meaningful_move=meaningful_move,
                promoted=has_mechanism,
                lookback_start=lookback_start,
                lookback_end=now,
                coverage=coverage,
                coverage_notes=coverage_notes,
                freshness=DataFreshness(
                    as_of=now,
                    provider="catalyst-service",
                    state=FreshnessState.UNAVAILABLE,
                ),
                quiet=not has_mechanism,
            )
            # Do not overwrite a good cache with empty UNAVAILABLE results.
            return result

        for provider in providers:
            name = getattr(provider, "name", "provider")
            health = None
            try:
                health = provider.health()
            except Exception:
                health = None
            state = getattr(health, "state", None) if health is not None else None
            if state is not None and str(state) == "not_configured":
                coverage_notes.append(f"{name} not configured")
                continue
            configured_count += 1
            try:
                payload = provider.news(normalized, scheduled_start, now)
            except Exception as error:
                providers_failed += 1
                coverage_notes.append(f"{name} unavailable: {type(error).__name__}")
                continue
            if payload is None:
                # Distinguish empty healthy vs unhealthy None.
                try:
                    health = provider.health()
                    state = getattr(health, "state", None)
                except Exception:
                    state = None
                if state is not None and str(state) in {"unavailable", "degraded"}:
                    providers_failed += 1
                    coverage_notes.append(f"{name} failed ({state})")
                else:
                    providers_ok += 1
                    coverage_notes.append(f"{name} returned no news in window (healthy empty)")
                continue
            providers_ok += 1
            raw_items.extend(parse_raw_news_payload(normalized, payload.provider, payload.value))

        if configured_count == 0:
            coverage = CoverageState.UNAVAILABLE
            coverage_notes.append("No configured news providers available")
        elif providers_ok == 0 and providers_failed > 0:
            coverage = CoverageState.OFFLINE
        elif providers_failed > 0:
            coverage = CoverageState.INCOMPLETE
        else:
            # Healthy empty is complete coverage with no items.
            coverage = CoverageState.COMPLETE
            coverage_notes = [
                note
                for note in coverage_notes
                if "not configured" not in note and "healthy empty" not in note
            ]

        # Offline with cache: return STALE cached result; do not overwrite good cache.
        if coverage is CoverageState.OFFLINE:
            cached = self.database.get_latest_symbol_catalyst(normalized)
            if cached is not None:
                try:
                    cached_result = SymbolCatalystResult.model_validate(cached)
                except Exception:
                    cached_result = None
                if cached_result is not None:
                    notes = list(
                        dict.fromkeys(
                            [
                                *coverage_notes,
                                "Using cached catalyst result; live providers are offline.",
                                f"Cached as of {cached_result.freshness.as_of.isoformat()}",
                            ]
                        )
                    )
                    return cached_result.model_copy(
                        update={
                            "coverage": CoverageState.OFFLINE,
                            "coverage_notes": notes,
                            "freshness": DataFreshness(
                                as_of=cached_result.freshness.as_of,
                                provider=cached_result.freshness.provider,
                                state=FreshnessState.STALE,
                            ),
                            "cached": True,
                            "quiet": cached_result.quiet,
                        }
                    )

        filtered: list[RawNewsItem] = []
        for item in raw_items:
            published = (
                item.published_at
                if item.published_at.tzinfo
                else item.published_at.replace(tzinfo=UTC)
            )
            if published >= lookback_start:
                filtered.append(item)
            elif published >= scheduled_start and (
                item.taxonomy in HIGH_IMPACT_TAXONOMIES
                or item.source_tier in {"company", "regulator", "government"}
            ):
                filtered.append(item)

        social_notes: list[SocialSideNote] = []
        evidence_items: list[RawNewsItem] = []
        for item in filtered:
            if item.source_tier == "social" or item.taxonomy is EventTaxonomy.SOCIAL:
                social_notes.append(
                    SocialSideNote(
                        note_id=stable_source_id(
                            catalyst_id=f"social:{normalized}",
                            provider=item.provider,
                            url=item.url,
                        ),
                        headline=item.title,
                        summary=item.excerpt or item.title,
                        source_name=item.source_name,
                        url=item.url,
                        published_at=item.published_at,
                    )
                )
            elif is_causal_candidate(item):
                evidence_items.append(item)
            # Weak/unknown/OTHER dropped from causal evidence (not social, not causal).

        catalysts = self._build_catalysts(normalized, evidence_items)
        feedback_adjustments = self._feedback_score_map(normalized)
        for catalyst in catalysts:
            catalyst.rank_score += feedback_adjustments.get(catalyst.catalyst_id, 0.0)
        catalysts.sort(
            key=lambda item: (-item.rank_score, -item.event_at.timestamp(), item.headline)
        )

        mechanisms, mechanism_coverage = self._option_mechanisms(normalized, now)
        high_impact = any(item.high_impact for item in catalysts)
        promoted = (
            meaningful_move or abnormal or high_impact or bool(mechanisms and not meaningful_move)
        )

        # Guard: incomplete/offline evidence cannot be sole causal proof of a new move.
        sole_causal_blocked = False
        if meaningful_move and coverage in {CoverageState.INCOMPLETE, CoverageState.OFFLINE}:
            if catalysts and not high_impact:
                sole_causal_blocked = True
                coverage_notes.append(
                    "Incomplete or offline coverage — evidence is not "
                    "sole causal proof of the move."
                )

        if catalysts and not sole_causal_blocked:
            primary = catalysts[0]
            confidence = primary.confidence
            attribution = primary.attribution
            summary = primary.summary
        elif sole_causal_blocked:
            confidence = CatalystConfidence.NO_CONFIRMED_CATALYST
            attribution = AttributionLevel.NONE
            summary = "No confirmed catalyst found"
            # Keep catalysts visible as supporting context but abstain on primary claim.
        elif mechanisms:
            confidence = CatalystConfidence.LIKELY
            attribution = AttributionLevel.OPTIONS_MARKET
            summary = mechanisms[0].summary
        else:
            confidence = CatalystConfidence.NO_CONFIRMED_CATALYST
            attribution = AttributionLevel.NONE
            summary = "No confirmed catalyst found"

        if sole_causal_blocked:
            confidence = CatalystConfidence.NO_CONFIRMED_CATALYST
            attribution = AttributionLevel.NONE
            summary = "No confirmed catalyst found"

        freshness_state = FreshnessState.FRESH
        if coverage is CoverageState.OFFLINE:
            freshness_state = FreshnessState.UNAVAILABLE
        elif coverage is CoverageState.UNAVAILABLE:
            freshness_state = FreshnessState.UNAVAILABLE

        result = SymbolCatalystResult(
            symbol=normalized,
            confidence=confidence,
            attribution=attribution,
            summary=summary,
            catalysts=catalysts if not sole_causal_blocked else [],
            option_mechanisms=mechanisms,
            option_mechanism_coverage=mechanism_coverage,
            social_side_notes=social_notes,
            move_percent=move_percent,
            prior_close=move.prior_close if move else None,
            last_price=move.last_price if move else None,
            meaningful_move=meaningful_move,
            promoted=promoted and not sole_causal_blocked,
            lookback_start=lookback_start,
            lookback_end=now,
            coverage=coverage,
            coverage_notes=list(dict.fromkeys(coverage_notes)),
            freshness=DataFreshness(
                as_of=now,
                provider="catalyst-service",
                state=freshness_state,
            ),
            quiet=not (promoted and not sole_causal_blocked),
        )
        # Never overwrite a good cache with an empty offline result.
        if (
            coverage is CoverageState.OFFLINE
            and not result.catalysts
            and not result.option_mechanisms
        ):
            return result
        self._persist_result(result, evidence_items)
        return result

    def event_markers(self, symbol: str) -> list[ChartEventMarker]:
        result = self.database.get_latest_symbol_catalyst(symbol.upper())
        if result is None:
            analyzed = self.analyze_symbol(symbol)
            events = analyzed.catalysts
        else:
            events = [CatalystEvent.model_validate(item) for item in result.get("catalysts", [])]
        return [
            ChartEventMarker(
                catalyst_id=event.catalyst_id,
                timestamp=event.event_at,
                headline=event.headline,
                confidence=event.confidence,
                attribution=event.attribution,
            )
            for event in events
        ]

    def submit_feedback(
        self,
        catalyst_id: str | None,
        kind: CatalystFeedbackKind,
        *,
        symbol: str | None = None,
        note: str = "",
    ) -> CatalystFeedbackEvent:
        resolved_symbol = symbol.upper() if symbol else None
        if catalyst_id and not resolved_symbol:
            resolved_symbol = self.database.symbol_for_catalyst(catalyst_id)
        if catalyst_id and not self.database.catalyst_exists(catalyst_id):
            # Allow missing-catalyst feedback without id; otherwise require known id.
            if kind is not CatalystFeedbackKind.MISSING_CATALYST:
                raise ValueError(f"Unknown catalyst_id: {catalyst_id}")
        if kind is not CatalystFeedbackKind.MISSING_CATALYST and not catalyst_id:
            raise ValueError("catalyst_id is required")
        if not resolved_symbol and kind is CatalystFeedbackKind.MISSING_CATALYST:
            raise ValueError("symbol is required for missing catalyst feedback")
        if not resolved_symbol:
            raise ValueError("symbol is required for feedback ranking")
        event = CatalystFeedbackEvent(
            feedback_id=str(uuid4()),
            kind=kind,
            catalyst_id=catalyst_id,
            symbol=resolved_symbol,
            note=note,
            recorded_at=self.clock(),
        )
        self.database.append_catalyst_feedback(event.model_dump(mode="json"))
        return event

    def feedback_history(
        self,
        catalyst_id: str | None = None,
        *,
        symbol: str | None = None,
    ) -> list[CatalystFeedbackEvent]:
        rows = self.database.list_catalyst_feedback(catalyst_id=catalyst_id, symbol=symbol)
        return [CatalystFeedbackEvent.model_validate(row) for row in rows]

    def apply_retention(self, *, now: datetime | None = None) -> dict[str, int]:
        current = now or self.clock()
        event_cutoff = current - timedelta(days=self.settings.catalyst_retention_days)
        article_cutoff = current - timedelta(days=self.settings.article_metadata_retention_days)
        return self.database.prune_catalysts(
            event_cutoff=event_cutoff,
            article_cutoff=article_cutoff,
        )

    def honor_removal_notice(self, *, provider: str, url: str) -> int:
        """Clear licensed content/excerpts and derived presentation; feedback stays immutable."""

        return self.database.apply_catalyst_removal(provider=provider, url=url)

    def public_settings(self) -> dict[str, Any]:
        stored = self.database.get_setting("catalysts", {}) or {}
        enabled = bool(stored.get("benzinga_enabled", self.settings.benzinga_enabled))
        key_present = (
            self.benzinga_api_key_present
            if self.benzinga_api_key_present is not None
            else bool(__import__("os").getenv("BENZINGA_API_KEY"))
        )
        if not enabled:
            benzinga_status = "disabled"
        elif not key_present:
            benzinga_status = "not_configured"
        else:
            benzinga_status = "enabled"
        return {
            "stock_move_threshold_pct": stored.get(
                "stock_move_threshold_pct", self.settings.stock_move_threshold_pct
            ),
            "etf_move_threshold_pct": stored.get(
                "etf_move_threshold_pct", self.settings.etf_move_threshold_pct
            ),
            "news_cadence_seconds": stored.get(
                "news_cadence_seconds", self.settings.news_cadence_seconds
            ),
            "benzinga": {
                "enabled": enabled and key_present,
                "status": benzinga_status,
            },
            "scheduled_window_hours": self.settings.scheduled_window_hours,
        }

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.database.get_setting("catalysts", {}) or {}
        allowed = {
            "stock_move_threshold_pct",
            "etf_move_threshold_pct",
            "news_cadence_seconds",
            "benzinga_enabled",
        }
        for key, value in payload.items():
            if key in allowed:
                current[key] = value
        self.database.set_setting("catalysts", current)
        if "stock_move_threshold_pct" in current:
            self.settings.stock_move_threshold_pct = float(current["stock_move_threshold_pct"])
        if "etf_move_threshold_pct" in current:
            self.settings.etf_move_threshold_pct = float(current["etf_move_threshold_pct"])
        if "news_cadence_seconds" in current:
            self.settings.news_cadence_seconds = int(current["news_cadence_seconds"])
        if "benzinga_enabled" in current:
            self.settings.benzinga_enabled = bool(current["benzinga_enabled"])
        return self.public_settings()

    def _move_input(self, symbol: str) -> SymbolMoveInput | None:
        if self.quote_source is None:
            return None
        try:
            move = self.quote_source(symbol)
        except Exception:
            return None
        if move is None:
            return None
        if move.is_broad_etf is None:
            move = move.model_copy(update={"is_broad_etf": is_broad_etf(symbol)})
        return move

    @staticmethod
    def _move_percent(move: SymbolMoveInput | None) -> float | None:
        if move is None or move.last_price is None or move.prior_close in (None, 0):
            return None
        return ((move.last_price - move.prior_close) / move.prior_close) * 100

    def _abnormal_intraday(self, move: SymbolMoveInput | None) -> bool:
        if move is None or move.last_price is None or move.prior_close in (None, 0):
            return False
        if move.session_high is None or move.session_low is None:
            return False
        session_range = move.session_high - move.session_low
        if session_range <= 0:
            return False
        move_abs = abs(move.last_price - move.prior_close)
        # Promote when the session range itself is large vs prior close threshold.
        threshold_pct = (
            self.settings.etf_move_threshold_pct
            if move.is_broad_etf
            else self.settings.stock_move_threshold_pct
        )
        range_pct = (session_range / move.prior_close) * 100
        large_range = range_pct >= threshold_pct * self.settings.abnormal_range_multiple
        sharp_move = (move_abs / session_range) >= self.settings.abnormal_range_multiple and abs(
            self._move_percent(move) or 0
        ) >= threshold_pct * 0.75
        return large_range or sharp_move

    def _build_catalysts(self, symbol: str, items: list[RawNewsItem]) -> list[CatalystEvent]:
        grouped: dict[str, list[RawNewsItem]] = {}
        for item in items:
            fingerprint = content_fingerprint(symbol, item.title, item.published_at)
            grouped.setdefault(fingerprint, []).append(item)

        catalysts: list[CatalystEvent] = []
        for fingerprint, group in grouped.items():
            group.sort(
                key=lambda item: (
                    -SOURCE_TIER_PRIORITY.get(item.source_tier, 0),
                    item.published_at.timestamp(),
                )
            )
            primary = group[0]
            confidence = confidence_for_item(primary)
            if confidence is None:
                continue
            attribution = attribution_for_taxonomy(primary.taxonomy)
            if attribution is AttributionLevel.NONE:
                continue
            catalyst_id = hashlib.sha256(f"{symbol}:{fingerprint}".encode()).hexdigest()[:32]
            sources = [
                CatalystSource(
                    source_id=stable_source_id(
                        catalyst_id=catalyst_id,
                        provider=item.provider,
                        url=item.url,
                    ),
                    name=item.source_name,
                    tier=item.source_tier,
                    url=item.url,
                    provider=item.provider,
                    published_at=item.published_at,
                    excerpt=(item.excerpt[:400] if item.excerpt else None),
                )
                for item in group
            ]
            rank = (
                SOURCE_TIER_PRIORITY.get(primary.source_tier, 0)
                + (30 if confidence is CatalystConfidence.CONFIRMED else 10)
                + (20 if is_high_impact(primary, confidence) else 0)
                + min(len(group) * 2, 10)
            )
            summary = primary.excerpt or primary.title
            catalysts.append(
                CatalystEvent(
                    catalyst_id=catalyst_id,
                    symbol=symbol,
                    headline=primary.title,
                    summary=summary if summary else primary.title,
                    taxonomy=primary.taxonomy,
                    confidence=confidence,
                    attribution=attribution,
                    evidence_kind=(
                        CatalystEvidenceKind.FILING
                        if primary.source_tier in {"regulator", "government"}
                        or primary.taxonomy is EventTaxonomy.CORPORATE_ACTION
                        else CatalystEvidenceKind.NEWS
                    ),
                    event_at=min(item.published_at for item in group),
                    sources=sources,
                    rank_score=float(rank),
                    high_impact=is_high_impact(primary, confidence),
                    fingerprint=fingerprint,
                )
            )
        return catalysts

    def _option_mechanisms(
        self, symbol: str, now: datetime
    ) -> tuple[list[OptionMechanism], list[OptionMechanismCoverage]]:
        coverage: list[OptionMechanismCoverage] = []
        mechanisms: list[OptionMechanism] = []
        if self.option_metrics_source is None:
            for kind, label in (
                (OptionMechanismKind.IV_EXPANSION, "IV expansion/crush"),
                (OptionMechanismKind.UNUSUAL_VOLUME, "Unusual options volume"),
                (OptionMechanismKind.OPEN_INTEREST, "Open interest"),
                (OptionMechanismKind.SKEW, "Skew"),
                (OptionMechanismKind.LIQUIDITY, "Liquidity"),
                (OptionMechanismKind.DIVIDEND, "Dividend proximity"),
                (OptionMechanismKind.EARNINGS_PROXIMITY, "Earnings proximity"),
                (OptionMechanismKind.EXPIRATION, "Expiration"),
                (OptionMechanismKind.GAMMA, "Gamma risk"),
            ):
                coverage.append(
                    OptionMechanismCoverage(
                        kind=kind,
                        label=label,
                        availability=MechanismAvailability.UNAVAILABLE,
                        detail="Option metrics source is not configured.",
                    )
                )
            return [], coverage

        try:
            metrics = self.option_metrics_source(symbol) or {}
        except Exception:
            metrics = {}

        def mark(
            kind: OptionMechanismKind,
            label: str,
            available: bool,
            detail: str,
        ) -> None:
            coverage.append(
                OptionMechanismCoverage(
                    kind=kind,
                    label=label,
                    availability=(
                        MechanismAvailability.AVAILABLE
                        if available
                        else MechanismAvailability.UNAVAILABLE
                    ),
                    detail=detail,
                )
            )

        iv_change = metrics.get("iv_change_pct")
        if isinstance(iv_change, (int, float)):
            mark(
                OptionMechanismKind.IV_EXPANSION,
                "IV expansion/crush",
                True,
                "iv_change_pct present",
            )
            if iv_change >= 10:
                mechanisms.append(
                    OptionMechanism(
                        kind=OptionMechanismKind.IV_EXPANSION,
                        label="IV expansion",
                        summary=(
                            f"Implied volatility expanded about {iv_change:.1f}% "
                            "(options-market mechanism, not news)."
                        ),
                        magnitude=float(iv_change),
                        observed_at=now,
                    )
                )
            elif iv_change <= -10:
                mechanisms.append(
                    OptionMechanism(
                        kind=OptionMechanismKind.IV_CRUSH,
                        label="IV crush",
                        summary=(
                            f"Implied volatility compressed about {abs(iv_change):.1f}% "
                            "(options-market mechanism, not news)."
                        ),
                        magnitude=float(iv_change),
                        observed_at=now,
                    )
                )
        else:
            mark(
                OptionMechanismKind.IV_EXPANSION,
                "IV expansion/crush",
                False,
                "iv_change_pct not available from providers",
            )

        volume_available = (
            metrics.get("unusual_volume") is not None
            or isinstance(metrics.get("volume_oi_ratio"), (int, float))
            or isinstance(metrics.get("option_volume"), (int, float))
        )
        if volume_available:
            mark(
                OptionMechanismKind.UNUSUAL_VOLUME,
                "Unusual options volume",
                True,
                "volume fields present",
            )
            ratio = metrics.get("volume_oi_ratio")
            if metrics.get("unusual_volume") or (
                isinstance(ratio, (int, float)) and float(ratio) >= 2.0
            ):
                mechanisms.append(
                    OptionMechanism(
                        kind=OptionMechanismKind.UNUSUAL_VOLUME,
                        label="Unusual options volume",
                        summary=(
                            f"Options volume/OI ratio {float(ratio):.1f}x "
                            if isinstance(ratio, (int, float))
                            else "Unusual options volume "
                        )
                        + "(options-market mechanism, not news).",
                        magnitude=float(ratio) if isinstance(ratio, (int, float)) else None,
                        observed_at=now,
                    )
                )
        else:
            mark(
                OptionMechanismKind.UNUSUAL_VOLUME,
                "Unusual options volume",
                False,
                "option volume/OI not available from providers",
            )

        oi_available = isinstance(metrics.get("open_interest"), (int, float)) or isinstance(
            metrics.get("volume_oi_ratio"), (int, float)
        )
        mark(
            OptionMechanismKind.OPEN_INTEREST,
            "Open interest",
            oi_available,
            "open interest present" if oi_available else "open interest not available",
        )
        if isinstance(metrics.get("open_interest"), (int, float)) and metrics.get(
            "unusual_open_interest"
        ):
            mechanisms.append(
                OptionMechanism(
                    kind=OptionMechanismKind.OPEN_INTEREST,
                    label="Unusual open interest",
                    summary=(
                        f"Open interest {float(metrics['open_interest']):.0f} "
                        "(options-market mechanism, not news)."
                    ),
                    magnitude=float(metrics["open_interest"]),
                    observed_at=now,
                )
            )

        skew = metrics.get("skew_shift")
        if isinstance(skew, (int, float)):
            mark(OptionMechanismKind.SKEW, "Skew", True, "skew_shift present")
            if abs(skew) >= 0.08:
                mechanisms.append(
                    OptionMechanism(
                        kind=OptionMechanismKind.SKEW,
                        label="Skew shift",
                        summary=(
                            f"Options skew shifted {skew:+.2f} "
                            "(options-market mechanism, not news)."
                        ),
                        magnitude=float(skew),
                        observed_at=now,
                    )
                )
        else:
            mark(OptionMechanismKind.SKEW, "Skew", False, "skew not available from providers")

        liquidity = metrics.get("liquidity_score")
        if isinstance(liquidity, (int, float)):
            mark(OptionMechanismKind.LIQUIDITY, "Liquidity", True, "liquidity_score present")
            if liquidity < 0.5:
                mechanisms.append(
                    OptionMechanism(
                        kind=OptionMechanismKind.LIQUIDITY,
                        label="Liquidity stress",
                        summary=(
                            "Option liquidity is impaired relative to normal "
                            "(options-market mechanism, not news)."
                        ),
                        magnitude=float(liquidity),
                        observed_at=now,
                    )
                )
        else:
            mark(
                OptionMechanismKind.LIQUIDITY,
                "Liquidity",
                False,
                "liquidity_score not available from providers",
            )

        dividend_date = metrics.get("dividend_date")
        mark(
            OptionMechanismKind.DIVIDEND,
            "Dividend proximity",
            bool(dividend_date),
            "dividend_date present" if dividend_date else "dividend date not available",
        )
        if dividend_date:
            mechanisms.append(
                OptionMechanism(
                    kind=OptionMechanismKind.DIVIDEND,
                    label="Dividend proximity",
                    summary=(
                        f"Dividend date {dividend_date} is nearby "
                        "(options-market mechanism, not news)."
                    ),
                    observed_at=now,
                )
            )

        earnings_date = metrics.get("earnings_date")
        mark(
            OptionMechanismKind.EARNINGS_PROXIMITY,
            "Earnings proximity",
            bool(earnings_date),
            "earnings_date present" if earnings_date else "earnings date not available",
        )
        if earnings_date:
            mechanisms.append(
                OptionMechanism(
                    kind=OptionMechanismKind.EARNINGS_PROXIMITY,
                    label="Earnings proximity",
                    summary=(
                        f"Earnings date {earnings_date} is nearby "
                        "(options-market mechanism, not news)."
                    ),
                    observed_at=now,
                )
            )

        dte = metrics.get("days_to_expiration")
        mark(
            OptionMechanismKind.EXPIRATION,
            "Expiration",
            isinstance(dte, int),
            "days_to_expiration present" if isinstance(dte, int) else "DTE not available",
        )
        if isinstance(dte, int) and dte <= 3:
            mechanisms.append(
                OptionMechanism(
                    kind=OptionMechanismKind.EXPIRATION,
                    label="Expiration proximity",
                    summary=(
                        f"{dte} DTE — expiration effects may dominate "
                        "(options-market mechanism, not news)."
                    ),
                    magnitude=float(dte),
                    observed_at=now,
                )
            )

        gamma = metrics.get("gamma_risk")
        mark(
            OptionMechanismKind.GAMMA,
            "Gamma risk",
            gamma is not None,
            "gamma_risk present" if gamma is not None else "gamma not available from providers",
        )
        if gamma:
            mechanisms.append(
                OptionMechanism(
                    kind=OptionMechanismKind.GAMMA,
                    label="Gamma risk",
                    summary=f"Gamma risk marked {gamma} (options-market mechanism, not news).",
                    observed_at=now,
                )
            )
        return mechanisms, coverage

    def _feedback_score_map(self, symbol: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        # Include feedback stored under this symbol and by catalyst id alone.
        rows = self.database.list_catalyst_feedback(symbol=symbol)
        for row in rows:
            catalyst_id = row.get("catalyst_id")
            if not catalyst_id:
                continue
            kind = row.get("kind")
            if kind == CatalystFeedbackKind.RELEVANT.value or kind == CatalystFeedbackKind.RELEVANT:
                scores[catalyst_id] = scores.get(catalyst_id, 0.0) + 25.0
            elif (
                kind == CatalystFeedbackKind.NOT_RELATED.value
                or kind == CatalystFeedbackKind.NOT_RELATED
            ):
                # Large demotion so trader feedback can outrank default source priority.
                scores[catalyst_id] = scores.get(catalyst_id, 0.0) - 100.0
        return scores

    def _persist_result(self, result: SymbolCatalystResult, raw_items: list[RawNewsItem]) -> None:
        now = self.clock()
        for catalyst in result.catalysts:
            self.database.upsert_catalyst_event(
                {
                    "catalyst_id": catalyst.catalyst_id,
                    "symbol": catalyst.symbol,
                    "fingerprint": catalyst.fingerprint,
                    "headline": catalyst.headline,
                    "summary": catalyst.summary,
                    "taxonomy": catalyst.taxonomy.value,
                    "confidence": catalyst.confidence.value,
                    "attribution": catalyst.attribution.value,
                    "evidence_kind": catalyst.evidence_kind.value,
                    "event_at": catalyst.event_at.isoformat(),
                    "rank_score": catalyst.rank_score,
                    "high_impact": catalyst.high_impact,
                    "payload_json": catalyst.model_dump(mode="json"),
                    "created_at": now.isoformat(),
                }
            )
            for source in catalyst.sources:
                self.database.upsert_catalyst_source(
                    {
                        "source_id": source.source_id,
                        "catalyst_id": catalyst.catalyst_id,
                        "name": source.name,
                        "tier": source.tier,
                        "url": source.url,
                        "provider": source.provider,
                        "published_at": source.published_at.isoformat(),
                        "excerpt": source.excerpt,
                    }
                )
        for item in raw_items:
            store_full = item.provider in self.settings.store_full_text_providers
            self.database.upsert_catalyst_article(
                {
                    "article_id": hashlib.sha256(
                        f"{item.provider}:{item.url}".encode()
                    ).hexdigest()[:32],
                    "symbol": result.symbol,
                    "provider": item.provider,
                    "url": item.url,
                    "title": item.title,
                    "source_name": item.source_name,
                    "published_at": item.published_at.isoformat(),
                    "excerpt": (item.excerpt[:500] if item.excerpt else None),
                    "full_text": item.full_text if store_full else None,
                    "stored_at": now.isoformat(),
                }
            )
        self.database.save_symbol_catalyst_snapshot(
            result.symbol,
            result.model_dump(mode="json"),
            captured_at=now,
        )
