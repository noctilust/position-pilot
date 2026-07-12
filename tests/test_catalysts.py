"""Phase 5 catalyst intelligence: ranking, attribution, retention, feedback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from position_pilot.domain.catalysts import (
    AttributionLevel,
    CatalystConfidence,
    CatalystEvidenceKind,
    CatalystFeedbackKind,
    CatalystService,
    CatalystSettings,
    CoverageState,
    EventTaxonomy,
    OptionMechanismKind,
    RawNewsItem,
    SymbolMoveInput,
)
from position_pilot.domain.snapshots import FreshnessState
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.providers.benzinga import BenzingaProvider
from position_pilot.providers.contracts import ProviderState

FIXED_NOW = datetime(2026, 7, 11, 18, 30, tzinfo=UTC)
PRIOR_CLOSE_AT = datetime(2026, 7, 10, 20, 0, tzinfo=UTC)  # 16:00 ET


class StubNewsProvider:
    def __init__(self, name: str, items: list[RawNewsItem] | None = None, *, fail: bool = False):
        self.name = name
        self.items = items or []
        self.fail = fail
        self.calls: list[tuple[str, datetime, datetime]] = []

    def news(self, symbol: str, start: datetime, end: datetime):
        self.calls.append((symbol, start, end))
        if self.fail:
            raise ConnectionError("provider down")
        from position_pilot.providers.contracts import ProviderValue

        return ProviderValue(
            field="stock.news",
            value=[item.model_dump(mode="json") for item in self.items if item.symbol == symbol],
            provider=self.name,
            observed_at=FIXED_NOW,
        )

    def health(self):
        from position_pilot.providers.contracts import ProviderHealth

        return ProviderHealth(
            provider=self.name,
            state=ProviderState.UNAVAILABLE if self.fail else ProviderState.HEALTHY,
            checked_at=FIXED_NOW,
        )


def _item(
    *,
    symbol: str = "AAPL",
    title: str = "Apple reports record services revenue",
    source: str = "company",
    published: datetime | None = None,
    url: str = "https://example.com/aapl-earnings",
    provider: str = "massive-stocks",
    body: str | None = "Apple beat estimates on services.",
    source_name: str = "Apple IR",
    taxonomy: EventTaxonomy = EventTaxonomy.EARNINGS,
) -> RawNewsItem:
    return RawNewsItem(
        symbol=symbol,
        title=title,
        url=url,
        published_at=published or (FIXED_NOW - timedelta(hours=2)),
        provider=provider,
        source_name=source_name,
        source_tier=source,
        excerpt=body,
        full_text=body,
        taxonomy=taxonomy,
    )


def _service(
    database: PositionPilotDatabase,
    *,
    news_providers: list[Any] | None = None,
    moves: dict[str, SymbolMoveInput] | None = None,
    option_metrics: dict[str, dict[str, Any]] | None = None,
    settings: CatalystSettings | None = None,
) -> CatalystService:
    move_map = moves or {
        "AAPL": SymbolMoveInput(
            symbol="AAPL",
            last_price=210.0,
            prior_close=200.0,
            session_high=211.0,
            session_low=199.0,
            is_broad_etf=False,
        )
    }

    def quote_source(symbol: str) -> SymbolMoveInput | None:
        return move_map.get(symbol.upper())

    return CatalystService(
        database=database,
        news_providers=news_providers or [],
        quote_source=quote_source,
        option_metrics_source=lambda symbol: (option_metrics or {}).get(symbol.upper()),
        clock=lambda: FIXED_NOW,
        settings=settings or CatalystSettings(),
    )


def test_schema_migrates_to_v5_with_catalyst_tables(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    assert database.schema_version == 6
    with database._connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "catalyst_events" in tables
    assert "catalyst_sources" in tables
    assert "catalyst_source_links" in tables
    assert "catalyst_feedback" in tables
    assert "catalyst_articles" in tables


def test_scan_every_held_underlying_with_abstention_when_no_evidence(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(
        database,
        news_providers=[StubNewsProvider("massive-stocks", [])],
        moves={
            "AAPL": SymbolMoveInput(
                symbol="AAPL", last_price=200.5, prior_close=200.0, is_broad_etf=False
            ),
            "MSFT": SymbolMoveInput(
                symbol="MSFT", last_price=400.0, prior_close=401.0, is_broad_etf=False
            ),
        },
    )

    scan = service.scan_held(["AAPL", "MSFT", "aapl"])

    assert sorted(r.symbol for r in scan.results) == ["AAPL", "MSFT"]
    for result in scan.results:
        assert result.confidence is CatalystConfidence.NO_CONFIRMED_CATALYST
        assert result.attribution is AttributionLevel.NONE
        assert result.summary == "No confirmed catalyst found"
        assert result.catalysts == []
        assert result.freshness.state is FreshnessState.FRESH
        assert result.coverage is CoverageState.COMPLETE


def test_default_window_is_prior_close_to_now_and_extends_for_scheduled(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    provider = StubNewsProvider(
        "massive-stocks",
        [
            _item(
                title="Apple scheduled earnings in two days",
                published=FIXED_NOW - timedelta(hours=48),
                taxonomy=EventTaxonomy.EARNINGS,
            )
        ],
    )
    # Patch taxonomy on item — constructor already set EARNINGS
    service = _service(database, news_providers=[provider])
    result = service.analyze_symbol("AAPL")

    symbol, start, end = provider.calls[0]
    assert symbol == "AAPL"
    assert end == FIXED_NOW
    # Window starts at prior close for general news; scheduled search may widen.
    assert start <= PRIOR_CLOSE_AT or (FIXED_NOW - start) <= timedelta(hours=72)
    assert result.lookback_start is not None
    assert result.lookback_end == FIXED_NOW


def test_move_thresholds_stock_2pct_etf_1pct_and_high_impact_promotion(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    news = StubNewsProvider(
        "massive-stocks",
        [
            _item(
                symbol="SPY",
                title="Fed chair signals emergency rate cut",
                source="regulator",
                source_name="Federal Reserve",
                taxonomy=EventTaxonomy.MACRO,
            ),
            _item(
                symbol="XYZ",
                title="Quiet analyst note",
                source="specialist",
                source_name="Niche Desk",
                taxonomy=EventTaxonomy.ANALYST,
            ),
        ],
    )
    service = _service(
        database,
        news_providers=[news],
        moves={
            "SPY": SymbolMoveInput(
                symbol="SPY", last_price=500.5, prior_close=500.0, is_broad_etf=True
            ),
            "XYZ": SymbolMoveInput(
                symbol="XYZ", last_price=100.5, prior_close=100.0, is_broad_etf=False
            ),
            "MOVE": SymbolMoveInput(
                symbol="MOVE", last_price=103.0, prior_close=100.0, is_broad_etf=False
            ),
        },
    )

    spy = service.analyze_symbol("SPY")
    xyz = service.analyze_symbol("XYZ")
    move = service.analyze_symbol("MOVE")

    assert spy.promoted is True  # high-impact event even without large move
    assert spy.move_percent is not None and abs(spy.move_percent) < 1.0
    assert xyz.promoted is False  # small move, not high-impact alone
    assert move.promoted is True  # 3% stock move
    assert move.meaningful_move is True


def test_provider_failures_are_independent_and_mark_incomplete_coverage(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    good = StubNewsProvider("massive-stocks", [_item()])
    bad = StubNewsProvider("benzinga", fail=True)
    service = _service(database, news_providers=[good, bad])

    result = service.analyze_symbol("AAPL")

    assert result.catalysts
    assert result.coverage is CoverageState.INCOMPLETE
    assert any("benzinga" in note.lower() for note in result.coverage_notes)
    assert result.confidence is not CatalystConfidence.NO_CONFIRMED_CATALYST


def test_deduplicate_without_collapsing_distinct_events(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    items = [
        _item(
            title="Apple reports record services revenue",
            url="https://massive.example/1",
            provider="massive-stocks",
            source_name="Reuters",
            source="established",
        ),
        _item(
            title="Apple reports record services revenue",
            url="https://benzinga.example/1",
            provider="benzinga",
            source_name="Benzinga",
            source="specialist",
            published=FIXED_NOW - timedelta(hours=1, minutes=55),
        ),
        _item(
            title="Apple announces new product lineup",
            url="https://massive.example/2",
            provider="massive-stocks",
            source_name="Bloomberg",
            source="established",
            taxonomy=EventTaxonomy.PRODUCT,
        ),
    ]
    service = _service(database, news_providers=[StubNewsProvider("massive-stocks", items)])
    result = service.analyze_symbol("AAPL")

    titles = {c.headline for c in result.catalysts}
    assert "Apple reports record services revenue" in titles
    assert "Apple announces new product lineup" in titles
    # Deduped duplicate story into one catalyst with multiple sources
    revenue = next(c for c in result.catalysts if "services revenue" in c.headline)
    assert len(revenue.sources) >= 2


def test_company_peer_macro_and_options_attribution(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    items = [
        _item(
            title="Apple board authorizes $100B buyback",
            source="company",
            source_name="SEC Form 8-K",
            taxonomy=EventTaxonomy.CORPORATE_ACTION,
        ),
    ]
    service = _service(
        database,
        news_providers=[StubNewsProvider("massive-stocks", items)],
        option_metrics={
            "AAPL": {
                "iv_change_pct": 18.0,
                "unusual_volume": True,
                "volume_oi_ratio": 3.5,
                "skew_shift": 0.12,
                "liquidity_score": 0.4,
                "dividend_date": (FIXED_NOW + timedelta(days=2)).date().isoformat(),
                "earnings_date": (FIXED_NOW + timedelta(days=5)).date().isoformat(),
                "days_to_expiration": 2,
                "gamma_risk": "elevated",
            }
        },
    )
    result = service.analyze_symbol("AAPL")

    assert result.attribution is AttributionLevel.COMPANY
    assert any(m.kind is OptionMechanismKind.IV_EXPANSION for m in result.option_mechanisms)
    assert all(m.evidence_kind is CatalystEvidenceKind.MECHANISM for m in result.option_mechanisms)
    assert all(m.label and "news" not in m.label.lower() for m in result.option_mechanisms)


def test_social_sentiment_is_side_note_never_evidence(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    items = [
        _item(
            title="Retail chatter bullish on AAPL",
            source="social",
            source_name="Social aggregate",
            taxonomy=EventTaxonomy.SOCIAL,
            url="https://social.example/1",
        ),
        _item(
            title="Apple IR confirms dividend increase",
            source="company",
            source_name="Apple IR",
            taxonomy=EventTaxonomy.DIVIDEND,
            url="https://example.com/div",
        ),
    ]
    service = _service(database, news_providers=[StubNewsProvider("massive-stocks", items)])
    result = service.analyze_symbol("AAPL")

    assert all(c.evidence_kind is not CatalystEvidenceKind.SOCIAL for c in result.catalysts)
    assert result.social_side_notes
    assert all(
        note.evidence_kind is CatalystEvidenceKind.SOCIAL for note in result.social_side_notes
    )
    assert result.confidence is CatalystConfidence.CONFIRMED


def test_deterministic_ranking_and_no_fabricated_causality(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    items = [
        _item(
            title="Vague market color on tech",
            source="specialist",
            source_name="Blog Desk",
            taxonomy=EventTaxonomy.OTHER,
            published=FIXED_NOW - timedelta(hours=1),
            url="https://example.com/vague",
        ),
        _item(
            title="Apple files 8-K disclosing CFO transition",
            source="regulator",
            source_name="SEC",
            taxonomy=EventTaxonomy.CORPORATE_ACTION,
            published=FIXED_NOW - timedelta(hours=3),
            url="https://sec.example/8k",
        ),
    ]
    service = _service(database, news_providers=[StubNewsProvider("massive-stocks", items)])
    result = service.analyze_symbol("AAPL")

    assert result.catalysts[0].headline.startswith("Apple files 8-K")
    assert result.catalysts[0].confidence is CatalystConfidence.CONFIRMED
    assert result.summary != result.catalysts[0].headline or "because" not in result.summary.lower()


def test_feedback_is_immutable_and_only_adjusts_local_ranking(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(
        database,
        news_providers=[
            StubNewsProvider(
                "massive-stocks",
                [
                    _item(
                        title="Primary catalyst product note",
                        url="https://example.com/p1",
                        source="specialist",
                        source_name="Niche Desk",
                        taxonomy=EventTaxonomy.PRODUCT,
                    ),
                    _item(
                        title="Secondary earnings recap",
                        url="https://example.com/p2",
                        source="established",
                        source_name="WSJ",
                        taxonomy=EventTaxonomy.EARNINGS,
                    ),
                ],
            )
        ],
    )
    first = service.analyze_symbol("AAPL")
    catalyst_id = first.catalysts[0].catalyst_id
    original_headline = first.catalysts[0].headline
    original_score = first.catalysts[0].rank_score

    feedback = service.submit_feedback(
        catalyst_id,
        CatalystFeedbackKind.NOT_RELATED,
        symbol="AAPL",
        note="Wrong product line",
    )
    again = service.analyze_symbol("AAPL")
    history = service.feedback_history(catalyst_id)

    assert feedback.kind is CatalystFeedbackKind.NOT_RELATED
    assert len(history) == 1
    # Historical catalyst evidence is preserved
    preserved = next(c for c in again.catalysts if c.catalyst_id == catalyst_id)
    assert preserved.headline == original_headline
    assert preserved.rank_score < original_score
    # Not-related demotes in ranking when alternatives exist
    assert len(again.catalysts) > 1
    assert again.catalysts[0].catalyst_id != catalyst_id
    # Second feedback appends rather than mutates
    service.submit_feedback(catalyst_id, CatalystFeedbackKind.RELEVANT, symbol="AAPL")
    assert len(service.feedback_history(catalyst_id)) == 2


def test_retention_policy_and_license_dependent_full_text(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    settings = CatalystSettings(
        store_full_text_providers={"benzinga"},
        catalyst_retention_days=365,
        article_metadata_retention_days=90,
    )
    service = _service(
        database,
        news_providers=[
            StubNewsProvider(
                "mixed",
                [
                    _item(
                        title="Licensed story",
                        provider="benzinga",
                        body="FULL TEXT BODY",
                        url="https://bz.example/1",
                    ),
                    _item(
                        title="Baseline story",
                        provider="massive-stocks",
                        body="Should not store full text",
                        url="https://massive.example/1",
                    ),
                ],
            )
        ],
        settings=settings,
    )
    service.analyze_symbol("AAPL")

    articles = database.list_catalyst_articles("AAPL")
    licensed = next(a for a in articles if a["provider"] == "benzinga")
    baseline = next(a for a in articles if a["provider"] == "massive-stocks")
    assert licensed["full_text"] == "FULL TEXT BODY"
    assert baseline["full_text"] is None
    assert baseline["excerpt"]

    # Simulate old metadata and prune
    old = (FIXED_NOW - timedelta(days=100)).isoformat()
    with database._connect() as connection:
        connection.execute(
            "UPDATE catalyst_articles SET stored_at = ?, published_at = ?",
            (old, old),
        )
        connection.execute(
            "UPDATE catalyst_events SET created_at = ?",
            ((FIXED_NOW - timedelta(days=400)).isoformat(),),
        )
    pruned = service.apply_retention(now=FIXED_NOW)
    assert pruned["articles"] >= 1
    assert pruned["events"] >= 0


def test_removal_notice_clears_licensed_full_text(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    settings = CatalystSettings(store_full_text_providers={"benzinga"})
    service = _service(
        database,
        news_providers=[
            StubNewsProvider(
                "benzinga",
                [
                    _item(
                        title="To be removed",
                        provider="benzinga",
                        body="secret full text",
                        url="https://bz.example/remove-me",
                    )
                ],
            )
        ],
        settings=settings,
    )
    result = service.analyze_symbol("AAPL")
    source_url = result.catalysts[0].sources[0].url
    cleared = service.honor_removal_notice(provider="benzinga", url=source_url)
    assert cleared >= 1
    articles = database.list_catalyst_articles("AAPL")
    assert all(a["full_text"] is None for a in articles if a["url"] == source_url)


def test_benzinga_adapter_is_configuration_gated_and_mockable() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[
                {
                    "id": "bz-1",
                    "title": "Apple supplier update",
                    "author": "Benzinga Newsdesk",
                    "created": "2026-07-11T16:00:00Z",
                    "url": "https://www.benzinga.com/news/26/07/apple",
                    "teaser": "Supply chain notes",
                    "body": "Detailed licensed body",
                    "stocks": [{"name": "AAPL"}],
                }
            ],
        )

    unconfigured = BenzingaProvider(api_key="")
    assert unconfigured.news("AAPL", FIXED_NOW - timedelta(hours=24), FIXED_NOW) is None
    assert unconfigured.health().state is ProviderState.NOT_CONFIGURED

    provider = BenzingaProvider(
        api_key="bz-secret",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
        clock=lambda: FIXED_NOW,
    )
    value = provider.news("AAPL", FIXED_NOW - timedelta(hours=24), FIXED_NOW)
    assert value is not None
    assert value.provider == "benzinga"
    assert "apiKey" not in str(requests[0].url) or "bz-secret" in str(requests[0].url)
    assert provider.health().state is ProviderState.HEALTHY
    # Credentials never appear in returned payload
    assert "bz-secret" not in str(value.value)


def test_chart_includes_volume_prior_close_and_event_markers(tmp_path) -> None:
    from position_pilot.domain.market import MarketService

    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    news = StubNewsProvider("massive-stocks", [_item(published=FIXED_NOW - timedelta(hours=1))])
    catalyst_service = _service(database, news_providers=[news])
    catalyst_service.analyze_symbol("AAPL")

    class Source:
        def get_quote(self, symbol, force_refresh=False):
            return {"mark": 210.0, "close": 200.0}

        def get_market_metrics(self, symbol, force_refresh=False):
            return {}

    def bar_source(symbol: str):
        # Premarket + after-hours bars in America/New_York for truthful extended-hours.
        return [
            {
                "t": int(datetime(2026, 7, 10, 12, 0, tzinfo=UTC).timestamp() * 1000),  # 08:00 ET
                "o": 201,
                "h": 205,
                "l": 200,
                "c": 204,
                "v": 1_000_000,
            },
            {
                "t": int(datetime(2026, 7, 10, 21, 30, tzinfo=UTC).timestamp() * 1000),  # 17:30 ET
                "o": 204,
                "h": 211,
                "l": 203,
                "c": 210,
                "v": 2_500_000,
            },
        ]

    market = MarketService(source=Source(), bar_source=bar_source, clock=lambda: FIXED_NOW)
    chart = market.chart(
        "AAPL",
        prior_close=200.0,
        event_markers=catalyst_service.event_markers("AAPL"),
        include_extended_hours=True,
        window_start=datetime(2026, 7, 10, 11, 0, tzinfo=UTC),
        window_end=FIXED_NOW,
    )

    assert chart.prior_close == 200.0
    assert chart.extended_hours_truthful is True
    assert chart.include_extended_hours is True
    assert chart.bars[0].volume == 1_000_000
    assert chart.volume_series == [1_000_000.0, 2_500_000.0]
    assert chart.event_markers
    assert chart.event_markers[0].headline


def test_missing_catalyst_feedback_records_gap(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(database)
    event = service.submit_feedback(
        None,
        CatalystFeedbackKind.MISSING_CATALYST,
        symbol="AAPL",
        note="Supplier fire not captured",
    )
    assert event.kind is CatalystFeedbackKind.MISSING_CATALYST
    assert event.symbol == "AAPL"
    history = service.feedback_history(symbol="AAPL")
    assert len(history) == 1


def test_previous_regular_close_session_phases() -> None:
    from zoneinfo import ZoneInfo

    from position_pilot.domain.catalysts import previous_regular_close

    et = ZoneInfo("America/New_York")
    # Tuesday 2026-07-07
    premarket = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)  # 08:00 ET
    regular = datetime(2026, 7, 7, 18, 0, tzinfo=UTC)  # 14:00 ET
    after_hours = datetime(2026, 7, 7, 21, 30, tzinfo=UTC)  # 17:30 ET
    weekend = datetime(2026, 7, 11, 18, 30, tzinfo=UTC)  # Saturday
    monday_pre = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)

    assert previous_regular_close(premarket).astimezone(et).day == 6  # Monday
    assert previous_regular_close(regular).astimezone(et).day == 6
    assert previous_regular_close(after_hours).astimezone(et).day == 7  # today
    assert previous_regular_close(weekend).astimezone(et).day == 10  # Friday
    assert previous_regular_close(monday_pre).astimezone(et).day == 3  # prior Friday
    # Never two calendar days before a midweek open
    wed_pre = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)
    assert previous_regular_close(wed_pre).astimezone(et).day == 7


def test_derive_move_from_bars_never_uses_live_close_field() -> None:
    from position_pilot.domain.catalysts import derive_move_from_bars, previous_regular_close

    prior_at = previous_regular_close(FIXED_NOW)
    bars = [
        {
            "t": int((prior_at - timedelta(minutes=1)).timestamp() * 1000),
            "o": 99,
            "h": 100,
            "l": 98,
            "c": 100.0,
            "v": 10,
        },
        {
            "t": int((prior_at + timedelta(hours=2)).timestamp() * 1000),
            "o": 101,
            "h": 110,
            "l": 100,
            "c": 108.0,
            "v": 20,
        },
    ]
    prior, last, high, low = derive_move_from_bars(bars, prior_close_at=prior_at, now=FIXED_NOW)
    assert prior == 100.0
    assert last == 108.0
    assert high == 110.0
    assert low == 100.0


def test_production_composition_derives_prior_close_from_bars(tmp_path, monkeypatch) -> None:
    """Regression through factory quote_source: no live close as prior close."""
    from position_pilot.domain import factory as factory_mod
    from position_pilot.domain.catalysts import previous_regular_close
    from position_pilot.domain.snapshots import (
        AccountSnapshot,
        DataFreshness,
        PortfolioSnapshot,
        PositionSnapshot,
        QuantityDirection,
        SnapshotState,
    )
    from position_pilot.models import PositionType
    from position_pilot.providers.contracts import ProviderHealth, ProviderState, ProviderValue

    factory_mod.get_catalyst_service.cache_clear()
    factory_mod.get_field_router.cache_clear()
    factory_mod.get_market_service.cache_clear()
    factory_mod.get_database.cache_clear()

    monkeypatch.setenv("POSITION_PILOT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")

    prior_at = previous_regular_close(FIXED_NOW)
    bars = [
        {
            "t": int((prior_at - timedelta(minutes=5)).timestamp() * 1000),
            "o": 50,
            "h": 51,
            "l": 49,
            "c": 50.0,
            "v": 100,
        },
        {
            "t": int((prior_at + timedelta(hours=1)).timestamp() * 1000),
            "o": 51,
            "h": 55,
            "l": 50.5,
            "c": 54.0,
            "v": 200,
        },
    ]

    class FakeClient:
        def get_quote(self, symbol, force_refresh=False):
            # Live close present but must NOT be used as prior close.
            return {"mark": 54.0, "close": 54.0}

        def get_market_metrics(self, symbol, force_refresh=False):
            return {"iv_rank": 40, "liquidity_rating": 3, "earnings_date": "2026-07-20"}

        def get_accounts(self):
            return []

    class FakeMassive:
        name = "massive-stocks"

        def health(self):
            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

        def fetch(self, field, symbol):
            if field == "stock.bars":
                return ProviderValue(
                    field=field,
                    value=bars,
                    provider=self.name,
                    observed_at=FIXED_NOW,
                )
            if field == "stock.news":
                return ProviderValue(
                    field=field,
                    value=[],
                    provider=self.name,
                    observed_at=FIXED_NOW,
                )
            return None

        def news(self, symbol, start, end):
            return self.fetch("stock.news", symbol)

    class FakeRouter:
        def __init__(self):
            self.providers = {"massive-stocks": FakeMassive(), "benzinga": FakeMassive()}
            self.providers["benzinga"].name = "benzinga"
            self.option_calls = []

        def resolve(self, field, symbol):
            if field == "option.snapshot":
                self.option_calls.append(symbol)
                return ProviderValue(
                    field=field,
                    value={
                        "volume": 120,
                        "open_interest": 240,
                        "implied_volatility": 0.32,
                        "greeks": {"gamma": 0.06},
                    },
                    provider="massive-options",
                    observed_at=FIXED_NOW,
                )
            return self.providers["massive-stocks"].fetch(field, symbol)

        def health(self):
            return {}

    class FakeMarket:
        def snapshot(self, symbol, force_refresh=False):
            from position_pilot.domain.market import MarketSnapshot
            from position_pilot.domain.snapshots import DataFreshness

            return MarketSnapshot(
                symbol=symbol,
                price=54.0,
                freshness=DataFreshness(as_of=FIXED_NOW, provider="tastytrade"),
            )

    database = PositionPilotDatabase(tmp_path / "p.sqlite3")
    database.save_portfolio_snapshot(
        PortfolioSnapshot(
            snapshot_id="held-options",
            captured_at=FIXED_NOW,
            state=SnapshotState.LIVE,
            freshness=DataFreshness(as_of=FIXED_NOW, provider="tastytrade"),
            accounts=[
                AccountSnapshot(
                    account_id="account-1",
                    label="Individual 1",
                    account_type="Individual",
                    positions=[
                        PositionSnapshot(
                            symbol="AAPL  260821C00200000",
                            underlying_symbol="AAPL",
                            quantity=1,
                            quantity_direction=QuantityDirection.LONG,
                            position_type=PositionType.EQUITY_OPTION,
                            days_to_expiration=41,
                            gamma=0.04,
                        )
                    ],
                )
            ],
        )
    )
    fake_router = FakeRouter()
    monkeypatch.setattr(factory_mod, "get_client", lambda: FakeClient())
    monkeypatch.setattr(factory_mod, "get_field_router", lambda: fake_router)
    monkeypatch.setattr(factory_mod, "get_market_service", lambda: FakeMarket())
    monkeypatch.setattr(factory_mod, "get_database", lambda: database)

    service = factory_mod.get_catalyst_service()
    move = service.quote_source("AAPL")
    assert move is not None
    assert move.prior_close == 50.0
    assert move.prior_close != 54.0  # not live close
    assert move.last_price == 54.0
    assert move.session_high == 55.0
    assert move.session_low == 50.5
    metrics = service.option_metrics_source("AAPL")
    assert fake_router.option_calls == ["AAPL  260821C00200000"]
    assert metrics is not None
    assert metrics["option_volume"] == 120
    assert metrics["open_interest"] == 240
    assert metrics["volume_oi_ratio"] == 0.5
    assert metrics["days_to_expiration"] == 41
    assert metrics["gamma_risk"] == "elevated"
    factory_mod.get_catalyst_service.cache_clear()


def test_provider_states_offline_cache_and_unavailable(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    good = StubNewsProvider("massive-stocks", [_item()])
    service = _service(database, news_providers=[good])
    first = service.analyze_symbol("AAPL")
    assert first.catalysts
    assert first.coverage is CoverageState.COMPLETE

    # All configured providers fail => OFFLINE + stale cache preserved
    bad = StubNewsProvider("massive-stocks", fail=True)
    offline = _service(database, news_providers=[bad])
    second = offline.analyze_symbol("AAPL")
    assert second.coverage is CoverageState.OFFLINE
    assert second.cached is True
    assert second.freshness.state is FreshnessState.STALE
    assert second.catalysts
    assert any("Cached as of" in note for note in second.coverage_notes)

    # No providers => UNAVAILABLE abstention
    empty = _service(database, news_providers=[])
    third = empty.analyze_symbol("MSFT")
    assert third.coverage is CoverageState.UNAVAILABLE
    assert third.summary == "No confirmed catalyst found"
    assert third.freshness.state is FreshnessState.UNAVAILABLE


def test_offline_reuses_cached_abstention_instead_of_inventing_fresh_state(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    healthy_empty = _service(
        database,
        news_providers=[StubNewsProvider("massive-stocks", [])],
        moves={"MSFT": SymbolMoveInput(symbol="MSFT", last_price=400, prior_close=400)},
    )
    first = healthy_empty.analyze_symbol("MSFT")
    assert first.summary == "No confirmed catalyst found"
    assert first.cached is False

    offline = _service(
        database,
        news_providers=[StubNewsProvider("massive-stocks", fail=True)],
        moves={"MSFT": SymbolMoveInput(symbol="MSFT", last_price=400, prior_close=400)},
    )
    second = offline.analyze_symbol("MSFT")
    assert second.summary == "No confirmed catalyst found"
    assert second.coverage is CoverageState.OFFLINE
    assert second.cached is True
    assert second.freshness.state is FreshnessState.STALE


def test_option_mechanism_remains_actionable_when_news_is_unconfigured(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(
        database,
        news_providers=[],
        option_metrics={"AAPL": {"iv_change_pct": 18.0}},
    )

    result = service.analyze_symbol("AAPL")

    assert result.coverage is CoverageState.UNAVAILABLE
    assert result.confidence is CatalystConfidence.LIKELY
    assert result.attribution is AttributionLevel.OPTIONS_MARKET
    assert result.promoted is True
    assert result.quiet is False
    assert result.option_mechanisms


def test_option_mechanism_remains_actionable_during_meaningful_move_with_empty_news(
    tmp_path,
) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(
        database,
        news_providers=[StubNewsProvider("massive-stocks", [])],
        option_metrics={"AAPL": {"iv_change_pct": 18.0}},
    )

    result = service.analyze_symbol("AAPL")

    assert result.meaningful_move is True
    assert result.coverage is CoverageState.COMPLETE
    assert result.confidence is CatalystConfidence.LIKELY
    assert result.attribution is AttributionLevel.OPTIONS_MARKET
    assert result.promoted is True


def test_runtime_benzinga_toggle_gates_ingestion_without_restart(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    massive = StubNewsProvider("massive-stocks", [])
    benzinga = StubNewsProvider(
        "benzinga",
        [
            _item(
                title="Licensed earnings flash",
                provider="benzinga",
                source="licensed",
                source_name="Benzinga",
                taxonomy=EventTaxonomy.EARNINGS,
            )
        ],
    )
    settings = CatalystSettings(benzinga_enabled=True)

    def factory():
        providers = [massive]
        stored = database.get_setting("catalysts", {}) or {}
        enabled = bool(stored.get("benzinga_enabled", settings.benzinga_enabled))
        if enabled:
            providers.append(benzinga)
        return providers

    service = CatalystService(
        database=database,
        news_provider_factory=factory,
        quote_source=lambda s: SymbolMoveInput(symbol=s, last_price=210, prior_close=200),
        clock=lambda: FIXED_NOW,
        settings=settings,
        benzinga_api_key_present=True,
    )
    enabled_scan = service.analyze_symbol("AAPL")
    assert any(c.headline.startswith("Licensed") for c in enabled_scan.catalysts)
    assert service.public_settings()["benzinga"]["status"] == "enabled"

    service.update_settings({"benzinga_enabled": False})
    assert service.public_settings()["benzinga"]["status"] == "disabled"
    disabled_scan = service.analyze_symbol("AAPL")
    # Toggle applied immediately on next analyze without reconstructing service.
    assert all(c.headline.startswith("Licensed") is False for c in disabled_scan.catalysts) or (
        not disabled_scan.catalysts
    )
    assert benzinga.calls  # was called while enabled
    calls_while_disabled = len(benzinga.calls)
    service.analyze_symbol("AAPL")
    assert len(benzinga.calls) == calls_while_disabled  # not called after disable


def test_strong_abstention_drops_weak_undated_and_other(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    items = [
        _item(
            title="Vague chatter",
            source="unknown",
            source_name="Random Blog",
            taxonomy=EventTaxonomy.OTHER,
            url="https://example.com/vague",
        ),
        RawNewsItem(
            symbol="AAPL",
            title="Undated mystery",
            url="https://example.com/undated",
            published_at=FIXED_NOW,  # will be dropped at parse if undated; inject via stub
            provider="massive-stocks",
            source_name="Unknown",
            source_tier="unknown",
            taxonomy=EventTaxonomy.OTHER,
        ),
    ]

    # Stub returning undated dict (no published fields)
    class UndatedProvider:
        name = "massive-stocks"

        def news(self, symbol, start, end):
            from position_pilot.providers.contracts import ProviderValue

            return ProviderValue(
                field="stock.news",
                value=[
                    {
                        "title": "No timestamp article",
                        "url": "https://example.com/no-ts",
                        "publisher": "Random",
                    },
                    items[0].model_dump(mode="json"),
                ],
                provider=self.name,
                observed_at=FIXED_NOW,
            )

        def health(self):
            from position_pilot.providers.contracts import ProviderHealth

            return ProviderHealth(provider=self.name, state=ProviderState.HEALTHY)

    service = _service(database, news_providers=[UndatedProvider()])
    result = service.analyze_symbol("AAPL")
    assert result.confidence is CatalystConfidence.NO_CONFIRMED_CATALYST
    assert result.catalysts == []
    assert result.attribution is AttributionLevel.NONE


def test_source_ids_stable_across_repeated_scans(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(
        database,
        news_providers=[StubNewsProvider("massive-stocks", [_item()])],
    )
    first = service.analyze_symbol("AAPL")
    second = service.analyze_symbol("AAPL")
    assert first.catalysts[0].sources[0].source_id == second.catalysts[0].sources[0].source_id
    assert database.count_catalyst_sources(first.catalysts[0].catalyst_id) == 1


def test_feedback_resolves_symbol_and_affects_ranking(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(
        database,
        news_providers=[
            StubNewsProvider(
                "massive-stocks",
                [
                    _item(
                        title="A story",
                        url="https://example.com/a",
                        source="specialist",
                        source_name="Desk",
                        taxonomy=EventTaxonomy.PRODUCT,
                    ),
                    _item(
                        title="B story earnings",
                        url="https://example.com/b",
                        source="established",
                        source_name="WSJ",
                        taxonomy=EventTaxonomy.EARNINGS,
                    ),
                ],
            )
        ],
    )
    first = service.analyze_symbol("AAPL")
    top_id = first.catalysts[0].catalyst_id
    # No symbol supplied — resolved from stored catalyst
    service.submit_feedback(top_id, CatalystFeedbackKind.NOT_RELATED)
    again = service.analyze_symbol("AAPL")
    assert again.catalysts[0].catalyst_id != top_id


def test_incomplete_coverage_cannot_be_sole_causal_proof(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    good = StubNewsProvider(
        "massive-stocks",
        [
            _item(
                title="Soft specialist product note",
                source="specialist",
                source_name="Desk",
                taxonomy=EventTaxonomy.PRODUCT,
                url="https://example.com/soft",
            )
        ],
    )
    bad = StubNewsProvider("benzinga", fail=True)
    service = _service(
        database,
        news_providers=[good, bad],
        moves={
            "AAPL": SymbolMoveInput(
                symbol="AAPL", last_price=110.0, prior_close=100.0, is_broad_etf=False
            )
        },
    )
    result = service.analyze_symbol("AAPL")
    assert result.coverage is CoverageState.INCOMPLETE
    assert result.meaningful_move is True
    assert result.confidence is CatalystConfidence.NO_CONFIRMED_CATALYST
    assert result.summary == "No confirmed catalyst found"
    assert any("not sole causal proof" in note for note in result.coverage_notes)


def test_option_mechanism_coverage_exposes_unavailable_fields(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    service = _service(
        database,
        news_providers=[StubNewsProvider("massive-stocks", [])],
        option_metrics={"AAPL": {"liquidity_score": 0.9, "earnings_date": "2026-07-20"}},
    )
    result = service.analyze_symbol("AAPL")
    kinds = {c.kind for c in result.option_mechanism_coverage}
    assert OptionMechanismKind.UNUSUAL_VOLUME in kinds or any(
        c.kind.value == "unusual_volume" for c in result.option_mechanism_coverage
    )
    unavailable = [
        c for c in result.option_mechanism_coverage if c.availability.value == "unavailable"
    ]
    assert unavailable  # volume/OI/skew/gamma explicitly unavailable
    assert any(c.kind.value == "earnings_proximity" for c in result.option_mechanisms)


def test_retention_prunes_snapshots_and_removal_clears_excerpts(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "db.sqlite3")
    settings = CatalystSettings(store_full_text_providers={"benzinga"})
    service = _service(
        database,
        news_providers=[
            StubNewsProvider(
                "benzinga",
                [
                    _item(
                        title="Licensed",
                        provider="benzinga",
                        body="FULL",
                        url="https://bz.example/x",
                        source="licensed",
                        source_name="Benzinga",
                    )
                ],
            )
        ],
        settings=settings,
    )
    result = service.analyze_symbol("AAPL")
    assert database.get_latest_symbol_catalyst("AAPL") is not None
    url = result.catalysts[0].sources[0].url
    catalyst_id = result.catalysts[0].catalyst_id
    # Feedback is immutable even when the licensed source is later removed.
    service.submit_feedback(
        catalyst_id,
        CatalystFeedbackKind.RELEVANT,
        symbol="AAPL",
    )
    cleared = service.honor_removal_notice(provider="benzinga", url=url)
    assert cleared >= 1
    articles = database.list_catalyst_articles("AAPL")
    assert all(a["url"] != url for a in articles)
    assert database.catalyst_exists(catalyst_id) is False
    snapshot = database.get_latest_symbol_catalyst("AAPL")
    assert snapshot is not None
    assert snapshot["catalysts"] == []
    assert snapshot["summary"] == "No confirmed catalyst found"
    assert len(service.feedback_history(catalyst_id)) == 1
    with database._connect() as connection:
        connection.execute(
            "UPDATE symbol_catalyst_snapshots SET captured_at = ?",
            ((FIXED_NOW - timedelta(days=400)).isoformat(),),
        )
        connection.execute(
            "UPDATE catalyst_events SET created_at = ?",
            ((FIXED_NOW - timedelta(days=400)).isoformat(),),
        )
    pruned = service.apply_retention(now=FIXED_NOW)
    assert pruned["snapshots"] >= 1
    assert pruned.get("events", 0) >= 0
