"""Application-service composition root."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..client import get_client
from ..models import Account, Position
from ..models.transaction import Order, Transaction
from ..persistence.sqlite import PositionPilotDatabase
from ..providers.benzinga import BenzingaProvider
from ..providers.codex import CodexCLIProvider, ExplicitApiKeyFallbackProvider
from ..providers.massive import MassiveProvider
from ..providers.router import FieldRouter
from ..providers.tastytrade import TastytradeProvider
from .alerts import AlertService
from .catalysts import (
    CatalystService,
    CatalystSettings,
    SymbolMoveInput,
    derive_move_from_bars,
    is_broad_etf,
    previous_regular_close,
)
from .market import MarketService, ProviderRoutedMarketSource
from .monitoring import MonitoringService
from .notifications import NotificationService
from .orders import OrderService
from .plans import PlansService
from .portfolio import PortfolioService
from .recommendations import RecommendationService
from .risk import RiskService
from .rolls import RollService
from .watchlist import WatchlistService


class TastytradePortfolioSource:
    """Checked adapter that turns swallowed transport failures into domain failures."""

    def __init__(self) -> None:
        self.client = get_client()

    def get_accounts(self) -> list[Account]:
        accounts = self.client.get_accounts()
        if not accounts:
            raise ConnectionError("Tastytrade account request failed")
        return accounts

    def get_account_balances(self, account_number: str) -> dict:
        balances = self.client.get_account_balances(account_number)
        if balances is None:
            raise ConnectionError("Tastytrade balance request failed")
        return balances

    def get_positions(self, account_number: str) -> list[Position]:
        succeeded, positions = self.client.get_positions_checked(account_number)
        if not succeeded:
            raise ConnectionError("Tastytrade position request failed")
        return positions

    def enrich_positions_greeks_batch(self, positions: list[Position]) -> list[Position]:
        return self.client.enrich_positions_greeks_batch(positions)

    def get_orders(self, account_number: str, *, limit: int = 100) -> list[Order]:
        return self.client.get_orders(account_number, limit=limit)

    def get_transactions(
        self,
        account_number: str,
        *,
        start_date=None,
        end_date=None,
    ) -> list[Transaction]:
        return self.client.get_transactions(
            account_number,
            start_date=start_date,
            end_date=end_date,
        )


def data_directory() -> Path:
    """Return the configurable local state directory."""

    return Path(
        os.getenv(
            "POSITION_PILOT_DATA_DIR",
            Path.home() / ".local" / "share" / "position-pilot",
        )
    )


@lru_cache(maxsize=1)
def get_database() -> PositionPilotDatabase:
    database = PositionPilotDatabase(
        data_directory() / "position-pilot.sqlite3",
        legacy_config_path=Path.home() / ".config" / "position-pilot" / "config.json",
        legacy_cache_directory=Path.home() / ".cache" / "position-pilot",
    )
    database.ensure_daily_backup()
    return database


@lru_cache(maxsize=1)
def get_portfolio_service() -> PortfolioService:
    return PortfolioService(
        database=get_database(),
        source=TastytradePortfolioSource(),
        field_router=get_field_router(),
    )


@lru_cache(maxsize=1)
def get_market_service() -> MarketService:
    router = get_field_router()

    def bar_source(symbol: str) -> list[dict] | None:
        value = router.resolve("stock.bars", symbol)
        if value is None:
            return None
        if isinstance(value.value, list):
            return value.value
        return None

    return MarketService(
        source=ProviderRoutedMarketSource(
            primary=get_client(),
            router=router,
        ),
        bar_source=bar_source,
    )


@lru_cache(maxsize=1)
def get_roll_service() -> RollService:
    service = RollService(
        get_database(),
        legacy_history_path=Path.home() / ".cache" / "position-pilot" / "roll_history.json",
    )
    service.migrate_legacy_cache()
    return service


@lru_cache(maxsize=1)
def get_risk_service() -> RiskService:
    return RiskService()


@lru_cache(maxsize=1)
def get_order_service() -> OrderService:
    return OrderService(database=get_database(), source=TastytradePortfolioSource())


@lru_cache(maxsize=1)
def get_plans_service() -> PlansService:
    return PlansService(get_database())


@lru_cache(maxsize=1)
def get_watchlist_service() -> WatchlistService:
    return WatchlistService(get_database(), get_market_service())


@lru_cache(maxsize=1)
def get_field_router() -> FieldRouter:
    tastytrade = TastytradeProvider(get_client())
    massive_stocks = MassiveProvider(
        api_key=os.getenv("MASSIVE_API_KEY", ""),
        capability="stocks",
    )
    massive_options = MassiveProvider(
        api_key=os.getenv("MASSIVE_API_KEY", ""),
        capability="options",
    )
    benzinga = BenzingaProvider(api_key=os.getenv("BENZINGA_API_KEY", ""))
    providers = {
        provider.name: provider
        for provider in (tastytrade, massive_stocks, massive_options, benzinga)
    }
    news_route = ["massive-stocks"]
    if os.getenv("BENZINGA_API_KEY"):
        news_route.append("benzinga")
    return FieldRouter(
        providers=providers,
        routes={
            "stock.quote": ["tastytrade", "massive-stocks"],
            "stock.bars": ["massive-stocks"],
            "stock.news": news_route,
            "option.mark": ["tastytrade", "massive-options"],
            "option.greeks": ["tastytrade", "massive-options"],
            "option.snapshot": ["massive-options"],
            "option.bars": ["massive-options"],
            "option.quotes": ["massive-options"],
        },
    )


@lru_cache(maxsize=1)
def get_catalyst_service() -> CatalystService:
    """Compose catalyst intelligence with injectable provider seams."""

    router = get_field_router()
    market = get_market_service()
    client = get_client()

    def news_provider_factory():
        providers = []
        massive = router.providers.get("massive-stocks")
        if massive is not None:
            providers.append(massive)
        stored = get_database().get_setting("catalysts", {}) or {}
        benzinga_enabled = bool(stored.get("benzinga_enabled", True))
        benzinga = router.providers.get("benzinga")
        if benzinga is not None and benzinga_enabled and os.getenv("BENZINGA_API_KEY"):
            providers.append(benzinga)
        return providers

    def quote_source(symbol: str) -> SymbolMoveInput | None:
        snapshot = market.snapshot(symbol)
        last_price = snapshot.price if snapshot else None
        try:
            raw_quote = client.get_quote(symbol) or {}
        except Exception:
            raw_quote = {}
        # Never treat live "close" as prior close — only explicit prior/previous fields.
        prior_close = None
        for key in ("prior_close", "previous_close", "prev_close", "prevClose", "previousClose"):
            value = raw_quote.get(key)
            if isinstance(value, (int, float)):
                prior_close = float(value)
                break
        session_high = raw_quote.get("session_high") or raw_quote.get("day_high")
        session_low = raw_quote.get("session_low") or raw_quote.get("day_low")
        if last_price is None and isinstance(raw_quote.get("mark"), (int, float)):
            last_price = float(raw_quote["mark"])

        # Fill gaps from extended-hours bars when quote lacks official prior close / range.
        need_bars = prior_close is None or session_high is None or session_low is None
        if need_bars:
            try:
                bar_value = router.resolve("stock.bars", symbol)
                raw_bars = (
                    bar_value.value if bar_value and isinstance(bar_value.value, list) else []
                )
            except Exception:
                raw_bars = []
            now = datetime.now(UTC)
            prior_at = previous_regular_close(now)
            bar_prior, bar_last, bar_high, bar_low = derive_move_from_bars(
                raw_bars,
                prior_close_at=prior_at,
                now=now,
            )
            if prior_close is None:
                prior_close = bar_prior
            if last_price is None:
                last_price = bar_last
            if session_high is None:
                session_high = bar_high
            if session_low is None:
                session_low = bar_low

        if last_price is None and prior_close is None:
            return None
        return SymbolMoveInput(
            symbol=symbol.upper(),
            last_price=last_price,
            prior_close=prior_close,
            session_high=float(session_high) if isinstance(session_high, (int, float)) else None,
            session_low=float(session_low) if isinstance(session_low, (int, float)) else None,
            is_broad_etf=is_broad_etf(symbol),
        )

    def option_metrics_source(symbol: str) -> dict | None:
        metrics: dict = {}
        try:
            market_metrics = client.get_market_metrics(symbol) or {}
        except Exception:
            market_metrics = {}
        for key in (
            "implied_volatility",
            "liquidity_rating",
            "iv_rank",
            "iv_percentile",
            "iv_change_pct",
            "iv_index",
            "option_volume",
            "open_interest",
            "volume_oi_ratio",
            "unusual_volume",
            "unusual_open_interest",
            "skew_shift",
            "skew",
            "gamma_risk",
            "days_to_expiration",
        ):
            if key in market_metrics and market_metrics[key] is not None:
                metrics[key] = market_metrics[key]
        if "skew" in metrics and "skew_shift" not in metrics:
            metrics["skew_shift"] = metrics["skew"]
        if market_metrics.get("earnings_date"):
            metrics["earnings_date"] = market_metrics["earnings_date"]
        if market_metrics.get("dividend_date") or market_metrics.get("ex_dividend_date"):
            metrics["dividend_date"] = market_metrics.get("dividend_date") or market_metrics.get(
                "ex_dividend_date"
            )
        liquidity = market_metrics.get("liquidity_rating")
        if isinstance(liquidity, (int, float)):
            metrics["liquidity_score"] = (
                float(liquidity) / 5.0 if liquidity > 1 else float(liquidity)
            )

        # Massive option snapshots require contract symbols, never an underlying root.
        # Aggregate only contracts the latest portfolio snapshot proves are held.
        portfolio = get_database().latest_portfolio_snapshot()
        held_contracts = []
        if portfolio is not None:
            held_contracts = [
                position
                for account in portfolio.accounts
                for position in account.positions
                if position.underlying_symbol.upper() == symbol.upper()
                and "option" in position.position_type.value.lower()
            ]
        snapshots: list[dict] = []
        for position in held_contracts:
            try:
                option_value = router.resolve("option.snapshot", position.symbol)
            except Exception:
                continue
            if option_value and isinstance(option_value.value, dict):
                snapshots.append(option_value.value)

        def numeric_values(key: str) -> list[float]:
            return [
                float(snapshot[key])
                for snapshot in snapshots
                if isinstance(snapshot.get(key), (int, float))
            ]

        volumes = numeric_values("volume")
        interests = numeric_values("open_interest")
        ivs = numeric_values("implied_volatility")
        gammas = [
            float(greeks["gamma"])
            for snapshot in snapshots
            if isinstance((greeks := snapshot.get("greeks")), dict)
            and isinstance(greeks.get("gamma"), (int, float))
        ]
        if metrics.get("option_volume") is None and volumes:
            metrics["option_volume"] = sum(volumes)
        if metrics.get("open_interest") is None and interests:
            metrics["open_interest"] = sum(interests)
        if metrics.get("implied_volatility") is None and ivs:
            metrics["implied_volatility"] = sum(ivs) / len(ivs)
        if metrics.get("days_to_expiration") is None:
            dtes = [
                position.days_to_expiration
                for position in held_contracts
                if position.days_to_expiration is not None
            ]
            if dtes:
                metrics["days_to_expiration"] = min(dtes)
        if metrics.get("gamma_risk") is None:
            held_gammas = [
                abs(float(position.gamma))
                for position in held_contracts
                if position.gamma is not None
            ]
            all_gammas = [*gammas, *held_gammas]
            if all_gammas:
                metrics["gamma_risk"] = (
                    "elevated" if max(abs(gamma) for gamma in all_gammas) >= 0.05 else "normal"
                )
        if (
            metrics.get("volume_oi_ratio") is None
            and isinstance(metrics.get("option_volume"), (int, float))
            and isinstance(metrics.get("open_interest"), (int, float))
            and metrics["open_interest"]
        ):
            metrics["volume_oi_ratio"] = float(metrics["option_volume"]) / float(
                metrics["open_interest"]
            )
        return metrics or None

    stored = get_database().get_setting("catalysts", {}) or {}
    settings = CatalystSettings(
        stock_move_threshold_pct=float(
            stored.get("stock_move_threshold_pct", CatalystSettings().stock_move_threshold_pct)
        ),
        etf_move_threshold_pct=float(
            stored.get("etf_move_threshold_pct", CatalystSettings().etf_move_threshold_pct)
        ),
        news_cadence_seconds=int(
            stored.get("news_cadence_seconds", CatalystSettings().news_cadence_seconds)
        ),
        benzinga_enabled=bool(stored.get("benzinga_enabled", True)),
    )
    return CatalystService(
        database=get_database(),
        news_provider_factory=news_provider_factory,
        quote_source=quote_source,
        option_metrics_source=option_metrics_source,
        settings=settings,
        benzinga_api_key_present=bool(os.getenv("BENZINGA_API_KEY")),
    )


@lru_cache(maxsize=1)
def get_codex_provider() -> CodexCLIProvider:
    return CodexCLIProvider()


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    return RecommendationService(
        database=get_database(),
        provider=get_codex_provider(),
        fallback_provider=ExplicitApiKeyFallbackProvider(),
    )


@lru_cache(maxsize=1)
def get_alert_service() -> AlertService:
    return AlertService(get_database())


@lru_cache(maxsize=1)
def get_notification_service() -> NotificationService:
    return NotificationService()


# Optional DXLink hub set by the web lifespan — never holds account identifiers.
_live_market_hub: Any | None = None


def set_live_market_hub(hub: Any | None) -> None:
    """Wire the streaming hub for redacted live market values (web lifespan)."""

    global _live_market_hub
    _live_market_hub = hub


def _market_fields_from_dxlink(symbol: str) -> dict | None:
    hub = _live_market_hub
    if hub is None:
        return None
    latest = getattr(hub, "latest_market", None)
    if not isinstance(latest, dict):
        return None
    by_type = latest.get(symbol.upper()) or latest.get(symbol)
    if not isinstance(by_type, dict) or not by_type:
        return None
    # Flatten latest event values (already redacted at publish time).
    merged: dict = {}
    for values in by_type.values():
        if isinstance(values, dict):
            merged.update(values)
    price = (
        merged.get("price")
        or merged.get("mark")
        or merged.get("last")
        or merged.get("bidPrice")
    )
    bid = merged.get("bid") or merged.get("bidPrice")
    ask = merged.get("ask") or merged.get("askPrice")
    if price is None and bid is None and ask is None:
        return None
    spread = None
    if (
        isinstance(bid, (int, float))
        and isinstance(ask, (int, float))
        and isinstance(price, (int, float))
        and price
    ):
        spread = ((ask - bid) / price) * 100
    return {
        "price": float(price) if isinstance(price, (int, float)) else None,
        "bid": float(bid) if isinstance(bid, (int, float)) else None,
        "ask": float(ask) if isinstance(ask, (int, float)) else None,
        "iv": merged.get("iv") or merged.get("implied_volatility"),
        "iv_rank": merged.get("iv_rank"),
        "iv_percentile": merged.get("iv_percentile"),
        "spread_percent": spread,
        "provider": "tastytrade-dxlink",
        "as_of": datetime.now(UTC).isoformat(),
    }


def build_market_context_loader() -> Any:
    """Prefer live DXLink hub values; fall back to MarketService with provenance."""

    def market_context_loader(symbol: str) -> dict | None:
        live = _market_fields_from_dxlink(symbol)
        if live is not None:
            return live
        try:
            snap = get_market_service().snapshot(symbol, force_refresh=True)
        except Exception:
            return None
        if snap is None:
            return None
        provider = snap.freshness.provider if snap.freshness else "market-service"
        return {
            "price": snap.price,
            "bid": snap.bid,
            "ask": snap.ask,
            "iv": snap.iv,
            "iv_rank": snap.iv_rank,
            "iv_percentile": snap.iv_percentile,
            "spread_percent": snap.spread_percent,
            "provider": provider,
            "as_of": snap.freshness.as_of.isoformat() if snap.freshness else None,
        }

    return market_context_loader


@lru_cache(maxsize=1)
def get_monitoring_service() -> MonitoringService:
    def portfolio_loader():
        try:
            return get_portfolio_service().latest()
        except Exception:
            return get_database().latest_portfolio_snapshot()

    def catalyst_loader(symbol: str) -> list[dict]:
        try:
            result = get_catalyst_service().analyze_symbol(symbol)
            return [event.model_dump(mode="json") for event in result.catalysts]
        except Exception:
            return []

    def thesis_loader(strategy_id: str) -> dict | None:
        thesis = get_plans_service().get_thesis(strategy_id)
        return thesis.model_dump(mode="json") if thesis else None

    def plan_loader(strategy_id: str) -> dict | None:
        plan = get_plans_service().get_trade_plan(strategy_id)
        return plan.model_dump(mode="json") if plan else None

    def risk_snapshot_loader(snapshot) -> dict | None:
        try:
            risk = get_risk_service().portfolio_risk(snapshot)
            top_share = 0.0
            if risk.concentration:
                top_share = float(risk.concentration[0].share_of_portfolio or 0.0)
            return {
                "total_delta": risk.total_delta,
                "total_gamma": risk.total_gamma,
                "total_theta": risk.total_theta,
                "total_vega": risk.total_vega,
                "unrealized_pnl": risk.unrealized_pnl,
                "concentration_top_share": top_share,
            }
        except Exception:
            return None

    return MonitoringService(
        database=get_database(),
        recommendations=get_recommendation_service(),
        alerts=get_alert_service(),
        notifications=get_notification_service(),
        portfolio_loader=portfolio_loader,
        catalyst_loader=catalyst_loader,
        thesis_loader=thesis_loader,
        plan_loader=plan_loader,
        risk_snapshot_loader=risk_snapshot_loader,
        market_context_loader=build_market_context_loader(),
    )
