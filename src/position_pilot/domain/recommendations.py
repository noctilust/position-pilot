"""Horizon-aware Codex recommendations with fingerprints, history, and decisions."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from ..persistence.sqlite import PositionPilotDatabase
from ..providers.codex import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    CodexCLIProvider,
    CodexInvocationResult,
    CodexProviderStatus,
    CodexStructuredOutput,
    ExplicitApiKeyFallbackProvider,
    RecommendationAction,
    RecommendationRisk,
    redact_secrets,
)
from .snapshots import PortfolioSnapshot, PositionHorizon, StrategySnapshot

CATALYST_ALLOWLIST = frozenset(
    {
        "catalyst_id",
        "id",
        "headline",
        "summary",
        "taxonomy",
        "confidence",
        "attribution",
        "event_at",
        "event_timestamp",
        "high_impact",
    }
)
THESIS_ALLOWLIST = frozenset(
    {
        "purpose",
        "expected_duration",
        "target_range",
        "invalidation",
        "income_or_hedge_intent",
        "events_to_watch",
    }
)
TRADE_PLAN_ALLOWLIST = frozenset(
    {
        "entry_thesis",
        "intended_duration",
        "profit_target",
        "max_loss",
        "roll_criteria",
        "event_exposure",
        "exit_deadline",
    }
)
EXPOSURE_ALLOWLIST = frozenset(
    {
        "net_delta",
        "net_theta",
        "net_vega",
        "net_gamma",
        "symbol_count",
        "strategy_count",
        "concentration",
        "total_delta",
        "total_theta",
        "account_count",
        "position_count",
    }
)
# Analytical market fields only — never raw provider payloads.
MARKET_CONTEXT_ALLOWLIST = frozenset(
    {
        "price",
        "bid",
        "ask",
        "iv",
        "iv_rank",
        "iv_percentile",
        "spread_percent",
        "provider",
        "as_of",
    }
)

_FORBIDDEN_KEY = re.compile(
    r"(?i)(account_number|account_id|account_name|broker|credential|password|secret|"
    r"token|refresh|authorization|api_key|client_secret|transfer|routing|ssn|"
    r"provider_payload|raw_payload|full_text|payload_json)"
)
# Broker/account shapes only — must NOT match pure strikes/prices like 500 or 500/505.
# Tastytrade-style: digit + 2+ letters + 3+ digits (e.g. 5WT00001).
# Opaque internal: acct- / acct_ + long hex.
_ACCOUNT_ID_VALUE = re.compile(
    r"(?i)\b("
    r"\d[A-Z]{2,}\d{3,}"
    r"|acct[-_][0-9a-f]{8,}"
    r")\b"
)
# Pure numerical / strike / price text — never treat as account identifiers.
_ANALYTICAL_NUMERIC = re.compile(
    r"^[\s$€£+-]*(?:\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*%?\s*$"
)


class SubjectType(StrEnum):
    STRATEGY = "strategy"
    EQUITY = "equity"
    PORTFOLIO = "portfolio"


class TraderDecisionKind(StrEnum):
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    DEFERRED = "deferred"
    HANDLED_IN_TASTYTRADE = "handled_in_tastytrade"


class HistoryEntryKind(StrEnum):
    MATERIAL_CHANGE = "material_change"
    AUDIT_CHANGE = "audit_change"
    DAILY_SUMMARY = "daily_summary"
    EVALUATION = "evaluation"
    PROVIDER_FAILURE = "provider_failure"


class RecommendationRecord(BaseModel):
    """Current recommendation state for one subject."""

    recommendation_id: str
    subject_type: SubjectType
    subject_id: str
    account_id: str | None = None
    symbol: str | None = None
    strategy_type: str | None = None
    horizon: PositionHorizon
    action: RecommendationAction | None = None
    urgency: int | None = Field(default=None, ge=1, le=5)
    risk: RecommendationRisk | None = None
    reasoning: str | None = None
    evidence: list[str] = Field(default_factory=list)
    catalyst_refs: list[str] = Field(default_factory=list)
    suggested_action: str | None = None
    input_fingerprint: str
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    last_evaluated_at: datetime
    recommendation_updated_at: datetime | None = None
    provider: str = "codex-cli"
    provider_status: CodexProviderStatus
    error: str | None = None
    codex_called: bool = False


class RecommendationHistoryEntry(BaseModel):
    history_id: str
    recommendation_id: str
    subject_type: SubjectType
    subject_id: str
    kind: HistoryEntryKind
    recorded_at: datetime
    action: RecommendationAction | None = None
    urgency: int | None = None
    risk: RecommendationRisk | None = None
    summary: str
    diff: dict[str, Any] = Field(default_factory=dict)
    input_fingerprint: str | None = None
    provider_status: CodexProviderStatus | None = None
    evaluation_count: int = 1


class TraderDecision(BaseModel):
    decision_id: str
    recommendation_id: str
    subject_type: SubjectType
    subject_id: str
    decision: TraderDecisionKind
    note: str = ""
    recorded_at: datetime


class RecommendationProvider(Protocol):
    def complete_recommendation(self, context: dict[str, Any]) -> CodexInvocationResult: ...

    def public_status(self) -> str: ...


def default_horizon_for_strategy(strategy: StrategySnapshot) -> PositionHorizon:
    """Equities/LEAPS default strategic; multi-leg options default tactical."""

    if strategy.horizon in (PositionHorizon.STRATEGIC, PositionHorizon.TACTICAL):
        return strategy.horizon
    strategy_type = strategy.strategy_type.lower()
    if strategy_type in {"long stock", "short stock", "stock"}:
        return PositionHorizon.STRATEGIC
    dte = strategy.days_to_expiration
    if dte is not None and dte >= 365:
        return PositionHorizon.STRATEGIC
    if "leap" in strategy_type:
        return PositionHorizon.STRATEGIC
    return PositionHorizon.TACTICAL


def is_stock_strategy(strategy: StrategySnapshot) -> bool:
    return strategy.strategy_type.lower() in {"long stock", "short stock", "stock"}


def equity_subject_id(account_id: str, symbol: str) -> str:
    """Canonical subject path for standalone equity (account + symbol)."""

    return f"equity:{account_id}:{symbol.upper()}"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_inputs(payload: dict[str, Any]) -> str:
    """Deterministic SHA-256 over account-safe analytical context only."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _looks_like_analytical_numeric(value: str) -> bool:
    """True for strikes, prices, DTE-like numbers, and slash-separated strike chains."""

    stripped = value.strip()
    if not stripped:
        return False
    if _ANALYTICAL_NUMERIC.match(stripped):
        return True
    # Allow common strike chain forms with optional $ and spaces: $500/$505
    if re.fullmatch(r"[$\s0-9./+\-]+", stripped) and any(ch.isdigit() for ch in stripped):
        return True
    return False


def _sanitize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        if _looks_like_analytical_numeric(value):
            return value[:2_000]
        if _ACCOUNT_ID_VALUE.search(value):
            return "[redacted]"
        if _SECRET_LIKE_SEARCH(value):
            return "[redacted]"
        return value[:2_000]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:500]


def _SECRET_LIKE_SEARCH(value: str) -> bool:
    return bool(
        re.search(
            r"(?i)(bearer\s+\S+|sk-[a-z0-9]{10,}|eyJ[a-zA-Z0-9_\-]{20,}"
            r"|password|client_secret|refresh_token|access_token)",
            value,
        )
    )


def sanitize_tree(value: Any) -> Any:
    """Recursively drop forbidden keys and redact credential-like values."""

    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _FORBIDDEN_KEY.search(key_str):
                continue
            cleaned[key_str] = sanitize_tree(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_tree(item) for item in value[:50]]
    return _sanitize_scalar(value)


def allowlist_dict(payload: dict[str, Any] | None, allowed: frozenset[str]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        key: sanitize_tree(value)
        for key, value in payload.items()
        if key in allowed and not _FORBIDDEN_KEY.search(key)
    }


def sanitize_catalysts(catalysts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not catalysts:
        return []
    cleaned: list[dict[str, Any]] = []
    for item in catalysts[:20]:
        if not isinstance(item, dict):
            continue
        row = allowlist_dict(item, CATALYST_ALLOWLIST)
        # Normalize id field for stable fingerprints.
        if "id" in row and "catalyst_id" not in row:
            row["catalyst_id"] = row.pop("id")
        if "event_timestamp" in row and "event_at" not in row:
            row["event_at"] = row.pop("event_timestamp")
        cleaned.append(row)
    return cleaned


def sanitize_exposure(exposure: dict[str, Any] | None) -> dict[str, Any]:
    if not exposure:
        return {}
    base = allowlist_dict(exposure, EXPOSURE_ALLOWLIST)
    if "concentration" in exposure and isinstance(exposure["concentration"], list):
        base["concentration"] = [
            {
                "symbol": str(row.get("symbol", ""))[:32],
                "market_value": row.get("market_value"),
                "share": row.get("share") or row.get("share_of_portfolio"),
            }
            for row in exposure["concentration"][:20]
            if isinstance(row, dict)
        ]
    return sanitize_tree(base)


def sanitize_market_context(market: dict[str, Any] | None) -> dict[str, Any]:
    """Allowlisted live/market analytical fields only."""

    if not market:
        return {}
    return allowlist_dict(market, MARKET_CONTEXT_ALLOWLIST)


def is_notification_material(
    previous: RecommendationRecord | None,
    action: RecommendationAction,
    urgency: int,
    risk: RecommendationRisk,
) -> bool:
    """Notify only when action, urgency, or risk changes."""

    if previous is None or previous.action is None:
        return True
    return (
        previous.action != action
        or previous.urgency != urgency
        or previous.risk != risk
    )


# Back-compat alias used by monitoring/tests.
is_material_change = is_notification_material


def audit_diff(
    previous: RecommendationRecord | None,
    *,
    action: RecommendationAction,
    urgency: int,
    risk: RecommendationRisk,
    reasoning: str,
    evidence: list[str],
    catalyst_refs: list[str],
    input_fingerprint: str,
) -> dict[str, Any]:
    """Diff covering action/urgency/risk/reasoning/evidence/catalysts/inputs."""

    before = {
        "action": previous.action.value if previous and previous.action else None,
        "urgency": previous.urgency if previous else None,
        "risk": previous.risk.value if previous and previous.risk else None,
        "reasoning": previous.reasoning if previous else None,
        "evidence": list(previous.evidence) if previous else [],
        "catalyst_refs": list(previous.catalyst_refs) if previous else [],
        "input_fingerprint": previous.input_fingerprint if previous else None,
    }
    after = {
        "action": action.value,
        "urgency": urgency,
        "risk": risk.value,
        "reasoning": reasoning,
        "evidence": evidence,
        "catalyst_refs": catalyst_refs,
        "input_fingerprint": input_fingerprint,
    }
    return {
        key: {"from": before[key], "to": after[key]}
        for key in after
        if before.get(key) != after.get(key)
    }


# Back-compat alias
material_diff = audit_diff


def has_audit_change(diff: dict[str, Any]) -> bool:
    return bool(diff)


def strategy_context(
    strategy: StrategySnapshot,
    *,
    catalysts: list[dict[str, Any]] | None = None,
    thesis: dict[str, Any] | None = None,
    trade_plan: dict[str, Any] | None = None,
    portfolio_exposure: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Account-safe analytical context for one strategy-within-account subject."""

    horizon = default_horizon_for_strategy(strategy)
    legs = []
    for leg in strategy.legs:
        legs.append(
            {
                "symbol": leg.symbol,
                "underlying": leg.underlying_symbol,
                "quantity": leg.quantity,
                "direction": leg.quantity_direction.value
                if hasattr(leg.quantity_direction, "value")
                else str(leg.quantity_direction),
                "position_type": leg.position_type.value
                if hasattr(leg.position_type, "value")
                else str(leg.position_type),
                "strike": leg.strike_price,
                "option_type": leg.option_type,
                "expiration": leg.expiration_date,
                "dte": leg.days_to_expiration,
                "mark": leg.mark_price,
                "delta": leg.delta,
                "gamma": leg.gamma,
                "theta": leg.theta,
                "vega": leg.vega,
                "iv": leg.implied_volatility,
                "unrealized_pnl": leg.unrealized_pnl,
            }
        )
    context = {
        "subject_type": SubjectType.STRATEGY.value,
        "symbol": strategy.underlying,
        "strategy_type": strategy.strategy_type,
        "horizon": horizon.value,
        "quantity": strategy.quantity,
        "strikes": strategy.strikes,
        "dte": strategy.days_to_expiration,
        "expiration": strategy.expiration_date,
        "unrealized_pnl": strategy.unrealized_pnl,
        "unrealized_pnl_percent": strategy.unrealized_pnl_percent,
        "total_delta": strategy.total_delta,
        "total_theta": strategy.total_theta,
        "legs": legs,
        "catalysts": sanitize_catalysts(catalysts),
        "thesis": allowlist_dict(thesis, THESIS_ALLOWLIST),
        "trade_plan": allowlist_dict(trade_plan, TRADE_PLAN_ALLOWLIST),
        "portfolio_exposure": sanitize_exposure(portfolio_exposure),
        "market": sanitize_market_context(market),
    }
    return sanitize_tree(context)


def equity_context(
    *,
    symbol: str,
    quantity: int,
    mark_price: float | None,
    unrealized_pnl: float,
    unrealized_pnl_percent: float | None,
    catalysts: list[dict[str, Any]] | None = None,
    portfolio_exposure: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = {
        "subject_type": SubjectType.EQUITY.value,
        "symbol": symbol,
        "horizon": PositionHorizon.STRATEGIC.value,
        "quantity": quantity,
        "mark": mark_price,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_percent": unrealized_pnl_percent,
        "catalysts": sanitize_catalysts(catalysts),
        "portfolio_exposure": sanitize_exposure(portfolio_exposure),
        "market": sanitize_market_context(market),
    }
    return sanitize_tree(context)


def portfolio_context(snapshot: PortfolioSnapshot) -> dict[str, Any]:
    concentration: dict[str, float] = {}
    for account in snapshot.accounts:
        for position in account.positions:
            symbol = position.underlying_symbol.upper()
            concentration[symbol] = concentration.get(symbol, 0.0) + abs(position.market_value)
    strategies = [
        {
            "symbol": strategy.underlying,
            "strategy_type": strategy.strategy_type,
            "horizon": default_horizon_for_strategy(strategy).value,
            "delta": strategy.total_delta,
            "theta": strategy.total_theta,
            "unrealized_pnl": strategy.unrealized_pnl,
            "dte": strategy.days_to_expiration,
        }
        for strategy in snapshot.strategies
    ]
    context = {
        "subject_type": SubjectType.PORTFOLIO.value,
        "account_count": len(snapshot.accounts),
        "strategy_count": len(snapshot.strategies),
        "position_count": sum(len(account.positions) for account in snapshot.accounts),
        "totals": {
            "net_liquidating_value": snapshot.totals.net_liquidating_value,
            "unrealized_pnl": snapshot.totals.unrealized_pnl,
        },
        "concentration": [
            {"symbol": symbol, "market_value": value}
            for symbol, value in sorted(concentration.items(), key=lambda item: -item[1])[:20]
        ],
        "strategies": strategies[:50],
    }
    return sanitize_tree(context)


class RecommendationService:
    """Evaluate, persist, and audit Codex recommendations without trading."""

    def __init__(
        self,
        database: PositionPilotDatabase,
        provider: CodexCLIProvider | RecommendationProvider | None = None,
        *,
        fallback_provider: ExplicitApiKeyFallbackProvider | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self.provider = provider or CodexCLIProvider()
        self.fallback_provider = fallback_provider or ExplicitApiKeyFallbackProvider()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._subject_locks: dict[str, threading.Lock] = {}
        self._subject_locks_guard = threading.Lock()

    def _lock_for(self, subject_type: str, subject_id: str) -> threading.Lock:
        key = f"{subject_type}:{subject_id}"
        with self._subject_locks_guard:
            lock = self._subject_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._subject_locks[key] = lock
            return lock

    def provider_public_status(self, *, force_refresh: bool = False) -> str:
        if hasattr(self.provider, "public_status"):
            try:
                return self.provider.public_status(force_refresh=force_refresh)  # type: ignore[call-arg]
            except TypeError:
                return self.provider.public_status()  # type: ignore[no-any-return]
        return "not_checked"

    def settings(self) -> dict[str, Any]:
        stored = self.database.get_setting("recommendations", {}) or {}
        return {
            "selected_provider": "codex-cli",
            "api_key_fallback_available": False,
            "api_key_fallback_enabled": False,
            "rich_notification_preview": bool(stored.get("rich_notification_preview", False)),
        }

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.settings()
        if "rich_notification_preview" in payload:
            current["rich_notification_preview"] = bool(payload["rich_notification_preview"])
        # Provider selection is fixed to codex-cli; ignore attempts to enable API-key path.
        current["selected_provider"] = "codex-cli"
        current["api_key_fallback_enabled"] = False
        current["api_key_fallback_available"] = False
        self.database.set_setting(
            "recommendations",
            {
                "rich_notification_preview": current["rich_notification_preview"],
                "selected_provider": "codex-cli",
                "api_key_fallback_enabled": False,
            },
        )
        return current

    def get(self, subject_type: SubjectType | str, subject_id: str) -> RecommendationRecord | None:
        payload = self.database.get_recommendation(str(subject_type), subject_id)
        return RecommendationRecord.model_validate(payload) if payload else None

    def list_for_account(self, account_id: str = "all") -> list[RecommendationRecord]:
        rows = self.database.list_recommendations(account_id=account_id)
        return [RecommendationRecord.model_validate(row) for row in rows]

    def history(
        self,
        subject_type: SubjectType | str,
        subject_id: str,
        *,
        limit: int = 50,
    ) -> list[RecommendationHistoryEntry]:
        rows = self.database.list_recommendation_history(
            str(subject_type),
            subject_id,
            limit=limit,
        )
        return [RecommendationHistoryEntry.model_validate(row) for row in rows]

    def record_decision(
        self,
        *,
        recommendation_id: str,
        decision: TraderDecisionKind | str,
        note: str = "",
    ) -> TraderDecision:
        current = self.database.get_recommendation_by_id(recommendation_id)
        if current is None:
            raise KeyError(recommendation_id)
        kind = TraderDecisionKind(decision)
        record = TraderDecision(
            decision_id=str(uuid4()),
            recommendation_id=recommendation_id,
            subject_type=SubjectType(current["subject_type"]),
            subject_id=current["subject_id"],
            decision=kind,
            note=(note or "")[:2_000],
            recorded_at=self._clock(),
        )
        self.database.append_trader_decision(record.model_dump(mode="json"))
        return record

    def list_decisions(
        self,
        subject_type: SubjectType | str | None = None,
        subject_id: str | None = None,
        *,
        limit: int = 50,
    ) -> list[TraderDecision]:
        rows = self.database.list_trader_decisions(
            subject_type=str(subject_type) if subject_type else None,
            subject_id=subject_id,
            limit=limit,
        )
        return [TraderDecision.model_validate(row) for row in rows]

    def evaluate_strategy(
        self,
        strategy: StrategySnapshot,
        *,
        catalysts: list[dict[str, Any]] | None = None,
        thesis: dict[str, Any] | None = None,
        trade_plan: dict[str, Any] | None = None,
        portfolio_exposure: dict[str, Any] | None = None,
        market: dict[str, Any] | None = None,
        force: bool = False,
        material_event: bool = False,
    ) -> RecommendationRecord:
        del material_event  # scheduling-only; never bypasses fingerprint skip
        context = strategy_context(
            strategy,
            catalysts=catalysts,
            thesis=thesis,
            trade_plan=trade_plan,
            portfolio_exposure=portfolio_exposure,
            market=market,
        )
        return self._evaluate(
            subject_type=SubjectType.STRATEGY,
            subject_id=strategy.strategy_id,
            account_id=strategy.account_id,
            symbol=strategy.underlying,
            strategy_type=strategy.strategy_type,
            horizon=default_horizon_for_strategy(strategy),
            context=context,
            force=force,
        )

    def evaluate_equity(
        self,
        *,
        subject_id: str,
        account_id: str,
        symbol: str,
        quantity: int,
        mark_price: float | None,
        unrealized_pnl: float,
        unrealized_pnl_percent: float | None = None,
        catalysts: list[dict[str, Any]] | None = None,
        portfolio_exposure: dict[str, Any] | None = None,
        market: dict[str, Any] | None = None,
        force: bool = False,
        material_event: bool = False,
    ) -> RecommendationRecord:
        del material_event
        context = equity_context(
            symbol=symbol,
            quantity=quantity,
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_percent,
            catalysts=catalysts,
            portfolio_exposure=portfolio_exposure,
            market=market,
        )
        return self._evaluate(
            subject_type=SubjectType.EQUITY,
            subject_id=subject_id,
            account_id=account_id,
            symbol=symbol,
            strategy_type="Long Stock",
            horizon=PositionHorizon.STRATEGIC,
            context=context,
            force=force,
        )

    def evaluate_portfolio(
        self,
        snapshot: PortfolioSnapshot,
        *,
        force: bool = False,
        material_event: bool = False,
    ) -> RecommendationRecord:
        del material_event
        context = portfolio_context(snapshot)
        subject_id = f"portfolio:{snapshot.selected_account_id}"
        return self._evaluate(
            subject_type=SubjectType.PORTFOLIO,
            subject_id=subject_id,
            account_id=snapshot.selected_account_id
            if snapshot.selected_account_id != "all"
            else None,
            symbol=None,
            strategy_type=None,
            horizon=PositionHorizon.STRATEGIC,
            context=context,
            force=force,
        )

    def _active_provider(self) -> RecommendationProvider:
        # Always Codex CLI — API-key path is not selectable.
        return self.provider  # type: ignore[return-value]

    def _evaluate(
        self,
        *,
        subject_type: SubjectType,
        subject_id: str,
        account_id: str | None,
        symbol: str | None,
        strategy_type: str | None,
        horizon: PositionHorizon,
        context: dict[str, Any],
        force: bool,
    ) -> RecommendationRecord:
        lock = self._lock_for(str(subject_type), subject_id)
        with lock:
            return self._evaluate_locked(
                subject_type=subject_type,
                subject_id=subject_id,
                account_id=account_id,
                symbol=symbol,
                strategy_type=strategy_type,
                horizon=horizon,
                context=context,
                force=force,
            )

    def _evaluate_locked(
        self,
        *,
        subject_type: SubjectType,
        subject_id: str,
        account_id: str | None,
        symbol: str | None,
        strategy_type: str | None,
        horizon: PositionHorizon,
        context: dict[str, Any],
        force: bool,
    ) -> RecommendationRecord:
        now = self._clock()
        fingerprint = fingerprint_inputs(context)
        previous = self.get(subject_type, subject_id)

        # Fingerprint short-circuit unless force=True. material_event never bypasses this.
        if (
            previous is not None
            and previous.input_fingerprint == fingerprint
            and previous.provider_status
            in {CodexProviderStatus.OK, CodexProviderStatus.SKIPPED_UNCHANGED}
            and previous.action is not None
            and not force
        ):
            updated = previous.model_copy(
                update={
                    "last_evaluated_at": now,
                    "provider_status": CodexProviderStatus.SKIPPED_UNCHANGED,
                    "codex_called": False,
                    "error": None,
                }
            )
            day = now.astimezone(UTC).date().isoformat()
            summary = RecommendationHistoryEntry(
                history_id=str(uuid4()),
                recommendation_id=updated.recommendation_id,
                subject_type=subject_type,
                subject_id=subject_id,
                kind=HistoryEntryKind.DAILY_SUMMARY,
                recorded_at=now,
                action=updated.action,
                urgency=updated.urgency,
                risk=updated.risk,
                summary=(
                    f"Unchanged evaluation on {day} "
                    f"({updated.action.value if updated.action else 'n/a'})"
                ),
                diff={"day": day},
                input_fingerprint=fingerprint,
                provider_status=CodexProviderStatus.SKIPPED_UNCHANGED,
                evaluation_count=1,
            )
            self.database.upsert_recommendation_atomic(
                updated.model_dump(mode="json"),
                daily_summary=summary.model_dump(mode="json"),
            )
            return updated

        invocation = self._active_provider().complete_recommendation(context)
        if invocation.status != CodexProviderStatus.OK or invocation.output is None:
            failure = RecommendationRecord(
                recommendation_id=previous.recommendation_id if previous else str(uuid4()),
                subject_type=subject_type,
                subject_id=subject_id,
                account_id=account_id,
                symbol=symbol,
                strategy_type=strategy_type,
                horizon=horizon,
                action=previous.action if previous else None,
                urgency=previous.urgency if previous else None,
                risk=previous.risk if previous else None,
                reasoning=previous.reasoning if previous else None,
                evidence=previous.evidence if previous else [],
                catalyst_refs=previous.catalyst_refs if previous else [],
                suggested_action=previous.suggested_action if previous else None,
                input_fingerprint=fingerprint,
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                last_evaluated_at=now,
                recommendation_updated_at=previous.recommendation_updated_at if previous else None,
                provider=invocation.provider,
                provider_status=invocation.status,
                error=redact_secrets(invocation.error),
                codex_called=True,
            )
            history = RecommendationHistoryEntry(
                history_id=str(uuid4()),
                recommendation_id=failure.recommendation_id,
                subject_type=subject_type,
                subject_id=subject_id,
                kind=HistoryEntryKind.PROVIDER_FAILURE,
                recorded_at=now,
                summary=f"Provider {invocation.status.value}: {failure.error or 'failed'}",
                diff={"status": invocation.status.value},
                input_fingerprint=fingerprint,
                provider_status=invocation.status,
            )
            self.database.upsert_recommendation_atomic(
                failure.model_dump(mode="json"),
                history_entry=history.model_dump(mode="json"),
            )
            return failure

        output: CodexStructuredOutput = invocation.output
        notify_material = is_notification_material(
            previous, output.action, output.urgency, output.risk
        )
        diff = audit_diff(
            previous,
            action=output.action,
            urgency=output.urgency,
            risk=output.risk,
            reasoning=output.reasoning,
            evidence=list(output.evidence),
            catalyst_refs=list(output.catalyst_refs),
            input_fingerprint=fingerprint,
        )
        recommendation_id = previous.recommendation_id if previous else str(uuid4())
        # recommendation_updated_at advances on notification-material change only.
        if notify_material or previous is None:
            updated_at = now
        else:
            updated_at = previous.recommendation_updated_at or now

        record = RecommendationRecord(
            recommendation_id=recommendation_id,
            subject_type=subject_type,
            subject_id=subject_id,
            account_id=account_id,
            symbol=symbol,
            strategy_type=strategy_type,
            horizon=horizon,
            action=output.action,
            urgency=output.urgency,
            risk=output.risk,
            reasoning=output.reasoning,
            evidence=list(output.evidence),
            catalyst_refs=list(output.catalyst_refs),
            suggested_action=output.suggested_action,
            input_fingerprint=fingerprint,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            last_evaluated_at=now,
            recommendation_updated_at=updated_at,
            provider=invocation.provider,
            provider_status=CodexProviderStatus.OK,
            error=None,
            codex_called=True,
        )

        history_entry: RecommendationHistoryEntry | None = None
        daily_summary: RecommendationHistoryEntry | None = None
        if has_audit_change(diff):
            kind = (
                HistoryEntryKind.MATERIAL_CHANGE
                if notify_material
                else HistoryEntryKind.AUDIT_CHANGE
            )
            history_entry = RecommendationHistoryEntry(
                history_id=str(uuid4()),
                recommendation_id=recommendation_id,
                subject_type=subject_type,
                subject_id=subject_id,
                kind=kind,
                recorded_at=now,
                action=output.action,
                urgency=output.urgency,
                risk=output.risk,
                summary=(
                    f"{output.action.value} · urgency {output.urgency} · {output.risk.value}"
                ),
                diff=diff,
                input_fingerprint=fingerprint,
                provider_status=CodexProviderStatus.OK,
            )
        else:
            day = now.astimezone(UTC).date().isoformat()
            daily_summary = RecommendationHistoryEntry(
                history_id=str(uuid4()),
                recommendation_id=recommendation_id,
                subject_type=subject_type,
                subject_id=subject_id,
                kind=HistoryEntryKind.DAILY_SUMMARY,
                recorded_at=now,
                action=output.action,
                urgency=output.urgency,
                risk=output.risk,
                summary=f"Unchanged evaluation on {day} ({output.action.value})",
                diff={"day": day},
                input_fingerprint=fingerprint,
                provider_status=CodexProviderStatus.OK,
                evaluation_count=1,
            )

        self.database.upsert_recommendation_atomic(
            record.model_dump(mode="json"),
            history_entry=history_entry.model_dump(mode="json") if history_entry else None,
            daily_summary=daily_summary.model_dump(mode="json") if daily_summary else None,
        )
        return record


def strategic_due(
    record: RecommendationRecord | None,
    *,
    now: datetime,
    material_event: bool,
    fingerprint: str | None = None,
) -> bool:
    """Strategic subjects: daily cadence or new material-event inputs (fingerprint change)."""

    if record is None:
        return True
    if fingerprint is not None and record.input_fingerprint != fingerprint:
        return True
    # Persistent high-impact without fingerprint change does not re-due.
    if material_event and fingerprint is None:
        # Legacy callers without fingerprint: treat as due only if never OK.
        if record.provider_status not in {
            CodexProviderStatus.OK,
            CodexProviderStatus.SKIPPED_UNCHANGED,
        }:
            return True
    last = record.last_evaluated_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return last.astimezone(UTC).date() < now.astimezone(UTC).date()


def tactical_due(
    record: RecommendationRecord | None,
    *,
    now: datetime,
    fingerprint: str,
    reevaluation_seconds: int = 30 * 60,
    material_event: bool = False,
) -> bool:
    """Tactical due when fingerprint changes or 30m elapsed — not sticky material_event."""

    del material_event  # fingerprint already incorporates event identity when present
    if record is None:
        return True
    if record.input_fingerprint != fingerprint:
        return True
    last = record.last_evaluated_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return (now - last).total_seconds() >= reevaluation_seconds


def today_utc(now: datetime | None = None) -> date:
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).date()
