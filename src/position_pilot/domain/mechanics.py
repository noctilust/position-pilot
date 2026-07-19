"""Tasty mechanics: versioned educational playbooks + deterministic evaluation.

Combines tastylive-derived educational defaults (metadata/source links only) with
read-only tastytrade-derived strategy facts. Decision support only — never creates,
stages, dry-runs, submits, replaces, or cancels broker orders.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..persistence.sqlite import PositionPilotDatabase
from .plans import PlansService
from .risk import RiskService, StrategyRisk
from .snapshots import PortfolioSnapshot, QuantityDirection, StrategySnapshot

logger = logging.getLogger(__name__)

SETTINGS_KEY = "mechanics"
PLAYBOOK_ID_V1 = "tastylive-short-premium.v1"

# Short-premium structures the v1 playbook can reason about.
# Debit butterflies are detected as "Butterfly" and are intentionally excluded.
SUPPORTED_SHORT_PREMIUM = frozenset(
    {
        "Iron Condor",
        "Iron Butterfly",
        "Bull Put Spread",
        "Bear Call Spread",
        "Short Strangle",
        "Short Straddle",
        "Short Call",
        "Short Put",
        "Jade Lizard",
        "Covered Call",
    }
)

UNDEFINED_RISK_TYPES = frozenset(
    {
        "Short Strangle",
        "Short Straddle",
        "Short Call",
        "Short Put",
        "Jade Lizard",
    }
)

# Wide bid/ask relative to mid (percent) triggers liquidity gate.
DEFAULT_WIDE_SPREAD_PCT = 15.0
# Market data older than this (seconds) is stale when as_of is available.
DEFAULT_STALE_SECONDS = 900
# Absolute short-leg delta default for "tested side" (settings override).
DEFAULT_TESTED_DELTA = 0.30


class RiskClass(StrEnum):
    DEFINED = "defined"
    UNDEFINED = "undefined"
    UNKNOWN = "unknown"


class RuleStatus(StrEnum):
    PASS = "pass"
    WATCH = "watch"
    DUE = "due"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class CandidateKind(StrEnum):
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"
    ROLL_REVIEW = "roll-review"
    MANUAL_REVIEW = "manual-review"


class CreditProvenance(StrEnum):
    LIFETIME_NET_CREDIT = "lifetime_net_credit"
    COST_BASIS_PROXY = "cost_basis_proxy"
    UNKNOWN = "unknown"


class PnlHistoryQuality(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    RAW = "raw"
    UNKNOWN = "unknown"


class CatalystAvailability(StrEnum):
    """Known empty/present list vs unavailable catalyst scan."""

    KNOWN = "known"
    UNKNOWN = "unknown"


class PlaybookSource(BaseModel):
    """Official source metadata — never store article bodies."""

    source_id: str
    title: str
    url: str
    publication_date: str | None = None
    reviewed_at: str
    rule_ids: list[str] = Field(default_factory=list)


class PlaybookRuleDef(BaseModel):
    rule_id: str
    name: str
    description: str
    category: str
    source_ids: list[str] = Field(default_factory=list)


class Playbook(BaseModel):
    """Versioned educational playbook (presets, not guarantees)."""

    playbook_id: str
    version: str
    title: str
    description: str
    sources: list[PlaybookSource]
    rules: list[PlaybookRuleDef]
    default_profit_capture_pct: float = 0.50
    default_manage_at_dte: int = 21
    default_tested_delta_threshold: float = DEFAULT_TESTED_DELTA
    default_defined_risk_cap_pct: float = 0.05
    default_undefined_bpr_cap_pct: float = 0.05
    credit_only_rolls: bool = True
    supported_strategy_types: list[str] = Field(
        default_factory=lambda: sorted(SUPPORTED_SHORT_PREMIUM)
    )
    disclaimer: str = (
        "Educational presets from tastylive-style short-premium mechanics. "
        "Defaults are not guaranteed or universally optimal. Execution remains "
        "manual in tastytrade; Position Pilot never places orders."
    )


class MechanicsSettings(BaseModel):
    """Persisted advisory settings — fail closed on invalid ranges."""

    enabled: bool = True
    advisory_only: bool = True
    shadow_mode: bool = True
    playbook_id: str = PLAYBOOK_ID_V1
    profit_target_pct: float = Field(default=0.50, ge=0.10, le=0.90)
    manage_at_dte: int = Field(default=21, ge=7, le=60)
    tested_delta_threshold: float = Field(default=DEFAULT_TESTED_DELTA, ge=0.10, le=0.50)
    defined_risk_cap_pct: float = Field(default=0.05, ge=0.005, le=0.50)
    undefined_bpr_cap_pct: float = Field(default=0.05, ge=0.005, le=0.50)
    credit_only_rolls: bool = True

    @field_validator("playbook_id")
    @classmethod
    def _known_playbook(cls, value: str) -> str:
        if value != PLAYBOOK_ID_V1:
            raise ValueError(f"Unsupported playbook_id: {value}")
        return value

    @model_validator(mode="after")
    def _advisory_invariant(self) -> MechanicsSettings:
        # Mechanics is always advisory / non-executing.
        self.advisory_only = True
        return self


class StrategyMechanicsFacts(BaseModel):
    """Account-safe deterministic facts; nulls mean unknown, never fabricated."""

    strategy_id: str
    strategy_type: str
    underlying: str
    supported: bool
    risk_class: RiskClass
    dte: int | None = None
    pnl_open: float | None = None
    pnl_history_quality: PnlHistoryQuality = PnlHistoryQuality.UNKNOWN
    original_credit: float | None = None
    credit_provenance: CreditProvenance = CreditProvenance.UNKNOWN
    profit_capture_ratio: float | None = None
    spot: float | None = None
    short_call_delta: float | None = None
    short_put_delta: float | None = None
    tested_side: str | None = None  # call | put | both | untested | unknown
    iv: float | None = None
    iv_rank: float | None = None
    # Underlying stock/ETF quote spread — NOT option-leg or complex-order liquidity.
    underlying_spread_pct: float | None = None
    # Option/complex liquidity is not available from current snapshots.
    option_liquidity_known: bool = False
    defined_max_loss: float | None = None
    # BPR or defined max-loss sizing only; never market-value/NLV as a BPR ratio.
    size_ratio: float | None = None
    size_basis: str | None = None  # max_loss_nlv | strategy_bpr_nlv | None
    # Informational only — never compared to undefined_bpr_cap_pct.
    market_value_nlv_ratio: float | None = None
    account_nlv: float | None = None
    account_buying_power: float | None = None
    # None when catalyst scan availability is unknown.
    high_impact_catalyst: bool | None = None
    catalyst_availability: CatalystAvailability = CatalystAvailability.UNKNOWN
    data_quality_flags: list[str] = Field(default_factory=list)
    plan_profit_target_pct: float | None = None
    plan_override_notes: list[str] = Field(default_factory=list)
    market_as_of: str | None = None
    market_freshness: str | None = None


class RuleResult(BaseModel):
    rule_id: str
    name: str
    status: RuleStatus
    observed: dict[str, Any] = Field(default_factory=dict)
    reason_code: str
    explanation: str
    source_ids: list[str] = Field(default_factory=list)
    playbook_id: str
    data_quality_notes: list[str] = Field(default_factory=list)


class LocalRiskCompare(BaseModel):
    """Defensible before/after risk for close/hold/reduce only."""

    current_pnl: float | None = None
    max_loss: float | None = None
    max_profit: float | None = None
    total_delta: float | None = None
    total_theta: float | None = None
    defined_risk: bool | None = None
    note: str | None = None


class AdvisoryCandidate(BaseModel):
    candidate_id: str
    kind: CandidateKind
    rule_hits: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    before_risk: LocalRiskCompare | None = None
    after_risk: LocalRiskCompare | None = None
    explanation: str
    priority: int = 100  # lower = higher priority


class MechanicsEvaluation(BaseModel):
    """Full offline-capable evaluation for one strategy."""

    schema_version: str = "mechanics.v1"
    strategy_id: str
    playbook_id: str
    playbook_version: str
    shadow_mode: bool
    enabled: bool
    evaluated_at: datetime
    facts: StrategyMechanicsFacts
    rules: list[RuleResult]
    candidates: list[AdvisoryCandidate]
    sources: list[PlaybookSource]
    settings_snapshot: dict[str, Any] = Field(default_factory=dict)
    execution_boundary: str = (
        "Advisory only. Execution remains manual in tastytrade. "
        "Position Pilot cannot create, stage, dry-run, submit, replace, or cancel orders."
    )
    fingerprint: str = ""


def evaluation_fingerprint(evaluation: MechanicsEvaluation) -> str:
    """Stable hash of rule statuses + candidate ids for reevaluation triggers."""

    payload = {
        "playbook_id": evaluation.playbook_id,
        "playbook_version": evaluation.playbook_version,
        "shadow_mode": evaluation.shadow_mode,
        "enabled": evaluation.enabled,
        "facts": {
            "risk_class": evaluation.facts.risk_class,
            "dte": evaluation.facts.dte,
            "profit_capture_ratio": evaluation.facts.profit_capture_ratio,
            "tested_side": evaluation.facts.tested_side,
            "size_ratio": evaluation.facts.size_ratio,
            "data_quality_flags": list(evaluation.facts.data_quality_flags),
            "pnl_history_quality": evaluation.facts.pnl_history_quality,
            "supported": evaluation.facts.supported,
        },
        "rules": [
            {"rule_id": rule.rule_id, "status": rule.status, "reason_code": rule.reason_code}
            for rule in sorted(evaluation.rules, key=lambda item: item.rule_id)
        ],
        "candidates": [item.candidate_id for item in evaluation.candidates],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compact_mechanics_context(evaluation: MechanicsEvaluation | None) -> dict[str, Any]:
    """Allowlisted compact payload for Codex recommendation context."""

    if evaluation is None:
        return {}
    return {
        "playbook_id": evaluation.playbook_id,
        "playbook_version": evaluation.playbook_version,
        "shadow_mode": evaluation.shadow_mode,
        "enabled": evaluation.enabled,
        "fingerprint": evaluation.fingerprint,
        "risk_class": evaluation.facts.risk_class.value,
        "dte": evaluation.facts.dte,
        "profit_capture_ratio": evaluation.facts.profit_capture_ratio,
        "credit_provenance": evaluation.facts.credit_provenance.value,
        "pnl_history_quality": evaluation.facts.pnl_history_quality.value,
        "tested_side": evaluation.facts.tested_side,
        "size_ratio": evaluation.facts.size_ratio,
        "size_basis": evaluation.facts.size_basis,
        "market_value_nlv_ratio": evaluation.facts.market_value_nlv_ratio,
        "underlying_spread_pct": evaluation.facts.underlying_spread_pct,
        "option_liquidity_known": evaluation.facts.option_liquidity_known,
        "catalyst_availability": evaluation.facts.catalyst_availability.value,
        "high_impact_catalyst": evaluation.facts.high_impact_catalyst,
        "data_quality_flags": list(evaluation.facts.data_quality_flags)[:12],
        "rule_hits": [
            {
                "rule_id": rule.rule_id,
                "status": rule.status.value,
                "reason_code": rule.reason_code,
            }
            for rule in evaluation.rules
            if rule.status in {RuleStatus.DUE, RuleStatus.WATCH, RuleStatus.BLOCKED}
        ][:20],
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "kind": item.kind.value,
                "rule_hits": list(item.rule_hits)[:8],
                "missing_inputs": list(item.missing_inputs)[:8],
                "blocking_reasons": list(item.blocking_reasons)[:8],
            }
            for item in evaluation.candidates[:10]
        ],
        "execution_boundary": evaluation.execution_boundary,
    }


# --- Built-in playbook -------------------------------------------------------

_REVIEWED = "2026-07-18"


def built_in_playbook_v1() -> Playbook:
    """tastylive-short-premium.v1 — source metadata only, no article text."""

    sources = [
        PlaybookSource(
            source_id="tl-dte-definition",
            title="Days to Expiration (DTE)",
            url="https://www.tastylive.com/definitions/days-to-expiration-dte",
            reviewed_at=_REVIEWED,
            rule_ids=["time.manage_at_dte", "gate.assignment_expiration"],
        ),
        PlaybookSource(
            source_id="tl-manage-21-dte-2022",
            title="Managing at 21 DTE (2022)",
            url=(
                "https://www.tastylive.com/shows/from-theory-to-practice/"
                "episodes/managing-at-21-dte-06-24-2022"
            ),
            publication_date="2022-06-24",
            reviewed_at=_REVIEWED,
            rule_ids=["time.manage_at_dte", "profit.manage_winner"],
        ),
        PlaybookSource(
            source_id="tl-key-mechanics",
            title="How to Use Options Strategies & Key Mechanics Takeaways",
            url=(
                "https://www.tastylive.com/news-insights/"
                "how-to-use-options-strategies-amp-key-mechanics-takeaways"
            ),
            reviewed_at=_REVIEWED,
            rule_ids=[
                "profit.manage_winner",
                "time.manage_at_dte",
                "size.small_position",
                "risk.defined_vs_undefined",
            ],
        ),
        PlaybookSource(
            source_id="tl-defending-positions",
            title="Defending Positions",
            url="https://www.tastylive.com/concepts-strategies/defending-positions",
            reviewed_at=_REVIEWED,
            rule_ids=["tested.side_review", "roll.credit_only"],
        ),
        PlaybookSource(
            source_id="tl-tested-side-2021",
            title="How to Interpret Adjustments to the Tested Side (2021)",
            url=(
                "https://www.tastylive.com/shows/from-theory-to-practice/"
                "episodes/how-to-interpret-adjustments-to-the-tested-side-08-03-2021"
            ),
            publication_date="2021-08-03",
            reviewed_at=_REVIEWED,
            rule_ids=["tested.side_review", "roll.untested_side"],
        ),
        PlaybookSource(
            source_id="tl-manage-21-dte-2023",
            title="Managing Positions at 21 DTE (2023)",
            url=(
                "https://www.tastylive.com/shows/from-theory-to-practice/"
                "episodes/managing-positions-at-21-dte-07-28-2023"
            ),
            publication_date="2023-07-28",
            reviewed_at=_REVIEWED,
            rule_ids=["time.manage_at_dte"],
        ),
        PlaybookSource(
            source_id="tl-number-of-occurrences",
            title="Number of Occurrences",
            url="https://www.tastylive.com/concepts-strategies/number-of-occurrences",
            reviewed_at=_REVIEWED,
            rule_ids=["size.small_position"],
        ),
        PlaybookSource(
            source_id="tl-trade-small-2025",
            title="Why Trade Small and Trade Often (2025)",
            url=(
                "https://www.tastylive.com/shows/market-measures/"
                "episodes/why-trade-small-and-trade-often-04-28-2025"
            ),
            publication_date="2025-04-28",
            reviewed_at=_REVIEWED,
            rule_ids=["size.small_position"],
        ),
        PlaybookSource(
            source_id="tt-api-read-only",
            title="tastytrade Basic API Usage (read-only boundary)",
            url="https://developer.tastytrade.com/basic-api-usage/",
            reviewed_at=_REVIEWED,
            rule_ids=["gate.manual_execution"],
        ),
    ]
    rules = [
        PlaybookRuleDef(
            rule_id="gate.strategy_supported",
            name="Supported short-premium strategy",
            description="Playbook applies only to supported short-premium structures.",
            category="gate",
            source_ids=["tl-key-mechanics"],
        ),
        PlaybookRuleDef(
            rule_id="gate.data_quality",
            name="Data quality / freshness",
            description=(
                "Stale quotes, wide spreads, incomplete roll history, or missing "
                "critical inputs block or downgrade advisory candidates."
            ),
            category="gate",
            source_ids=["tt-api-read-only"],
        ),
        PlaybookRuleDef(
            rule_id="gate.event_exposure",
            name="Event / catalyst exposure",
            description="High-impact catalysts warrant review before mechanical management.",
            category="gate",
            source_ids=["tl-key-mechanics"],
        ),
        PlaybookRuleDef(
            rule_id="gate.assignment_expiration",
            name="Assignment / expiration proximity",
            description="Short options near expiration carry assignment and pin risk.",
            category="gate",
            source_ids=["tl-dte-definition"],
        ),
        PlaybookRuleDef(
            rule_id="gate.manual_execution",
            name="Manual execution boundary",
            description="All actions are advisory; execution is manual in tastytrade.",
            category="gate",
            source_ids=["tt-api-read-only"],
        ),
        PlaybookRuleDef(
            rule_id="profit.manage_winner",
            name="Profit management (short premium)",
            description=(
                "Educational default: consider managing eligible short-premium winners "
                "around 50% of original credit when that basis is known."
            ),
            category="profit",
            source_ids=["tl-key-mechanics", "tl-manage-21-dte-2022"],
        ),
        PlaybookRuleDef(
            rule_id="time.manage_at_dte",
            name="Time management (~21 DTE)",
            description=(
                "Educational default: stronger attention around 21 DTE, especially for "
                "undefined-risk short premium."
            ),
            category="time",
            source_ids=[
                "tl-dte-definition",
                "tl-manage-21-dte-2022",
                "tl-manage-21-dte-2023",
            ],
        ),
        PlaybookRuleDef(
            rule_id="size.small_position",
            name="Small position sizing",
            description=(
                "Prefer small size: defined-risk vs NLV and undefined-risk buying-power "
                "usage are capped separately (educational presets)."
            ),
            category="size",
            source_ids=["tl-number-of-occurrences", "tl-trade-small-2025"],
        ),
        PlaybookRuleDef(
            rule_id="tested.side_review",
            name="Tested-side review",
            description="When a short leg is tested (elevated absolute delta), review defense.",
            category="tested",
            source_ids=["tl-defending-positions", "tl-tested-side-2021"],
        ),
        PlaybookRuleDef(
            rule_id="roll.untested_side",
            name="Untested-side roll consideration",
            description=(
                "Strategy-specific possibility of rolling the untested side for credit; "
                "never invents strikes or quotes."
            ),
            category="roll",
            source_ids=["tl-tested-side-2021", "tl-defending-positions"],
        ),
        PlaybookRuleDef(
            rule_id="roll.credit_only",
            name="Roll for credit only (default)",
            description=(
                "Default advisory constraint: prefer credit rolls when economics are known; "
                "unknown roll credit blocks verification."
            ),
            category="roll",
            source_ids=["tl-defending-positions"],
        ),
        PlaybookRuleDef(
            rule_id="risk.defined_vs_undefined",
            name="Defined vs undefined risk",
            description=(
                "Risk class changes time and size urgency; undefined gets stronger attention."
            ),
            category="risk",
            source_ids=["tl-key-mechanics"],
        ),
    ]
    return Playbook(
        playbook_id=PLAYBOOK_ID_V1,
        version="v1",
        title="tastylive short premium mechanics",
        description=(
            "Educational short-premium management playbook: profit targets, ~21 DTE, "
            "small size, tested-side review, and credit-only roll preference."
        ),
        sources=sources,
        rules=rules,
    )


def get_playbook(playbook_id: str = PLAYBOOK_ID_V1) -> Playbook:
    if playbook_id != PLAYBOOK_ID_V1:
        raise ValueError(f"Unknown playbook: {playbook_id}")
    return built_in_playbook_v1()


# --- Fact derivation ---------------------------------------------------------


def parse_plan_profit_target_pct(profit_target: str | None) -> float | None:
    """Parse free-text trade plan profit target into a 0–1 fraction when obvious."""

    if not profit_target or not str(profit_target).strip():
        return None
    text = str(profit_target).strip()
    # Prefer explicit percent: "50%", "take 50 percent"
    pct_match = re.search(r"(?i)(\d{1,2}(?:\.\d+)?)\s*%", text)
    if pct_match:
        value = float(pct_match.group(1)) / 100.0
        if 0.05 <= value <= 0.95:
            return value
    # "0.5 of credit" / "half" / "50 percent of credit"
    if re.search(r"(?i)\bhalf\b|50\s*percent|fifty\s*percent", text):
        return 0.50
    frac = re.search(r"(?i)(?:^|[^\d])(0?\.\d{1,3})\s*(?:of|credit)?", text)
    if frac:
        value = float(frac.group(1))
        if 0.05 <= value <= 0.95:
            return value
    return None


def _short_option_legs(strategy: StrategySnapshot) -> list:
    return [
        leg
        for leg in strategy.legs
        if leg.quantity_direction == QuantityDirection.SHORT
        and leg.option_type in {"C", "P"}
        and leg.strike_price is not None
    ]


def derive_risk_class(
    strategy: StrategySnapshot,
    risk: StrategyRisk | None,
) -> RiskClass:
    if strategy.strategy_type in UNDEFINED_RISK_TYPES:
        return RiskClass.UNDEFINED
    if risk is not None:
        if risk.defined_risk:
            return RiskClass.DEFINED
        if risk.max_loss is None and any(
            leg.quantity_direction == QuantityDirection.SHORT and leg.option_type in {"C", "P"}
            for leg in strategy.legs
        ):
            # Undefined max loss from structure analysis.
            return RiskClass.UNDEFINED
        if risk.defined_risk is False and risk.max_loss is None:
            return RiskClass.UNDEFINED
    # Covered call: equity long + short call — max loss not purely defined by options.
    if strategy.strategy_type == "Covered Call":
        return RiskClass.UNDEFINED
    if strategy.strategy_type in SUPPORTED_SHORT_PREMIUM:
        return RiskClass.DEFINED if risk and risk.defined_risk else RiskClass.UNKNOWN
    return RiskClass.UNKNOWN


def derive_original_credit(
    strategy: StrategySnapshot,
) -> tuple[float | None, CreditProvenance, PnlHistoryQuality]:
    """Never fabricate credit; use roll-complete lifetime or refuse."""

    legs = strategy.legs
    if not legs:
        return None, CreditProvenance.UNKNOWN, PnlHistoryQuality.UNKNOWN

    statuses = {leg.roll_history_status for leg in legs}
    if "partial" in statuses:
        # Incomplete roll history: do not invent lifetime credit.
        lifetime_vals = [
            leg.lifetime_net_credit
            for leg in legs
            if leg.lifetime_net_credit is not None and leg.roll_history_status == "complete"
        ]
        if lifetime_vals and all(leg.roll_history_status == "complete" for leg in legs):
            credit = sum(lifetime_vals)
            # Short premium: original credit is positive cash received.
            if credit > 0:
                return credit, CreditProvenance.LIFETIME_NET_CREDIT, PnlHistoryQuality.COMPLETE
        return None, CreditProvenance.UNKNOWN, PnlHistoryQuality.PARTIAL

    if all(leg.roll_history_status == "complete" for leg in legs):
        lifetime = [leg.lifetime_net_credit for leg in legs if leg.lifetime_net_credit is not None]
        if len(lifetime) == len(legs):
            credit = sum(lifetime)
            if credit > 0:
                return credit, CreditProvenance.LIFETIME_NET_CREDIT, PnlHistoryQuality.COMPLETE
            # Zero/negative: not a usable short-premium credit basis.
            return None, CreditProvenance.UNKNOWN, PnlHistoryQuality.COMPLETE
        return None, CreditProvenance.UNKNOWN, PnlHistoryQuality.PARTIAL

    # No rolls: cost_basis on short premium is often negative credit (broker convention varies).
    # Prefer strategy pnl_open_basis when present and history is none.
    if strategy.pnl_open_basis is not None and strategy.pnl_open_basis > 0:
        # pnl_open_basis is absolute lifetime/cost denominator — only treat as credit
        # when all short and net credit structure: use abs of sum of short cost bases if negative.
        short_legs = _short_option_legs(strategy)
        if short_legs:
            # Unrolled short premium: opening credit ≈ abs(short cost_basis) net of longs.
            credits: list[float] = []
            for leg in short_legs:
                # cost_basis on short options: often stored as positive premium sold * mult * qty
                # or negative; never invent — only use when cost_basis is non-zero.
                if leg.cost_basis and leg.cost_basis != 0:
                    credits.append(abs(leg.cost_basis))
            if credits and len(credits) == len(short_legs):
                # For multi-leg defined risk, net credit needs long debit subtracted.
                long_debits = [
                    abs(leg.cost_basis)
                    for leg in strategy.legs
                    if leg.quantity_direction == QuantityDirection.LONG
                    and leg.cost_basis
                    and leg.option_type in {"C", "P"}
                ]
                net = sum(credits) - sum(long_debits)
                if net > 0:
                    return net, CreditProvenance.COST_BASIS_PROXY, PnlHistoryQuality.RAW

    return (
        None,
        CreditProvenance.UNKNOWN,
        (
            PnlHistoryQuality.RAW
            if all(leg.roll_history_status == "none" for leg in legs)
            else PnlHistoryQuality.UNKNOWN
        ),
    )


def derive_pnl_open(
    strategy: StrategySnapshot,
    history_quality: PnlHistoryQuality,
) -> tuple[float | None, PnlHistoryQuality]:
    if history_quality == PnlHistoryQuality.PARTIAL:
        # Explicit: do not present roll-adjusted P/L as complete.
        return strategy.unrealized_pnl, PnlHistoryQuality.PARTIAL
    if strategy.pnl_open is not None:
        quality = history_quality
        if quality == PnlHistoryQuality.UNKNOWN:
            quality = (
                PnlHistoryQuality.COMPLETE
                if all(leg.roll_history_status in {"none", "complete"} for leg in strategy.legs)
                else PnlHistoryQuality.UNKNOWN
            )
        return strategy.pnl_open, quality
    return strategy.unrealized_pnl, (
        PnlHistoryQuality.RAW if history_quality == PnlHistoryQuality.UNKNOWN else history_quality
    )


def derive_tested_side(
    strategy: StrategySnapshot,
    *,
    spot: float | None,
    threshold: float,
) -> tuple[str | None, float | None, float | None]:
    """Return (tested_side, short_call_delta, short_put_delta)."""

    short_call_delta: float | None = None
    short_put_delta: float | None = None
    for leg in _short_option_legs(strategy):
        if leg.option_type == "C" and leg.delta is not None:
            if short_call_delta is None or abs(leg.delta) > abs(short_call_delta):
                short_call_delta = leg.delta
        elif leg.option_type == "P" and leg.delta is not None:
            if short_put_delta is None or abs(leg.delta) > abs(short_put_delta):
                short_put_delta = leg.delta

    call_tested = short_call_delta is not None and abs(short_call_delta) >= threshold
    put_tested = short_put_delta is not None and abs(short_put_delta) >= threshold

    # Spot vs strike fallback when delta missing.
    if spot is not None and short_call_delta is None and short_put_delta is None:
        for leg in _short_option_legs(strategy):
            if leg.strike_price is None:
                continue
            if leg.option_type == "C" and spot >= leg.strike_price:
                call_tested = True
            if leg.option_type == "P" and spot <= leg.strike_price:
                put_tested = True

    if short_call_delta is None and short_put_delta is None and spot is None:
        return "unknown", None, None
    if call_tested and put_tested:
        return "both", short_call_delta, short_put_delta
    if call_tested:
        return "call", short_call_delta, short_put_delta
    if put_tested:
        return "put", short_call_delta, short_put_delta
    if short_call_delta is not None or short_put_delta is not None or spot is not None:
        return "untested", short_call_delta, short_put_delta
    return "unknown", short_call_delta, short_put_delta


def derive_facts(
    strategy: StrategySnapshot,
    *,
    risk: StrategyRisk | None = None,
    market: dict[str, Any] | None = None,
    trade_plan: dict[str, Any] | None = None,
    account_nlv: float | None = None,
    account_buying_power: float | None = None,
    catalysts: list[dict[str, Any]] | None = None,
    settings: MechanicsSettings | None = None,
    now: datetime | None = None,
) -> StrategyMechanicsFacts:
    settings = settings or MechanicsSettings()
    market = market or {}
    flags: list[str] = []
    supported = strategy.strategy_type in SUPPORTED_SHORT_PREMIUM

    if not supported:
        flags.append("unsupported_strategy")

    original_credit, credit_prov, hist_quality = derive_original_credit(strategy)
    if original_credit is None:
        flags.append("missing_opening_credit")
    if hist_quality == PnlHistoryQuality.PARTIAL:
        flags.append("partial_roll_history")

    pnl_open, hist_quality = derive_pnl_open(strategy, hist_quality)

    profit_ratio: float | None = None
    if original_credit is not None and original_credit > 0 and pnl_open is not None:
        # Short premium winners: positive P/L captures fraction of credit.
        profit_ratio = pnl_open / original_credit

    spot = None
    if isinstance(market.get("price"), (int, float)):
        spot = float(market["price"])
    elif risk is not None and risk.underlying_price is not None:
        spot = risk.underlying_price
    if spot is None:
        flags.append("missing_spot")

    tested_side, sc_delta, sp_delta = derive_tested_side(
        strategy,
        spot=spot,
        threshold=settings.tested_delta_threshold,
    )
    if (
        sc_delta is None
        and sp_delta is None
        and any(
            leg.option_type in {"C", "P"} and leg.quantity_direction == QuantityDirection.SHORT
            for leg in strategy.legs
        )
    ):
        flags.append("missing_short_delta")

    iv = market.get("iv")
    iv_rank = market.get("iv_rank")
    if isinstance(iv, (int, float)):
        iv = float(iv)
    else:
        iv = risk.combined.average_iv if risk and risk.combined.average_iv is not None else None
    if isinstance(iv_rank, (int, float)):
        iv_rank = float(iv_rank)
    else:
        iv_rank = None

    # Underlying market spread only — never treated as option/complex liquidity.
    underlying_spread: float | None = None
    has_market_input = bool(market)
    if has_market_input:
        spread_raw = market.get("spread_percent")
        bid = market.get("bid")
        ask = market.get("ask")
        if isinstance(spread_raw, (int, float)):
            underlying_spread = float(spread_raw)
            if underlying_spread >= DEFAULT_WIDE_SPREAD_PCT:
                flags.append("wide_underlying_spread")
        elif isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
            mid = (float(bid) + float(ask)) / 2.0
            if mid > 0:
                underlying_spread = abs(float(ask) - float(bid)) / mid * 100.0
                if underlying_spread >= DEFAULT_WIDE_SPREAD_PCT:
                    flags.append("wide_underlying_spread")
            else:
                flags.append("incomplete_underlying_quote")
        else:
            # Price-only or missing bid/ask/spread is incomplete market quality.
            flags.append("incomplete_underlying_quote")
    else:
        flags.append("missing_underlying_quote")
    # Option leg / complex-order liquidity is not on StrategySnapshot today.
    option_liquidity_known = False
    flags.append("missing_option_liquidity")

    market_as_of = market.get("as_of") if has_market_input else None
    market_freshness = (
        (market.get("freshness") or market.get("state")) if has_market_input else None
    )
    if has_market_input:
        if market_as_of:
            try:
                as_of_dt = datetime.fromisoformat(str(market_as_of).replace("Z", "+00:00"))
                if as_of_dt.tzinfo is None:
                    # Naive timestamps are ambiguous — never silently assume a timezone.
                    flags.append("invalid_market_timestamp")
                    market_freshness = "invalid"
                else:
                    age = (now or datetime.now(UTC)) - as_of_dt.astimezone(UTC)
                    if age.total_seconds() > DEFAULT_STALE_SECONDS:
                        flags.append("stale_market")
                        market_freshness = market_freshness or "stale"
            except (TypeError, ValueError):
                flags.append("invalid_market_timestamp")
                market_freshness = "invalid"
        else:
            # Presence of market data without freshness/as_of is fail-closed.
            flags.append("missing_market_freshness")
            market_freshness = market_freshness or "unknown"
    if str(market_freshness or "").lower() in {"stale", "unavailable"}:
        if "stale_market" not in flags:
            flags.append("stale_market")

    defined_max_loss = risk.max_loss if risk else None
    risk_class = derive_risk_class(strategy, risk)

    # Defined-risk size uses max_loss/NLV. Undefined-risk size requires true strategy BPR
    # (never invented; never use market value or account remaining BP as BPR).
    size_ratio: float | None = None
    size_basis: str | None = None
    market_value_nlv_ratio: float | None = None
    if account_nlv is not None and account_nlv > 0:
        mv = sum(abs(leg.market_value) for leg in strategy.legs)
        if mv > 0:
            market_value_nlv_ratio = mv / account_nlv
        if defined_max_loss is not None and risk_class == RiskClass.DEFINED:
            size_ratio = defined_max_loss / account_nlv
            size_basis = "max_loss_nlv"
        else:
            # No per-strategy BPR on snapshots — leave size_ratio unknown.
            flags.append("missing_bpr")
    else:
        flags.append("missing_nlv")
        if risk_class != RiskClass.DEFINED or defined_max_loss is None:
            flags.append("missing_bpr")

    # catalysts=None → unknown availability; catalysts=[] → known empty scan.
    catalyst_availability = CatalystAvailability.UNKNOWN
    high_impact: bool | None = None
    if catalysts is None:
        flags.append("catalysts_unknown")
        catalyst_availability = CatalystAvailability.UNKNOWN
        high_impact = None
    else:
        catalyst_availability = CatalystAvailability.KNOWN
        high_impact = any(isinstance(item, dict) and item.get("high_impact") for item in catalysts)

    plan_notes: list[str] = []
    plan_pct = None
    if trade_plan:
        plan_pct = parse_plan_profit_target_pct(trade_plan.get("profit_target"))
        if plan_pct is not None and abs(plan_pct - settings.profit_target_pct) > 0.01:
            plan_notes.append(
                f"trade_plan_profit_target={plan_pct:.0%} overrides playbook "
                f"default {settings.profit_target_pct:.0%}"
            )
        roll_criteria = str(trade_plan.get("roll_criteria") or "")
        if re.search(r"(?i)debit|pay\s+to\s+roll|accept\s+debit", roll_criteria):
            plan_notes.append("trade_plan allows debit rolls (conflicts with credit-only default)")
        hold_text = (
            str(trade_plan.get("exit_deadline") or "")
            + " "
            + str(trade_plan.get("profit_target") or "")
        )
        if re.search(r"(?i)hold\s+through|hold\s+to\s+expir", hold_text):
            plan_notes.append("trade_plan prefers holding longer than mechanical profit target")

    dte = strategy.days_to_expiration
    if dte is None and risk and risk.combined.nearest_dte is not None:
        dte = risk.combined.nearest_dte

    return StrategyMechanicsFacts(
        strategy_id=strategy.strategy_id,
        strategy_type=strategy.strategy_type,
        underlying=strategy.underlying,
        supported=supported,
        risk_class=risk_class,
        dte=dte,
        pnl_open=pnl_open,
        pnl_history_quality=hist_quality,
        original_credit=original_credit,
        credit_provenance=credit_prov,
        profit_capture_ratio=profit_ratio,
        spot=spot,
        short_call_delta=sc_delta,
        short_put_delta=sp_delta,
        tested_side=tested_side,
        iv=iv if isinstance(iv, (int, float)) else None,
        iv_rank=iv_rank if isinstance(iv_rank, (int, float)) else None,
        underlying_spread_pct=underlying_spread,
        option_liquidity_known=option_liquidity_known,
        defined_max_loss=defined_max_loss,
        size_ratio=size_ratio,
        size_basis=size_basis,
        market_value_nlv_ratio=market_value_nlv_ratio,
        account_nlv=account_nlv,
        account_buying_power=account_buying_power,
        high_impact_catalyst=high_impact,
        catalyst_availability=catalyst_availability,
        data_quality_flags=sorted(set(flags)),
        plan_profit_target_pct=plan_pct,
        plan_override_notes=plan_notes,
        market_as_of=str(market_as_of) if market_as_of else None,
        market_freshness=str(market_freshness) if market_freshness else None,
    )


# --- Rule evaluation & candidates --------------------------------------------


def _rule_meta(playbook: Playbook, rule_id: str) -> PlaybookRuleDef:
    for rule in playbook.rules:
        if rule.rule_id == rule_id:
            return rule
    return PlaybookRuleDef(
        rule_id=rule_id,
        name=rule_id,
        description="",
        category="unknown",
        source_ids=[],
    )


def _result(
    playbook: Playbook,
    rule_id: str,
    status: RuleStatus,
    reason_code: str,
    explanation: str,
    *,
    observed: dict[str, Any] | None = None,
    data_quality_notes: list[str] | None = None,
) -> RuleResult:
    meta = _rule_meta(playbook, rule_id)
    return RuleResult(
        rule_id=rule_id,
        name=meta.name,
        status=status,
        observed=observed or {},
        reason_code=reason_code,
        explanation=explanation,
        source_ids=list(meta.source_ids),
        playbook_id=playbook.playbook_id,
        data_quality_notes=data_quality_notes or [],
    )


def evaluate_rules(
    facts: StrategyMechanicsFacts,
    *,
    playbook: Playbook,
    settings: MechanicsSettings,
) -> list[RuleResult]:
    results: list[RuleResult] = []

    # gate.strategy_supported
    if facts.supported:
        results.append(
            _result(
                playbook,
                "gate.strategy_supported",
                RuleStatus.PASS,
                "supported",
                f"{facts.strategy_type} is in the short-premium playbook.",
                observed={"strategy_type": facts.strategy_type},
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "gate.strategy_supported",
                RuleStatus.BLOCKED,
                "unsupported_strategy",
                f"{facts.strategy_type} is not supported by {playbook.playbook_id}; fail closed.",
                observed={"strategy_type": facts.strategy_type},
            )
        )

    # gate.data_quality
    # Hard blocks: incomplete/stale market quality, incomplete roll history, and
    # missing option/complex quote evidence (underlying spread is not a substitute).
    # Soft watches: catalysts unknown, etc.
    blocking_flags = {
        "stale_market",
        "invalid_market_timestamp",
        "missing_market_freshness",
        "partial_roll_history",
        "missing_underlying_quote",
        "incomplete_underlying_quote",
        "missing_option_liquidity",
    }
    hit = [flag for flag in facts.data_quality_flags if flag in blocking_flags]
    soft = [flag for flag in facts.data_quality_flags if flag not in blocking_flags]
    if not facts.supported:
        results.append(
            _result(
                playbook,
                "gate.data_quality",
                RuleStatus.NOT_APPLICABLE,
                "skipped_unsupported",
                "Data-quality gate skipped for unsupported strategy.",
            )
        )
    elif hit:
        results.append(
            _result(
                playbook,
                "gate.data_quality",
                RuleStatus.BLOCKED,
                "data_quality_block",
                "Critical market/history data issues block or downgrade advisory actions.",
                observed={"flags": facts.data_quality_flags},
                data_quality_notes=hit,
            )
        )
    elif soft:
        results.append(
            _result(
                playbook,
                "gate.data_quality",
                RuleStatus.WATCH,
                "data_quality_watch",
                "Some inputs are incomplete; treat results as review-only where noted.",
                observed={"flags": facts.data_quality_flags},
                data_quality_notes=soft,
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "gate.data_quality",
                RuleStatus.PASS,
                "data_ok",
                "No critical data-quality flags.",
                observed={"flags": []},
            )
        )

    # gate.event_exposure
    if not facts.supported:
        results.append(
            _result(
                playbook,
                "gate.event_exposure",
                RuleStatus.NOT_APPLICABLE,
                "skipped_unsupported",
                "Event gate skipped for unsupported strategy.",
            )
        )
    elif facts.catalyst_availability == CatalystAvailability.UNKNOWN:
        results.append(
            _result(
                playbook,
                "gate.event_exposure",
                RuleStatus.WATCH,
                "catalysts_unknown",
                "Catalyst availability unknown — do not treat as confirmed absence of events.",
                observed={
                    "catalyst_availability": facts.catalyst_availability.value,
                    "high_impact_catalyst": None,
                },
                data_quality_notes=["catalysts_unknown"],
            )
        )
    elif facts.high_impact_catalyst:
        results.append(
            _result(
                playbook,
                "gate.event_exposure",
                RuleStatus.WATCH,
                "high_impact_catalyst",
                "High-impact catalyst present — review before mechanical management.",
                observed={
                    "catalyst_availability": facts.catalyst_availability.value,
                    "high_impact_catalyst": True,
                },
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "gate.event_exposure",
                RuleStatus.PASS,
                "no_high_impact",
                "No high-impact catalyst flagged (known empty/confirmed scan).",
                observed={
                    "catalyst_availability": facts.catalyst_availability.value,
                    "high_impact_catalyst": False,
                },
            )
        )

    # gate.assignment_expiration
    if not facts.supported:
        results.append(
            _result(
                playbook,
                "gate.assignment_expiration",
                RuleStatus.NOT_APPLICABLE,
                "skipped_unsupported",
                "Assignment gate skipped for unsupported strategy.",
            )
        )
    elif facts.dte is None:
        results.append(
            _result(
                playbook,
                "gate.assignment_expiration",
                RuleStatus.WATCH,
                "missing_dte",
                "DTE unknown; cannot assess assignment/expiration proximity.",
                observed={"dte": None},
                data_quality_notes=["missing_dte"],
            )
        )
    elif facts.dte <= 2:
        results.append(
            _result(
                playbook,
                "gate.assignment_expiration",
                RuleStatus.DUE,
                "expiration_imminent",
                f"DTE={facts.dte}: assignment/expiration risk is elevated.",
                observed={"dte": facts.dte},
            )
        )
    elif facts.dte <= 5:
        results.append(
            _result(
                playbook,
                "gate.assignment_expiration",
                RuleStatus.WATCH,
                "expiration_near",
                f"DTE={facts.dte}: approaching expiration — review assignment risk.",
                observed={"dte": facts.dte},
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "gate.assignment_expiration",
                RuleStatus.PASS,
                "expiration_ok",
                f"DTE={facts.dte} is outside imminent assignment window.",
                observed={"dte": facts.dte},
            )
        )

    # gate.manual_execution (always pass informational)
    results.append(
        _result(
            playbook,
            "gate.manual_execution",
            RuleStatus.PASS,
            "manual_only",
            "Execution remains manual in tastytrade. No order APIs are invoked.",
            observed={"advisory_only": True, "shadow_mode": settings.shadow_mode},
        )
    )

    if not facts.supported:
        for rule_id in (
            "profit.manage_winner",
            "time.manage_at_dte",
            "size.small_position",
            "tested.side_review",
            "roll.untested_side",
            "roll.credit_only",
            "risk.defined_vs_undefined",
        ):
            results.append(
                _result(
                    playbook,
                    rule_id,
                    RuleStatus.NOT_APPLICABLE,
                    "unsupported_strategy",
                    "Rule not applied — strategy outside playbook scope.",
                )
            )
        return results

    # risk.defined_vs_undefined
    results.append(
        _result(
            playbook,
            "risk.defined_vs_undefined",
            RuleStatus.PASS if facts.risk_class != RiskClass.UNKNOWN else RuleStatus.WATCH,
            f"risk_class_{facts.risk_class.value}",
            (
                f"Risk class={facts.risk_class.value}. Undefined-risk short premium receives "
                "stronger time and size attention."
                if facts.risk_class != RiskClass.UNKNOWN
                else "Risk class could not be determined from structure bounds."
            ),
            observed={"risk_class": facts.risk_class.value},
        )
    )

    # profit.manage_winner
    target = (
        facts.plan_profit_target_pct
        if facts.plan_profit_target_pct is not None
        else settings.profit_target_pct
    )
    hold_conflict = any("holding longer" in note for note in facts.plan_override_notes)
    if facts.original_credit is None or facts.profit_capture_ratio is None:
        results.append(
            _result(
                playbook,
                "profit.manage_winner",
                RuleStatus.BLOCKED,
                "missing_credit_basis",
                "Cannot evaluate profit capture without a known original/lifetime credit.",
                observed={
                    "original_credit": facts.original_credit,
                    "profit_capture_ratio": facts.profit_capture_ratio,
                    "credit_provenance": facts.credit_provenance.value,
                },
                data_quality_notes=["missing_opening_credit"],
            )
        )
    elif hold_conflict and facts.profit_capture_ratio >= target:
        results.append(
            _result(
                playbook,
                "profit.manage_winner",
                RuleStatus.WATCH,
                "plan_conflict_hold",
                (
                    f"Profit capture {facts.profit_capture_ratio:.0%} ≥ target {target:.0%}, "
                    "but trade plan prefers holding — manual review."
                ),
                observed={
                    "profit_capture_ratio": facts.profit_capture_ratio,
                    "target_pct": target,
                    "plan_notes": facts.plan_override_notes,
                },
            )
        )
    elif facts.profit_capture_ratio >= target:
        results.append(
            _result(
                playbook,
                "profit.manage_winner",
                RuleStatus.DUE,
                "profit_target_reached",
                (
                    f"Profit capture {facts.profit_capture_ratio:.0%} ≥ target {target:.0%} "
                    f"(basis={facts.credit_provenance.value}). "
                    "Educational default — not a guarantee."
                ),
                observed={
                    "profit_capture_ratio": facts.profit_capture_ratio,
                    "target_pct": target,
                    "pnl_open": facts.pnl_open,
                    "original_credit": facts.original_credit,
                },
            )
        )
    elif facts.profit_capture_ratio >= target * 0.8:
        results.append(
            _result(
                playbook,
                "profit.manage_winner",
                RuleStatus.WATCH,
                "profit_approaching_target",
                f"Profit capture {facts.profit_capture_ratio:.0%} approaching target {target:.0%}.",
                observed={
                    "profit_capture_ratio": facts.profit_capture_ratio,
                    "target_pct": target,
                },
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "profit.manage_winner",
                RuleStatus.PASS,
                "below_profit_target",
                f"Profit capture {facts.profit_capture_ratio:.0%} below target {target:.0%}.",
                observed={
                    "profit_capture_ratio": facts.profit_capture_ratio,
                    "target_pct": target,
                },
            )
        )

    # time.manage_at_dte
    manage_dte = settings.manage_at_dte
    if facts.dte is None:
        results.append(
            _result(
                playbook,
                "time.manage_at_dte",
                RuleStatus.WATCH,
                "missing_dte",
                "DTE unknown; time-management rule cannot fully evaluate.",
                observed={"manage_at_dte": manage_dte},
                data_quality_notes=["missing_dte"],
            )
        )
    elif facts.dte <= manage_dte:
        strength = (
            "stronger attention (undefined risk)"
            if facts.risk_class == RiskClass.UNDEFINED
            else "review"
        )
        results.append(
            _result(
                playbook,
                "time.manage_at_dte",
                RuleStatus.DUE,
                "dte_threshold_crossed",
                f"DTE={facts.dte} ≤ manage-at {manage_dte}; {strength}. Educational preset.",
                observed={
                    "dte": facts.dte,
                    "manage_at_dte": manage_dte,
                    "risk_class": facts.risk_class.value,
                },
            )
        )
    elif facts.dte <= manage_dte + 5:
        results.append(
            _result(
                playbook,
                "time.manage_at_dte",
                RuleStatus.WATCH,
                "dte_approaching",
                f"DTE={facts.dte} approaching manage-at {manage_dte}.",
                observed={"dte": facts.dte, "manage_at_dte": manage_dte},
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "time.manage_at_dte",
                RuleStatus.PASS,
                "dte_ok",
                f"DTE={facts.dte} above manage-at {manage_dte}.",
                observed={"dte": facts.dte, "manage_at_dte": manage_dte},
            )
        )

    # size.small_position — never compare market_value/NLV to undefined BPR cap.
    if facts.risk_class == RiskClass.DEFINED:
        cap = settings.defined_risk_cap_pct
        if facts.size_ratio is None or facts.size_basis != "max_loss_nlv":
            results.append(
                _result(
                    playbook,
                    "size.small_position",
                    RuleStatus.WATCH,
                    "size_unknown",
                    (
                        "Defined-risk size unknown (missing NLV or max loss). "
                        "Cannot verify small-size preset."
                    ),
                    observed={
                        "cap_pct": cap,
                        "size_basis": facts.size_basis,
                        "market_value_nlv_ratio": facts.market_value_nlv_ratio,
                    },
                    data_quality_notes=["missing_size_inputs"],
                )
            )
        elif facts.size_ratio > cap:
            results.append(
                _result(
                    playbook,
                    "size.small_position",
                    RuleStatus.WATCH,
                    "size_above_cap",
                    (
                        f"Defined max-loss/NLV ratio {facts.size_ratio:.1%} exceeds "
                        f"cap {cap:.1%}. Educational small-size preset."
                    ),
                    observed={
                        "size_ratio": facts.size_ratio,
                        "cap_pct": cap,
                        "size_basis": facts.size_basis,
                    },
                )
            )
        else:
            results.append(
                _result(
                    playbook,
                    "size.small_position",
                    RuleStatus.PASS,
                    "size_within_cap",
                    f"Defined max-loss/NLV ratio {facts.size_ratio:.1%} within cap {cap:.1%}.",
                    observed={
                        "size_ratio": facts.size_ratio,
                        "cap_pct": cap,
                        "size_basis": facts.size_basis,
                    },
                )
            )
    else:
        # Undefined risk: only true strategy BPR may be compared to undefined_bpr_cap_pct.
        cap = settings.undefined_bpr_cap_pct
        if facts.size_ratio is None or facts.size_basis != "strategy_bpr_nlv":
            results.append(
                _result(
                    playbook,
                    "size.small_position",
                    RuleStatus.WATCH,
                    "size_unknown",
                    (
                        "Undefined-risk BPR size unknown (per-strategy BPR not available). "
                        "Market-value/NLV is informational only and is not compared to the BPR cap."
                    ),
                    observed={
                        "cap_pct": cap,
                        "size_basis": facts.size_basis,
                        "size_ratio": facts.size_ratio,
                        "market_value_nlv_ratio": facts.market_value_nlv_ratio,
                        "missing_bpr": True,
                    },
                    data_quality_notes=["missing_bpr"],
                )
            )
        elif facts.size_ratio > cap:
            results.append(
                _result(
                    playbook,
                    "size.small_position",
                    RuleStatus.DUE,
                    "size_above_cap",
                    (
                        f"Strategy BPR/NLV ratio {facts.size_ratio:.1%} exceeds "
                        f"cap {cap:.1%}. Educational small-size preset."
                    ),
                    observed={
                        "size_ratio": facts.size_ratio,
                        "cap_pct": cap,
                        "size_basis": facts.size_basis,
                    },
                )
            )
        else:
            results.append(
                _result(
                    playbook,
                    "size.small_position",
                    RuleStatus.PASS,
                    "size_within_cap",
                    f"Strategy BPR/NLV ratio {facts.size_ratio:.1%} within cap {cap:.1%}.",
                    observed={
                        "size_ratio": facts.size_ratio,
                        "cap_pct": cap,
                        "size_basis": facts.size_basis,
                    },
                )
            )

    # tested.side_review
    if facts.tested_side in {None, "unknown"}:
        results.append(
            _result(
                playbook,
                "tested.side_review",
                RuleStatus.WATCH,
                "tested_side_unknown",
                "Tested side could not be determined (missing delta/spot).",
                observed={
                    "tested_side": facts.tested_side,
                    "threshold": settings.tested_delta_threshold,
                },
                data_quality_notes=["missing_tested_side_inputs"],
            )
        )
    elif facts.tested_side == "untested":
        results.append(
            _result(
                playbook,
                "tested.side_review",
                RuleStatus.PASS,
                "untested",
                "Short legs appear untested relative to delta threshold.",
                observed={
                    "tested_side": facts.tested_side,
                    "short_call_delta": facts.short_call_delta,
                    "short_put_delta": facts.short_put_delta,
                    "threshold": settings.tested_delta_threshold,
                },
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "tested.side_review",
                RuleStatus.DUE,
                "tested_side_active",
                f"Tested side={facts.tested_side}; review defense / adjustments.",
                observed={
                    "tested_side": facts.tested_side,
                    "short_call_delta": facts.short_call_delta,
                    "short_put_delta": facts.short_put_delta,
                    "threshold": settings.tested_delta_threshold,
                },
            )
        )

    # roll.untested_side
    if facts.tested_side in {"call", "put"} and facts.strategy_type in {
        "Iron Condor",
        "Iron Butterfly",
        "Short Strangle",
        "Jade Lizard",
    }:
        untested = "put" if facts.tested_side == "call" else "call"
        results.append(
            _result(
                playbook,
                "roll.untested_side",
                RuleStatus.WATCH,
                "consider_untested_roll",
                (
                    f"Tested={facts.tested_side}; strategy may allow reviewing a credit roll "
                    f"on the untested {untested} side. No strikes/quotes invented."
                ),
                observed={"tested_side": facts.tested_side, "untested_side": untested},
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "roll.untested_side",
                RuleStatus.NOT_APPLICABLE
                if facts.tested_side in {None, "unknown", "untested", "both"}
                else RuleStatus.PASS,
                "untested_roll_not_applicable",
                "No untested-side roll consideration for current state/strategy.",
                observed={"tested_side": facts.tested_side},
            )
        )

    # roll.credit_only
    debit_plan = any("debit rolls" in note for note in facts.plan_override_notes)
    if debit_plan:
        results.append(
            _result(
                playbook,
                "roll.credit_only",
                RuleStatus.WATCH,
                "plan_overrides_credit_only",
                "Trade plan may allow debit rolls; playbook default remains credit-only advisory.",
                observed={
                    "credit_only_preference": settings.credit_only_rolls,
                    "plan_notes": facts.plan_override_notes,
                },
            )
        )
    elif settings.credit_only_rolls:
        results.append(
            _result(
                playbook,
                "roll.credit_only",
                RuleStatus.PASS,
                "credit_only_default",
                (
                    "Credit-only roll preference active. Without a quoted chain candidate, "
                    "roll economics remain unknown and cannot be verified."
                ),
                observed={"credit_only_preference": True, "roll_credit_known": False},
            )
        )
    else:
        results.append(
            _result(
                playbook,
                "roll.credit_only",
                RuleStatus.WATCH,
                "credit_only_disabled",
                "Credit-only preference disabled in settings; still never invents roll quotes.",
                observed={"credit_only_preference": False},
            )
        )

    return results


def _before_risk(risk: StrategyRisk | None, facts: StrategyMechanicsFacts) -> LocalRiskCompare:
    if risk is None:
        return LocalRiskCompare(
            current_pnl=facts.pnl_open,
            note="Strategy risk unavailable",
        )
    return LocalRiskCompare(
        current_pnl=facts.pnl_open if facts.pnl_open is not None else risk.current_pnl,
        max_loss=risk.max_loss,
        max_profit=risk.max_profit,
        total_delta=risk.combined.delta,
        total_theta=risk.combined.theta,
        defined_risk=risk.defined_risk,
    )


def generate_candidates(
    facts: StrategyMechanicsFacts,
    rules: list[RuleResult],
    *,
    risk: StrategyRisk | None,
    settings: MechanicsSettings,
) -> list[AdvisoryCandidate]:
    by_id = {rule.rule_id: rule for rule in rules}
    candidates: list[AdvisoryCandidate] = []
    before = _before_risk(risk, facts)

    if not facts.supported:
        candidates.append(
            AdvisoryCandidate(
                candidate_id="manual-review:unsupported",
                kind=CandidateKind.MANUAL_REVIEW,
                rule_hits=["gate.strategy_supported"],
                missing_inputs=[],
                blocking_reasons=["unsupported_strategy"],
                before_risk=before,
                after_risk=None,
                explanation=(
                    f"{facts.strategy_type} is outside {PLAYBOOK_ID_V1}. "
                    "No mechanics advisory candidates beyond manual review."
                ),
                priority=10,
            )
        )
        return candidates

    data_gate = by_id.get("gate.data_quality")
    data_blocked = data_gate is not None and data_gate.status == RuleStatus.BLOCKED
    missing_credit = facts.original_credit is None
    catalysts_unknown = facts.catalyst_availability == CatalystAvailability.UNKNOWN

    profit = by_id.get("profit.manage_winner")
    time_rule = by_id.get("time.manage_at_dte")
    tested = by_id.get("tested.side_review")
    size = by_id.get("size.small_position")
    event = by_id.get("gate.event_exposure")
    assign = by_id.get("gate.assignment_expiration")
    untested_roll = by_id.get("roll.untested_side")
    credit_only = by_id.get("roll.credit_only")

    def _event_blockers() -> list[str]:
        if catalysts_unknown:
            return ["catalysts_unknown"]
        if (
            event
            and event.status == RuleStatus.WATCH
            and event.reason_code == "high_impact_catalyst"
        ):
            return ["high_impact_catalyst_review"]
        return []

    # Close candidate for profit target
    if profit and profit.status == RuleStatus.DUE:
        blockers: list[str] = []
        missing: list[str] = []
        if data_blocked:
            blockers.append("data_quality_block")
        blockers.extend(_event_blockers())
        after = LocalRiskCompare(
            current_pnl=0.0,
            max_loss=0.0,
            max_profit=0.0,
            total_delta=0.0,
            total_theta=0.0,
            defined_risk=True,
            note="After full close: residual options risk removed (local comparison only).",
        )
        kind = CandidateKind.CLOSE if not blockers else CandidateKind.MANUAL_REVIEW
        candidates.append(
            AdvisoryCandidate(
                candidate_id=(
                    "close:profit-target"
                    if kind == CandidateKind.CLOSE
                    else "manual-review:profit-target"
                ),
                kind=kind,
                rule_hits=["profit.manage_winner"],
                missing_inputs=missing,
                blocking_reasons=blockers,
                before_risk=before,
                after_risk=after if kind == CandidateKind.CLOSE else None,
                explanation=profit.explanation,
                priority=20 if kind == CandidateKind.CLOSE else 25,
            )
        )
    elif (
        profit and profit.status == RuleStatus.WATCH and profit.reason_code == "plan_conflict_hold"
    ):
        candidates.append(
            AdvisoryCandidate(
                candidate_id="manual-review:plan-conflict",
                kind=CandidateKind.MANUAL_REVIEW,
                rule_hits=["profit.manage_winner"],
                missing_inputs=[],
                blocking_reasons=["trade_plan_conflict"],
                before_risk=before,
                after_risk=None,
                explanation=profit.explanation,
                priority=30,
            )
        )
    elif profit and profit.status == RuleStatus.BLOCKED:
        candidates.append(
            AdvisoryCandidate(
                candidate_id="manual-review:missing-credit",
                kind=CandidateKind.MANUAL_REVIEW,
                rule_hits=["profit.manage_winner"],
                missing_inputs=["original_credit"],
                blocking_reasons=["missing_opening_credit"],
                before_risk=before,
                after_risk=None,
                explanation=profit.explanation,
                priority=40,
            )
        )

    # Time management → close/reduce/manual
    if time_rule and time_rule.status == RuleStatus.DUE:
        blockers = []
        if data_blocked:
            blockers.append("data_quality_block")
        blockers.extend(_event_blockers())
        if missing_credit and facts.risk_class == RiskClass.UNDEFINED:
            blockers.append("missing_credit_for_undefined_time_mgmt")
        kind = (
            CandidateKind.CLOSE
            if facts.risk_class == RiskClass.DEFINED and not blockers
            else CandidateKind.MANUAL_REVIEW
            if blockers or facts.risk_class == RiskClass.UNDEFINED
            else CandidateKind.CLOSE
        )
        if facts.risk_class == RiskClass.UNDEFINED and not blockers:
            kind = CandidateKind.MANUAL_REVIEW
        after = None
        if kind == CandidateKind.CLOSE:
            after = LocalRiskCompare(
                current_pnl=0.0,
                max_loss=0.0,
                max_profit=0.0,
                total_delta=0.0,
                total_theta=0.0,
                defined_risk=True,
                note="After full close: residual options risk removed (local comparison only).",
            )
        candidates.append(
            AdvisoryCandidate(
                candidate_id=f"{kind.value}:dte-management",
                kind=kind,
                rule_hits=["time.manage_at_dte", "risk.defined_vs_undefined"],
                missing_inputs=["original_credit"] if missing_credit else [],
                blocking_reasons=blockers
                + (
                    ["undefined_risk_time_review"]
                    if (
                        facts.risk_class == RiskClass.UNDEFINED
                        and kind == CandidateKind.MANUAL_REVIEW
                    )
                    else []
                ),
                before_risk=before,
                after_risk=after,
                explanation=time_rule.explanation,
                priority=35 if facts.risk_class == RiskClass.UNDEFINED else 28,
            )
        )

    # Reduce only when size_above_cap from true defined max-loss or strategy BPR.
    if (
        size
        and size.status in {RuleStatus.DUE, RuleStatus.WATCH}
        and size.reason_code == "size_above_cap"
    ):
        blockers = []
        if data_blocked:
            blockers.append("data_quality_block")
        blockers.extend(_event_blockers())
        kind = CandidateKind.REDUCE if not blockers else CandidateKind.MANUAL_REVIEW
        after = None
        if kind == CandidateKind.REDUCE and before.max_loss is not None:
            after = LocalRiskCompare(
                current_pnl=None,
                max_loss=before.max_loss * 0.5 if before.max_loss is not None else None,
                max_profit=before.max_profit * 0.5 if before.max_profit is not None else None,
                total_delta=before.total_delta * 0.5 if before.total_delta is not None else None,
                total_theta=before.total_theta * 0.5 if before.total_theta is not None else None,
                defined_risk=before.defined_risk,
                note="Illustrative 50% size reduction — not an order. Exact fill unknown.",
            )
        candidates.append(
            AdvisoryCandidate(
                candidate_id=f"{kind.value}:size-cap",
                kind=kind,
                rule_hits=["size.small_position"],
                missing_inputs=[],
                blocking_reasons=blockers,
                before_risk=before,
                after_risk=after,
                explanation=size.explanation,
                priority=45,
            )
        )

    # Tested side + roll review
    if tested and tested.status == RuleStatus.DUE:
        roll_blockers = ["roll_economics_unknown"]
        if settings.credit_only_rolls:
            roll_blockers.append("credit_only_unverified")
        if data_blocked:
            roll_blockers.append("data_quality_block")
        hits = ["tested.side_review"]
        if untested_roll and untested_roll.status == RuleStatus.WATCH:
            hits.append("roll.untested_side")
        if credit_only:
            hits.append("roll.credit_only")
        candidates.append(
            AdvisoryCandidate(
                candidate_id="roll-review:tested-side",
                kind=CandidateKind.ROLL_REVIEW,
                rule_hits=hits,
                missing_inputs=["option_chain_quote", "candidate_roll_credit"],
                blocking_reasons=roll_blockers,
                before_risk=before,
                after_risk=LocalRiskCompare(
                    note=(
                        "After-risk unknown without a quoted roll candidate. "
                        "Never invent strikes, expirations, or credits."
                    )
                ),
                explanation=(
                    f"{tested.explanation} Roll review only — no chain candidate supplied; "
                    "economics and after-risk remain unknown."
                ),
                priority=32,
            )
        )

    # Assignment
    if assign and assign.status == RuleStatus.DUE:
        candidates.append(
            AdvisoryCandidate(
                candidate_id="manual-review:expiration",
                kind=CandidateKind.MANUAL_REVIEW,
                rule_hits=["gate.assignment_expiration"],
                missing_inputs=[],
                blocking_reasons=["expiration_imminent"],
                before_risk=before,
                after_risk=None,
                explanation=assign.explanation,
                priority=15,
            )
        )

    # Default hold when nothing urgent and data ok
    urgent = {
        RuleStatus.DUE,
        RuleStatus.BLOCKED,
    }
    has_urgent = any(
        by_id.get(rid) and by_id[rid].status in urgent
        for rid in (
            "profit.manage_winner",
            "time.manage_at_dte",
            "tested.side_review",
            "gate.assignment_expiration",
            "size.small_position",
        )
    )
    if not has_urgent and not data_blocked:
        candidates.append(
            AdvisoryCandidate(
                candidate_id="hold:within-mechanics",
                kind=CandidateKind.HOLD,
                rule_hits=[
                    rid
                    for rid in ("profit.manage_winner", "time.manage_at_dte", "tested.side_review")
                    if by_id.get(rid) and by_id[rid].status == RuleStatus.PASS
                ],
                missing_inputs=[],
                blocking_reasons=[],
                before_risk=before,
                after_risk=before.model_copy(
                    update={"note": "Hold: risk profile unchanged (local comparison)."}
                ),
                explanation=(
                    "No due mechanics thresholds crossed; "
                    "hold remains consistent with playbook defaults."
                ),
                priority=90,
            )
        )
    elif data_blocked and not candidates:
        candidates.append(
            AdvisoryCandidate(
                candidate_id="manual-review:data-quality",
                kind=CandidateKind.MANUAL_REVIEW,
                rule_hits=["gate.data_quality"],
                missing_inputs=list(facts.data_quality_flags),
                blocking_reasons=["data_quality_block"],
                before_risk=before,
                after_risk=None,
                explanation="Critical data-quality issues — abstain from mechanical actions.",
                priority=12,
            )
        )

    # Stable sort: priority then candidate_id. Never emit ADD (not in CandidateKind).
    candidates.sort(key=lambda item: (item.priority, item.candidate_id))
    return candidates


def evaluate_mechanics(
    strategy: StrategySnapshot,
    *,
    risk: StrategyRisk | None = None,
    market: dict[str, Any] | None = None,
    trade_plan: dict[str, Any] | None = None,
    account_nlv: float | None = None,
    account_buying_power: float | None = None,
    catalysts: list[dict[str, Any]] | None = None,
    settings: MechanicsSettings | None = None,
    playbook: Playbook | None = None,
    now: datetime | None = None,
) -> MechanicsEvaluation:
    """Pure offline evaluation — no network, no Codex, no broker writes."""

    settings = settings or MechanicsSettings()
    playbook = playbook or get_playbook(settings.playbook_id)
    evaluated_at = now or datetime.now(UTC)

    if not settings.enabled:
        facts = StrategyMechanicsFacts(
            strategy_id=strategy.strategy_id,
            strategy_type=strategy.strategy_type,
            underlying=strategy.underlying,
            supported=False,
            risk_class=RiskClass.UNKNOWN,
            data_quality_flags=["mechanics_disabled"],
        )
        evaluation = MechanicsEvaluation(
            strategy_id=strategy.strategy_id,
            playbook_id=playbook.playbook_id,
            playbook_version=playbook.version,
            shadow_mode=settings.shadow_mode,
            enabled=False,
            evaluated_at=evaluated_at,
            facts=facts,
            rules=[],
            candidates=[],
            sources=playbook.sources,
            settings_snapshot=settings.model_dump(mode="json"),
        )
        return evaluation.model_copy(update={"fingerprint": evaluation_fingerprint(evaluation)})

    # Local risk if not supplied
    if risk is None:
        spot = None
        if market and isinstance(market.get("price"), (int, float)):
            spot = float(market["price"])
        risk = RiskService().strategy_risk(strategy, underlying_price=spot)

    facts = derive_facts(
        strategy,
        risk=risk,
        market=market,
        trade_plan=trade_plan,
        account_nlv=account_nlv,
        account_buying_power=account_buying_power,
        catalysts=catalysts,
        settings=settings,
        now=evaluated_at,
    )
    rules = evaluate_rules(facts, playbook=playbook, settings=settings)
    candidates = generate_candidates(facts, rules, risk=risk, settings=settings)

    # Attach source rule citations to sources already on playbook
    evaluation = MechanicsEvaluation(
        strategy_id=strategy.strategy_id,
        playbook_id=playbook.playbook_id,
        playbook_version=playbook.version,
        shadow_mode=settings.shadow_mode,
        enabled=True,
        evaluated_at=evaluated_at,
        facts=facts,
        rules=rules,
        candidates=candidates,
        sources=playbook.sources,
        settings_snapshot=settings.model_dump(mode="json"),
    )
    return evaluation.model_copy(update={"fingerprint": evaluation_fingerprint(evaluation)})


class MechanicsService:
    """Domain service for settings, evaluation, and offline replay."""

    def __init__(
        self,
        database: PositionPilotDatabase,
        *,
        risk_service: RiskService | None = None,
        plans_service: PlansService | None = None,
        clock: Any | None = None,
    ) -> None:
        self.database = database
        self.risk_service = risk_service or RiskService()
        self.plans_service = plans_service or PlansService(database)
        self._clock = clock or (lambda: datetime.now(UTC))

    def get_settings(self) -> MechanicsSettings:
        # Distinguish missing key (default {}) from present malformed falsy values
        # like [], "", 0, False — never coerce those into enabled defaults.
        if not self.database.has_setting(SETTINGS_KEY):
            return MechanicsSettings()
        stored = self.database.get_setting(SETTINGS_KEY, default=None)
        if not isinstance(stored, dict):
            logger.warning(
                "Mechanics settings stored value is not an object; disabling until corrected"
            )
            return MechanicsSettings(enabled=False, shadow_mode=True, advisory_only=True)
        try:
            return MechanicsSettings.model_validate(stored)
        except Exception:
            # Fail closed: disable until stored payload is corrected. Do not overwrite.
            logger.warning(
                "Invalid mechanics settings payload; disabling mechanics until corrected"
            )
            return MechanicsSettings(enabled=False, shadow_mode=True, advisory_only=True)

    def public_settings(self) -> dict[str, Any]:
        settings = self.get_settings()
        playbook = get_playbook(settings.playbook_id)
        return {
            **settings.model_dump(mode="json"),
            "available_playbooks": [
                {
                    "playbook_id": playbook.playbook_id,
                    "version": playbook.version,
                    "title": playbook.title,
                }
            ],
            "disclaimer": playbook.disclaimer,
            "execution_boundary": ("Advisory only. Execution remains manual in tastytrade."),
        }

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings().model_dump(mode="json")
        allowed = {
            "enabled",
            "shadow_mode",
            "playbook_id",
            "profit_target_pct",
            "manage_at_dte",
            "tested_delta_threshold",
            "defined_risk_cap_pct",
            "undefined_bpr_cap_pct",
            "credit_only_rolls",
        }
        for key, value in payload.items():
            if key in allowed and value is not None:
                current[key] = value
        current["advisory_only"] = True
        try:
            settings = MechanicsSettings.model_validate(current)
        except Exception as error:
            raise ValueError(f"Invalid mechanics settings: {error}") from error
        serialized = settings.model_dump(mode="json")
        previous = self.database.get_setting(SETTINGS_KEY, {}) or {}
        self.database.set_setting(SETTINGS_KEY, serialized)
        # Audit settings changes (immutable audit stream). Write already succeeded;
        # log audit failures without claiming success or rolling back settings.
        try:
            self.plans_service.record_event(
                strategy_id="__mechanics_settings__",
                action="mechanics_settings_updated",
                summary="Mechanics settings updated",
                details={"previous": previous, "current": serialized},
            )
        except Exception:
            logger.exception(
                "Mechanics settings persisted but audit event failed (no account identifiers)"
            )
        return self.public_settings()

    def evaluate_strategy(
        self,
        strategy: StrategySnapshot,
        *,
        portfolio: PortfolioSnapshot | None = None,
        risk: StrategyRisk | None = None,
        market: dict[str, Any] | None = None,
        trade_plan: dict[str, Any] | None = None,
        catalysts: list[dict[str, Any]] | None = None,
        settings: MechanicsSettings | None = None,
    ) -> MechanicsEvaluation:
        settings = settings or self.get_settings()
        if trade_plan is None:
            plan = self.plans_service.get_trade_plan(strategy.strategy_id)
            trade_plan = plan.model_dump(mode="json") if plan else None

        account_nlv = None
        account_bp = None
        if portfolio is not None:
            # Only the strategy's own account capital — never aggregate portfolio totals.
            for account in portfolio.accounts:
                if account.account_id == strategy.account_id:
                    account_nlv = account.net_liquidating_value
                    account_bp = account.buying_power
                    break

        if risk is None:
            spot = None
            if market and isinstance(market.get("price"), (int, float)):
                spot = float(market["price"])
            risk = self.risk_service.strategy_risk(strategy, underlying_price=spot)

        return evaluate_mechanics(
            strategy,
            risk=risk,
            market=market,
            trade_plan=trade_plan,
            account_nlv=account_nlv,
            account_buying_power=account_bp,
            catalysts=catalysts,
            settings=settings,
            now=self._clock(),
        )

    def replay(
        self,
        strategy: StrategySnapshot,
        *,
        risk: StrategyRisk | None = None,
        market: dict[str, Any] | None = None,
        trade_plan: dict[str, Any] | None = None,
        account_nlv: float | None = None,
        account_buying_power: float | None = None,
        catalysts: list[dict[str, Any]] | None = None,
        settings: MechanicsSettings | None = None,
        playbook_id: str = PLAYBOOK_ID_V1,
    ) -> MechanicsEvaluation:
        """Fixture-driven offline evaluator (no network / Codex)."""

        settings = settings or MechanicsSettings(playbook_id=playbook_id)
        return evaluate_mechanics(
            strategy,
            risk=risk,
            market=market,
            trade_plan=trade_plan,
            account_nlv=account_nlv,
            account_buying_power=account_buying_power,
            catalysts=catalysts,
            settings=settings,
            playbook=get_playbook(playbook_id),
            now=self._clock(),
        )

    def playbook_public(self, playbook_id: str = PLAYBOOK_ID_V1) -> dict[str, Any]:
        playbook = get_playbook(playbook_id)
        return playbook.model_dump(mode="json")
