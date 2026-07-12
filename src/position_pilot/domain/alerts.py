"""Alert center domain: risk, catalyst, recommendation, and provider-health."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from ..persistence.sqlite import PositionPilotDatabase
from ..providers.codex import RecommendationAction, RecommendationRisk


class AlertCategory(StrEnum):
    RISK = "risk"
    CATALYST = "catalyst"
    RECOMMENDATION = "recommendation"
    PROVIDER_HEALTH = "provider_health"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertResolution(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    SNOOZED = "snoozed"
    MUTED = "muted"
    RESOLVED = "resolved"


class AlertRecord(BaseModel):
    alert_id: str
    category: AlertCategory
    severity: AlertSeverity
    alert_type: str
    title: str
    summary: str
    account_id: str | None = None
    symbol: str | None = None
    strategy_type: str | None = None
    subject_type: str | None = None
    subject_id: str | None = None
    source: str
    created_at: datetime
    updated_at: datetime
    resolution: AlertResolution = AlertResolution.OPEN
    snoozed_until: datetime | None = None
    mute_rule: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class MuteRule(BaseModel):
    rule_id: str
    category: AlertCategory | None = None
    alert_type: str | None = None
    symbol: str | None = None
    strategy_type: str | None = None
    created_at: datetime


def recommendation_severity(urgency: int | None, risk: RecommendationRisk | None) -> AlertSeverity:
    if risk == RecommendationRisk.CRITICAL or (urgency or 0) >= 5:
        return AlertSeverity.CRITICAL
    if risk == RecommendationRisk.HIGH or (urgency or 0) >= 4:
        return AlertSeverity.HIGH
    if risk == RecommendationRisk.MODERATE or (urgency or 0) >= 3:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


def privacy_safe_notification_title(
    *,
    symbol: str | None,
    strategy_type: str | None,
    alert_type: str,
) -> str:
    parts = [part for part in (symbol, strategy_type, alert_type) if part]
    return " · ".join(parts) if parts else alert_type


def privacy_safe_notification_body(
    alert: AlertRecord,
    *,
    rich_preview: bool,
) -> str:
    if not rich_preview:
        return privacy_safe_notification_title(
            symbol=alert.symbol,
            strategy_type=alert.strategy_type,
            alert_type=alert.alert_type,
        )
    # Rich preview still excludes account ids, quantities, balances, and P/L.
    return f"{alert.title}: {alert.summary}"[:240]


class AlertService:
    """Persist and resolve account-safe alerts."""

    def __init__(
        self,
        database: PositionPilotDatabase,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self._clock = clock or (lambda: datetime.now(UTC))

    def list_alerts(
        self,
        *,
        account_id: str = "all",
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[AlertRecord]:
        rows = self.database.list_alerts(
            account_id=account_id,
            include_resolved=include_resolved,
            limit=limit,
        )
        now = self._clock()
        alerts = [AlertRecord.model_validate(row) for row in rows]
        visible: list[AlertRecord] = []
        for alert in alerts:
            if alert.resolution == AlertResolution.SNOOZED and alert.snoozed_until:
                until = alert.snoozed_until
                if until.tzinfo is None:
                    until = until.replace(tzinfo=UTC)
                if until > now:
                    continue
            if alert.resolution == AlertResolution.MUTED:
                continue
            if self._is_muted(alert):
                continue
            visible.append(alert)
        return visible

    def raise_alert(
        self,
        *,
        category: AlertCategory,
        severity: AlertSeverity,
        alert_type: str,
        title: str,
        summary: str,
        source: str,
        account_id: str | None = None,
        symbol: str | None = None,
        strategy_type: str | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> AlertRecord | None:
        if dedupe_key:
            existing = self.database.find_open_alert(dedupe_key)
            if existing:
                return AlertRecord.model_validate(existing)
        now = self._clock()
        alert = AlertRecord(
            alert_id=str(uuid4()),
            category=category,
            severity=severity,
            alert_type=alert_type,
            title=title[:200],
            summary=summary[:1_000],
            account_id=account_id,
            symbol=symbol,
            strategy_type=strategy_type,
            subject_type=subject_type,
            subject_id=subject_id,
            source=source,
            created_at=now,
            updated_at=now,
            payload={**(payload or {}), "dedupe_key": dedupe_key},
        )
        if self._is_muted(alert):
            return None
        self.database.insert_alert(alert.model_dump(mode="json"))
        return alert

    def raise_recommendation_change(
        self,
        *,
        action: RecommendationAction,
        urgency: int,
        risk: RecommendationRisk,
        account_id: str | None,
        symbol: str | None,
        strategy_type: str | None,
        subject_type: str,
        subject_id: str,
        recommendation_id: str,
    ) -> AlertRecord | None:
        return self.raise_alert(
            category=AlertCategory.RECOMMENDATION,
            severity=recommendation_severity(urgency, risk),
            alert_type="recommendation_change",
            title=f"{symbol or 'Portfolio'} recommendation changed",
            summary=f"{action.value} · urgency {urgency} · {risk.value}",
            source="recommendation-service",
            account_id=account_id,
            symbol=symbol,
            strategy_type=strategy_type,
            subject_type=subject_type,
            subject_id=subject_id,
            payload={"recommendation_id": recommendation_id},
            dedupe_key=f"rec-change:{subject_type}:{subject_id}:{action.value}:{urgency}:{risk.value}",
        )

    def raise_provider_health(
        self,
        *,
        provider: str,
        status: str,
        detail: str,
    ) -> AlertRecord | None:
        severity = (
            AlertSeverity.CRITICAL
            if status in {"signed_out", "unavailable", "not_installed"}
            else AlertSeverity.WARNING
        )
        return self.raise_alert(
            category=AlertCategory.PROVIDER_HEALTH,
            severity=severity,
            alert_type=f"provider_{status}",
            title=f"{provider} {status.replace('_', ' ')}",
            summary=detail[:500],
            source=provider,
            payload={"provider": provider, "status": status},
            dedupe_key=f"provider:{provider}:{status}",
        )

    def acknowledge(self, alert_id: str) -> AlertRecord:
        return self._set_resolution(alert_id, AlertResolution.ACKNOWLEDGED)

    def resolve(self, alert_id: str) -> AlertRecord:
        return self._set_resolution(alert_id, AlertResolution.RESOLVED)

    def snooze(self, alert_id: str, *, minutes: int = 60) -> AlertRecord:
        alert = self._require(alert_id)
        now = self._clock()
        alert.resolution = AlertResolution.SNOOZED
        alert.snoozed_until = now + timedelta(minutes=max(1, minutes))
        alert.updated_at = now
        self.database.update_alert(alert.model_dump(mode="json"))
        return alert

    def mute_by_rule(
        self,
        *,
        category: AlertCategory | str | None = None,
        alert_type: str | None = None,
        symbol: str | None = None,
        strategy_type: str | None = None,
    ) -> MuteRule:
        rule = MuteRule(
            rule_id=str(uuid4()),
            category=AlertCategory(category) if category else None,
            alert_type=alert_type,
            symbol=symbol.upper() if symbol else None,
            strategy_type=strategy_type,
            created_at=self._clock(),
        )
        self.database.insert_mute_rule(rule.model_dump(mode="json"))
        return rule

    def list_mute_rules(self) -> list[MuteRule]:
        return [MuteRule.model_validate(row) for row in self.database.list_mute_rules()]

    def _set_resolution(self, alert_id: str, resolution: AlertResolution) -> AlertRecord:
        alert = self._require(alert_id)
        alert.resolution = resolution
        alert.updated_at = self._clock()
        if resolution != AlertResolution.SNOOZED:
            alert.snoozed_until = None
        self.database.update_alert(alert.model_dump(mode="json"))
        return alert

    def _require(self, alert_id: str) -> AlertRecord:
        row = self.database.get_alert(alert_id)
        if row is None:
            raise KeyError(alert_id)
        return AlertRecord.model_validate(row)

    def _is_muted(self, alert: AlertRecord) -> bool:
        for rule in self.list_mute_rules():
            if rule.category and rule.category != alert.category:
                continue
            if rule.alert_type and rule.alert_type != alert.alert_type:
                continue
            if rule.symbol and rule.symbol != (alert.symbol or "").upper():
                continue
            if rule.strategy_type and rule.strategy_type != alert.strategy_type:
                continue
            return True
        return False
