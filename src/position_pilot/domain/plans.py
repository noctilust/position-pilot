"""Trader-authored theses, trade plans, and immutable audit history."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..persistence.sqlite import PositionPilotDatabase


class Thesis(BaseModel):
    strategy_id: str
    purpose: str = ""
    expected_duration: str = ""
    target_range: str = ""
    invalidation: str = ""
    income_or_hedge_intent: str = ""
    events_to_watch: list[str] = Field(default_factory=list)
    updated_at: datetime


class TradePlan(BaseModel):
    strategy_id: str
    entry_thesis: str = ""
    intended_duration: str = ""
    profit_target: str = ""
    max_loss: str = ""
    roll_criteria: str = ""
    event_exposure: str = ""
    exit_deadline: str = ""
    updated_at: datetime


class AuditEvent(BaseModel):
    event_id: str
    strategy_id: str
    action: str
    summary: str
    recorded_at: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class PlansService:
    """Persist editable trader notes without touching brokerage data."""

    def __init__(self, database: PositionPilotDatabase) -> None:
        self.database = database

    def get_thesis(self, strategy_id: str) -> Thesis | None:
        payload = self.database.get_strategy_document("thesis", strategy_id)
        return Thesis.model_validate(payload) if payload else None

    def save_thesis(self, strategy_id: str, payload: dict[str, Any]) -> Thesis:
        thesis = Thesis(
            strategy_id=strategy_id,
            purpose=str(payload.get("purpose", "")),
            expected_duration=str(payload.get("expected_duration", "")),
            target_range=str(payload.get("target_range", "")),
            invalidation=str(payload.get("invalidation", "")),
            income_or_hedge_intent=str(payload.get("income_or_hedge_intent", "")),
            events_to_watch=list(payload.get("events_to_watch") or []),
            updated_at=datetime.now(UTC),
        )
        self.database.save_strategy_document("thesis", strategy_id, thesis.model_dump(mode="json"))
        self._audit(
            strategy_id,
            "thesis_saved",
            f"Thesis updated: {thesis.purpose or 'no purpose set'}",
            thesis.model_dump(mode="json"),
        )
        return thesis

    def get_trade_plan(self, strategy_id: str) -> TradePlan | None:
        payload = self.database.get_strategy_document("trade_plan", strategy_id)
        return TradePlan.model_validate(payload) if payload else None

    def save_trade_plan(self, strategy_id: str, payload: dict[str, Any]) -> TradePlan:
        plan = TradePlan(
            strategy_id=strategy_id,
            entry_thesis=str(payload.get("entry_thesis", "")),
            intended_duration=str(payload.get("intended_duration", "")),
            profit_target=str(payload.get("profit_target", "")),
            max_loss=str(payload.get("max_loss", "")),
            roll_criteria=str(payload.get("roll_criteria", "")),
            event_exposure=str(payload.get("event_exposure", "")),
            exit_deadline=str(payload.get("exit_deadline", "")),
            updated_at=datetime.now(UTC),
        )
        self.database.save_strategy_document(
            "trade_plan",
            strategy_id,
            plan.model_dump(mode="json"),
        )
        self._audit(
            strategy_id,
            "trade_plan_saved",
            f"Trade plan updated: {plan.entry_thesis or 'no thesis set'}",
            plan.model_dump(mode="json"),
        )
        return plan

    def audit_history(self, strategy_id: str) -> list[AuditEvent]:
        return [
            AuditEvent.model_validate(payload)
            for payload in self.database.list_audit_events(strategy_id)
        ]

    def record_event(
        self,
        strategy_id: str,
        action: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        return self._audit(strategy_id, action, summary, details or {})

    def _audit(
        self,
        strategy_id: str,
        action: str,
        summary: str,
        details: dict[str, Any],
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid4()),
            strategy_id=strategy_id,
            action=action,
            summary=summary,
            recorded_at=datetime.now(UTC),
            details=details,
        )
        self.database.append_audit_event(event.model_dump(mode="json"))
        return event
