"""Thesis, trade plan, and immutable audit history."""

from position_pilot.domain.plans import PlansService
from position_pilot.persistence.sqlite import PositionPilotDatabase


def test_thesis_and_trade_plan_persist_with_audit_history(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    service = PlansService(database)

    thesis = service.save_thesis(
        "strat-1",
        {
            "purpose": "Core equity exposure",
            "expected_duration": "Multi-year",
            "target_range": "180-220",
            "invalidation": "Break of 160 on weekly close",
            "income_or_hedge_intent": "Income via covered calls",
            "events_to_watch": ["earnings", "product launch"],
        },
    )
    plan = service.save_trade_plan(
        "strat-2",
        {
            "entry_thesis": "Sell premium into elevated IV",
            "intended_duration": "21-45 DTE",
            "profit_target": "50% of credit",
            "max_loss": "2x credit",
            "roll_criteria": "21 DTE or 21 delta",
            "event_exposure": "Avoid earnings week",
            "exit_deadline": "2026-08-15",
        },
    )

    assert thesis.strategy_id == "strat-1"
    assert plan.strategy_id == "strat-2"
    assert service.get_thesis("strat-1") is not None
    assert service.get_trade_plan("strat-2") is not None

    audit = service.audit_history("strat-1")
    assert len(audit) >= 1
    assert audit[0].action == "thesis_saved"
    assert "Core equity exposure" in audit[0].summary
