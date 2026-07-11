from datetime import UTC, datetime

from position_pilot.domain.portfolio import PortfolioService
from position_pilot.domain.snapshots import PositionHorizon, SnapshotState
from position_pilot.models import Account, Position, PositionType
from position_pilot.persistence.sqlite import PositionPilotDatabase


class FakePortfolioSource:
    def get_accounts(self) -> list[Account]:
        return [
            Account(
                account_number="5WT12345",
                account_type="Individual",
                nickname="Forrest trading",
            ),
            Account(account_number="5WT67890", account_type="IRA"),
        ]

    def get_account_balances(self, account_number: str) -> dict[str, float]:
        return {
            "net_liquidating_value": 10_000 if account_number.endswith("12345") else 20_000,
            "cash_balance": 2_000,
            "buying_power": 4_000,
        }

    def get_positions(self, account_number: str) -> list[Position]:
        direction = "Long" if account_number.endswith("12345") else "Short"
        return [
            Position(
                symbol="SPY",
                underlying_symbol="SPY",
                quantity=100,
                quantity_direction=direction,
                position_type=PositionType.EQUITY,
                mark_price=550,
                market_value=55_000,
                unrealized_pnl=1_250,
            )
        ]


def test_refresh_builds_one_atomic_snapshot_without_cross_account_grouping(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    service = PortfolioService(
        database=database,
        source=FakePortfolioSource(),
        clock=lambda: datetime(2026, 7, 11, 16, 30, tzinfo=UTC),
    )

    snapshot = service.refresh()

    assert snapshot.state is SnapshotState.LIVE
    assert len(snapshot.accounts) == 2
    assert len(snapshot.strategies) == 2
    assert {strategy.account_id for strategy in snapshot.strategies} == {
        account.account_id for account in snapshot.accounts
    }
    serialized = snapshot.model_dump_json()
    assert "5WT12345" not in serialized
    assert "5WT67890" not in serialized
    assert "Forrest trading" not in serialized
    assert database.latest_portfolio_snapshot() == snapshot


def test_refresh_failure_returns_the_last_snapshot_as_cached(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    service = PortfolioService(database=database, source=FakePortfolioSource())
    live = service.refresh()

    class OfflineSource(FakePortfolioSource):
        def get_accounts(self) -> list[Account]:
            raise OSError("network unreachable")

    cached = PortfolioService(database=database, source=OfflineSource()).refresh()

    assert cached.snapshot_id == live.snapshot_id
    assert cached.state is SnapshotState.CACHED
    assert cached.freshness.state == "stale"
    assert cached.notice is not None
    assert live.captured_at.isoformat() in cached.notice


def test_empty_provider_response_does_not_replace_a_valid_snapshot(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    live = PortfolioService(database=database, source=FakePortfolioSource()).refresh()

    class EmptySource(FakePortfolioSource):
        def get_accounts(self) -> list[Account]:
            return []

    cached = PortfolioService(database=database, source=EmptySource()).refresh()

    assert cached.snapshot_id == live.snapshot_id
    assert cached.state is SnapshotState.CACHED
    assert database.latest_portfolio_snapshot() == live


def test_primary_account_and_edited_horizon_are_durable(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    service = PortfolioService(database=database, source=FakePortfolioSource())
    snapshot = service.refresh()
    account_id = snapshot.accounts[0].account_id
    strategy_id = snapshot.strategies[0].strategy_id

    service.set_primary_account(account_id)
    updated = service.set_strategy_horizon(strategy_id, PositionHorizon.STRATEGIC)
    refreshed = service.refresh()

    assert service.primary_account_id() == account_id
    assert updated.horizon is PositionHorizon.STRATEGIC
    assert (
        next(item for item in refreshed.strategies if item.strategy_id == strategy_id).horizon
        is PositionHorizon.STRATEGIC
    )
