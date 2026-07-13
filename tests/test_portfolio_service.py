from datetime import UTC, datetime

from position_pilot.domain.portfolio import PortfolioService
from position_pilot.domain.snapshots import FreshnessState, PositionHorizon, SnapshotState
from position_pilot.models import Account, Greeks, Position, PositionType
from position_pilot.persistence.sqlite import PositionPilotDatabase
from position_pilot.providers.contracts import ProviderHealth, ProviderState, ProviderValue
from position_pilot.providers.router import FieldRouter


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
    assert all(account.positions[0].delta == 1.0 for account in snapshot.accounts)
    assert all(
        account.positions[0].provenance["delta"].provider == "position-pilot"
        for account in snapshot.accounts
    )
    assert sorted(strategy.total_delta for strategy in snapshot.strategies) == [-100, 100]


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
    assert cached.freshness.state is FreshnessState.STALE
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


def test_missing_option_fields_use_routed_fallback_with_provenance(tmp_path) -> None:
    observed_at = datetime(2026, 7, 11, 16, 29, tzinfo=UTC)

    class OptionSource(FakePortfolioSource):
        def get_accounts(self) -> list[Account]:
            return [Account(account_number="5WT12345", account_type="Individual")]

        def get_positions(self, account_number: str) -> list[Position]:
            return [
                Position(
                    symbol="SPY   260821C00550000",
                    underlying_symbol="SPY",
                    quantity=1,
                    position_type=PositionType.EQUITY_OPTION,
                    average_open_price=3.0,
                    cost_basis=300.0,
                    greeks=Greeks(delta=0.4),
                    multiplier=100,
                )
            ]

    class EmptyProvider:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            return None

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    class OptionsProvider(EmptyProvider):
        name = "massive-options"

        def fetch(self, field: str, symbol: str) -> ProviderValue:
            value = 4.25 if field == "option.mark" else {"delta": 0.52, "theta": -0.08}
            return ProviderValue(
                field=field,
                value=value,
                provider=self.name,
                observed_at=observed_at,
            )

    providers = [EmptyProvider(), OptionsProvider()]
    router = FieldRouter(
        providers={provider.name: provider for provider in providers},
        routes={
            "option.mark": ["tastytrade", "massive-options"],
            "option.greeks": ["tastytrade", "massive-options"],
        },
    )
    service = PortfolioService(
        database=PositionPilotDatabase(tmp_path / "position-pilot.sqlite3"),
        source=OptionSource(),
        field_router=router,
    )

    snapshot = service.refresh()
    position = snapshot.accounts[0].positions[0]

    assert position.mark_price == 4.25
    assert position.market_value == 425.0
    assert position.unrealized_pnl == 125.0
    assert position.unrealized_pnl_percent == 125 / 300 * 100
    assert snapshot.totals.unrealized_pnl == 125.0
    assert position.delta == 0.4
    assert position.theta == -0.08
    assert position.provenance["mark_price"].provider == "massive-options"
    assert position.provenance["mark_price"].fallback_reason == ("tastytrade returned no value")
    assert position.provenance["delta"].provider == "tastytrade"
    assert snapshot.strategies[0].legs[0].provenance["theta"].provider == "massive-options"


def test_invalid_fallback_mark_does_not_record_mark_provenance(tmp_path) -> None:
    """Rejected fallback marks must not claim provenance for unchanged fields."""
    observed_at = datetime(2026, 7, 11, 16, 29, tzinfo=UTC)

    class OptionSource(FakePortfolioSource):
        def get_accounts(self) -> list[Account]:
            return [Account(account_number="5WT12345", account_type="Individual")]

        def get_positions(self, account_number: str) -> list[Position]:
            return [
                Position(
                    symbol="SPY   260821C00550000",
                    underlying_symbol="SPY",
                    quantity=1,
                    position_type=PositionType.EQUITY_OPTION,
                    average_open_price=3.0,
                    cost_basis=300.0,
                    mark_price=None,
                    market_value=0.0,
                    unrealized_pnl=0.0,
                    greeks=Greeks(delta=0.4, theta=-0.05, gamma=0.01, vega=0.1),
                    multiplier=100,
                )
            ]

    class EmptyProvider:
        name = "tastytrade"

        def fetch(self, field: str, symbol: str):
            return None

        def health(self) -> ProviderHealth:
            return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    class InvalidMarkProvider(EmptyProvider):
        name = "massive-options"

        def fetch(self, field: str, symbol: str) -> ProviderValue | None:
            if field != "option.mark":
                return None
            return ProviderValue(
                field=field,
                value=float("nan"),
                provider=self.name,
                observed_at=observed_at,
            )

    providers = [EmptyProvider(), InvalidMarkProvider()]
    router = FieldRouter(
        providers={provider.name: provider for provider in providers},
        routes={
            "option.mark": ["tastytrade", "massive-options"],
            "option.greeks": ["tastytrade", "massive-options"],
        },
    )
    service = PortfolioService(
        database=PositionPilotDatabase(tmp_path / "position-pilot.sqlite3"),
        source=OptionSource(),
        field_router=router,
    )

    snapshot = service.refresh()
    position = snapshot.accounts[0].positions[0]

    # Accounting stays at broker/parse values; invalid NaN mark is not applied.
    assert position.mark_price is None
    assert position.market_value == 0.0
    assert position.unrealized_pnl == 0.0
    # No fallback provenance for mark accounting (provider would be massive-options).
    assert "mark_price" not in position.provenance
    assert "unrealized_pnl_percent" not in position.provenance
    for field in ("market_value", "unrealized_pnl"):
        prov = position.provenance.get(field)
        if prov is not None:
            assert prov.provider != "massive-options"
            assert prov.fallback_reason is None
