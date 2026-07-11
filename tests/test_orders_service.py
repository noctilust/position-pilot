"""Read-only order activity with redacted identities and fill linkage."""

from datetime import UTC, datetime

from position_pilot.domain.orders import OrderService
from position_pilot.models.transaction import Order, OrderStatus, Transaction, TransactionType
from position_pilot.persistence.sqlite import PositionPilotDatabase


class FakeOrderSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.transaction_start: datetime | None = None

    def get_orders(self, account_number: str, *, limit: int = 100) -> list[Order]:
        self.calls.append((account_number, limit))
        return [
            Order(
                order_id="broker-order-99",
                account_number=account_number,
                symbol="SPY  260821C00500000",
                action="Sell to Open",
                quantity=1,
                order_type="Limit",
                status=OrderStatus.FILLED,
                created_at=datetime(2026, 7, 10, 14, 0, tzinfo=UTC),
                updated_at=datetime(2026, 7, 10, 14, 5, tzinfo=UTC),
                filled_quantity=1,
                average_fill_price=2.15,
                commissions=1.0,
                underlying_symbol="SPY",
            )
        ]

    def get_transactions(
        self,
        account_number: str,
        *,
        start_date=None,
        end_date=None,
    ) -> list[Transaction]:
        self.transaction_start = start_date
        return [
            Transaction(
                transaction_id="txn-broker-1",
                transaction_type=TransactionType.ORDER_FILL,
                transaction_date=datetime(2026, 7, 10, 14, 5, tzinfo=UTC),
                description="Sold 1 SPY 500C",
                amount=215,
                symbol="SPY  260821C00500000",
                quantity=1,
                price=2.15,
                commission=1.0,
                order_id="broker-order-99",
                account_number=account_number,
                action="Sell to Open",
            )
        ]


def test_orders_are_redacted_and_linked_to_fills(tmp_path) -> None:
    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    identity = database.account_identity("5WT12345", "Individual")
    source = FakeOrderSource()
    service = OrderService(database=database, source=source)

    orders = service.list_orders(identity.account_id, limit=25)

    assert len(orders) == 1
    order = orders[0]
    assert order.account_id == identity.account_id
    assert order.order_id != "broker-order-99"
    assert "5WT12345" not in order.model_dump_json()
    assert "broker-order" not in order.model_dump_json()
    assert order.status == "filled"
    assert order.underlying_symbol == "SPY"
    assert len(order.fills) == 1
    assert order.fills[0].price == 2.15
    assert order.fills[0].fill_id != "txn-broker-1"
    assert source.transaction_start == datetime(2026, 7, 9, 14, 0, tzinfo=UTC)
