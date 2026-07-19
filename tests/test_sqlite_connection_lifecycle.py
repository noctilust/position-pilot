"""Regression tests for SQLite connection / file-descriptor lifecycle.

Python's ``sqlite3.Connection`` context manager only commits or rolls back; it
does not close the connection. Prior code used ``with sqlite3.connect(...)`` and
``with self._connect()`` without an explicit close, which leaked FDs under
repeated repository use. These tests assert connections are unusable after each
operation (closed on success and error paths) without depending on GC timing.
"""

from __future__ import annotations

import sqlite3

import pytest

from position_pilot.domain.operations import OperationsService
from position_pilot.persistence.sqlite import (
    PositionPilotDatabase,
    managed_sqlite_connection,
)


def _assert_closed(connection: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_managed_sqlite_connection_closes_on_success(tmp_path) -> None:
    path = tmp_path / "managed-success.sqlite3"
    held: sqlite3.Connection | None = None

    with managed_sqlite_connection(path) as connection:
        held = connection
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO items(id) VALUES (1)")

    assert held is not None
    _assert_closed(held)

    with managed_sqlite_connection(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_managed_sqlite_connection_closes_and_rolls_back_on_error(tmp_path) -> None:
    path = tmp_path / "managed-error.sqlite3"
    with managed_sqlite_connection(path) as connection:
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY)")

    held: sqlite3.Connection | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with managed_sqlite_connection(path) as connection:
            held = connection
            connection.execute("BEGIN")
            connection.execute("INSERT INTO items(id) VALUES (1)")
            raise RuntimeError("boom")

    assert held is not None
    _assert_closed(held)

    with managed_sqlite_connection(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0


def test_database_operations_close_connections_without_gc(tmp_path, monkeypatch) -> None:
    """Repeated repository calls must close each connection immediately."""

    real_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(
        "position_pilot.persistence.sqlite.sqlite3.connect", tracking_connect
    )

    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    for index in range(40):
        database.set_setting(f"key-{index}", index)
        assert database.get_setting(f"key-{index}") == index

    identity = database.account_identity("5WT99999", "Individual")
    assert database.account_identity("5WT99999", "Individual") == identity

    assert opened, "factory should have been invoked"
    for connection in opened:
        _assert_closed(connection)


def test_database_connect_closes_when_body_raises(tmp_path, monkeypatch) -> None:
    real_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(
        "position_pilot.persistence.sqlite.sqlite3.connect", tracking_connect
    )

    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    baseline = len(opened)

    with pytest.raises(sqlite3.OperationalError):
        with database._connect() as connection:
            connection.execute("SELECT * FROM definitely_missing_table")

    assert len(opened) > baseline
    for connection in opened:
        _assert_closed(connection)


def test_operations_sqlite_helpers_close_connections(tmp_path, monkeypatch) -> None:
    real_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(
        "position_pilot.persistence.sqlite.sqlite3.connect", tracking_connect
    )

    database = PositionPilotDatabase(tmp_path / "position-pilot.sqlite3")
    path = database.path
    assert OperationsService._verify_sqlite(path) is True

    service = OperationsService(
        database=database,
        portfolio=None,  # type: ignore[arg-type]
        data_directory=tmp_path,
    )
    schema = service._read_backup_schema_version(path)
    assert schema == database.schema_version

    assert opened
    for connection in opened:
        _assert_closed(connection)
