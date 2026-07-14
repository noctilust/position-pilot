from typer.testing import CliRunner

from position_pilot.cli import app
from position_pilot.web.launcher import DEFAULT_DASHBOARD_PORT


def test_dashboard_command_launches_the_web_experience_by_default(monkeypatch) -> None:
    launches: list[dict[str, object]] = []

    def capture(*, open_browser: bool, port: int = DEFAULT_DASHBOARD_PORT) -> None:
        launches.append({"open_browser": open_browser, "port": port})

    monkeypatch.setattr("position_pilot.web.launcher.run_web_dashboard", capture)

    result = CliRunner().invoke(app, ["dashboard", "--no-browser"])

    assert result.exit_code == 0
    assert launches == [{"open_browser": False, "port": DEFAULT_DASHBOARD_PORT}]


def test_dashboard_command_passes_custom_port(monkeypatch) -> None:
    launches: list[dict[str, object]] = []

    def capture(*, open_browser: bool, port: int = DEFAULT_DASHBOARD_PORT) -> None:
        launches.append({"open_browser": open_browser, "port": port})

    monkeypatch.setattr("position_pilot.web.launcher.run_web_dashboard", capture)

    result = CliRunner().invoke(app, ["dashboard", "--no-browser", "--port", "9123"])

    assert result.exit_code == 0
    assert launches == [{"open_browser": False, "port": 9123}]


def test_dashboard_command_rejects_out_of_range_port() -> None:
    result = CliRunner().invoke(app, ["dashboard", "--no-browser", "--port", "70000"])

    assert result.exit_code != 0
    combined = ((result.stdout or "") + (result.stderr or "")).lower()
    assert "port" in combined or "65535" in combined or "range" in combined


def test_dashboard_command_rejects_retired_tui_flag() -> None:
    result = CliRunner().invoke(app, ["dashboard", "--tui"])

    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert (
        "tui" in combined.lower() or "no such option" in combined.lower() or result.exit_code == 2
    )
