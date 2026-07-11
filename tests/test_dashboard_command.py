from typer.testing import CliRunner

from position_pilot.cli import app


def test_dashboard_command_launches_the_web_experience_by_default(monkeypatch) -> None:
    launches: list[bool] = []
    monkeypatch.setattr(
        "position_pilot.web.launcher.run_web_dashboard",
        lambda *, open_browser: launches.append(open_browser),
    )

    result = CliRunner().invoke(app, ["dashboard", "--no-browser"])

    assert result.exit_code == 0
    assert launches == [False]


def test_dashboard_command_preserves_the_tui_fallback(monkeypatch) -> None:
    launches: list[str] = []
    monkeypatch.setattr(
        "position_pilot.dashboard.run_dashboard",
        lambda: launches.append("tui"),
    )

    result = CliRunner().invoke(app, ["dashboard", "--tui"])

    assert result.exit_code == 0
    assert launches == ["tui"]
