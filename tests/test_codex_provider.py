"""Codex CLI provider: cwd isolation, schema validation, auth, retry, privacy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from position_pilot.providers.codex import (
    SCHEMA_VERSION,
    CodexCLIProvider,
    CodexProviderStatus,
    ExplicitApiKeyFallbackProvider,
    RecommendationAction,
    build_recommendation_prompt,
    redact_secrets,
    structured_output_json_schema,
)


def _ok_payload(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "action": "hold",
        "urgency": 2,
        "risk": "low",
        "reasoning": "Theta still favorable with distance to short strike.",
        "evidence": ["DTE 21", "delta -0.12"],
        "catalyst_refs": [],
        "suggested_action": "Hold through next session",
    }
    payload.update(overrides)
    return payload


def test_redact_secrets_strips_token_shapes() -> None:
    text = "Bearer sk-abc1234567890 and eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aaa.bbb"
    cleaned = redact_secrets(text)
    assert cleaned is not None
    assert "sk-abc" not in cleaned
    assert "eyJ" not in cleaned
    assert "[redacted]" in cleaned


def test_prompt_excludes_account_identifiers() -> None:
    prompt = build_recommendation_prompt(
        {
            "symbol": "SPY",
            "strategy_type": "Iron Condor",
            "quantity": 2,
            "unrealized_pnl": 120.5,
        }
    )
    assert "5WT" not in prompt
    assert "account_number" not in prompt
    assert "SPY" in prompt
    assert "password" not in prompt
    assert "refresh_token" not in prompt


def test_structured_schema_is_versioned() -> None:
    schema = structured_output_json_schema()
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_not_installed_status() -> None:
    provider = CodexCLIProvider(binary="codex-missing-binary", which=lambda _: None)
    result = provider.complete_recommendation({"symbol": "AAPL"})
    assert result.status == CodexProviderStatus.NOT_INSTALLED
    assert result.output is None


def test_signed_out_is_explicit() -> None:
    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 1, "", "Not logged in")
        return subprocess.CompletedProcess(command, 1, "", "unexpected")

    provider = CodexCLIProvider(binary="codex", runner=runner, which=lambda _: "/usr/bin/codex")
    result = provider.complete_recommendation({"symbol": "AAPL"})
    assert result.status == CodexProviderStatus.SIGNED_OUT


def test_timeout_is_unavailable_not_signed_out() -> None:
    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 124, "", "codex invocation timed out")
        return subprocess.CompletedProcess(command, 1, "", "x")

    provider = CodexCLIProvider(binary="codex", runner=runner, which=lambda _: "/usr/bin/codex")
    health = provider.health()
    assert health.state.value == "unavailable"
    assert "signed out" not in (health.error or "").lower()
    assert "timed out" in (health.error or "").lower()


def test_unknown_successful_login_output_fails_closed() -> None:
    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 0, "status: ok", "")
        return subprocess.CompletedProcess(command, 1, "", "x")

    provider = CodexCLIProvider(binary="codex", runner=runner, which=lambda _: "/usr/bin/codex")
    health = provider.health()
    assert health.state.value == "unavailable"
    assert "could not be confirmed" in (health.error or "").lower()


def test_auth_status_ttl_cache_avoids_repeat_probes() -> None:
    calls = {"n": 0}

    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            calls["n"] += 1
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        return subprocess.CompletedProcess(command, 1, "", "x")

    provider = CodexCLIProvider(
        binary="codex",
        runner=runner,
        which=lambda _: "/usr/bin/codex",
        auth_cache_ttl_seconds=60,
    )
    assert provider.public_status() == "configured"
    assert provider.public_status() == "configured"
    assert provider.health().state.value == "healthy"
    assert calls["n"] == 1


def test_exec_uses_isolated_temp_cwd_and_stdin_prompt() -> None:
    captured: dict = {}

    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        captured["cwd"] = kwargs.get("cwd")
        captured["input"] = kwargs.get("input")
        captured["command"] = command
        if "--output-last-message" in command:
            out_idx = command.index("--output-last-message") + 1
            Path(command[out_idx]).write_text(json.dumps(_ok_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    provider = CodexCLIProvider(binary="codex", runner=runner, which=lambda _: "/usr/bin/codex")
    result = provider.complete_recommendation({"symbol": "SPY"})
    assert result.status == CodexProviderStatus.OK
    assert captured["cwd"] is not None
    cwd = Path(captured["cwd"])
    assert cwd.exists() or True  # may be cleaned after return
    assert "position-pilot-codex-" in str(captured["cwd"])
    # Isolated temp dir name, not the repository checkout root.
    assert "codex" in Path(captured["cwd"]).name
    assert captured["input"] is not None and "CONTEXT=" in captured["input"]
    assert captured["command"][-1] == "-"
    assert "--ignore-user-config" in captured["command"]
    assert "--ignore-rules" in captured["command"]
    assert ".env" not in str(captured["command"])


def test_valid_structured_output_parses() -> None:
    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if "--output-last-message" in command:
            out_idx = command.index("--output-last-message") + 1
            Path(command[out_idx]).write_text(json.dumps(_ok_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    provider = CodexCLIProvider(binary="codex", runner=runner, which=lambda _: "/usr/bin/codex")
    result = provider.complete_recommendation({"symbol": "SPY", "strategy_type": "Short Put"})
    assert result.status == CodexProviderStatus.OK
    assert result.output is not None
    assert result.output.action == RecommendationAction.HOLD


def test_invalid_output_is_explicit_no_retry() -> None:
    calls = {"exec": 0}

    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if "exec" in command:
            calls["exec"] += 1
            if "--output-last-message" in command:
                out_idx = command.index("--output-last-message") + 1
                Path(command[out_idx]).write_text("not-json", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "not-json", "")
        return subprocess.CompletedProcess(command, 1, "", "x")

    provider = CodexCLIProvider(
        binary="codex",
        runner=runner,
        which=lambda _: "/usr/bin/codex",
        max_retries=2,
    )
    result = provider.complete_recommendation({"symbol": "QQQ"})
    assert result.status == CodexProviderStatus.INVALID_OUTPUT
    assert calls["exec"] == 1


def test_rate_limited_retries_bounded() -> None:
    calls = {"exec": 0}
    sleeps: list[float] = []

    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if "exec" in command:
            calls["exec"] += 1
            return subprocess.CompletedProcess(command, 1, "", "Error: rate limit exceeded 429")
        return subprocess.CompletedProcess(command, 1, "", "x")

    provider = CodexCLIProvider(
        binary="codex",
        runner=runner,
        which=lambda _: "/usr/bin/codex",
        max_retries=2,
        retry_backoff_seconds=0.01,
        sleeper=sleeps.append,
    )
    result = provider.complete_recommendation({"symbol": "IWM"})
    assert result.status == CodexProviderStatus.RATE_LIMITED
    assert calls["exec"] == 3  # initial + 2 retries
    assert len(sleeps) == 2


def test_signed_out_does_not_retry() -> None:
    calls = {"exec": 0}

    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if "exec" in command:
            calls["exec"] += 1
            return subprocess.CompletedProcess(command, 1, "", "Error: not logged in")
        return subprocess.CompletedProcess(command, 1, "", "x")

    provider = CodexCLIProvider(
        binary="codex",
        runner=runner,
        which=lambda _: "/usr/bin/codex",
        max_retries=3,
    )
    result = provider.complete_recommendation({"symbol": "IWM"})
    assert result.status == CodexProviderStatus.SIGNED_OUT
    assert calls["exec"] == 1


def test_api_key_fallback_always_disabled() -> None:
    provider = ExplicitApiKeyFallbackProvider(enabled=True)
    result = provider.complete_recommendation({"symbol": "SPY"})
    assert result.status == CodexProviderStatus.DISABLED
    assert provider.enabled is False


def test_provider_never_reads_token_files(tmp_path: Path, monkeypatch) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    secret = codex_home / "auth.json"
    secret.write_text('{"access_token":"super-secret-token"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    opened: list[str] = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    def runner(command, **kwargs):
        if command[:2] == ["codex", "login"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if "--output-last-message" in command:
            out_idx = command.index("--output-last-message") + 1
            Path(command[out_idx]).write_text(
                json.dumps(_ok_payload(action="review")),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    provider = CodexCLIProvider(binary="codex", runner=runner, which=lambda _: "/usr/bin/codex")
    result = provider.complete_recommendation({"symbol": "AAPL"})
    assert result.status == CodexProviderStatus.OK
    assert not any("auth.json" in path for path in opened)
