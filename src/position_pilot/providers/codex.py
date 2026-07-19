"""Local Codex CLI provider — never reads or persists OAuth tokens."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .contracts import ProviderHealth, ProviderState

PROMPT_VERSION = "recommendation-prompt.v1"
SCHEMA_VERSION = "recommendation.v1"
AUTH_CACHE_TTL_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_SECONDS = 0.05

# Token-shaped strings that must never appear in logs or stored errors.
_SECRET_LIKE = re.compile(
    r"(?i)(bearer\s+[a-z0-9._\-]+|sk-[a-z0-9]{10,}|eyJ[a-zA-Z0-9_\-]{20,}"
    r"|refresh[_-]?token\s*[:=]\s*\S+|access[_-]?token\s*[:=]\s*\S+)"
)


class CodexProviderStatus(StrEnum):
    """Explicit provider outcome — never silently remapped to success."""

    OK = "ok"
    SIGNED_OUT = "signed_out"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    INVALID_OUTPUT = "invalid_output"
    NOT_INSTALLED = "not_installed"
    DISABLED = "disabled"
    SKIPPED_UNCHANGED = "skipped_unchanged"


class RecommendationAction(StrEnum):
    HOLD = "hold"
    ROLL = "roll"
    CLOSE = "close"
    REDUCE = "reduce"
    ADD = "add"
    HEDGE = "hedge"
    REVIEW = "review"


class RecommendationRisk(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CodexStructuredOutput(BaseModel):
    """Versioned structured schema returned by the Codex CLI."""

    schema_version: Literal["recommendation.v1"] = SCHEMA_VERSION
    action: RecommendationAction
    urgency: int = Field(ge=1, le=5)
    risk: RecommendationRisk
    reasoning: str = Field(min_length=1, max_length=4_000)
    evidence: list[str] = Field(default_factory=list, max_length=20)
    catalyst_refs: list[str] = Field(default_factory=list, max_length=20)
    suggested_action: str | None = Field(default=None, max_length=1_000)

    @field_validator("evidence", "catalyst_refs", mode="before")
    @classmethod
    def _coerce_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("expected a list of strings")
        return [str(item)[:500] for item in value][:20]

    @field_validator("reasoning", "suggested_action", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class CodexInvocationResult(BaseModel):
    """Outcome of one Codex CLI invocation — never includes secrets."""

    status: CodexProviderStatus
    output: CodexStructuredOutput | None = None
    error: str | None = None
    provider: str = "codex-cli"
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = None
    schema_version: str = SCHEMA_VERSION
    prompt_version: str = PROMPT_VERSION
    attempts: int = 1


def redact_secrets(text: str | None, *, limit: int = 400) -> str | None:
    """Strip token-like substrings before any logging or API return."""

    if text is None:
        return None
    cleaned = _SECRET_LIKE.sub("[redacted]", text)
    cleaned = cleaned.replace("\n", " ").strip()
    if len(cleaned) > limit:
        return cleaned[: limit - 1] + "…"
    return cleaned or None


def structured_output_json_schema() -> dict[str, Any]:
    """JSON Schema file content for `codex exec --output-schema`."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "action",
            "urgency",
            "risk",
            "reasoning",
            "evidence",
            "catalyst_refs",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "action": {
                "type": "string",
                "enum": [item.value for item in RecommendationAction],
            },
            "urgency": {"type": "integer", "minimum": 1, "maximum": 5},
            "risk": {
                "type": "string",
                "enum": [item.value for item in RecommendationRisk],
            },
            "reasoning": {"type": "string", "minLength": 1, "maxLength": 4000},
            "evidence": {
                "type": "array",
                "items": {"type": "string", "maxLength": 500},
                "maxItems": 20,
            },
            "catalyst_refs": {
                "type": "array",
                "items": {"type": "string", "maxLength": 500},
                "maxItems": 20,
            },
            "suggested_action": {"type": ["string", "null"], "maxLength": 1000},
        },
    }


def build_recommendation_prompt(context: dict[str, Any]) -> str:
    """Build a minimized analytical prompt with no account identifiers."""

    payload = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    mechanics = context.get("mechanics") if isinstance(context, dict) else None
    mechanics_block = ""
    if isinstance(mechanics, dict) and mechanics.get("enabled") is not False:
        candidates = mechanics.get("candidates") or []
        # Default/missing shadow_mode is observational — do not constrain action selection.
        shadow = bool(mechanics.get("shadow_mode", True))
        if candidates and not shadow:
            mechanics_block = (
                "MECHANICS_CONSTRAINTS: When mechanics.candidates is present, explain and compare "
                "ONLY those supplied advisory candidates (by candidate_id/kind). "
                "Do not invent strikes, expirations, quotes, order legs, hedges, or ADD actions. "
                "If required facts are missing or candidates list blocking_reasons/missing_inputs, "
                "prefer action=review and say what is unknown. "
                "Mechanics is advisory decision support; execution is manual in tastytrade. "
                "Do not claim an order was or will be placed.\n"
            )
        elif candidates and shadow:
            mechanics_block = (
                "MECHANICS_OBSERVATION: mechanics.candidates are shadow-mode context only. "
                "You may reference them for explanation, but they must NOT change or constrain "
                "the selected action. Do not invent strikes, expirations, quotes, or order legs. "
                "Execution remains manual in tastytrade.\n"
            )
    return (
        "You are Position Pilot's options-trading analyst. "
        "Return ONLY JSON matching the provided schema. "
        "Capital preservation takes priority over profit maximization. "
        "Do not invent market data. Do not request credentials or account numbers. "
        "Use only the analytical context below.\n"
        f"{mechanics_block}\n"
        f"PROMPT_VERSION={PROMPT_VERSION}\n"
        f"SCHEMA_VERSION={SCHEMA_VERSION}\n"
        f"CONTEXT={payload}\n"
    )


class CodexCLIProvider:
    """Invoke the installed Codex CLI via subprocess; never open auth token files."""

    name = "codex-cli"

    def __init__(
        self,
        *,
        binary: str | None = None,
        timeout_seconds: float = 90.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        which: Callable[[str], str | None] | None = None,
        auth_cache_ttl_seconds: float = AUTH_CACHE_TTL_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.binary = binary or os.getenv("POSITION_PILOT_CODEX_BIN", "codex")
        self.timeout_seconds = timeout_seconds
        self._runner = runner or subprocess.run
        self._which = which or shutil.which
        self.auth_cache_ttl_seconds = auth_cache_ttl_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._sleeper = sleeper or time.sleep
        self._last_health: ProviderHealth | None = None
        self._auth_cache: tuple[float, ProviderHealth] | None = None

    def is_installed(self) -> bool:
        if Path(self.binary).is_file():
            return True
        return self._which(self.binary) is not None

    def clear_auth_cache(self) -> None:
        self._auth_cache = None

    def health(self, *, force_refresh: bool = False) -> ProviderHealth:
        """Probe install + ChatGPT sign-in without reading OAuth material."""

        now_mono = time.monotonic()
        if (
            not force_refresh
            and self._auth_cache is not None
            and (now_mono - self._auth_cache[0]) < self.auth_cache_ttl_seconds
        ):
            return self._auth_cache[1]

        now = datetime.now(UTC)
        if not self.is_installed():
            health = ProviderHealth(
                provider=self.name,
                state=ProviderState.NOT_CONFIGURED,
                checked_at=now,
                error="Codex CLI is not installed",
            )
            self._store_auth_cache(health)
            return health

        try:
            result = self._run_command(
                [self.binary, "login", "status"],
                timeout=15.0,
                capture_input=None,
                cwd=None,
            )
        except Exception as error:  # pragma: no cover - runner should not raise
            health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=now,
                error=redact_secrets(f"Codex status probe failed: {error}"),
            )
            self._store_auth_cache(health)
            return health

        stdout = (result.stdout or "").lower()
        stderr = (result.stderr or "").lower()
        combined = f"{stdout}\n{stderr}"

        if result.returncode == 124 or "timed out" in combined:
            health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=now,
                error="Codex authentication status timed out",
            )
            self._store_auth_cache(health)
            return health

        signed_out_markers = (
            "not logged",
            "signed out",
            "login required",
            "unauthenticated",
            "not authenticated",
            "please log in",
            "please login",
        )
        if any(marker in combined for marker in signed_out_markers):
            health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=now,
                error="Codex is signed out",
            )
            self._store_auth_cache(health)
            return health

        if result.returncode != 0:
            # Non-zero without explicit signed-out language → unavailable (IO/runtime).
            health = ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                checked_at=now,
                error=redact_secrets(result.stderr or result.stdout or "Codex status unavailable"),
            )
            self._store_auth_cache(health)
            return health

        success_markers = ("logged in", "chatgpt", "authenticated")
        if any(marker in combined for marker in success_markers):
            health = ProviderHealth(
                provider=self.name,
                state=ProviderState.HEALTHY,
                checked_at=now,
                last_success_at=now,
            )
            self._store_auth_cache(health)
            return health

        # Exit 0 with unknown/empty output fails closed — do not assume healthy.
        health = ProviderHealth(
            provider=self.name,
            state=ProviderState.UNAVAILABLE,
            checked_at=now,
            error="Codex authentication status could not be confirmed",
        )
        self._store_auth_cache(health)
        return health

    def public_status(self, *, force_refresh: bool = False) -> str:
        """Bootstrap-safe provider status string (cached)."""

        health = self.health(force_refresh=force_refresh)
        if health.state == ProviderState.NOT_CONFIGURED:
            return "not_configured"
        if health.state == ProviderState.HEALTHY:
            return "configured"
        if health.error and "signed out" in (health.error or "").lower():
            return "signed_out"
        return "unavailable"

    def complete_recommendation(self, context: dict[str, Any]) -> CodexInvocationResult:
        """Run one structured recommendation; fail closed on any provider issue."""

        started = datetime.now(UTC)
        if not self.is_installed():
            return CodexInvocationResult(
                status=CodexProviderStatus.NOT_INSTALLED,
                error="Codex CLI is not installed on PATH",
                checked_at=started,
            )

        auth = self.health()
        if auth.state == ProviderState.NOT_CONFIGURED:
            return CodexInvocationResult(
                status=CodexProviderStatus.NOT_INSTALLED,
                error=auth.error,
                checked_at=started,
            )
        if auth.state != ProviderState.HEALTHY:
            status = (
                CodexProviderStatus.SIGNED_OUT
                if auth.error and "signed out" in auth.error.lower()
                else CodexProviderStatus.UNAVAILABLE
            )
            return CodexInvocationResult(
                status=status,
                error=redact_secrets(auth.error),
                checked_at=started,
            )

        prompt = build_recommendation_prompt(context)
        schema = structured_output_json_schema()
        attempts = 0
        last_result: CodexInvocationResult | None = None

        while attempts <= self.max_retries:
            attempts += 1
            last_result = self._invoke_once(prompt, schema, started=started, attempts=attempts)
            if last_result.status == CodexProviderStatus.OK:
                return last_result
            if last_result.status not in {
                CodexProviderStatus.UNAVAILABLE,
                CodexProviderStatus.RATE_LIMITED,
            }:
                return last_result
            if attempts > self.max_retries:
                break
            if self.retry_backoff_seconds:
                self._sleeper(self.retry_backoff_seconds * attempts)

        assert last_result is not None
        return last_result

    def _invoke_once(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        started: datetime,
        attempts: int,
    ) -> CodexInvocationResult:
        with tempfile.TemporaryDirectory(prefix="position-pilot-codex-") as tmp:
            tmp_path = Path(tmp)
            schema_path = tmp_path / "recommendation.schema.json"
            output_path = tmp_path / "last_message.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")

            command = [
                self.binary,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--color",
                "never",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-",  # prompt strictly from stdin
            ]
            completed = self._run_command(
                command,
                timeout=self.timeout_seconds,
                capture_input=prompt,
                cwd=str(tmp_path),
            )
            duration_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
            classified = self._classify_process_failure(completed)
            if classified is not None:
                classified.duration_ms = duration_ms
                classified.attempts = attempts
                return classified

            raw_text = self._read_output(output_path, completed.stdout)
            try:
                parsed = self._parse_structured_output(raw_text)
            except (ValidationError, json.JSONDecodeError, ValueError) as error:
                return CodexInvocationResult(
                    status=CodexProviderStatus.INVALID_OUTPUT,
                    error=redact_secrets(f"Invalid Codex structured output: {error}"),
                    checked_at=datetime.now(UTC),
                    duration_ms=duration_ms,
                    attempts=attempts,
                )

            return CodexInvocationResult(
                status=CodexProviderStatus.OK,
                output=parsed,
                checked_at=datetime.now(UTC),
                duration_ms=duration_ms,
                attempts=attempts,
            )

    def _store_auth_cache(self, health: ProviderHealth) -> None:
        self._last_health = health
        self._auth_cache = (time.monotonic(), health)

    def _run_command(
        self,
        command: list[str],
        *,
        timeout: float,
        capture_input: str | None,
        cwd: str | None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                command,
                input=capture_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=self._safe_env(),
                cwd=cwd,
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                command,
                returncode=127,
                stdout="",
                stderr="codex binary not found",
            )
        except subprocess.TimeoutExpired as error:
            return subprocess.CompletedProcess(
                command,
                returncode=124,
                stdout=error.stdout or "",
                stderr="codex invocation timed out",
            )
        except OSError as error:
            return subprocess.CompletedProcess(
                command,
                returncode=1,
                stdout="",
                stderr=f"codex invocation failed: {error}",
            )

    def _safe_env(self) -> dict[str, str]:
        """Pass a minimal environment; do not inject or echo API keys for Codex auth."""

        env: dict[str, str] = {}
        for key, value in os.environ.items():
            if key in {"PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "TERM", "CODEX_HOME"}:
                env[key] = value
            elif key.startswith("LC_"):
                env[key] = value
        for banned in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_ACCESS_TOKEN",
            "CHATGPT_ACCESS_TOKEN",
            "CODEX_API_KEY",
            "XAI_API_KEY",
            "TASTYTRADE_CLIENT_SECRET",
            "TASTYTRADE_REFRESH_TOKEN",
        ):
            env.pop(banned, None)
        return env

    def _classify_process_failure(
        self,
        completed: subprocess.CompletedProcess[str],
    ) -> CodexInvocationResult | None:
        combined = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
        if completed.returncode == 0:
            return None
        if completed.returncode == 127 or "not found" in combined:
            return CodexInvocationResult(
                status=CodexProviderStatus.NOT_INSTALLED,
                error=redact_secrets(completed.stderr or "Codex CLI not found"),
            )
        if any(
            marker in combined
            for marker in ("rate limit", "rate_limit", "too many requests", "429")
        ):
            return CodexInvocationResult(
                status=CodexProviderStatus.RATE_LIMITED,
                error="Codex rate limit reached",
            )
        if any(
            marker in combined
            for marker in (
                "not logged",
                "signed out",
                "login required",
                "unauthenticated",
                "not authenticated",
                "please log in",
                "please login",
            )
        ):
            return CodexInvocationResult(
                status=CodexProviderStatus.SIGNED_OUT,
                error="Codex is signed out",
            )
        if completed.returncode in {401, 403} or re.search(r"\b(401|403)\b", combined):
            return CodexInvocationResult(
                status=CodexProviderStatus.SIGNED_OUT,
                error="Codex is signed out",
            )
        if completed.returncode == 124 or "timed out" in combined:
            return CodexInvocationResult(
                status=CodexProviderStatus.UNAVAILABLE,
                error="Codex invocation timed out",
            )
        return CodexInvocationResult(
            status=CodexProviderStatus.UNAVAILABLE,
            error=redact_secrets(completed.stderr or completed.stdout or "Codex unavailable"),
        )

    def _read_output(self, output_path: Path, stdout: str | None) -> str:
        if output_path.exists():
            text = output_path.read_text(encoding="utf-8").strip()
            if text:
                return text
        return (stdout or "").strip()

    def _parse_structured_output(self, raw_text: str) -> CodexStructuredOutput:
        if not raw_text:
            raise ValueError("empty Codex response")
        candidates = [raw_text]
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
        if fence:
            candidates.insert(0, fence.group(1))
        brace = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if brace:
            candidates.append(brace.group(0))
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                return CodexStructuredOutput.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = error
                continue
        assert last_error is not None
        raise last_error


class ExplicitApiKeyFallbackProvider:
    """Internal disabled abstraction — never a silent or selectable production path.

    Exists only so the application never invents an implicit Anthropic/OpenAI
    fallback. Always returns DISABLED / UNAVAILABLE.
    """

    name = "api-key-fallback"

    def __init__(self, *, enabled: bool = False, api_key: str | None = None) -> None:
        # Intentionally ignored: abstraction is not user-selectable as functional.
        self.enabled = False
        self._api_key = None
        del enabled, api_key

    def public_status(self) -> str:
        return "unavailable"

    def complete_recommendation(self, context: dict[str, Any]) -> CodexInvocationResult:
        del context
        return CodexInvocationResult(
            status=CodexProviderStatus.DISABLED,
            provider=self.name,
            error=(
                "API-key recommendation provider is not available. "
                "Use the local Codex CLI with ChatGPT sign-in."
            ),
        )
