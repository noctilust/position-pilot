---
name: position-pilot-codex-orchestrator
description: >
  Codex orchestration for Position Pilot. Use for nontrivial coding tasks:
  defining contracts, delegating to Grok Build, worktrees, review, repair
  passes, independent validation, and final integration. Prefer Grok for
  implementation; Codex reviews diffs and integrates. Trigger on implement,
  fix, refactor, delegate to Grok, or multi-file production changes.
---

# Position Pilot — Codex orchestrator

Read root [`AGENTS.md`](../../../AGENTS.md) for shared policy (billing, git safety, quality, security). This skill is **Codex-only**.

**Model:** Codex defines the contract → Grok implements → Codex reviews the actual diff → Grok repairs → Codex validates critically → Codex integrates and reports.

Do not consume OpenAI/xAI API credits. Invoke Grok with subscription auth only (`env -u XAI_API_KEY -u OPENAI_API_KEY grok ...`).

---

## Responsibilities

**Codex owns:** interpret user objective; define requirements, constraints, acceptance criteria, and prohibited scope; decide delegation; create worktrees; write Grok assignments; inspect actual repo state/diffs (not just Grok summaries); independently validate high-risk behavior; request concrete repair passes; accept/reject/integrate; produce the final user-facing summary.

**Delegate to Grok:** exploration, technical planning, production edits, tests, targeted runs, debugging, repair, implementation self-review.

**Codex may handle directly:** explain code, narrow Q&A, review-only, one-line/config/typo fixes, emergency repair when Grok cannot run. When uncertain on code changes, prefer Grok.

**Codex may take over implementation only if:** Grok CLI/auth unavailable; credits appear exhausted; unrecoverable Grok errors; missing local capability; two failed well-specified repair passes; trivial integration left; or immediate risk to user work. State the reason; do not pretend Grok finished the work.

---

## Efficiency rules

1. **One large assignment** — one contract → one Grok session → one review → one resumed repair → final validation. Avoid many tiny Grok calls.
2. **Delegate discovery** — let Grok find files, abstractions, tests, and validation commands unless safety/scope requires Codex.
3. **No duplicate analysis** — do not fully design/implement then ask Grok to reproduce.
4. **No duplicate patches** — do not privately write the full patch then have Grok retype it.
5. **Minimal Codex edits** after Grok — typos, imports, format, trivial conflicts, mechanical fixes only. Nontrivial logic → back to Grok.
6. **Explanations sparingly** — only for architecture, defects, security/compat risk, or non-obvious behavior.

---

## Execution workflow

### 1. Repository safety

```bash
git status --short
git branch --show-current
git rev-parse --show-toplevel
```

Preserve all user-authored changes. Prefer an isolated worktree for multi-file, behavior-changing, dirty-tree, autonomous Grok, refactor, multi-pass, or rollback-sensitive work.

Destructive commands (`reset --hard`, `clean -fd`, `checkout -- .`, `restore .`, `push --force`, `branch -D`, etc.) require explicit user authorization.

### 2. Implementation contract

Define: objective, user-visible requirements, acceptance criteria, constraints, prohibited changes, compatibility, test expectations, security/data integrity, public-interface change permission. Inspect only enough context to write an accurate assignment.

### 3. Worktree (default for nontrivial tasks)

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$REPO_ROOT")"
TASK_ID="$(date +%Y%m%d-%H%M%S)"
BRANCH_NAME="grok/${TASK_ID}"
WORKTREE_PATH="$(dirname "$REPO_ROOT")/${REPO_NAME}-grok-${TASK_ID}"

git -C "$REPO_ROOT" worktree add -b "$BRANCH_NAME" "$WORKTREE_PATH" HEAD
git -C "$WORKTREE_PATH" status --short
git -C "$WORKTREE_PATH" branch --show-current
```

Retain `REPO_ROOT`, `TASK_ID`, `BRANCH_NAME`, `WORKTREE_PATH`. If a worktree is unsuitable, allow Grok in the current tree only after confirming user changes stay safe and distinguishable.

### 4. Grok task file

Write `/tmp/grok-task-${TASK_ID}.md` comprehensive enough for one session. Point Grok at [`.grok/position-pilot-implementation.md`](../../../.grok/position-pilot-implementation.md) and root `AGENTS.md`.

Include: Role (primary implementer; complete the change), Objective, User-visible requirements, Acceptance criteria, Constraints (architecture, no remote writes, no API keys, no unrelated cleanup), Required workflow (inspect → plan → implement → test → lint/types → diff → repair → report), Completion report fields.

### 5. Invoke Grok

`--session-id` requires a **valid UUID** for a new conversation (must not already exist). Prefer omitting it and recording the session id Grok assigns (`grok sessions` if needed). Do not use unsupported flags such as `--no-auto-update`.

```bash
# Preferred: let Grok assign the session id; record it for repair passes
env -u XAI_API_KEY -u OPENAI_API_KEY \
  grok --cwd "$WORKTREE_PATH" \
  -p "$(cat "$TASK_FILE")" \
  --output-format plain --always-approve

# Optional: explicit new session UUID
SESSION_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
env -u XAI_API_KEY -u OPENAI_API_KEY \
  grok --cwd "$WORKTREE_PATH" \
  --session-id "$SESSION_ID" \
  -p "$(cat "$TASK_FILE")" \
  --output-format plain --always-approve
```

Adapt flags via `grok --help` if needed. One write-capable Grok session per worktree. Capture exit status; keep useful stdout/stderr. Do not poll/restart without reason.

### 6. Inspect results

```bash
git -C "$WORKTREE_PATH" status --short
git -C "$WORKTREE_PATH" diff --stat
git -C "$WORKTREE_PATH" diff --check
git -C "$WORKTREE_PATH" diff
git -C "$WORKTREE_PATH" diff --cached --stat
git -C "$WORKTREE_PATH" diff --cached
```

Verify: acceptance criteria, no unrelated files, user work intact, tests exercise behavior, error paths, compatibility, no secrets/junk/placeholders/silent failures. Review **code**, not only stats.

### 7. Independent validation

Rerun only high-value checks: targeted tests, one smoke/integration, types/lint on affected files, build when critical, bug reproduction. Avoid re-running Grok’s full expensive suite unless high risk, ambiguous output, untrusted results, repo policy, or user request.

### 8. Repair passes (up to two by default)

Write `/tmp/grok-review-${TASK_ID}.md` with concrete defects: location, observed vs required behavior, evidence. Constraints: fix listed issues; preserve correct work; add tests that would have caught each defect.

```bash
# Resume by UUID only if the prior run used / recorded a valid SESSION_ID
env -u XAI_API_KEY -u OPENAI_API_KEY \
  grok --cwd "$WORKTREE_PATH" \
  --resume "$SESSION_ID" \
  -p "$(cat "$REVIEW_FILE")" \
  --output-format plain --always-approve

# Otherwise: look up id via `grok sessions`, or continue most recent for this cwd
env -u XAI_API_KEY -u OPENAI_API_KEY \
  grok --cwd "$WORKTREE_PATH" \
  --continue \
  -p "$(cat "$REVIEW_FILE")" \
  --output-format plain --always-approve
```

Third pass only if substantial progress, narrow remaining defect, still cheaper than takeover, and no integrity risk.

### 9. Accept and integrate

Accept only when criteria met, critical tests pass, no high-severity defects, no unexplained unrelated changes, risks understood, integration won’t overwrite user work.

Options: cherry-pick, merge agent branch, apply reviewed patch, trivial manual fix. Know exactly what will integrate. Do not integrate unexplained files. Remove worktree only after safe integration or intentional discard:

```bash
git -C "$REPO_ROOT" worktree remove "$WORKTREE_PATH"
git -C "$REPO_ROOT" branch -d "$BRANCH_NAME"
```

---

## Grok failure handling

| Case | Action |
|------|--------|
| Unsupported CLI flags | `grok --help`; adapt; don’t retry same bad command |
| Auth failure | Confirm subscription auth; never fall back to xAI API key |
| Credit exhaustion | Preserve partial work; inspect diff; salvage or complete only if needed; record that Grok stopped |
| Transient process failure | One retry if warranted; no unbounded loops |
| Implementation failure | Two concrete repair passes; then takeover or report blocker |
| Any failure | Always re-inspect `status` / `diff` / `diff --cached` |

---

## Decision defaults

- Code change / implementation exploration / technical plan → Grok
- Inferable product requirements → Codex resolves from context
- Dirty tree or multi-file production change → worktree
- Nontrivial review defect → Grok repair; two failures → Codex may take over
- Paid API would be required → stop and report
- Remote write → explicit user authorization
- Expensive verification → Grok main suite; Codex critical checks only
- User work at risk → stop and preserve

---

## Operational checklist

```text
[ ] Confirm repository and Git state; preserve user changes
[ ] Define objective, constraints, acceptance criteria
[ ] Create isolated Grok worktree when appropriate
[ ] Write one comprehensive Grok assignment
[ ] Invoke subscription-authenticated Grok without API keys
[ ] Let Grok explore, implement, test, debug
[ ] Inspect status, staged, and full diff
[ ] Independently run critical validation
[ ] Send concrete defects to same Grok session (≤2 repairs default)
[ ] Codex implementation only as fallback / trivial integration
[ ] Integrate only reviewed changes
[ ] Report validation and residual risks accurately
```

---

## Final report to user

Concise and concrete: what shipped, key files, validation run, whether Grok implemented, limitations/risks, commit/worktree state.

Do not expose chain-of-thought, raw prompts (unless useful and requested), credentials, private env data, or noise logs. Do not claim tests passed or credits exhausted without evidence.
