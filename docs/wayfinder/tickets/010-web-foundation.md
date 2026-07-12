---
status: closed
type: prototype
blocks: []
---

# Establish the web application foundation

## Question

What is the smallest production-shaped FastAPI and React foundation that can replace the dashboard incrementally without breaking existing workflows?

## Exit evidence

- Backend application factory and health tests.
- Packaged frontend build with a distinctive workstation shell.
- `pilot dashboard` web launcher and temporary `--tui` fallback.
- Reproducible lint, test, type, and build commands.

## Resolution

Delivered the loopback-only FastAPI service, one-time browser session exchange,
packaged React workstation shell, web-default CLI launcher, CSP and private-cache
headers, TUI escape hatch, and automated Python, browser, and accessibility checks.
The wheel build contains the compiled frontend assets.
