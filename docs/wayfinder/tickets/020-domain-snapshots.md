---
status: closed
type: implementation
blocked_by:
  - Establish the web application foundation
---

# Extract domain services and versioned snapshots

## Question

How should portfolio, strategy, provider provenance, freshness, SQLite persistence, and atomic snapshots be modeled so CLI and web consumers share one source of truth?

## Resolution

Delivered focused account, portfolio, strategy, market, and roll application
services; browser-safe account and roll identities; field provenance and panel
freshness; SQLite migrations, settings, consistent backups, and legacy cache
imports; atomic live/cached portfolio snapshots; durable account scope and
horizon edits; and consolidated account switching. CLI positions and the web
dashboard now consume the shared portfolio service. Verified against live
read-only Tastytrade data and automated service, API, browser, and accessibility
tests.
