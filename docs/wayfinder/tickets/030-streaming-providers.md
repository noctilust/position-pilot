---
status: closed
type: research
blocked_by:
  - Extract domain services and versioned snapshots
---

# Implement streaming and provider reconciliation

## Question

How should Tastytrade streaming, REST reconciliation, provider fallbacks, gaps, stale state, and field-level provenance behave under normal and degraded conditions?

## Resolution

- DXLink follows the documented setup, authentication, channel, feed setup, and subscription sequence, with periodic keepalives, OCC-symbol normalization, portfolio-driven subscription resets, stale detection, and independently observable reconnect state.
- The account streamer uses the production websocket with Bearer authentication, heartbeats, full-object notifications, sequence-gap detection, and an isolated supervisor.
- OAuth and quote-stream credentials are renewed for each connection attempt so reconnects do not reuse expired tokens.
- REST snapshots remain authoritative at startup, every five minutes, after reconnects, after sequence gaps, after impossible-state signals, and through bounded account-event refreshes.
- Market and account streams fail independently. Browser-visible events redact brokerage identifiers recursively and expose opaque account IDs only.
- Field routing is Tastytrade-first, records provenance and the reason for fallback, never averages discrepant sources, exposes independent provider health, and backs production stock quotes plus held-option marks and Greeks.
- Massive Stocks, Options, and News capabilities are configuration-gated. Missing credentials produce an explicit `not_configured` state rather than silent failure.

The protocol evidence and primary documentation links are recorded in [Streaming protocol decisions](../../STREAMING_PROTOCOL.md). Regression coverage exercises message construction, feed decoding, credential renewal, reconciliation triggers, fallback routing, persistence, SSE state, and browser redaction.
