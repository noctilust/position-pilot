# Streaming protocol decisions

Position Pilot implements Tastytrade's two independent read-only websocket
paths:

- Market data uses the DXLink URL and 24-hour token returned by
  `GET /api-quote-tokens`. The client performs `SETUP`, `AUTH`,
  `CHANNEL_REQUEST`, `FEED_SETUP`, and `FEED_SUBSCRIPTION` in order, then sends
  a `KEEPALIVE` every 25 seconds independently of inbound traffic. Broker OCC
  symbols are normalized to DXLink option notation at the subscription boundary.
  Each authoritative portfolio refresh resets the subscription when its held
  symbol set changes.
- Account data uses `wss://streamer.tastyworks.com`. The client sends `connect`
  for the selected account numbers before starting 25-second heartbeats.
  OAuth access tokens require the same `Bearer <token>` form used by the HTTP
  Authorization header. Notifications contain full objects rather than diffs.

Authoritative references:

- https://developer.tastytrade.com/streaming-market-data/
- https://developer.tastytrade.com/streaming-account-data/
- https://developer.tastytrade.com/open-api-spec/accounts-and-customers/

Both handshakes are bounded to 15 seconds. A connection with no inbound protocol
message for 90 seconds is marked stale and reconnected with bounded backoff.

REST snapshots remain authoritative. Reconciliation runs at startup, every five
minutes, after reconnects, after an observed sequence gap, after an impossible
state, and on a five-second-bounded account-event trigger. In-flight trigger
bursts retain one trailing refresh. Stream failures are tracked independently
and never prevent a cached snapshot from loading.
