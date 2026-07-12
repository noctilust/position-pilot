import { expect, test } from "@playwright/test";

import {
  buildRollLedger,
  dedupeRollChains,
  flattenRollEvents,
  mergeAccountRollResults,
  sortRollEventsNewestFirst,
  summarizeRollLedger,
} from "../src/rollLedger";
import type { RollChain } from "../src/types";

function chain(overrides: Partial<RollChain> & Pick<RollChain, "chain_id">): RollChain {
  return {
    account_id: "acct-a",
    underlying: "SPY",
    strategy_type: "Short Put",
    original_open_credit: 2.5,
    chain_total_credit: 3.0,
    rolls: [],
    ...overrides,
  };
}

function event(
  roll_id: string,
  timestamp: string,
  premium_effect: number,
  extra: Record<string, unknown> = {},
) {
  return {
    roll_id,
    timestamp,
    old_strike: 500,
    new_strike: 495,
    old_dte: 14,
    new_dte: 35,
    roll_pnl: 50,
    premium_effect,
    old_expiration: "2026-07-18",
    new_expiration: "2026-08-15",
    ...extra,
  };
}

test.describe("roll ledger pure helpers", () => {
  test("dedupeRollChains keeps first occurrence of chain_id", () => {
    const rows = dedupeRollChains([
      chain({ chain_id: "c1", underlying: "SPY" }),
      chain({ chain_id: "c1", underlying: "QQQ" }),
      chain({ chain_id: "c2", underlying: "IWM" }),
    ]);
    expect(rows).toHaveLength(2);
    expect(rows[0]?.underlying).toBe("SPY");
    expect(rows[1]?.underlying).toBe("IWM");
  });

  test("flattenRollEvents emits every event with chain context", () => {
    const rows = flattenRollEvents([
      chain({
        chain_id: "c1",
        rolls: [
          event("r1", "2026-07-09T16:00:00Z", 0.3),
          event("r2", "2026-07-10T16:00:00Z", -0.1),
        ],
      }),
      chain({
        chain_id: "c2",
        underlying: "QQQ",
        strategy_type: "Iron Condor",
        rolls: [event("r3", "2026-07-11T12:00:00Z", 0.5, { underlying: "QQQ" })],
      }),
    ]);
    expect(rows).toHaveLength(3);
    expect(rows.map((r) => r.roll_id).sort()).toEqual(["r1", "r2", "r3"]);
    expect(rows.find((r) => r.roll_id === "r3")?.strategy_type).toBe("Iron Condor");
  });

  test("flattenRollEvents dedupes by roll_id", () => {
    const rows = flattenRollEvents([
      chain({
        chain_id: "c1",
        rolls: [event("same", "2026-07-09T16:00:00Z", 0.3)],
      }),
      chain({
        chain_id: "c2",
        rolls: [event("same", "2026-07-09T16:00:00Z", 0.9)],
      }),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.premium_effect).toBe(0.3);
  });

  test("sortRollEventsNewestFirst orders by timestamp desc", () => {
    const sorted = sortRollEventsNewestFirst(
      flattenRollEvents([
        chain({
          chain_id: "c1",
          rolls: [
            event("old", "2026-07-01T10:00:00Z", 0.1),
            event("new", "2026-07-12T10:00:00Z", 0.2),
            event("mid", "2026-07-05T10:00:00Z", 0.15),
          ],
        }),
      ]),
    );
    expect(sorted.map((r) => r.roll_id)).toEqual(["new", "mid", "old"]);
  });

  test("buildRollLedger flattens and sorts newest-first", () => {
    const ledger = buildRollLedger([
      chain({
        chain_id: "c1",
        rolls: [
          event("a", "2026-07-01T00:00:00Z", 1),
          event("b", "2026-07-10T00:00:00Z", 2),
        ],
      }),
    ]);
    expect(ledger.map((r) => r.roll_id)).toEqual(["b", "a"]);
  });

  test("summarizeRollLedger nets premium_effect and counts events/chains", () => {
    const summary = summarizeRollLedger([
      chain({
        chain_id: "c1",
        original_open_credit: 2.5,
        chain_total_credit: 2.9,
        rolls: [
          event("r1", "2026-07-09T16:00:00Z", 0.5),
          event("r2", "2026-07-10T16:00:00Z", -0.1),
        ],
      }),
      chain({
        chain_id: "c2",
        original_open_credit: 1.0,
        chain_total_credit: 1.2,
        rolls: [event("r3", "2026-07-11T12:00:00Z", 0.2)],
      }),
    ]);
    // Net roll credit excludes opening credits: 0.5 - 0.1 + 0.2 = 0.6
    expect(summary.netRollCredit).toBeCloseTo(0.6, 8);
    expect(summary.eventCount).toBe(3);
    expect(summary.chainCount).toBe(2);
    // Lifetime includes opening: 2.9 + 1.2 = 4.1
    expect(summary.knownLifetimeCredit).toBeCloseTo(4.1, 8);
    expect(summary.lifetimePartial).toBe(false);
  });

  test("summarizeRollLedger marks lifetime partial when any chain total is null", () => {
    const summary = summarizeRollLedger([
      chain({
        chain_id: "c1",
        chain_total_credit: 2.0,
        rolls: [event("r1", "2026-07-09T16:00:00Z", 0.3)],
      }),
      chain({
        chain_id: "c2",
        original_open_credit: null,
        chain_total_credit: null,
        rolls: [event("r2", "2026-07-10T16:00:00Z", 0.1)],
      }),
    ]);
    expect(summary.knownLifetimeCredit).toBeCloseTo(2.0, 8);
    expect(summary.lifetimePartial).toBe(true);
    expect(summary.netRollCredit).toBeCloseTo(0.4, 8);
  });

  test("summarizeRollLedger with only null totals is partial and unknown", () => {
    const summary = summarizeRollLedger([
      chain({
        chain_id: "c1",
        original_open_credit: null,
        chain_total_credit: null,
        rolls: [event("r1", "2026-07-09T16:00:00Z", 0.3)],
      }),
    ]);
    expect(summary.knownLifetimeCredit).toBeNull();
    expect(summary.lifetimePartial).toBe(true);
  });

  test("mergeAccountRollResults keeps successes when some accounts fail", () => {
    const results: PromiseSettledResult<RollChain[]>[] = [
      {
        status: "fulfilled",
        value: [
          chain({
            chain_id: "c1",
            rolls: [event("r1", "2026-07-09T16:00:00Z", 0.3)],
          }),
        ],
      },
      { status: "rejected", reason: new Error("timeout") },
      {
        status: "fulfilled",
        value: [
          chain({
            chain_id: "c2",
            account_id: "acct-b",
            rolls: [event("r2", "2026-07-10T16:00:00Z", 0.1)],
          }),
        ],
      },
    ];
    const merged = mergeAccountRollResults(results);
    expect(merged.map((c) => c.chain_id).sort()).toEqual(["c1", "c2"]);
  });
});
