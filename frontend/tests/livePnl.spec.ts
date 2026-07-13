import { expect, test } from "@playwright/test";

import {
  applyMarketEventToMarks,
  fromDxlinkSymbol,
  heldMatchKeysFromStrategies,
  isValidPrice,
  liveLegOpenPnl,
  liveLegOpenPnlPercent,
  markFromQuote,
  matchKeyFromEventSymbol,
  matchKeyFromLegSymbol,
  normalizeMatchSymbol,
  overlayLivePnlOnStrategies,
  pruneLiveMarks,
  rawUnrealizedFromMark,
  retireLiveMarksIfSnapshotChanged,
  type LiveMarkMap,
  type LiveMarkState,
} from "../src/livePnl";
import type { PositionLeg, Strategy } from "../src/types";

function leg(overrides: Partial<PositionLeg> = {}): PositionLeg {
  return {
    symbol: "MU    260731C01400000",
    underlying_symbol: "MU",
    quantity: 1,
    quantity_direction: "Short",
    position_type: "Equity Option",
    strike_price: 1400,
    option_type: "C",
    expiration_date: "2026-07-31",
    days_to_expiration: 18,
    mark_price: 4.05,
    market_value: 405,
    cost_basis: 1601,
    unrealized_pnl: 1196,
    unrealized_pnl_percent: (1196 / 1601) * 100,
    pnl_open: 1557,
    pnl_open_percent: (1557 / 1962) * 100,
    pnl_open_basis: 1962,
    roll_adjustment: 361,
    roll_count: 1,
    roll_history_status: "complete",
    delta: 0.1,
    gamma: 0.01,
    theta: 0.1,
    vega: -0.1,
    implied_volatility: 0.4,
    multiplier: 100,
    horizon: "tactical",
    ...overrides,
  };
}

function strategy(overrides: Partial<Strategy> = {}, legs?: PositionLeg[]): Strategy {
  const strategyLegs = legs ?? [leg()];
  return {
    strategy_id: "mu-strangle",
    account_id: "public-account-id",
    underlying: "MU",
    strategy_type: "Short Strangle",
    expiration_date: "2026-07-31",
    days_to_expiration: 18,
    quantity: 1,
    strikes: "$800/$1400",
    unrealized_pnl: strategyLegs.reduce((s, l) => s + l.unrealized_pnl, 0),
    unrealized_pnl_percent: 12,
    pnl_open: strategyLegs.reduce(
      (s, l) => s + (l.pnl_open ?? l.unrealized_pnl),
      0,
    ),
    pnl_open_percent: 18,
    roll_adjustment: strategyLegs.reduce((s, l) => s + (l.roll_adjustment ?? 0), 0),
    roll_count: 1,
    total_delta: -0.1,
    total_theta: 20,
    horizon: "tactical",
    legs: strategyLegs,
    ...overrides,
  };
}

test("symbol normalization: equity passthrough and OCC/DXLink matching", () => {
  expect(fromDxlinkSymbol("SPY")).toBe("SPY");
  expect(fromDxlinkSymbol(".MU260731C1400")).toBe("MU    260731C01400000");
  expect(fromDxlinkSymbol(".BRK.B260821C450.5")).toBe("BRK.B 260821C00450500");
  expect(matchKeyFromEventSymbol(".MU260731C1400")).toBe("MU 260731C01400000");
  expect(matchKeyFromLegSymbol("MU    260731C01400000")).toBe("MU 260731C01400000");
  expect(matchKeyFromLegSymbol("MU 260731C01400000")).toBe("MU 260731C01400000");
  expect(normalizeMatchSymbol("  spy  ")).toBe("SPY");
});

test("isValidPrice: zero valid; negatives NaN infinity rejected", () => {
  expect(isValidPrice(0)).toBe(true);
  expect(isValidPrice(0.0)).toBe(true);
  expect(isValidPrice(3.725)).toBe(true);
  expect(isValidPrice(-0.01)).toBe(false);
  expect(isValidPrice(Number.NaN)).toBe(false);
  expect(isValidPrice(Number.POSITIVE_INFINITY)).toBe(false);
  expect(isValidPrice(Number.NEGATIVE_INFINITY)).toBe(false);
  expect(isValidPrice(null)).toBe(false);
  expect(isValidPrice("1")).toBe(false);
});

test("markFromQuote: midpoint, single side, zero bid valid; invalid rejected", () => {
  expect(markFromQuote(1.0, 1.2)).toBeCloseTo(1.1);
  expect(markFromQuote(0, 1.0)).toBeCloseTo(0.5);
  expect(markFromQuote(2.5, null)).toBe(2.5);
  expect(markFromQuote(undefined, 3.0)).toBe(3.0);
  expect(markFromQuote(null, null)).toBeNull();
  expect(markFromQuote(Number.NaN, 1)).toBe(1);
  expect(markFromQuote(-1, 2)).toBe(2);
  expect(markFromQuote(-1, -2)).toBeNull();
  expect(markFromQuote(Number.POSITIVE_INFINITY, 1)).toBe(1);
});

test("invalid trade and quote prices never enter mark map", () => {
  const marks: LiveMarkMap = new Map();
  expect(
    applyMarketEventToMarks(marks, "market.Trade", ".MU260731C1400", { price: -1 }),
  ).toBe(false);
  expect(marks.size).toBe(0);
  expect(
    applyMarketEventToMarks(marks, "market.Trade", ".MU260731C1400", {
      price: Number.NaN,
    }),
  ).toBe(false);
  expect(
    applyMarketEventToMarks(marks, "market.Quote", ".MU260731C1400", {
      bidPrice: -1,
      askPrice: Number.NEGATIVE_INFINITY,
    }),
  ).toBe(false);
  expect(marks.size).toBe(0);

  // Zero bid midpoint remains valid.
  expect(
    applyMarketEventToMarks(marks, "market.Quote", ".MU260731C1400", {
      bidPrice: 0,
      askPrice: 1,
    }),
  ).toBe(true);
  expect(marks.get("MU 260731C01400000")?.mark).toBeCloseTo(0.5);
});

test("raw P/L long and short from live mark", () => {
  const longLeg = leg({
    quantity_direction: "Long",
    cost_basis: 200,
    mark_price: 2,
    roll_adjustment: 0,
  });
  expect(rawUnrealizedFromMark(longLeg, 3.5)).toBe(150);

  const shortLeg = leg({ cost_basis: 1601, quantity_direction: "Short" });
  expect(rawUnrealizedFromMark(shortLeg, 3.725)).toBeCloseTo(1228.5);
});

test("rolled complete carry applied once; partial/no-roll unchanged", () => {
  const complete = leg({ roll_adjustment: 361, roll_history_status: "complete" });
  expect(liveLegOpenPnl(complete, 3.725)).toBeCloseTo(1228.5 + 361);

  const partial = leg({
    roll_adjustment: 0,
    roll_history_status: "partial",
    pnl_open: 1196,
  });
  expect(liveLegOpenPnl(partial, 3.725)).toBeCloseTo(1228.5);

  const none = leg({ roll_adjustment: 0, roll_history_status: "none" });
  expect(liveLegOpenPnl(none, 3.725)).toBeCloseTo(1228.5);
});

test("percentage uses pnl_open_basis with fallback", () => {
  const withBasis = leg({ pnl_open_basis: 1962 });
  const open = liveLegOpenPnl(withBasis, 3.725);
  const raw = rawUnrealizedFromMark(withBasis, 3.725);
  expect(liveLegOpenPnlPercent(withBasis, open, raw)).toBeCloseTo((open / 1962) * 100);

  const noBasis = leg({
    pnl_open_basis: null,
    cost_basis: 1601,
    pnl_open_percent: 99,
  });
  expect(liveLegOpenPnlPercent(noBasis, open, raw)).toBeCloseTo((raw / 1601) * 100);

  const noCost = leg({
    pnl_open_basis: 0,
    cost_basis: 0,
    pnl_open_percent: 12.5,
    unrealized_pnl_percent: 5,
  });
  expect(liveLegOpenPnlPercent(noCost, open, raw)).toBe(12.5);
});

test("quote wins over later trade; trade fills when no quote", () => {
  const marks: LiveMarkMap = new Map();
  applyMarketEventToMarks(marks, "market.Trade", ".MU260731C1400", { price: 3.9 });
  expect(marks.get("MU 260731C01400000")?.mark).toBe(3.9);
  expect(marks.get("MU 260731C01400000")?.hasQuote).toBe(false);

  applyMarketEventToMarks(marks, "market.Quote", ".MU260731C1400", {
    bidPrice: 3.7,
    askPrice: 3.75,
  });
  expect(marks.get("MU 260731C01400000")?.mark).toBeCloseTo(3.725);
  expect(marks.get("MU 260731C01400000")?.hasQuote).toBe(true);

  applyMarketEventToMarks(marks, "market.Trade", ".MU260731C1400", { price: 3.5 });
  // Trade must not overwrite quote quality.
  expect(marks.get("MU 260731C01400000")?.mark).toBeCloseTo(3.725);
});

test("overlay recalculates leg, combined strategy, and leaves sibling stable", () => {
  const put = leg({
    symbol: "MU    260731P00800000",
    strike_price: 800,
    option_type: "P",
    mark_price: 4.62,
    market_value: 462,
    cost_basis: 250,
    unrealized_pnl: -212,
    pnl_open: -212,
    pnl_open_percent: null,
    pnl_open_basis: 250,
    roll_adjustment: 0,
    roll_count: 0,
    roll_history_status: "none",
  });
  const call = leg();
  const original = strategy({}, [put, call]);
  const marks = new Map<string, LiveMarkState>([
    ["MU 260731C01400000", { mark: 3.725, hasQuote: true }],
  ]);

  const overlaid = overlayLivePnlOnStrategies([original], marks, {
    streamUsable: true,
  });
  expect(overlaid).not.toBe(original as Strategy[]);
  // Source snapshot must not be mutated.
  expect(original.legs[1]!.mark_price).toBe(4.05);
  expect(original.legs[1]!.pnl_open).toBe(1557);

  const next = overlaid[0]!;
  expect(next.legs[0]!.pnl_open).toBe(-212);
  expect(next.legs[0]!.mark_price).toBe(4.62);
  expect(next.legs[1]!.mark_price).toBeCloseTo(3.725);
  expect(next.legs[1]!.unrealized_pnl).toBeCloseTo(1228.5);
  expect(next.legs[1]!.pnl_open).toBeCloseTo(1228.5 + 361);
  expect(next.pnl_open).toBeCloseTo(-212 + 1228.5 + 361);
});

test("degraded stream falls back to snapshot without overlay", () => {
  const original = strategy();
  const input = [original];
  const marks = new Map<string, LiveMarkState>([
    ["MU 260731C01400000", { mark: 3.725, hasQuote: true }],
  ]);
  const overlaid = overlayLivePnlOnStrategies(input, marks, {
    streamUsable: false,
  });
  expect(overlaid).toBe(input);
  expect(overlaid[0]!.legs[0]!.mark_price).toBe(4.05);
  expect(overlaid[0]!.legs[0]!.pnl_open).toBe(1557);
});

test("pruneLiveMarks drops symbols not in holdings", () => {
  const marks: LiveMarkMap = new Map([
    ["MU 260731C01400000", { mark: 1, hasQuote: true }],
    ["SPY", { mark: 500, hasQuote: true }],
  ]);
  pruneLiveMarks(marks, heldMatchKeysFromStrategies([strategy()]));
  expect(marks.has("MU 260731C01400000")).toBe(true);
  expect(marks.has("SPY")).toBe(false);
});

test("retireLiveMarksIfSnapshotChanged clears only when snapshot_id changes", () => {
  const marks: LiveMarkMap = new Map([
    ["MU 260731C01400000", { mark: 3.725, hasQuote: true }],
  ]);
  // First snapshot / same id must not clear (render churn).
  expect(retireLiveMarksIfSnapshotChanged(marks, null, "snap-a")).toBe(false);
  expect(marks.size).toBe(1);
  expect(retireLiveMarksIfSnapshotChanged(marks, "snap-a", "snap-a")).toBe(false);
  expect(marks.size).toBe(1);

  // Authoritative snapshot B retires prior live marks.
  expect(retireLiveMarksIfSnapshotChanged(marks, "snap-a", "snap-b")).toBe(true);
  expect(marks.size).toBe(0);

  // After retirement, overlay falls through to snapshot B values.
  const snapB = strategy({
    pnl_open: 2000,
    legs: [
      leg({
        mark_price: 2.0,
        market_value: 200,
        unrealized_pnl: 1401,
        pnl_open: 1762,
      }),
    ],
  });
  const overlaid = overlayLivePnlOnStrategies([snapB], marks, { streamUsable: true });
  expect(overlaid[0]!.pnl_open).toBe(2000);
  expect(overlaid[0]!.legs[0]!.mark_price).toBe(2.0);
  expect(overlaid[0]!.legs[0]!.pnl_open).toBe(1762);
});
