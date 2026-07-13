/**
 * Pure helpers for live Positions P/L overlay from DXLink SSE market events.
 *
 * REST portfolio snapshots remain authoritative for holdings, quantities, bases,
 * roll adjustments, and strategy grouping. Live ticks only overlay mark/P&L for
 * display while streaming is healthy — never persisted and never mutates inputs.
 */

import type { PositionLeg, Strategy } from "./types";

/** In-memory live mark for one instrument, derived from Quote/Trade events. */
export type LiveMarkState = {
  /** Deterministic usable mark (quote midpoint preferred). */
  mark: number | null;
  /** True once a usable quote (bid and/or ask) has been observed. */
  hasQuote: boolean;
};

export type LiveMarkMap = Map<string, LiveMarkState>;

const BASIS_EPSILON = 1e-9;

/**
 * Usable market price: finite and >= 0.
 * Zero is valid (e.g. zero bid); negatives, NaN, and infinity are rejected.
 */
export function isValidPrice(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/**
 * Retire all live marks when the authoritative snapshot identity changes.
 * Same snapshot_id leaves the map intact (React re-renders must not clear).
 * Returns true when marks were cleared.
 */
export function retireLiveMarksIfSnapshotChanged(
  marks: LiveMarkMap,
  previousSnapshotId: string | null | undefined,
  nextSnapshotId: string | null | undefined,
): boolean {
  if (
    previousSnapshotId != null &&
    nextSnapshotId != null &&
    previousSnapshotId !== nextSnapshotId
  ) {
    marks.clear();
    return true;
  }
  return false;
}

/**
 * Normalize portfolio or event symbols for matching.
 * Collapses whitespace so padded OCC and single-spaced OCC share one key.
 */
export function normalizeMatchSymbol(symbol: string | null | undefined): string {
  if (!symbol) return "";
  return symbol.trim().replace(/\s+/g, " ").toUpperCase();
}

/**
 * Convert DXLink option notation (`.MU260731C1400`) to broker OCC
 * (`MU    260731C01400000`). Equities and already-OCC symbols pass through.
 */
export function fromDxlinkSymbol(symbol: string): string {
  const normalized = symbol.trim().toUpperCase();
  if (!normalized.startsWith(".")) {
    return normalized;
  }
  const match = /^\.([A-Z0-9.]+?)(\d{6})([CP])(\d+(?:\.\d+)?)$/.exec(normalized);
  if (!match) {
    return normalized;
  }
  const [, root, expiration, optionType, strikeText] = match;
  const encodedStrike = Math.round(Number(strikeText) * 1000);
  const rootField = root.padEnd(6, " ");
  return `${rootField}${expiration}${optionType}${String(encodedStrike).padStart(8, "0")}`;
}

/** Stable match key for SSE event symbols (DXLink or OCC) and portfolio legs. */
export function matchKeyFromEventSymbol(symbol: string): string {
  return normalizeMatchSymbol(fromDxlinkSymbol(symbol));
}

export function matchKeyFromLegSymbol(symbol: string): string {
  return normalizeMatchSymbol(symbol);
}

/**
 * Usable mark from quote sides: midpoint when both present (zero bid is valid),
 * otherwise the single valid side.
 */
export function markFromQuote(
  bid: unknown,
  ask: unknown,
): number | null {
  const bidOk = isValidPrice(bid);
  const askOk = isValidPrice(ask);
  if (bidOk && askOk) {
    return (bid + ask) / 2;
  }
  if (bidOk) return bid;
  if (askOk) return ask;
  return null;
}

/**
 * Apply a market.Quote or market.Trade event into the mark map.
 * Quote marks take priority; trades only fill when no usable quote exists.
 * Returns true when the stored mark changed.
 */
export function applyMarketEventToMarks(
  marks: LiveMarkMap,
  eventType: string,
  symbol: string,
  values: Record<string, unknown>,
): boolean {
  const key = matchKeyFromEventSymbol(symbol);
  if (!key) return false;

  const previous = marks.get(key) ?? { mark: null, hasQuote: false };
  let next: LiveMarkState = previous;

  if (eventType === "market.Quote" || eventType === "Quote") {
    const quoteMark = markFromQuote(values.bidPrice ?? values.bid, values.askPrice ?? values.ask);
    if (quoteMark != null) {
      next = { mark: quoteMark, hasQuote: true };
    }
  } else if (eventType === "market.Trade" || eventType === "Trade" || eventType === "market.TradeETH") {
    // Do not overwrite a higher-quality quote mark with a trade print.
    if (!previous.hasQuote && isValidPrice(values.price)) {
      next = { mark: values.price, hasQuote: false };
    }
  } else {
    return false;
  }

  if (next.mark === previous.mark && next.hasQuote === previous.hasQuote) {
    return false;
  }
  marks.set(key, next);
  return true;
}

/** Prune marks for symbols no longer held; keeps map bounded to current book. */
export function pruneLiveMarks(
  marks: LiveMarkMap,
  heldSymbols: Iterable<string>,
): LiveMarkMap {
  const held = new Set(
    [...heldSymbols].map(matchKeyFromLegSymbol).filter(Boolean),
  );
  for (const key of marks.keys()) {
    if (!held.has(key)) {
      marks.delete(key);
    }
  }
  return marks;
}

/** Collect match keys for all legs in a strategy list. */
export function heldMatchKeysFromStrategies(
  strategies: readonly Pick<Strategy, "legs">[],
): string[] {
  const keys: string[] = [];
  for (const strategy of strategies) {
    for (const leg of strategy.legs ?? []) {
      const key = matchKeyFromLegSymbol(leg.symbol);
      if (key) keys.push(key);
    }
  }
  return keys;
}

function isShortLeg(leg: Pick<PositionLeg, "quantity" | "quantity_direction">): boolean {
  const direction = (leg.quantity_direction ?? "").trim();
  if (direction === "Short") return true;
  if (direction === "Long") return false;
  return leg.quantity < 0;
}

/**
 * Raw broker-style unrealized P/L from a live mark (no roll carry).
 * Matches snapshot accounting: market_value = mark * |qty| * mult;
 * short: cost_basis - market_value; long: market_value - cost_basis.
 */
export function rawUnrealizedFromMark(
  leg: Pick<
    PositionLeg,
    "quantity" | "quantity_direction" | "cost_basis" | "multiplier"
  >,
  mark: number,
): number {
  const multiplier = leg.multiplier || 1;
  const marketValue = mark * Math.abs(leg.quantity) * multiplier;
  const costBasis = leg.cost_basis ?? 0;
  if (isShortLeg(leg)) {
    return costBasis - marketValue;
  }
  return marketValue - costBasis;
}

/**
 * Display P/L Open for a leg with a live mark: raw + roll_adjustment once.
 * Partial/no-roll chains already carry roll_adjustment === 0 on the snapshot.
 */
export function liveLegOpenPnl(
  leg: Pick<
    PositionLeg,
    | "quantity"
    | "quantity_direction"
    | "cost_basis"
    | "multiplier"
    | "roll_adjustment"
  >,
  mark: number,
): number {
  const raw = rawUnrealizedFromMark(leg, mark);
  const adjustment = leg.roll_adjustment ?? 0;
  return raw + adjustment;
}

/**
 * Percent for live open P/L using snapshot pnl_open_basis; falls back like server.
 */
export function liveLegOpenPnlPercent(
  leg: Pick<PositionLeg, "pnl_open_basis" | "cost_basis" | "pnl_open_percent" | "unrealized_pnl_percent">,
  liveOpenPnl: number,
  liveRawPnl: number,
): number | null {
  const basis = leg.pnl_open_basis;
  if (basis != null && Number.isFinite(basis) && Math.abs(basis) > BASIS_EPSILON) {
    return (liveOpenPnl / Math.abs(basis)) * 100;
  }
  const cost = leg.cost_basis;
  if (cost != null && Number.isFinite(cost) && Math.abs(cost) > BASIS_EPSILON) {
    return (liveRawPnl / Math.abs(cost)) * 100;
  }
  if (leg.pnl_open_percent != null && Number.isFinite(leg.pnl_open_percent)) {
    return leg.pnl_open_percent;
  }
  if (leg.unrealized_pnl_percent != null && Number.isFinite(leg.unrealized_pnl_percent)) {
    return leg.unrealized_pnl_percent;
  }
  return null;
}

function overlayLeg(leg: PositionLeg, marks: ReadonlyMap<string, LiveMarkState>): PositionLeg {
  const key = matchKeyFromLegSymbol(leg.symbol);
  const state = key ? marks.get(key) : undefined;
  if (state?.mark == null || !isValidPrice(state.mark)) {
    return leg;
  }
  // Without cost basis we cannot recompute raw P/L safely — keep snapshot.
  if (leg.cost_basis == null || !Number.isFinite(leg.cost_basis)) {
    return leg;
  }
  const mark = state.mark;
  const multiplier = leg.multiplier || 1;
  const marketValue = mark * Math.abs(leg.quantity) * multiplier;
  const raw = rawUnrealizedFromMark(leg, mark);
  const pnlOpen = liveLegOpenPnl(leg, mark);
  const rawPercent =
    leg.cost_basis != null && Math.abs(leg.cost_basis) > BASIS_EPSILON
      ? (raw / Math.abs(leg.cost_basis)) * 100
      : null;
  return {
    ...leg,
    mark_price: mark,
    market_value: marketValue,
    unrealized_pnl: raw,
    unrealized_pnl_percent: rawPercent,
    pnl_open: pnlOpen,
    pnl_open_percent: liveLegOpenPnlPercent(leg, pnlOpen, raw),
  };
}

function overlayStrategy(
  strategy: Strategy,
  marks: ReadonlyMap<string, LiveMarkState>,
): Strategy {
  const legs = (strategy.legs ?? []).map((leg) => overlayLeg(leg, marks));
  const anyLive = legs.some((leg, index) => leg !== (strategy.legs ?? [])[index]);
  if (!anyLive) {
    return strategy;
  }
  const unrealized = legs.reduce((sum, leg) => sum + (leg.unrealized_pnl ?? 0), 0);
  const pnlOpen = legs.reduce((sum, leg) => {
    if (leg.pnl_open != null && Number.isFinite(leg.pnl_open)) {
      return sum + leg.pnl_open;
    }
    return sum + (leg.unrealized_pnl ?? 0);
  }, 0);
  const basisSum = legs.reduce((sum, leg) => {
    if (leg.pnl_open_basis != null && Number.isFinite(leg.pnl_open_basis)) {
      return sum + leg.pnl_open_basis;
    }
    return sum;
  }, 0);
  const hasBasis = legs.some(
    (leg) => leg.pnl_open_basis != null && Number.isFinite(leg.pnl_open_basis),
  );
  let pnlOpenPercent: number | null;
  if (hasBasis && Math.abs(basisSum) > BASIS_EPSILON) {
    pnlOpenPercent = (pnlOpen / Math.abs(basisSum)) * 100;
  } else if (strategy.pnl_open_percent != null && Number.isFinite(strategy.pnl_open_percent)) {
    pnlOpenPercent = strategy.pnl_open_percent;
  } else {
    pnlOpenPercent = strategy.unrealized_pnl_percent ?? null;
  }
  const rollAdjustment = legs.reduce((sum, leg) => sum + (leg.roll_adjustment ?? 0), 0);
  return {
    ...strategy,
    legs,
    unrealized_pnl: unrealized,
    unrealized_pnl_percent:
      strategy.unrealized_pnl_percent != null
        ? strategy.unrealized_pnl_percent
        : null,
    pnl_open: pnlOpen,
    pnl_open_percent: pnlOpenPercent,
    pnl_open_basis: hasBasis ? Math.abs(basisSum) : strategy.pnl_open_basis,
    roll_adjustment: rollAdjustment,
  };
}

/**
 * Overlay live marks onto a strategy list without mutating the snapshot.
 * When streaming is not usable, returns the input array reference unchanged.
 */
export function overlayLivePnlOnStrategies(
  strategies: readonly Strategy[],
  marks: ReadonlyMap<string, LiveMarkState>,
  options: { streamUsable: boolean },
): Strategy[] {
  if (!options.streamUsable || marks.size === 0 || strategies.length === 0) {
    return strategies as Strategy[];
  }
  let changed = false;
  const next = strategies.map((strategy) => {
    const overlaid = overlayStrategy(strategy, marks);
    if (overlaid !== strategy) changed = true;
    return overlaid;
  });
  return changed ? next : (strategies as Strategy[]);
}

/** Resolve a display mark from the live map for a leg symbol. */
export function liveMarkForSymbol(
  marks: ReadonlyMap<string, LiveMarkState>,
  symbol: string,
): number | null {
  const state = marks.get(matchKeyFromLegSymbol(symbol));
  return state?.mark ?? null;
}
