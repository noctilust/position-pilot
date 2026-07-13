/**
 * Presentation helpers for grouping and filtering portfolio strategies
 * by underlying symbol and stock vs options category.
 *
 * Pure and side-effect free — no backend fields or endpoints.
 */

import type { PositionLeg, Strategy } from "./types";

/** Visibility category for a strategy row. */
export type StrategyCategory = "stock" | "options";

export type CategoryVisibility = {
  showStock: boolean;
  showOptions: boolean;
};

/** One underlying symbol and its currently visible strategies. */
export type SymbolStrategyGroup = {
  /** Stable uppercase display symbol (blank underlyings → UNKNOWN). */
  symbol: string;
  strategies: Strategy[];
  stockCount: number;
  optionsCount: number;
  totalCount: number;
  unrealizedPnl: number;
};

/** Stock-only strategy type names used when legs are absent. */
const STOCK_STRATEGY_TYPES = new Set(["long stock", "short stock", "stock"]);

/** Display / grouping label for missing or whitespace-only underlyings. */
export const UNKNOWN_UNDERLYING_SYMBOL = "UNKNOWN";

type EquityLegFields = Pick<
  PositionLeg,
  "position_type" | "option_type" | "strike_price" | "expiration_date"
>;

/**
 * Normalize an underlying for grouping and display: trim + uppercase.
 * Empty/whitespace/null becomes the stable visible label UNKNOWN so blank
 * broker data does not create empty group headers or fragile empty ids.
 */
export function normalizeUnderlying(underlying: string | null | undefined): string {
  const trimmed = (underlying ?? "").trim().toUpperCase();
  return trimmed || UNKNOWN_UNDERLYING_SYMBOL;
}

/**
 * Deterministic HTML id fragment for a normalized symbol.
 *
 * Safe pass-through is only A–Z, a–z, 0–9, and `-`. Every other character
 * (including `_`) becomes `_xx_` (hex code point). Underscore must be encoded
 * too: otherwise `A/B` → `A_2f_B` collides with a literal symbol `A_2f_B`.
 * Distinct normalized symbols therefore produce distinct fragments.
 * Ordinary tickers like SPY stay readable.
 */
export function sanitizeSymbolDomId(symbol: string): string {
  const key = normalizeUnderlying(symbol);
  let out = "";
  for (const ch of key) {
    if (/[A-Za-z0-9-]/.test(ch)) {
      out += ch;
    } else {
      out += `_${ch.codePointAt(0)!.toString(16)}_`;
    }
  }
  return out || UNKNOWN_UNDERLYING_SYMBOL;
}

/** Full disclosure panel id for a symbol group (`aria-controls` target). */
export function symbolGroupPanelId(symbol: string): string {
  return `symbol-group-panel-${sanitizeSymbolDomId(symbol)}`;
}

/**
 * Compact split copy for a group header, e.g. "1 stock · 2 options".
 * Uses singular "option" when count is 1; stock stays "N stock".
 */
export function formatSymbolGroupSplit(
  stockCount: number,
  optionsCount: number,
): string | null {
  const parts: string[] = [];
  if (stockCount > 0) {
    parts.push(`${stockCount} stock`);
  }
  if (optionsCount > 0) {
    parts.push(optionsCount === 1 ? "1 option" : `${optionsCount} options`);
  }
  return parts.length ? parts.join(" · ") : null;
}

/**
 * True when a leg is a non-option equity (stock/ETF share).
 * Option instrument types and legs with option fields are never equity.
 */
export function isEquityLeg(leg: EquityLegFields): boolean {
  const positionType = (leg.position_type ?? "").trim().toLowerCase();
  if (positionType.includes("option")) {
    return false;
  }
  if (leg.option_type) {
    return false;
  }
  if (leg.strike_price != null || leg.expiration_date) {
    return false;
  }
  // Equity / Stock (and common equivalents). Unknown types are not treated as equity
  // so mixed/unknown risk falls through to the Options category.
  return (
    positionType === "equity" ||
    positionType === "stock" ||
    positionType === "common stock" ||
    positionType === "etf"
  );
}

/**
 * Classify a strategy as stock-only or options (including mixed structures).
 *
 * - Prefer legs: stock only when every leg is non-option equity.
 * - Any option leg (Covered Call, Collar, Protective Put, etc.) → options.
 * - Without legs: recognize Long Stock / Short Stock / Stock; everything else → options
 *   so option risk is never silently hidden by the Stock filter.
 */
export function classifyStrategy(
  strategy: Pick<Strategy, "strategy_type" | "legs">,
): StrategyCategory {
  const legs = strategy.legs ?? [];
  if (legs.length > 0) {
    return legs.every(isEquityLeg) ? "stock" : "options";
  }
  const typeName = (strategy.strategy_type ?? "").trim().toLowerCase();
  if (STOCK_STRATEGY_TYPES.has(typeName)) {
    return "stock";
  }
  return "options";
}

/** Count strategies in each category (independent of visibility toggles). */
export function countStrategiesByCategory(
  strategies: readonly Pick<Strategy, "strategy_type" | "legs">[],
): { stock: number; options: number } {
  let stock = 0;
  let options = 0;
  for (const strategy of strategies) {
    if (classifyStrategy(strategy) === "stock") {
      stock += 1;
    } else {
      options += 1;
    }
  }
  return { stock, options };
}

/**
 * Filter strategies by independent Stock / Options visibility.
 * Does not mutate portfolio data — presentation only.
 */
export function filterStrategiesByCategory<T extends Pick<Strategy, "strategy_type" | "legs">>(
  strategies: readonly T[],
  visibility: CategoryVisibility,
): T[] {
  return strategies.filter((strategy) => {
    const category = classifyStrategy(strategy);
    return category === "stock" ? visibility.showStock : visibility.showOptions;
  });
}

/**
 * Group strategies by normalized underlying. Groups are sorted alphabetically
 * by display symbol. Within a group, original relative order is preserved.
 */
export function groupStrategiesBySymbol(
  strategies: readonly Strategy[],
): SymbolStrategyGroup[] {
  const buckets = new Map<string, Strategy[]>();
  for (const strategy of strategies) {
    const symbol = normalizeUnderlying(strategy.underlying);
    const list = buckets.get(symbol);
    if (list) {
      list.push(strategy);
    } else {
      buckets.set(symbol, [strategy]);
    }
  }

  const symbols = [...buckets.keys()].sort((a, b) => a.localeCompare(b));
  return symbols.map((symbol) => {
    const rows = buckets.get(symbol) ?? [];
    let stockCount = 0;
    let optionsCount = 0;
    let unrealizedPnl = 0;
    for (const row of rows) {
      if (classifyStrategy(row) === "stock") {
        stockCount += 1;
      } else {
        optionsCount += 1;
      }
      unrealizedPnl += row.unrealized_pnl ?? 0;
    }
    return {
      symbol,
      strategies: rows,
      stockCount,
      optionsCount,
      totalCount: rows.length,
      unrealizedPnl,
    };
  });
}

/**
 * Prune collapse state to symbols that still exist in the portfolio.
 * Unknown symbols are dropped; known collapsed symbols are kept.
 */
export function pruneCollapsedSymbols(
  collapsed: ReadonlySet<string>,
  presentSymbols: readonly string[],
): Set<string> {
  const present = new Set(presentSymbols.map(normalizeUnderlying));
  const next = new Set<string>();
  for (const symbol of collapsed) {
    const key = normalizeUnderlying(symbol);
    if (present.has(key)) {
      next.add(key);
    }
  }
  return next;
}

/** Stable sorted list of present symbols from a strategy list (unfiltered). */
export function presentSymbolsFromStrategies(
  strategies: readonly Pick<Strategy, "underlying">[],
): string[] {
  const set = new Set<string>();
  for (const strategy of strategies) {
    set.add(normalizeUnderlying(strategy.underlying));
  }
  return [...set].sort((a, b) => a.localeCompare(b));
}

type OptionLegFields = EquityLegFields;

/**
 * True when a leg is an option instrument (or has option fields).
 * Complements isEquityLeg for mixed Covered Call / Collar detection.
 */
export function isOptionLeg(leg: OptionLegFields): boolean {
  const positionType = (leg.position_type ?? "").trim().toLowerCase();
  if (positionType.includes("option")) {
    return true;
  }
  if (leg.option_type) {
    return true;
  }
  if (leg.strike_price != null || leg.expiration_date) {
    return true;
  }
  return false;
}

/**
 * A multi-leg option structure detected by Position Pilot (backend StrategyService).
 *
 * Qualifies when classified as Options, has more than one leg, and includes at least
 * one option leg. Mixed stock+option structures (Covered Call, Collar, Protective Put)
 * qualify; pure single-leg stock/options do not. Does not invent combinations — only
 * makes backend-detected multi-leg strategies explicit for progressive disclosure.
 */
export function isCombinedOptionStrategy(
  strategy: Pick<Strategy, "strategy_type" | "legs">,
): boolean {
  const legs = strategy.legs ?? [];
  if (legs.length <= 1) {
    return false;
  }
  if (classifyStrategy(strategy) !== "options") {
    return false;
  }
  return legs.some(isOptionLeg);
}

/** Compact indicator, e.g. "4 legs" / "1 leg". */
export function formatLegCountLabel(legCount: number): string {
  return legCount === 1 ? "1 leg" : `${legCount} legs`;
}

/**
 * Human-readable instrument label for a leg ledger row.
 * Equity for stock legs; Call / Put for options; fallback to position_type.
 */
export function formatLegInstrument(leg: EquityLegFields): string {
  if (isEquityLeg(leg)) {
    return "Equity";
  }
  const opt = (leg.option_type ?? "").trim().toUpperCase();
  if (opt === "C" || opt === "CALL") {
    return "Call";
  }
  if (opt === "P" || opt === "PUT") {
    return "Put";
  }
  const positionType = (leg.position_type ?? "").trim();
  return positionType || "Option";
}

/**
 * Side + quantity for a leg, e.g. "Long 1" / "Short 100".
 * Uses quantity_direction when present; otherwise infers from signed quantity.
 */
export function formatLegSideQuantity(
  leg: Pick<PositionLeg, "quantity" | "quantity_direction">,
): string {
  const qty = Math.abs(leg.quantity);
  const direction = (leg.quantity_direction ?? "").trim();
  if (direction === "Long" || direction === "Short") {
    return `${direction} ${qty}`;
  }
  if (leg.quantity < 0) {
    return `Short ${qty}`;
  }
  return `Long ${qty}`;
}

/** Strike display when applicable; null for equity / missing strike. */
export function formatLegStrike(leg: Pick<PositionLeg, "strike_price">): string | null {
  if (leg.strike_price == null || Number.isNaN(leg.strike_price)) {
    return null;
  }
  return `$${leg.strike_price}`;
}

/**
 * Deterministic HTML id fragment for a strategy_id.
 * Same encoding rules as sanitizeSymbolDomId so ids stay collision-free.
 */
export function sanitizeStrategyDomId(strategyId: string): string {
  const key = (strategyId ?? "").trim();
  if (!key) {
    return "unknown-strategy";
  }
  let out = "";
  for (const ch of key) {
    if (/[A-Za-z0-9-]/.test(ch)) {
      out += ch;
    } else {
      out += `_${ch.codePointAt(0)!.toString(16)}_`;
    }
  }
  return out || "unknown-strategy";
}

/** Full disclosure panel id for a strategy's leg ledger (`aria-controls` target). */
export function strategyLegsPanelId(strategyId: string): string {
  return `strategy-legs-panel-${sanitizeStrategyDomId(strategyId)}`;
}

/**
 * Prune expanded strategy ids to those still present in the portfolio.
 * Unknown ids are dropped; known expanded ids are kept across filter/refresh.
 */
export function pruneExpandedStrategyIds(
  expanded: ReadonlySet<string>,
  presentStrategyIds: readonly string[],
): Set<string> {
  const present = new Set(presentStrategyIds);
  const next = new Set<string>();
  for (const id of expanded) {
    if (present.has(id)) {
      next.add(id);
    }
  }
  return next;
}

/** Stable sorted list of strategy_ids currently in the book (unfiltered). */
export function presentStrategyIdsFromStrategies(
  strategies: readonly Pick<Strategy, "strategy_id">[],
): string[] {
  return [...new Set(strategies.map((s) => s.strategy_id))].sort((a, b) => a.localeCompare(b));
}
