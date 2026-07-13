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
 * Presentation sort rank for horizon: Strategic before Tactical (and other).
 * Does not mutate broker data — used only for within-group display order.
 */
export function horizonDisplayRank(horizon: string | null | undefined): number {
  return (horizon ?? "").trim().toLowerCase() === "strategic" ? 0 : 1;
}

/**
 * Sort strategies within one symbol group for the Positions table:
 * 1. Stock/equity strategies before every options strategy
 * 2. Within each asset bucket, Strategic before Tactical
 * 3. Equal-priority entries keep input relative order (stable sort)
 *
 * Multi-leg structures stay intact as single strategy entries; legs are never
 * reordered relative to their parent (legs render nested under the parent row).
 */
export function sortStrategiesForSymbolGroup(
  strategies: readonly Strategy[],
): Strategy[] {
  return strategies
    .map((strategy, index) => ({ strategy, index }))
    .sort((a, b) => {
      const catA = classifyStrategy(a.strategy) === "stock" ? 0 : 1;
      const catB = classifyStrategy(b.strategy) === "stock" ? 0 : 1;
      if (catA !== catB) return catA - catB;
      const horA = horizonDisplayRank(a.strategy.horizon);
      const horB = horizonDisplayRank(b.strategy.horizon);
      if (horA !== horB) return horA - horB;
      return a.index - b.index;
    })
    .map(({ strategy }) => strategy);
}

/**
 * Group strategies by normalized underlying. Groups are sorted alphabetically
 * by display symbol. Within a group: stock before options, then Strategic
 * before Tactical; equal-priority order is stable.
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
    const rows = sortStrategiesForSymbolGroup(buckets.get(symbol) ?? []);
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
 * Compact signed quantity for dense leg identity, e.g. "+1" / "−100".
 * Uses quantity_direction when present; otherwise infers from signed quantity.
 * Uses Unicode minus (U+2212) for short so it is visually distinct from a hyphen.
 */
export function formatSignedQuantity(
  leg: Pick<PositionLeg, "quantity" | "quantity_direction">,
): string {
  const qty = Math.abs(leg.quantity);
  const direction = (leg.quantity_direction ?? "").trim();
  const isShort =
    direction === "Short" || (direction !== "Long" && leg.quantity < 0);
  return isShort ? `−${qty}` : `+${qty}`;
}

/** C / P code for option legs; null when not an option or unknown. */
export function formatOptionTypeCode(
  optionType: string | null | undefined,
): string | null {
  const opt = (optionType ?? "").trim().toUpperCase();
  if (opt === "C" || opt === "CALL") return "C";
  if (opt === "P" || opt === "PUT") return "P";
  return null;
}

/**
 * Compact calendar expiration for contract identity, e.g. "Aug 15".
 * Date-only values (YYYY-MM-DD) format in UTC so the calendar day is stable.
 */
export function formatCompactExpiration(
  value: string | null | undefined,
): string | null {
  if (!value) return null;
  try {
    const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(value);
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      timeZone: dateOnly ? "UTC" : undefined,
    }).format(new Date(dateOnly ? `${value}T00:00:00Z` : value));
  } catch {
    return value;
  }
}

/** Compact DTE label, e.g. "18d"; null when unavailable. */
export function formatDteLabel(dte: number | null | undefined): string | null {
  if (dte == null || Number.isNaN(dte)) return null;
  return `${dte}d`;
}

/**
 * Clamp only the visual P/L bar fill length (0–100%). Does not alter the
 * displayed percentage value.
 */
export function clampPnlBarPercent(percent: number, max = 100): number {
  if (!Number.isFinite(percent)) return 0;
  const abs = Math.abs(percent);
  return Math.min(max, abs);
}

/** Segmented contract / equity identity for a position leg (dense first cell). */
export type LegIdentitySegments = {
  /** Signed quantity for visual density, e.g. "+1" / "−100". */
  signedQuantity: string;
  /** Long/Short + abs qty for screen readers and side column fallbacks. */
  sideQuantity: string;
  /** True when the leg is equity rather than an option. */
  isEquity: boolean;
  /** "Equity" / "Call" / "Put" / fallback instrument label. */
  instrument: string;
  /** Compact expiration, e.g. "Aug 15"; null for equity / missing. */
  expiration: string | null;
  /** Compact DTE, e.g. "18d"; null when missing. */
  dte: string | null;
  /** Strike with $, e.g. "$470"; null for equity / missing. */
  strike: string | null;
  /** Option type code C/P; null for equity / unknown. */
  optionType: string | null;
  /**
   * Single accessible phrase covering all segments, e.g.
   * "Short 1, Aug 15, 18 days, strike $470, Put".
   */
  accessibleLabel: string;
};

/**
 * Build compact identity segments for a backend PositionLeg.
 * Does not invent broker fields — only formats what the leg already carries.
 */
export function getLegIdentitySegments(leg: PositionLeg): LegIdentitySegments {
  const isEquity = isEquityLeg(leg);
  const instrument = formatLegInstrument(leg);
  const sideQuantity = formatLegSideQuantity(leg);
  const signedQuantity = formatSignedQuantity(leg);
  const expiration = isEquity ? null : formatCompactExpiration(leg.expiration_date);
  const dte = isEquity ? null : formatDteLabel(leg.days_to_expiration);
  const strike = isEquity ? null : formatLegStrike(leg);
  const optionType = isEquity ? null : formatOptionTypeCode(leg.option_type);

  const parts: string[] = [sideQuantity];
  if (isEquity) {
    parts.push(instrument);
  } else {
    if (expiration) parts.push(expiration);
    if (leg.days_to_expiration != null && !Number.isNaN(leg.days_to_expiration)) {
      parts.push(
        `${leg.days_to_expiration} day${leg.days_to_expiration === 1 ? "" : "s"} to expiration`,
      );
    }
    if (strike) parts.push(`strike ${strike}`);
    if (instrument) parts.push(instrument);
  }

  return {
    signedQuantity,
    sideQuantity,
    isEquity,
    instrument,
    expiration,
    dte,
    strike,
    optionType,
    accessibleLabel: parts.join(", "),
  };
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

/** Stable id for the first disclosed leg row (`aria-controls` target). */
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
