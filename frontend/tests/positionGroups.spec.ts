import { expect, test } from "@playwright/test";

import {
  clampPnlBarPercent,
  classifyStrategy,
  countStrategiesByCategory,
  filterStrategiesByCategory,
  formatCompactExpiration,
  formatDteLabel,
  formatLegCountLabel,
  formatLegInstrument,
  formatLegSideQuantity,
  formatLegStrike,
  formatOptionTypeCode,
  formatSignedQuantity,
  formatSymbolGroupSplit,
  getLegIdentitySegments,
  groupStrategiesBySymbol,
  isCombinedOptionStrategy,
  isEquityLeg,
  isOptionLeg,
  normalizeUnderlying,
  presentStrategyIdsFromStrategies,
  presentSymbolsFromStrategies,
  pruneCollapsedSymbols,
  pruneExpandedStrategyIds,
  sanitizeStrategyDomId,
  sanitizeSymbolDomId,
  strategyLegsPanelId,
  symbolGroupPanelId,
  UNKNOWN_UNDERLYING_SYMBOL,
} from "../src/positionGroups";
import type { PositionLeg, Strategy } from "../src/types";

function leg(overrides: Partial<PositionLeg> = {}): PositionLeg {
  return {
    symbol: "SPY",
    underlying_symbol: "SPY",
    quantity: 100,
    quantity_direction: "Long",
    position_type: "Equity",
    strike_price: null,
    option_type: null,
    expiration_date: null,
    days_to_expiration: null,
    mark_price: 500,
    market_value: 50000,
    unrealized_pnl: 100,
    unrealized_pnl_percent: 1,
    delta: 1,
    gamma: 0,
    theta: 0,
    vega: 0,
    implied_volatility: null,
    multiplier: 1,
    horizon: "strategic",
    ...overrides,
  };
}

function optionLeg(overrides: Partial<PositionLeg> = {}): PositionLeg {
  return leg({
    symbol: "SPY  260821C00500000",
    position_type: "Equity Option",
    strike_price: 500,
    option_type: "C",
    expiration_date: "2026-08-21",
    days_to_expiration: 21,
    multiplier: 100,
    delta: 0.4,
    theta: -0.05,
    ...overrides,
  });
}

function strategy(overrides: Partial<Strategy> & Pick<Strategy, "strategy_id">): Strategy {
  return {
    account_id: "acct",
    underlying: "SPY",
    strategy_type: "Short Put",
    expiration_date: "2026-08-21",
    days_to_expiration: 21,
    quantity: 1,
    strikes: "$500",
    unrealized_pnl: 40,
    unrealized_pnl_percent: 10,
    total_delta: -20,
    total_theta: 4,
    horizon: "tactical",
    legs: [],
    ...overrides,
  };
}

test.describe("normalizeUnderlying", () => {
  test("trims and uppercases for stable display keys", () => {
    expect(normalizeUnderlying("  spy ")).toBe("SPY");
    expect(normalizeUnderlying("qQq")).toBe("QQQ");
  });

  test("maps missing or whitespace-only underlyings to UNKNOWN", () => {
    expect(normalizeUnderlying("")).toBe(UNKNOWN_UNDERLYING_SYMBOL);
    expect(normalizeUnderlying("   ")).toBe(UNKNOWN_UNDERLYING_SYMBOL);
    expect(normalizeUnderlying(null)).toBe(UNKNOWN_UNDERLYING_SYMBOL);
    expect(normalizeUnderlying(undefined)).toBe(UNKNOWN_UNDERLYING_SYMBOL);
  });

  test("preserves distinct valid symbols including special characters", () => {
    expect(normalizeUnderlying("brk/b")).toBe("BRK/B");
    expect(normalizeUnderlying(" spx.w ")).toBe("SPX.W");
    expect(normalizeUnderlying("^vix")).toBe("^VIX");
  });
});

test.describe("symbol DOM id sanitizer", () => {
  test("leaves plain symbols intact inside panel ids", () => {
    expect(sanitizeSymbolDomId("SPY")).toBe("SPY");
    expect(symbolGroupPanelId("spy")).toBe("symbol-group-panel-SPY");
  });

  test("encodes fragile characters deterministically and uniquely", () => {
    expect(sanitizeSymbolDomId("BRK/B")).toBe("BRK_2f_B");
    expect(sanitizeSymbolDomId("SPX.W")).toBe("SPX_2e_W");
    expect(sanitizeSymbolDomId("^VIX")).toBe("_5e_VIX");
    expect(sanitizeSymbolDomId("FOO BAR")).toBe("FOO_20_BAR");
    expect(symbolGroupPanelId("brk/b")).toBe("symbol-group-panel-BRK_2f_B");
    // Distinct normalized symbols never collide after sanitizing.
    expect(sanitizeSymbolDomId("A/B")).not.toBe(sanitizeSymbolDomId("A.B"));
    expect(sanitizeSymbolDomId("A/B")).not.toBe(sanitizeSymbolDomId("A_B"));
    // Regression: slash-encoding must not collide with a literal `_2f_` symbol.
    // (normalize uppercases first, so A_2f_B → A_2F_B before encoding underscores.)
    expect(sanitizeSymbolDomId("A/B")).toBe("A_2f_B");
    expect(sanitizeSymbolDomId("A_2f_B")).toBe("A_5f_2F_5f_B");
    expect(sanitizeSymbolDomId("A/B")).not.toBe(sanitizeSymbolDomId("A_2f_B"));
  });

  test("blank underlyings produce a stable UNKNOWN panel id", () => {
    expect(sanitizeSymbolDomId("")).toBe(UNKNOWN_UNDERLYING_SYMBOL);
    expect(symbolGroupPanelId("  ")).toBe("symbol-group-panel-UNKNOWN");
  });
});

test.describe("formatSymbolGroupSplit", () => {
  test("uses singular option and stock copy correctly", () => {
    expect(formatSymbolGroupSplit(1, 1)).toBe("1 stock · 1 option");
    expect(formatSymbolGroupSplit(2, 3)).toBe("2 stock · 3 options");
    expect(formatSymbolGroupSplit(0, 1)).toBe("1 option");
    expect(formatSymbolGroupSplit(1, 0)).toBe("1 stock");
    expect(formatSymbolGroupSplit(0, 0)).toBeNull();
  });
});

test.describe("classifyStrategy", () => {
  test("stock-only when every leg is equity", () => {
    expect(
      classifyStrategy(
        strategy({
          strategy_id: "s1",
          strategy_type: "Long Stock",
          legs: [leg()],
        }),
      ),
    ).toBe("stock");
  });

  test("options when any leg is an option", () => {
    expect(
      classifyStrategy(
        strategy({
          strategy_id: "s2",
          strategy_type: "Short Put",
          legs: [optionLeg()],
        }),
      ),
    ).toBe("options");
  });

  test("mixed stock+option structures classify as options", () => {
    expect(
      classifyStrategy(
        strategy({
          strategy_id: "s3",
          strategy_type: "Covered Call",
          legs: [leg({ quantity: 100 }), optionLeg({ quantity_direction: "Short" })],
        }),
      ),
    ).toBe("options");
    expect(
      classifyStrategy(
        strategy({
          strategy_id: "s4",
          strategy_type: "Collar",
          legs: [
            leg(),
            optionLeg({ option_type: "P", strike_price: 480 }),
            optionLeg({ option_type: "C", strike_price: 520 }),
          ],
        }),
      ),
    ).toBe("options");
    expect(
      classifyStrategy(
        strategy({
          strategy_id: "s5",
          strategy_type: "Protective Put",
          legs: [leg(), optionLeg({ option_type: "P" })],
        }),
      ),
    ).toBe("options");
  });

  test("missing-leg fallback recognizes stock type names", () => {
    expect(
      classifyStrategy(strategy({ strategy_id: "a", strategy_type: "Long Stock", legs: [] })),
    ).toBe("stock");
    expect(
      classifyStrategy(strategy({ strategy_id: "b", strategy_type: "Short Stock", legs: [] })),
    ).toBe("stock");
    expect(
      classifyStrategy(strategy({ strategy_id: "c", strategy_type: "Stock", legs: [] })),
    ).toBe("stock");
  });

  test("unknown and option strategy types without legs fall back to options", () => {
    expect(
      classifyStrategy(strategy({ strategy_id: "d", strategy_type: "Short Put", legs: [] })),
    ).toBe("options");
    expect(
      classifyStrategy(strategy({ strategy_id: "e", strategy_type: "Mystery", legs: [] })),
    ).toBe("options");
    expect(
      classifyStrategy(strategy({ strategy_id: "f", strategy_type: "Iron Condor", legs: [] })),
    ).toBe("options");
  });

  test("isEquityLeg rejects option instruments and option-like fields", () => {
    expect(isEquityLeg(leg())).toBe(true);
    expect(isEquityLeg(optionLeg())).toBe(false);
    expect(isEquityLeg(leg({ position_type: "Future Option" }))).toBe(false);
    expect(isEquityLeg(leg({ option_type: "P" }))).toBe(false);
    expect(isEquityLeg(leg({ strike_price: 100 }))).toBe(false);
  });
});

test.describe("groupStrategiesBySymbol", () => {
  test("groups by normalized underlying and sorts symbols alphabetically", () => {
    const groups = groupStrategiesBySymbol([
      strategy({ strategy_id: "1", underlying: "qqq", strategy_type: "Iron Condor", unrealized_pnl: 10 }),
      strategy({ strategy_id: "2", underlying: " SPY ", strategy_type: "Long Stock", unrealized_pnl: 20 }),
      strategy({ strategy_id: "3", underlying: "spy", strategy_type: "Short Put", unrealized_pnl: 30 }),
      strategy({ strategy_id: "4", underlying: "AAPL", strategy_type: "Covered Call", unrealized_pnl: -5 }),
    ]);
    expect(groups.map((g) => g.symbol)).toEqual(["AAPL", "QQQ", "SPY"]);
    const spy = groups.find((g) => g.symbol === "SPY");
    expect(spy?.totalCount).toBe(2);
    expect(spy?.strategies.map((s) => s.strategy_id)).toEqual(["2", "3"]);
    expect(spy?.unrealizedPnl).toBe(50);
  });

  test("blank underlyings collapse into a single UNKNOWN group", () => {
    const groups = groupStrategiesBySymbol([
      strategy({ strategy_id: "blank-a", underlying: "", strategy_type: "Short Put", unrealized_pnl: 1 }),
      strategy({ strategy_id: "blank-b", underlying: "  ", strategy_type: "Long Stock", unrealized_pnl: 2 }),
      strategy({ strategy_id: "spy", underlying: "SPY", strategy_type: "Short Put", unrealized_pnl: 3 }),
    ]);
    expect(groups.map((g) => g.symbol)).toEqual(["SPY", UNKNOWN_UNDERLYING_SYMBOL]);
    const unknown = groups.find((g) => g.symbol === UNKNOWN_UNDERLYING_SYMBOL);
    expect(unknown?.totalCount).toBe(2);
    expect(unknown?.strategies.map((s) => s.strategy_id)).toEqual(["blank-a", "blank-b"]);
  });

  test("same-symbol stock and option rows share one group with split counts", () => {
    const groups = groupStrategiesBySymbol([
      strategy({
        strategy_id: "stock",
        underlying: "SPY",
        strategy_type: "Long Stock",
        legs: [leg()],
        unrealized_pnl: 100,
      }),
      strategy({
        strategy_id: "opt",
        underlying: "SPY",
        strategy_type: "Short Put",
        legs: [optionLeg()],
        unrealized_pnl: 40,
      }),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0]?.symbol).toBe("SPY");
    expect(groups[0]?.stockCount).toBe(1);
    expect(groups[0]?.optionsCount).toBe(1);
    expect(groups[0]?.totalCount).toBe(2);
    expect(groups[0]?.unrealizedPnl).toBe(140);
  });
});

test.describe("filterStrategiesByCategory", () => {
  const rows = [
    strategy({
      strategy_id: "stock",
      strategy_type: "Long Stock",
      legs: [leg()],
    }),
    strategy({
      strategy_id: "put",
      strategy_type: "Short Put",
      legs: [optionLeg()],
    }),
    strategy({
      strategy_id: "cc",
      strategy_type: "Covered Call",
      legs: [leg(), optionLeg({ quantity_direction: "Short" })],
    }),
  ];

  test("Stock toggle hides stock-only without hiding option/mixed", () => {
    const filtered = filterStrategiesByCategory(rows, { showStock: false, showOptions: true });
    expect(filtered.map((s) => s.strategy_id)).toEqual(["put", "cc"]);
  });

  test("Options toggle hides option/mixed without hiding stock-only", () => {
    const filtered = filterStrategiesByCategory(rows, { showStock: true, showOptions: false });
    expect(filtered.map((s) => s.strategy_id)).toEqual(["stock"]);
  });

  test("both off yields empty list", () => {
    expect(filterStrategiesByCategory(rows, { showStock: false, showOptions: false })).toEqual([]);
  });

  test("countStrategiesByCategory reports zeros for empty categories", () => {
    expect(countStrategiesByCategory(rows)).toEqual({ stock: 1, options: 2 });
    expect(countStrategiesByCategory([])).toEqual({ stock: 0, options: 0 });
  });
});

test.describe("collapse state pruning", () => {
  test("pruneCollapsedSymbols drops symbols that disappear", () => {
    const collapsed = new Set(["SPY", "QQQ", "IWM"]);
    const next = pruneCollapsedSymbols(collapsed, ["spy", "AAPL"]);
    expect([...next].sort()).toEqual(["SPY"]);
  });

  test("presentSymbolsFromStrategies is sorted and normalized", () => {
    expect(
      presentSymbolsFromStrategies([
        { underlying: "qqq" },
        { underlying: " SPY" },
        { underlying: "spy" },
      ]),
    ).toEqual(["QQQ", "SPY"]);
  });
});

test.describe("isCombinedOptionStrategy", () => {
  test("four-leg Iron Condor qualifies as combined", () => {
    const ic = strategy({
      strategy_id: "ic",
      strategy_type: "Iron Condor",
      legs: [
        optionLeg({ strike_price: 470, option_type: "P", quantity_direction: "Long" }),
        optionLeg({ strike_price: 475, option_type: "P", quantity_direction: "Short" }),
        optionLeg({ strike_price: 505, option_type: "C", quantity_direction: "Short" }),
        optionLeg({ strike_price: 510, option_type: "C", quantity_direction: "Long" }),
      ],
    });
    expect(isCombinedOptionStrategy(ic)).toBe(true);
    expect(classifyStrategy(ic)).toBe("options");
    expect(formatLegCountLabel(ic.legs.length)).toBe("4 legs");
  });

  test("mixed Covered Call qualifies and keeps Options classification", () => {
    const cc = strategy({
      strategy_id: "cc",
      strategy_type: "Covered Call",
      legs: [leg({ quantity: 100 }), optionLeg({ quantity_direction: "Short", option_type: "C" })],
    });
    expect(isCombinedOptionStrategy(cc)).toBe(true);
    expect(classifyStrategy(cc)).toBe("options");
    expect(formatLegCountLabel(cc.legs.length)).toBe("2 legs");
  });

  test("single-leg stock and single-leg option do not qualify", () => {
    expect(
      isCombinedOptionStrategy(
        strategy({
          strategy_id: "stock",
          strategy_type: "Long Stock",
          legs: [leg()],
        }),
      ),
    ).toBe(false);
    expect(
      isCombinedOptionStrategy(
        strategy({
          strategy_id: "put",
          strategy_type: "Short Put",
          legs: [optionLeg()],
        }),
      ),
    ).toBe(false);
    expect(
      isCombinedOptionStrategy(
        strategy({ strategy_id: "empty", strategy_type: "Iron Condor", legs: [] }),
      ),
    ).toBe(false);
  });

  test("multi-leg pure stock does not qualify as combined option strategy", () => {
    expect(
      isCombinedOptionStrategy(
        strategy({
          strategy_id: "two-stock",
          strategy_type: "Long Stock",
          legs: [leg({ symbol: "SPY" }), leg({ symbol: "SPY", quantity: 50 })],
        }),
      ),
    ).toBe(false);
  });
});

test.describe("leg presentation helpers", () => {
  test("formatLegInstrument distinguishes Equity, Call, Put", () => {
    expect(formatLegInstrument(leg())).toBe("Equity");
    expect(formatLegInstrument(optionLeg({ option_type: "C" }))).toBe("Call");
    expect(formatLegInstrument(optionLeg({ option_type: "P" }))).toBe("Put");
    expect(formatLegInstrument(optionLeg({ option_type: "call" }))).toBe("Call");
  });

  test("formatLegSideQuantity and formatLegStrike", () => {
    expect(formatLegSideQuantity(leg({ quantity: 100, quantity_direction: "Long" }))).toBe(
      "Long 100",
    );
    expect(
      formatLegSideQuantity(optionLeg({ quantity: 1, quantity_direction: "Short" })),
    ).toBe("Short 1");
    expect(formatLegStrike(optionLeg({ strike_price: 500 }))).toBe("$500");
    expect(formatLegStrike(leg())).toBeNull();
  });

  test("isOptionLeg complements isEquityLeg", () => {
    expect(isOptionLeg(leg())).toBe(false);
    expect(isOptionLeg(optionLeg())).toBe(true);
    expect(isOptionLeg(leg({ option_type: "P" }))).toBe(true);
  });

  test("formatSignedQuantity and option type / DTE helpers", () => {
    expect(formatSignedQuantity(leg({ quantity: 100, quantity_direction: "Long" }))).toBe(
      "+100",
    );
    expect(
      formatSignedQuantity(optionLeg({ quantity: 1, quantity_direction: "Short" })),
    ).toBe("−1");
    expect(formatOptionTypeCode("C")).toBe("C");
    expect(formatOptionTypeCode("put")).toBe("P");
    expect(formatOptionTypeCode(null)).toBeNull();
    expect(formatDteLabel(18)).toBe("18d");
    expect(formatDteLabel(null)).toBeNull();
    expect(formatCompactExpiration("2026-08-15")).toMatch(/Aug/);
    expect(formatCompactExpiration("2026-08-15")).toMatch(/15/);
    expect(formatCompactExpiration(null)).toBeNull();
  });

  test("getLegIdentitySegments builds option and equity compact identities", () => {
    const put = getLegIdentitySegments(
      optionLeg({
        quantity: 1,
        quantity_direction: "Short",
        option_type: "P",
        strike_price: 475,
        expiration_date: "2026-08-15",
        days_to_expiration: 18,
      }),
    );
    expect(put.isEquity).toBe(false);
    expect(put.signedQuantity).toBe("−1");
    expect(put.sideQuantity).toBe("Short 1");
    expect(put.expiration).toMatch(/Aug/);
    expect(put.dte).toBe("18d");
    expect(put.strike).toBe("$475");
    expect(put.optionType).toBe("P");
    expect(put.instrument).toBe("Put");
    expect(put.accessibleLabel).toContain("Short 1");
    expect(put.accessibleLabel).toContain("Put");
    expect(put.accessibleLabel).not.toContain("null");
    expect(put.accessibleLabel).not.toContain("undefined");

    const equity = getLegIdentitySegments(leg({ quantity: 100, quantity_direction: "Long" }));
    expect(equity.isEquity).toBe(true);
    expect(equity.signedQuantity).toBe("+100");
    expect(equity.instrument).toBe("Equity");
    expect(equity.expiration).toBeNull();
    expect(equity.dte).toBeNull();
    expect(equity.strike).toBeNull();
    expect(equity.optionType).toBeNull();
    expect(equity.accessibleLabel).toContain("Long 100");
    expect(equity.accessibleLabel).toContain("Equity");
  });

  test("clampPnlBarPercent clamps visual bar only", () => {
    expect(clampPnlBarPercent(10)).toBe(10);
    expect(clampPnlBarPercent(-8)).toBe(8);
    expect(clampPnlBarPercent(250)).toBe(100);
    expect(clampPnlBarPercent(-999)).toBe(100);
    expect(clampPnlBarPercent(Number.NaN)).toBe(0);
  });
});

test.describe("strategy leg panel ids and expand pruning", () => {
  test("sanitizeStrategyDomId encodes fragile characters uniquely", () => {
    expect(sanitizeStrategyDomId("strat-qqq")).toBe("strat-qqq");
    expect(strategyLegsPanelId("strat-qqq")).toBe("strategy-legs-panel-strat-qqq");
    expect(sanitizeStrategyDomId("acct:SPY:ic")).toBe("acct_3a_SPY_3a_ic");
    expect(sanitizeStrategyDomId("a/b")).not.toBe(sanitizeStrategyDomId("a_b"));
    expect(sanitizeStrategyDomId("")).toBe("unknown-strategy");
  });

  test("pruneExpandedStrategyIds drops strategies that disappear", () => {
    const expanded = new Set(["strat-a", "strat-b", "strat-gone"]);
    const next = pruneExpandedStrategyIds(expanded, ["strat-a", "strat-c"]);
    expect([...next].sort()).toEqual(["strat-a"]);
  });

  test("presentStrategyIdsFromStrategies is sorted and unique", () => {
    expect(
      presentStrategyIdsFromStrategies([
        { strategy_id: "z" },
        { strategy_id: "a" },
        { strategy_id: "z" },
      ]),
    ).toEqual(["a", "z"]);
  });
});
