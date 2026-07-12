import { expect, test } from "@playwright/test";

import {
  catalystRiskActiveSegments,
  catalystRiskAriaLabel,
  catalystRiskLabel,
  catalystRiskTokenClass,
  deriveEventCatalystRisk,
  deriveSymbolCatalystRisk,
  type CatalystEventRiskInput,
  type SymbolCatalystRiskInput,
} from "../src/catalystRisk";

function event(
  overrides: Partial<CatalystEventRiskInput> = {},
): CatalystEventRiskInput {
  return {
    high_impact: false,
    confidence: "supporting",
    ...overrides,
  };
}

function symbol(
  overrides: Partial<SymbolCatalystRiskInput> = {},
): SymbolCatalystRiskInput {
  return {
    confidence: "no_confirmed_catalyst_found",
    meaningful_move: false,
    promoted: false,
    catalysts: [],
    ...overrides,
  };
}

test.describe("symbol catalyst risk policy", () => {
  test("high when any event has high_impact, regardless of other flags", () => {
    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "no_confirmed_catalyst_found",
          meaningful_move: false,
          promoted: false,
          catalysts: [event({ high_impact: true, confidence: "supporting" })],
        }),
      ),
    ).toBe("high");

    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "confirmed",
          meaningful_move: true,
          promoted: true,
          catalysts: [
            event({ high_impact: false, confidence: "likely" }),
            event({ high_impact: true, confidence: "supporting" }),
          ],
        }),
      ),
    ).toBe("high");
  });

  test("high-impact overrides low confidence and quiet aggregate signals", () => {
    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "supporting",
          meaningful_move: false,
          promoted: false,
          catalysts: [event({ high_impact: true })],
          coverage: "offline",
          cached: true,
          freshness: { state: "stale" },
        }),
      ),
    ).toBe("high");
  });

  test("medium when meaningful_move and not high", () => {
    expect(
      deriveSymbolCatalystRisk(
        symbol({
          meaningful_move: true,
          confidence: "no_confirmed_catalyst_found",
          catalysts: [],
        }),
      ),
    ).toBe("medium");
  });

  test("medium when promoted and not high", () => {
    expect(
      deriveSymbolCatalystRisk(
        symbol({
          promoted: true,
          confidence: "supporting",
          catalysts: [event({ confidence: "supporting" })],
        }),
      ),
    ).toBe("medium");
  });

  test("medium when aggregate confidence is confirmed", () => {
    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "confirmed",
          catalysts: [event({ confidence: "confirmed", high_impact: false })],
        }),
      ),
    ).toBe("medium");
  });

  test("medium when aggregate confidence is likely", () => {
    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "likely",
          catalysts: [event({ confidence: "likely", high_impact: false })],
        }),
      ),
    ).toBe("medium");
  });

  test("low for quiet / no-confirmed / supporting-only results", () => {
    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "no_confirmed_catalyst_found",
          catalysts: [],
        }),
      ),
    ).toBe("low");

    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "supporting",
          catalysts: [event({ confidence: "supporting" })],
        }),
      ),
    ).toBe("low");

    expect(
      deriveSymbolCatalystRisk(
        symbol({
          confidence: "unknown_bucket",
          catalysts: [event({ confidence: "rumor" })],
        }),
      ),
    ).toBe("low");
  });

  test("stale / offline / incomplete coverage does not alter risk", () => {
    const base = symbol({
      confidence: "no_confirmed_catalyst_found",
      catalysts: [],
      coverage: "complete",
      cached: false,
      freshness: { state: "fresh" },
    });
    const staleOffline = symbol({
      confidence: "no_confirmed_catalyst_found",
      catalysts: [],
      coverage: "offline",
      cached: true,
      freshness: { state: "stale" },
    });
    expect(deriveSymbolCatalystRisk(base)).toBe("low");
    expect(deriveSymbolCatalystRisk(staleOffline)).toBe("low");

    const mediumBase = symbol({
      confidence: "likely",
      coverage: "incomplete",
      cached: true,
      freshness: { state: "stale" },
      catalysts: [event({ confidence: "likely" })],
    });
    expect(deriveSymbolCatalystRisk(mediumBase)).toBe("medium");
  });
});

test.describe("event catalyst risk policy", () => {
  test("high when high_impact", () => {
    expect(
      deriveEventCatalystRisk(event({ high_impact: true, confidence: "supporting" })),
    ).toBe("high");
    expect(
      deriveEventCatalystRisk(event({ high_impact: true, confidence: "confirmed" })),
    ).toBe("high");
  });

  test("medium when confirmed or likely and not high", () => {
    expect(
      deriveEventCatalystRisk(event({ high_impact: false, confidence: "confirmed" })),
    ).toBe("medium");
    expect(
      deriveEventCatalystRisk(event({ high_impact: false, confidence: "likely" })),
    ).toBe("medium");
  });

  test("low otherwise", () => {
    expect(
      deriveEventCatalystRisk(event({ high_impact: false, confidence: "supporting" })),
    ).toBe("low");
    expect(
      deriveEventCatalystRisk(
        event({ high_impact: false, confidence: "no_confirmed_catalyst_found" }),
      ),
    ).toBe("low");
  });
});

test.describe("risk presentation helpers", () => {
  test("labels and aria text", () => {
    expect(catalystRiskLabel("low")).toBe("Low");
    expect(catalystRiskLabel("medium")).toBe("Medium");
    expect(catalystRiskLabel("high")).toBe("High");
    expect(catalystRiskAriaLabel("low")).toBe("Catalyst risk: Low");
    expect(catalystRiskAriaLabel("medium")).toBe("Catalyst risk: Medium");
    expect(catalystRiskAriaLabel("high")).toBe("Catalyst risk: High");
  });

  test("semantic token classes map low→good, medium→warn, high→danger", () => {
    expect(catalystRiskTokenClass("low")).toBe("good");
    expect(catalystRiskTokenClass("medium")).toBe("warn");
    expect(catalystRiskTokenClass("high")).toBe("danger");
  });

  test("segment counts are 1 / 2 / 3", () => {
    expect(catalystRiskActiveSegments("low")).toBe(1);
    expect(catalystRiskActiveSegments("medium")).toBe(2);
    expect(catalystRiskActiveSegments("high")).toBe(3);
  });
});
