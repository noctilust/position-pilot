/**
 * Presentation policy for aggregate and per-event catalyst market risk.
 *
 * Risk is distinct from evidence confidence, coverage, and freshness.
 * Coverage / offline / stale-cache states must never alter this derivation.
 */

export type CatalystRiskLevel = "low" | "medium" | "high";

/** Minimal event shape used for risk derivation. */
export type CatalystEventRiskInput = {
  high_impact: boolean;
  confidence: string;
};

/** Minimal symbol-result shape used for aggregate risk derivation. */
export type SymbolCatalystRiskInput = {
  confidence: string;
  meaningful_move: boolean;
  promoted: boolean;
  catalysts: readonly CatalystEventRiskInput[];
  /** Intentionally unused — coverage must not change market-risk labels. */
  coverage?: string;
  cached?: boolean;
  freshness?: { state?: string } | null;
};

const ELEVATED_CONFIDENCE = new Set(["confirmed", "likely"]);

/** Human-readable risk label shown next to the gauge. */
export function catalystRiskLabel(level: CatalystRiskLevel): "Low" | "Medium" | "High" {
  if (level === "high") return "High";
  if (level === "medium") return "Medium";
  return "Low";
}

/**
 * Semantic token class suffix for CSS (`--good` / `--warn` / `--danger`).
 * low → good (green), medium → warn (yellow/amber), high → danger (red).
 */
export function catalystRiskTokenClass(
  level: CatalystRiskLevel,
): "good" | "warn" | "danger" {
  if (level === "high") return "danger";
  if (level === "medium") return "warn";
  return "good";
}

/** Active segment count for a three-step gauge: low=1, medium=2, high=3. */
export function catalystRiskActiveSegments(level: CatalystRiskLevel): 1 | 2 | 3 {
  if (level === "high") return 3;
  if (level === "medium") return 2;
  return 1;
}

/** Accessible name: "Catalyst risk: Low|Medium|High". */
export function catalystRiskAriaLabel(level: CatalystRiskLevel): string {
  return `Catalyst risk: ${catalystRiskLabel(level)}`;
}

/**
 * Event-level catalyst risk.
 * High when high_impact; medium when confidence is confirmed/likely; else low.
 */
export function deriveEventCatalystRisk(
  event: CatalystEventRiskInput,
): CatalystRiskLevel {
  if (event.high_impact) return "high";
  if (ELEVATED_CONFIDENCE.has(event.confidence)) return "medium";
  return "low";
}

/**
 * Aggregate catalyst risk for a held-symbol result.
 *
 * Precedence:
 * 1. High — any event with high_impact === true
 * 2. Medium — meaningful_move, promoted, or aggregate confidence confirmed/likely
 * 3. Low — quiet / no-confirmed / supporting-only / otherwise
 *
 * coverage, cached, and freshness are ignored by design.
 */
export function deriveSymbolCatalystRisk(
  result: SymbolCatalystRiskInput,
): CatalystRiskLevel {
  if (result.catalysts.some((event) => event.high_impact)) return "high";
  if (
    result.meaningful_move ||
    result.promoted ||
    ELEVATED_CONFIDENCE.has(result.confidence)
  ) {
    return "medium";
  }
  return "low";
}
