import {
  catalystRiskActiveSegments,
  catalystRiskAriaLabel,
  catalystRiskLabel,
  catalystRiskTokenClass,
  type CatalystRiskLevel,
} from "./catalystRisk";

type CatalystRiskGaugeProps = {
  level: CatalystRiskLevel;
  /** Compact variant for dense event rows. Default is the standard metadata size. */
  size?: "sm" | "md";
  className?: string;
};

/**
 * Compact three-step catalyst risk gauge.
 * Color uses semantic tokens (--good / --warn / --danger); text always labels Low/Medium/High.
 */
export function CatalystRiskGauge({
  level,
  size = "md",
  className = "",
}: CatalystRiskGaugeProps) {
  const label = catalystRiskLabel(level);
  const token = catalystRiskTokenClass(level);
  const active = catalystRiskActiveSegments(level);
  const aria = catalystRiskAriaLabel(level);

  return (
    <span
      className={[
        "catalyst-risk-gauge",
        `catalyst-risk-${level}`,
        `catalyst-risk-token-${token}`,
        size === "sm" ? "catalyst-risk-gauge-sm" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      role="img"
      aria-label={aria}
      data-risk-level={level}
      data-risk-token={token}
      data-risk-segments={String(active)}
    >
      <span className="catalyst-risk-segments" aria-hidden="true">
        {[1, 2, 3].map((index) => (
          <span
            key={index}
            className={
              index <= active
                ? "catalyst-risk-seg is-active"
                : "catalyst-risk-seg"
            }
          />
        ))}
      </span>
      <span className="catalyst-risk-text">{label}</span>
    </span>
  );
}
