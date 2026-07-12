/**
 * Presentation-only policy for which brokerage accounts appear in the web UI.
 *
 * This never alters portfolio totals, persistence, broker calls, or backend data.
 * Missing optional balance fields are treated as zero for the display decision.
 */

/** Conservative USD threshold for “nearly empty” residual balances. */
export const NEARLY_EMPTY_BALANCE_THRESHOLD_USD = 100;

/** Minimal account shape needed for display eligibility. */
export type AccountDisplayInput = {
  account_id: string;
  net_liquidating_value: number;
  cash_balance: number;
  buying_power: number;
  maintenance_excess?: number | null;
  day_trading_buying_power?: number | null;
  positions?: readonly unknown[] | null;
};

/** Strategy reference used only to detect open strategies per account. */
export type StrategyAccountRef = {
  account_id: string;
};

export type FilterDisplayableAccountsOptions = {
  /** Override the default $100 nearly-empty threshold. */
  thresholdUsd?: number;
  /**
   * When the server snapshot is explicitly scoped to this account id, keep that
   * option even if the account would otherwise be classified inactive.
   * Prefer preserving an explicit scope so the selector never silently mislabels
   * the user’s requested account or enters an invalid <select> state.
   */
  preserveAccountId?: string | null;
};

/**
 * Relevant absolute-value balance fields for the nearly-empty check.
 * Day P/L is intentionally excluded — residual P/L alone does not keep an
 * otherwise empty account visible.
 */
function relevantAbsoluteBalances(account: AccountDisplayInput): number[] {
  return [
    account.net_liquidating_value,
    account.cash_balance,
    account.buying_power,
    account.maintenance_excess ?? 0,
    account.day_trading_buying_power ?? 0,
  ].map((value) => Math.abs(value));
}

/**
 * Returns true when the account should appear in user-visible account lists.
 *
 * Hide only when all of the following are true:
 * - no open positions
 * - no detected strategies for the account
 * - every relevant balance/exposure field is negligible in absolute value
 */
export function isAccountDisplayEligible(
  account: AccountDisplayInput,
  strategies: readonly StrategyAccountRef[] = [],
  thresholdUsd: number = NEARLY_EMPTY_BALANCE_THRESHOLD_USD,
): boolean {
  if ((account.positions?.length ?? 0) > 0) {
    return true;
  }

  if (strategies.some((strategy) => strategy.account_id === account.account_id)) {
    return true;
  }

  return relevantAbsoluteBalances(account).some((abs) => abs >= thresholdUsd);
}

/**
 * Filter a portfolio account collection for presentation lists and counts.
 * Does not recompute monetary totals — callers must keep using server totals.
 */
export function filterDisplayableAccounts<T extends AccountDisplayInput>(
  accounts: readonly T[],
  strategies: readonly StrategyAccountRef[] = [],
  options: FilterDisplayableAccountsOptions = {},
): T[] {
  const thresholdUsd = options.thresholdUsd ?? NEARLY_EMPTY_BALANCE_THRESHOLD_USD;
  const preserve =
    options.preserveAccountId && options.preserveAccountId !== "all"
      ? options.preserveAccountId
      : null;

  return accounts.filter((account) => {
    if (preserve && account.account_id === preserve) {
      return true;
    }
    return isAccountDisplayEligible(account, strategies, thresholdUsd);
  });
}
