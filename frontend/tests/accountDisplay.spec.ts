import { expect, test } from "@playwright/test";

import {
  filterDisplayableAccounts,
  isAccountDisplayEligible,
  NEARLY_EMPTY_BALANCE_THRESHOLD_USD,
} from "../src/accountDisplay";
import type { AccountDisplayInput } from "../src/accountDisplay";

function account(
  overrides: Partial<AccountDisplayInput> & Pick<AccountDisplayInput, "account_id">,
): AccountDisplayInput {
  return {
    net_liquidating_value: 0,
    cash_balance: 0,
    buying_power: 0,
    maintenance_excess: null,
    day_trading_buying_power: null,
    positions: [],
    ...overrides,
  };
}

test.describe("account display eligibility policy", () => {
  test("default nearly-empty threshold is $100 USD", () => {
    expect(NEARLY_EMPTY_BALANCE_THRESHOLD_USD).toBe(100);
  });

  test("empty account with zero balances is hidden", () => {
    expect(isAccountDisplayEligible(account({ account_id: "empty" }), [])).toBe(false);
  });

  test("residual balances under $100 with no positions/strategies are hidden", () => {
    expect(
      isAccountDisplayEligible(
        account({
          account_id: "residual",
          net_liquidating_value: 99.99,
          cash_balance: 50,
          buying_power: 25,
        }),
        [],
      ),
    ).toBe(false);
  });

  test("exactly $100 on any relevant balance field is visible", () => {
    const fields: Array<keyof AccountDisplayInput> = [
      "net_liquidating_value",
      "cash_balance",
      "buying_power",
      "maintenance_excess",
      "day_trading_buying_power",
    ];
    for (const field of fields) {
      expect(
        isAccountDisplayEligible(
          account({ account_id: `edge-${field}`, [field]: 100 }),
          [],
        ),
        `${field} at $100 should be visible`,
      ).toBe(true);
    }
  });

  test("balances above $100 are visible", () => {
    expect(
      isAccountDisplayEligible(
        account({ account_id: "funded", net_liquidating_value: 100.01 }),
        [],
      ),
    ).toBe(true);
  });

  test("negative meaningful balances remain visible by absolute value", () => {
    expect(
      isAccountDisplayEligible(
        account({ account_id: "debt", cash_balance: -150, net_liquidating_value: -150 }),
        [],
      ),
    ).toBe(true);
    expect(
      isAccountDisplayEligible(
        account({ account_id: "small-debt", cash_balance: -99.99 }),
        [],
      ),
    ).toBe(false);
  });

  test("missing optional balance fields are treated as zero", () => {
    expect(
      isAccountDisplayEligible(
        {
          account_id: "sparse",
          net_liquidating_value: 0,
          cash_balance: 0,
          buying_power: 0,
          // maintenance_excess and day_trading_buying_power omitted
          positions: [],
        },
        [],
      ),
    ).toBe(false);
  });

  test("zero-balance account with any position is visible", () => {
    expect(
      isAccountDisplayEligible(
        account({
          account_id: "with-pos",
          positions: [{ symbol: "SPY" }],
        }),
        [],
      ),
    ).toBe(true);
  });

  test("zero-balance account with a strategy for its id is visible", () => {
    expect(
      isAccountDisplayEligible(account({ account_id: "with-strat" }), [
        { account_id: "with-strat" },
      ]),
    ).toBe(true);
    expect(
      isAccountDisplayEligible(account({ account_id: "with-strat" }), [
        { account_id: "other-account" },
      ]),
    ).toBe(false);
  });

  test("day P/L alone does not keep an empty account visible", () => {
    // pnl_today is not part of AccountDisplayInput relevant fields
    expect(
      isAccountDisplayEligible(
        account({
          account_id: "pnl-only",
          net_liquidating_value: 0,
          cash_balance: 0,
          buying_power: 0,
        }),
        [],
      ),
    ).toBe(false);
  });

  test("filterDisplayableAccounts applies policy and preserves explicit scope", () => {
    const accounts = [
      account({ account_id: "active", net_liquidating_value: 5000 }),
      account({ account_id: "inactive" }),
      account({ account_id: "residual", cash_balance: 12 }),
    ];

    expect(filterDisplayableAccounts(accounts, []).map((row) => row.account_id)).toEqual([
      "active",
    ]);

    expect(
      filterDisplayableAccounts(accounts, [], {
        preserveAccountId: "inactive",
      }).map((row) => row.account_id),
    ).toEqual(["active", "inactive"]);

    expect(
      filterDisplayableAccounts(accounts, [], { preserveAccountId: "all" }).map(
        (row) => row.account_id,
      ),
    ).toEqual(["active"]);

    expect(filterDisplayableAccounts(accounts, [], { preserveAccountId: null })).toEqual(
      filterDisplayableAccounts(accounts, []),
    );
  });

  test("all-filtered collection yields an empty list for empty-state UI", () => {
    const accounts = [
      account({ account_id: "a" }),
      account({ account_id: "b", net_liquidating_value: 50 }),
    ];
    expect(filterDisplayableAccounts(accounts, [])).toEqual([]);
  });
});
