/**
 * Pure helpers for the Positions roll-event ledger.
 * Flatten, sort, dedupe, and aggregate roll chains without side effects.
 */

import type { RollChain, RollEvent } from "./types";

/** Flattened ledger row: one event, with chain context always present. */
export type RollLedgerEvent = {
  chain_id: string;
  account_id: string;
  underlying: string;
  strategy_type: string;
  roll_id: string;
  timestamp: string;
  old_strike: number;
  new_strike: number;
  old_dte: number;
  new_dte: number;
  old_expiration: string | null;
  new_expiration: string | null;
  old_symbol: string | null;
  new_symbol: string | null;
  premium_effect: number;
  roll_pnl: number;
  commission: number;
  reason: string | null;
  notes: string | null;
};

export type RollLedgerSummary = {
  /** Signed sum of every event premium_effect (credit +, debit −). Excludes opening credit. */
  netRollCredit: number;
  eventCount: number;
  chainCount: number;
  /**
   * Sum of finite non-null chain_total_credit values (includes original opening credits).
   * Null when no chain has a known total.
   */
  knownLifetimeCredit: number | null;
  /** True when at least one chain is missing chain_total_credit. */
  lifetimePartial: boolean;
};

function eventUnderlying(event: RollEvent, chain: RollChain): string {
  return event.underlying?.trim() || chain.underlying;
}

function eventStrategy(event: RollEvent, chain: RollChain): string {
  return event.strategy_type?.trim() || chain.strategy_type;
}

/**
 * Deduplicate chains by chain_id (first occurrence wins).
 * Useful when aggregating per-account fetches that may overlap.
 */
export function dedupeRollChains(chains: readonly RollChain[]): RollChain[] {
  const seen = new Set<string>();
  const out: RollChain[] = [];
  for (const chain of chains) {
    const key = chain.chain_id || `${chain.account_id}:${chain.underlying}:${chain.strategy_type}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(chain);
  }
  return out;
}

/**
 * Flatten every roll event from every chain into ledger rows.
 * Dedupes events by roll_id (first wins) when the same id appears twice.
 */
export function flattenRollEvents(chains: readonly RollChain[]): RollLedgerEvent[] {
  const uniqueChains = dedupeRollChains(chains);
  const seenEvents = new Set<string>();
  const rows: RollLedgerEvent[] = [];

  for (const chain of uniqueChains) {
    for (const event of chain.rolls ?? []) {
      const eventKey =
        event.roll_id ||
        `${chain.chain_id}:${event.timestamp}:${event.old_strike}:${event.new_strike}`;
      if (seenEvents.has(eventKey)) continue;
      seenEvents.add(eventKey);
      rows.push({
        chain_id: chain.chain_id,
        account_id: chain.account_id,
        underlying: eventUnderlying(event, chain),
        strategy_type: eventStrategy(event, chain),
        roll_id: event.roll_id,
        timestamp: event.timestamp,
        old_strike: event.old_strike,
        new_strike: event.new_strike,
        old_dte: event.old_dte,
        new_dte: event.new_dte,
        old_expiration: event.old_expiration ?? null,
        new_expiration: event.new_expiration ?? null,
        old_symbol: event.old_symbol ?? null,
        new_symbol: event.new_symbol ?? null,
        premium_effect: event.premium_effect,
        roll_pnl: event.roll_pnl,
        commission: event.commission ?? 0,
        reason: event.reason ?? null,
        notes: event.notes ?? null,
      });
    }
  }
  return rows;
}

/** Newest event first; stable secondary sort by roll_id for determinism. */
export function sortRollEventsNewestFirst(events: readonly RollLedgerEvent[]): RollLedgerEvent[] {
  return [...events].sort((a, b) => {
    const byTime = b.timestamp.localeCompare(a.timestamp);
    if (byTime !== 0) return byTime;
    return a.roll_id.localeCompare(b.roll_id);
  });
}

/** Flatten + sort: complete event ledger for the rail. */
export function buildRollLedger(chains: readonly RollChain[]): RollLedgerEvent[] {
  return sortRollEventsNewestFirst(flattenRollEvents(chains));
}

/**
 * Aggregate header stats for the roll rail.
 * Net roll credit excludes original opening credits (event premium_effect only).
 * Known lifetime credit sums chain_total_credit and flags partial when any is null.
 */
export function summarizeRollLedger(chains: readonly RollChain[]): RollLedgerSummary {
  const uniqueChains = dedupeRollChains(chains);
  const events = flattenRollEvents(uniqueChains);

  let netRollCredit = 0;
  for (const event of events) {
    netRollCredit += event.premium_effect;
  }

  let knownLifetimeCredit: number | null = null;
  let lifetimePartial = false;
  let knownSum = 0;
  let knownCount = 0;

  for (const chain of uniqueChains) {
    const total = chain.chain_total_credit;
    if (total == null || !Number.isFinite(total)) {
      lifetimePartial = true;
      continue;
    }
    knownSum += total;
    knownCount += 1;
  }

  if (knownCount > 0) {
    knownLifetimeCredit = knownSum;
  }
  // If we have chains but none have totals, lifetime is partial and unknown.
  if (uniqueChains.length > 0 && knownCount === 0) {
    lifetimePartial = true;
  }
  // Zero chains: not partial, just empty.
  if (uniqueChains.length === 0) {
    lifetimePartial = false;
  }

  return {
    netRollCredit,
    eventCount: events.length,
    chainCount: uniqueChains.length,
    knownLifetimeCredit,
    lifetimePartial,
  };
}

/** Merge settled per-account fetches; keep successes when some accounts fail. */
export function mergeAccountRollResults(
  results: readonly PromiseSettledResult<RollChain[]>[],
): RollChain[] {
  const merged: RollChain[] = [];
  for (const result of results) {
    if (result.status === "fulfilled" && Array.isArray(result.value)) {
      merged.push(...result.value);
    }
  }
  return dedupeRollChains(merged);
}
