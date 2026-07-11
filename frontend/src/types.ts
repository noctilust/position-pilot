export type ProviderState = "configured" | "not_configured" | "not_checked";

export type BootstrapPayload = {
  application: {
    name: string;
    version: string;
    phase: string;
  };
  providers: Record<"tastytrade" | "codex" | "massive" | "benzinga", ProviderState>;
  monitoring: {
    market_timezone: string;
    window_start: string;
    window_end: string;
    evaluation_minutes: number;
    risk_refresh_seconds: number;
  };
  navigation: string[];
  primary_account_id: string;
  data_state: string;
  server_time: string;
};

export type PositionLeg = {
  symbol: string;
  underlying_symbol: string;
  quantity: number;
  quantity_direction: "Long" | "Short";
  position_type: string;
  strike_price: number | null;
  option_type: string | null;
  expiration_date: string | null;
  days_to_expiration: number | null;
  mark_price: number | null;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number | null;
  delta: number | null;
  gamma: number | null;
  theta: number | null;
  vega: number | null;
  implied_volatility: number | null;
  multiplier: number;
  horizon: string;
};

export type Strategy = {
  strategy_id: string;
  account_id: string;
  underlying: string;
  strategy_type: string;
  expiration_date: string | null;
  days_to_expiration: number | null;
  quantity: number;
  strikes: string;
  unrealized_pnl: number;
  unrealized_pnl_percent: number | null;
  total_delta: number;
  total_theta: number;
  horizon: string;
  legs: PositionLeg[];
};

export type PortfolioAccount = {
  account_id: string;
  label: string;
  account_type: string;
  net_liquidating_value: number;
  cash_balance: number;
  buying_power: number;
  maintenance_excess: number | null;
  day_trading_buying_power: number | null;
  pnl_today: number;
  positions: PositionLeg[];
};

export type PortfolioSnapshot = {
  snapshot_id: string;
  captured_at: string;
  state: "live" | "cached";
  accounts: PortfolioAccount[];
  strategies: Strategy[];
  totals: {
    net_liquidating_value: number;
    cash_balance: number;
    buying_power: number;
    unrealized_pnl: number;
  };
  selected_account_id: string;
  notice: string | null;
};

export type PortfolioRisk = {
  total_delta: number;
  total_gamma: number;
  total_theta: number;
  total_vega: number;
  unrealized_pnl: number;
  net_liquidating_value: number;
  account_count: number;
  strategy_count: number;
  position_count: number;
  concentration: Array<{
    underlying: string;
    market_value: number;
    share_of_portfolio: number;
    strategy_count: number;
    net_delta: number;
  }>;
  stress: Array<{
    name: string;
    label: string;
    estimated_pnl_change: number;
    description: string;
  }>;
};

export type MarketQuote = {
  symbol: string;
  price: number;
  bid: number | null;
  ask: number | null;
  iv: number | null;
  iv_rank: number | null;
  iv_percentile: number | null;
  liquidity_rating: number | string | null;
  iv_environment: string;
  spread_percent: number | null;
};

export type MarketOverview = {
  captured_at: string;
  quotes: MarketQuote[];
  iv_summary: Record<string, number>;
};

export type WatchlistSnapshot = {
  symbols: string[];
  quotes: MarketQuote[];
};

export type RollChain = {
  chain_id: string;
  account_id: string;
  underlying: string;
  strategy_type: string;
  original_open_credit: number | null;
  chain_total_credit: number | null;
  rolls: Array<{
    roll_id: string;
    timestamp: string;
    old_strike: number;
    new_strike: number;
    old_dte: number;
    new_dte: number;
    roll_pnl: number;
    premium_effect: number;
  }>;
};

export type RollPatterns = {
  account_id: string;
  symbol: string | null;
  avg_dte_at_roll: number;
  typical_roll_days: number[];
  win_rate: number;
  total_rolls: number;
  avg_roll_pnl: number;
  total_pnl: number;
  best_dte_window: [number, number];
  avg_days_between_rolls: number;
};

export type RollHeatmap = {
  account_id: string;
  underlying: string;
  cells: Array<{
    strike: number;
    dte_bucket: string;
    count: number;
  }>;
  strikes: number[];
  buckets: string[];
  total_rolls: number;
};

export type OrderRow = {
  order_id: string;
  account_id: string;
  symbol: string;
  underlying_symbol: string | null;
  action: string;
  quantity: number;
  order_type: string;
  status: string;
  created_at: string;
  filled_quantity: number;
  average_fill_price: number | null;
  fills: Array<{
    fill_id: string;
    filled_at: string;
    symbol: string;
    quantity: number | null;
    price: number | null;
    amount: number;
  }>;
};

export type StrategyDetail = {
  strategy: Strategy;
  risk: {
    max_profit: number | null;
    max_loss: number | null;
    breakevens: number[];
    distance_to_nearest_strike: number | null;
    underlying_price: number | null;
    current_pnl: number;
    defined_risk: boolean;
    valuation_basis: "current_mark";
    combined: {
      delta: number | null;
      gamma: number | null;
      theta: number | null;
      vega: number | null;
      average_iv: number | null;
      nearest_dte: number | null;
    };
    stress: Array<{
      name: string;
      label: string;
      estimated_pnl_change: number;
      description: string;
    }>;
  };
  market: MarketQuote | null;
  chart: {
    symbol: string;
    bars: Array<{
      timestamp: string;
      open: number;
      high: number;
      low: number;
      close: number;
    }>;
    source: string;
    notice: string | null;
  };
  thesis: {
    purpose: string;
    expected_duration: string;
    target_range: string;
    invalidation: string;
    income_or_hedge_intent: string;
    events_to_watch: string[];
  } | null;
  trade_plan: {
    entry_thesis: string;
    intended_duration: string;
    profit_target: string;
    max_loss: string;
    roll_criteria: string;
    event_exposure: string;
    exit_deadline: string;
  } | null;
  audit: Array<{
    event_id: string;
    action: string;
    summary: string;
    recorded_at: string;
  }>;
  rolls: RollChain[];
  events: Array<{
    kind: string;
    timestamp: string;
    summary: string;
    action: string;
  }>;
};

export type LiveMarketTick = {
  symbol: string;
  price: number | null;
};
