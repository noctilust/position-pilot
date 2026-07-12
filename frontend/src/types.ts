export type ProviderState =
  | "configured"
  | "not_configured"
  | "not_checked"
  | "signed_out"
  | "unavailable";

export type CatalystSettings = {
  stock_move_threshold_pct: number;
  etf_move_threshold_pct: number;
  news_cadence_seconds: number;
  benzinga: {
    enabled: boolean;
    status: string;
  };
  scheduled_window_hours: number;
};

export type RecommendationSettings = {
  selected_provider: "codex-cli" | string;
  api_key_fallback_available?: boolean;
  api_key_fallback_enabled?: boolean;
  rich_notification_preview: boolean;
};

export type EnvDiagnostic = {
  path: string;
  exists: boolean;
  gitignored: boolean;
  tracked_by_git: boolean | null;
  permission_mode: string | null;
  broadly_readable: boolean;
  warnings: string[];
  note: string;
};

export type RetentionSettings = {
  portfolio_snapshots_days: number;
  catalyst_events_days: number;
  article_metadata_days: number;
  recommendation_history_days: number;
  transaction_history: string;
  updated_at?: string | null;
};

export type RetentionPreview = {
  settings: RetentionSettings;
  candidates: Record<string, number>;
  audit_critical_preserved: string[];
  would_delete: Record<string, number>;
  disclaimer: string;
};

export type BackupInfo = {
  backup_id: string;
  filename: string;
  path: string;
  size_bytes: number;
  created_at: string;
  reason: string;
  schema_version: number | null;
  app_version: string | null;
  sha256: string;
  integrity_ok: boolean;
};

export type RestoreResult = {
  restored: boolean;
  backup_id: string;
  pre_restore_backup_id: string | null;
  schema_version: number | null;
  message: string;
  disclaimer: string;
};

export type UpdateReadiness = {
  current_version: string;
  latest_version: string | null;
  update_available: boolean;
  schema_version: number;
  schema_migrations_pending: boolean;
  backup_required_before_update: boolean;
  monitoring_active: boolean;
  blocked_reason: string | null;
  reversible_instructions: string[];
  auto_install: boolean;
  note: string;
  disclaimer: string;
};

export type DiagnosticBundle = {
  generated_at: string;
  app_version: string;
  schema_version: number;
  provider_status: Record<string, string>;
  settings_redacted: Record<string, unknown>;
  env_diagnostics: EnvDiagnostic;
  monitoring: Record<string, unknown>;
  counts: Record<string, number>;
  redaction: {
    excluded: string[];
    redacted_keys: string[];
    policy: string;
  };
  disclaimer: string;
};

export type MonitoringStatus = {
  market_timezone: string;
  window_start: string;
  window_end: string;
  evaluation_minutes: number;
  risk_refresh_seconds: number;
  enabled?: boolean;
  consented?: boolean;
  inside_window?: boolean;
  is_trading_day?: boolean;
  is_holiday?: boolean;
  is_early_close?: boolean;
  provider_status?: string;
  running?: boolean;
  notice?: string | null;
  last_evaluation_at?: string | null;
};

export type BootstrapPayload = {
  application: {
    name: string;
    version: string;
    phase: string;
  };
  providers: Record<"tastytrade" | "codex" | "massive" | "benzinga", ProviderState>;
  monitoring: MonitoringStatus;
  recommendations?: RecommendationSettings;
  catalysts?: CatalystSettings;
  navigation: string[];
  primary_account_id: string;
  data_state: string;
  server_time: string;
};

export type RecommendationRecord = {
  recommendation_id: string;
  subject_type: string;
  subject_id: string;
  account_id: string | null;
  symbol: string | null;
  strategy_type: string | null;
  horizon: string;
  action: string | null;
  urgency: number | null;
  risk: string | null;
  reasoning: string | null;
  evidence: string[];
  catalyst_refs: string[];
  suggested_action: string | null;
  input_fingerprint: string;
  prompt_version: string;
  schema_version: string;
  last_evaluated_at: string;
  recommendation_updated_at: string | null;
  provider: string;
  provider_status: string;
  error: string | null;
  codex_called: boolean;
};

export type RecommendationHistoryEntry = {
  history_id: string;
  recommendation_id: string;
  subject_type: string;
  subject_id: string;
  kind: string;
  recorded_at: string;
  action: string | null;
  urgency: number | null;
  risk: string | null;
  summary: string;
  evaluation_count: number;
  diff?: Record<string, { from?: unknown; to?: unknown } | unknown>;
};

export type TraderDecision = {
  decision_id: string;
  recommendation_id: string;
  subject_type: string;
  subject_id: string;
  decision: string;
  note: string;
  recorded_at: string;
};

export type AlertRecord = {
  alert_id: string;
  category: string;
  severity: string;
  alert_type: string;
  title: string;
  summary: string;
  account_id: string | null;
  symbol: string | null;
  strategy_type: string | null;
  subject_type: string | null;
  subject_id: string | null;
  source: string;
  created_at: string;
  updated_at: string;
  resolution: string;
  snoozed_until: string | null;
};

export type CatalystSource = {
  source_id: string;
  name: string;
  tier: string;
  url: string;
  provider: string;
  published_at: string;
  excerpt: string | null;
};

export type CatalystEvent = {
  catalyst_id: string;
  symbol: string;
  headline: string;
  summary: string;
  taxonomy: string;
  confidence: string;
  attribution: string;
  evidence_kind: string;
  event_at: string;
  sources: CatalystSource[];
  rank_score: number;
  high_impact: boolean;
};

export type OptionMechanism = {
  kind: string;
  label: string;
  summary: string;
  evidence_kind: string;
  magnitude: number | null;
};

export type SocialSideNote = {
  note_id: string;
  headline: string;
  summary: string;
  evidence_kind: string;
  source_name: string;
  url: string | null;
  confidence_note: string;
};

export type OptionMechanismCoverage = {
  kind: string;
  label: string;
  availability: string;
  detail: string;
};

export type SymbolCatalystResult = {
  symbol: string;
  confidence: string;
  attribution: string;
  summary: string;
  catalysts: CatalystEvent[];
  option_mechanisms: OptionMechanism[];
  option_mechanism_coverage?: OptionMechanismCoverage[];
  social_side_notes: SocialSideNote[];
  move_percent: number | null;
  prior_close: number | null;
  last_price: number | null;
  meaningful_move: boolean;
  promoted: boolean;
  coverage: string;
  coverage_notes: string[];
  quiet: boolean;
  cached?: boolean;
  freshness: {
    as_of: string;
    provider: string;
    state: string;
  };
};

export type CatalystScanSnapshot = {
  captured_at: string;
  results: SymbolCatalystResult[];
  settings: CatalystSettings;
  coverage: string;
  coverage_notes: string[];
  freshness: {
    as_of: string;
    provider: string;
    state: string;
  };
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
  state: "live" | "cached" | "daily";
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
      volume?: number | null;
    }>;
    source: string;
    notice: string | null;
    prior_close?: number | null;
    include_extended_hours?: boolean;
    extended_hours_truthful?: boolean;
    volume_series?: number[];
    window_start?: string | null;
    window_end?: string | null;
    event_markers?: Array<{
      catalyst_id: string;
      timestamp: string;
      headline: string;
      confidence: string;
      attribution: string;
    }>;
  };
  catalyst?: SymbolCatalystResult | null;
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
  recommendation?: RecommendationRecord | null;
  recommendation_history?: RecommendationHistoryEntry[];
  decisions?: TraderDecision[];
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
