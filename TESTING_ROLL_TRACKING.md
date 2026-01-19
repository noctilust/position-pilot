# Roll Tracking Feature - Testing Guide

## Overview
This guide covers testing the new roll tracking feature that tracks roll history, displays P/L Open, analyzes rolling patterns, and provides AI-powered insights.

## Prerequisites
- Tastytrade account with transaction history
- Active options positions or past rolled positions
- Position Pilot installed and configured

## Setup

### 1. Initial Roll History Population

Before using the dashboard features, populate your roll history cache:

```bash
# Fetch 1 year of transaction history and detect rolls
pilot rolls fetch

# Fetch 90 days of history
pilot rolls fetch --days 90

# Force refresh from API (bypass cache)
pilot rolls fetch --force
```

Expected output:
```
✓ Fetched 523 transactions
✓ Detected 3 roll chains
✓ Stored 12 rolls from 3 roll chains

Roll History Summary:
  SPY    put_spread     : +$1,250.50 (5 rolls)
  SPY    call_spread    : +$485.20 (4 rolls)
  IWM    iron_condor    : -$125.30 (3 rolls)
```

### 2. Verify Roll History

Check that rolls were detected correctly:

```bash
# View roll history for SPY
pilot rolls history SPY

# View roll history for specific strategy
pilot rolls history SPY --strategy put_spread
```

Expected output:
```
SPY Put Spread Roll History (last 365 days)
══════════════════════════════════════════

Date        Roll              From          To            P/L      Notes
─────────── ──────────────── ───────────── ───────────── ──────── ───────
2026-01-15  $400→$415        21→35 DTE     +$125.50
2026-01-08  $390→$400        10→30 DTE     +$85.30
2026-01-02  Initial entry    -             +$350.00

─────────────────────────────────────────────────────────────────
Total Rolls: 2        P/L Open: +$560.80    Win Rate: 75%
```

## Dashboard Testing

### 1. Launch Dashboard

```bash
pilot dashboard
```

### 2. Test Roll History Panel (press 'h')

1. Navigate to a strategy row in the positions table
2. Press `h` to view roll history

Expected behavior:
- Roll History panel appears/updates
- Shows roll count, total P/L, P/L Open
- Displays last 5 rolls with strikes, DTE, and P/L
- Shows "No roll history available" if no rolls detected

### 3. Test Roll Insights Panel (press 'i')

1. Navigate to a strategy row
2. Press `i` to generate AI-powered roll insights

Expected behavior:
- Status bar shows "Generating roll insights..."
- Roll Insights panel populates with:
  - Pattern statistics (win rate, typical DTE targets)
  - AI recommendations for optimal roll timing
  - Pattern alignment score
  - Risk and opportunity identification
- Shows "No roll history available" if no rolls detected

### 4. Test Option Chain Heatmap (press 'C')

1. Navigate to a strategy row
2. Press `C` to view option chain heatmap

Expected behavior:
- Positions table hides, heatmap appears
- Visual grid showing roll frequency by strike and DTE
- Current position highlighted with ←
- Most profitable combination marked with ★
- Color-coded cells (░ ░ ▒ ▒ ▓ ▓ ██) based on roll count
- Press `C` again or navigate away to return to positions table

### 5. Test Key Bindings

| Key | Action | Expected Result |
|-----|--------|-----------------|
| `h` | Show roll history | Updates Roll History panel |
| `i` | Show roll insights | Generates AI insights |
| `C` | Show chain heatmap | Toggles heatmap view |
| `r` | Refresh | Reloads all data |
| `q` | Quit | Exits dashboard |

## CLI Commands Testing

### 1. Roll History Command

```bash
pilot rolls history SPY --days 90
```

Verify:
- Correct roll detection
- Accurate P/L calculations
- Proper strike and DTE information

### 2. Option Chain Command

```bash
pilot rolls chain SPY --days 365
```

Verify:
- Heatmap displays correctly
- Roll counts are accurate
- Current position is highlighted
- Best combination is marked

### 3. Patterns Command

```bash
pilot rolls patterns SPY --days 365
```

Verify:
- Pattern statistics are calculated correctly
- Win rate is accurate
- Typical roll days are identified
- Best DTE window is reasonable

## Edge Cases to Test

### 1. Empty Roll History
- Test with account that has no rolls
- Expected: Graceful "No roll history found" message

### 2. Single Roll
- Test with only one roll detected
- Expected: Pattern analysis shows limited data

### 3. Large History (1 year)
- Test with 1000+ transactions
- Expected: Performance remains acceptable (<5 seconds)

### 4. Invalid OCC Symbols
- Test with malformed option symbols
- Expected: Error handling logs warnings, continues processing

### 5. Missing Transaction Fields
- Test with incomplete transaction data
- Expected: Uses safe defaults, logs warnings

## Performance Benchmarks

| Operation | Expected Time |
|-----------|---------------|
| Roll fetch (1 year) | <10 seconds |
| Dashboard load | <3 seconds |
| Roll history panel | <1 second |
| Roll insights generation | <5 seconds |
| Heatmap rendering | <2 seconds |

## Bug Reporting

If you encounter issues, report:
1. Command or action performed
2. Expected vs actual behavior
3. Error messages (if any)
4. Account/symbol (can be sanitized)
5. Roll history file: `~/.cache/position-pilot/roll_history.json`

## Success Criteria

- ✓ All key bindings work correctly
- ✓ Roll history displays accurately
- ✓ P/L Open calculations match manual calculations
- ✓ Pattern analysis provides insights
- ✓ Dashboard performance is acceptable
- ✓ Error handling is graceful
- ✓ CLI commands work as expected

## Next Steps After Testing

1. Report bugs or issues
2. Suggest improvements or feature requests
3. Share feedback on UI/UX
4. Provide real-world testing results
