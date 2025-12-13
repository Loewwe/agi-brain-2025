# Microalpha Funding Mode - Technical Specification

**Mode:** `microalpha_funding`  
**Purpose:** Automated search for funding rate arbitrage strategies (F001-F003)  
**Anti-Overfitting:** Strict IS/OOS validation with fixed parameter grid

---

## 0. Goal

> **Objective:** Run autonomous background search for F001/F002/F003 strategies using funding/OI data, applying strict in-sample/out-of-sample validation, and saving only configurations that pass:
> - PF ≥ 1.25, WR ≥ 50%, ≥50 trades on OOS
> - Robustness test with 1.5x commission
> - Consistency check: PF_OOS/PF_IS ≥ 0.7

---

## 1. Input Data & Directories

**Data Sources:**
```
lab/microalpha/data/
├── funding_1h.parquet           # Funding rates + OI (1h resolution)
├── basis_spread.parquet         # Perpetual-spot basis
└── price_index.parquet          # Mark/index prices
```

**Features:**
```
lab/microalpha/features/
├── BTCUSDT_15m_features.parquet
├── ETHUSDT_15m_features.parquet
└── SOLUSDT_15m_features.parquet

Columns:
- timestamp
- funding_zscore (7d, 14d rolling)
- oi_change_pct (24h)
- basis_spread
- price_sma_50
- volatility_24h
```

**Outputs:**
```
lab/microalpha/results/
├── funding_search_history.csv    # All search iterations
└── iteration_YYYYMMDD_HHMM.json  # Detailed results per run

lab/microalpha/STRATEGY_SPECS/
└── F001_BTCUSDT_15m_20251213.json  # Candidate specs
```

---

## 2. Mode Implementation in Auto-Research Brain

### 2.1. Mode Activation

```python
# In auto_research_brain.py
mode = os.getenv("RESEARCH_MODE", "classic")  # or "microalpha_funding"

if mode == "microalpha_funding":
    run_microalpha_funding_search()
```

### 2.2. Search Cycle

**One Iteration = Test all (strategy × symbol × timeframe × params)**

```python
def run_microalpha_funding_once(config):
    """Single iteration of funding strategy search"""
    
    # Step 1: Update data
    update_funding_data()          # Run funding_backfill.py
    
    # Step 2: Build features
    build_features_for_all()       # Calculate z-scores, percentiles
    
    candidates = []
    
    # Step 3: Test all combinations
    for strategy_cls in [F001, F002, F003]:
        for symbol in config.symbols:
            for tf in config.timeframes:
                
                # Load features
                features = load_features(symbol, tf)
                
                # Generate signals
                signals = strategy_cls().generate_signals(features)
                
                # Grid search (fixed, limited)
                for params in FIXED_PARAM_GRID[strategy_cls]:
                    
                    # Backtest IS/OOS
                    metrics = backtest_IS_OOS(signals, params, config)
                    
                    # Check if passes criteria
                    if is_candidate(metrics):
                        spec = build_strategy_spec(
                            strategy_cls, symbol, tf, params, metrics
                        )
                        save_spec(spec)
                        log_to_csv(spec)
                        candidates.append(spec)
    
    return candidates
```

### 2.3. Loop Control

```python
def auto_search_loop(config):
    """Continuous autonomous search"""
    
    while True:
        logger.info("Starting microalpha funding search iteration")
        
        candidates = run_microalpha_funding_once(config)
        
        if len(candidates) >= config.min_new_candidates:
            logger.info(f"SUCCESS: Found {len(candidates)} candidates")
            send_alert(f"New candidates: {[c['strategy_id'] for c in candidates]}")
        else:
            logger.info("NO_CANDIDATES this iteration")
        
        # Wait before next iteration
        time.sleep(config.loop_interval_hours * 3600)
```

---

## 3. Fixed Parameter Grid (Anti-Overfitting)

### F001: Extreme Funding Reversal (SHORT)

```python
F001_GRID = {
    'tp_pct': [0.010, 0.015, 0.020],        # 3 values
    'sl_pct': [0.020, 0.025, 0.030],        # 3 values
    'funding_threshold': ['p90', 'p95'],    # 2 values
    'oi_change_threshold': [0.15, 0.20]     # 2 values
}
# Total: 3 × 3 × 2 × 2 = 36 combinations
```

### F002: Negative Funding Momentum (LONG)

```python
F002_GRID = {
    'tp_pct': [0.015, 0.020, 0.025],
    'sl_pct': [0.025, 0.030, 0.035],
    'funding_threshold': ['p5', 'p10'],
    'price_filter': ['below_ma50', 'below_ma100']
}
# Total: 3 × 3 × 2 × 2 = 36 combinations
```

### F003: Funding Divergence (Pairs)

```python
F003_GRID = {
    'tp_pct': [0.008, 0.010, 0.012],
    'sl_pct': [0.015, 0.020],
    'divergence_threshold': ['2std', '2.5std'],
    'lookback_days': [7, 14]
}
# Total: 3 × 2 × 2 × 2 = 24 combinations
```

**Critical:** Grid is fixed in code, NOT optimized per iteration

---

## 4. IS/OOS Validation Protocol

### 4.1. Data Split

```python
def split_is_oos(data, is_days=60, oos_days=30):
    """
    IS (In-Sample): First 60 days
    OOS (Out-of-Sample): Last 30 days
    
    FORBIDDEN:
    - Optimize on OOS
    - Change split boundaries per strategy
    - Mix IS/OOS data
    """
    total_days = is_days + oos_days
    cutoff = data.index[-1] - timedelta(days=oos_days)
    
    is_data = data[data.index < cutoff]
    oos_data = data[data.index >= cutoff]
    
    return is_data, oos_data
```

### 4.2. Candidate Criteria

**MUST pass ALL:**

```python
def is_candidate(metrics):
    """Strict criteria for L2 candidate status"""
    
    # Minimum statistics
    if metrics['trades_IS'] < 40:
        return False, "Insufficient IS trades"
    if metrics['trades_OOS'] < 50:
        return False, "Insufficient OOS trades"
    
    # IS quality
    if metrics['PF_IS'] < 1.3:
        return False, f"IS PF too low: {metrics['PF_IS']}"
    if metrics['WR_IS'] < 0.52:
        return False, f"IS WR too low: {metrics['WR_IS']}"
    
    # OOS quality
    if metrics['PF_OOS'] < 1.25:
        return False, f"OOS PF too low: {metrics['PF_OOS']}"
    if metrics['WR_OOS'] < 0.50:
        return False, f"OOS WR too low: {metrics['WR_OOS']}"
    if metrics['MaxDD_OOS'] > 0.15:
        return False, f"OOS DD too high: {metrics['MaxDD_OOS']}"
    
    # Consistency check
    pf_ratio = metrics['PF_OOS'] / metrics['PF_IS']
    if pf_ratio < 0.7:
        return False, f"PF degradation: {pf_ratio:.2f}"
    
    # Robustness (1.5x commission)
    if metrics['PF_OOS_robust'] < 1.15:
        return False, f"Not robust to fees: {metrics['PF_OOS_robust']}"
    
    return True, "PASS"
```

---

## 5. Strategy Specification Format

### 5.1. JSON Schema

```json
{
  "strategy_id": "F001",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "params": {
    "tp_pct": 0.015,
    "sl_pct": 0.025,
    "funding_threshold": "p95",
    "oi_change_threshold": 0.20
  },
  "metrics": {
    "PF_IS": 1.42,
    "PF_OOS": 1.28,
    "WR_IS": 0.58,
    "WR_OOS": 0.54,
    "trades_IS": 48,
    "trades_OOS": 56,
    "MaxDD_OOS": 0.12,
    "Sharpe_OOS": 1.45
  },
  "robustness": {
    "PF_OOS_commission_1_5x": 1.18,
    "WR_OOS_commission_1_5x": 0.52
  },
  "data_ranges": {
    "IS_start": "2025-09-13",
    "IS_end": "2025-11-12",
    "OOS_start": "2025-11-13",
    "OOS_end": "2025-12-13"
  },
  "created_at": "2025-12-13T12:30:00Z",
  "status": "L2_CANDIDATE"
}
```

### 5.2. CSV Log

```csv
timestamp,strategy_id,symbol,timeframe,PF_OOS,WR_OOS,trades_OOS,MaxDD_OOS,status
2025-12-13T12:30:00Z,F001,BTCUSDT,15m,1.28,0.54,56,0.12,CANDIDATE
2025-12-13T12:31:00Z,F001,ETHUSDT,15m,1.08,0.48,42,0.18,REJECT
2025-12-13T12:32:00Z,F002,BTCUSDT,15m,1.35,0.56,61,0.09,CANDIDATE
```

---

## 6. What Brain Must Learn

**Key Differences from Classic Mode:**

1. **Data Sources:**
   - Classic: OHLCV candles only
   - Microalpha: funding_rate, open_interest, basis_spread

2. **Time Horizons:**
   - Classic: Minutes to hours
   - Microalpha: Hours to days (funding updated 8h)

3. **Validation:**
   - Classic: Single period test
   - Microalpha: Strict IS/OOS split, robustness tests

4. **Parameter Search:**
   - Classic: May use optimization
   - Microalpha: Fixed grid only, no adaptive search

5. **Logging:**
   - Must track: which combos tested, why rejected
   - Prevent: re-testing same combos endlessly

---

## 7. Stop/Continue Logic

```python
def should_continue(iteration_results):
    """Decide if search should continue or pause"""
    
    # Success: Found candidates
    if len(iteration_results['candidates']) >= MIN_CANDIDATES:
        return {
            'continue': True,
            'wait_hours': 24,  # Daily search
            'reason': 'SUCCESS'
        }
    
    # No candidates but system healthy
    if iteration_results['data_quality'] == 'OK':
        return {
            'continue': True,
            'wait_hours': 6,   # Try again in 6h
            'reason': 'NO_CANDIDATES_RETRY'
        }
    
    # Data issues detected
    if iteration_results['data_quality'] == 'ERROR':
        return {
            'continue': False,
            'wait_hours': None,
            'reason': 'DATA_ERROR'
        }
    
    # Default: retry after delay
    return {
        'continue': True,
        'wait_hours': 12,
        'reason': 'DEFAULT'
    }
```

---

## 8. Configuration Example

```yaml
# lab/microalpha/config.yaml
mode: microalpha_funding

data:
  source: lab/microalpha/data/funding_1h.parquet
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  timeframes: [15m, 1h]

search:
  strategies: [F001, F002, F003]
  is_days: 60
  oos_days: 30
  min_new_candidates: 1
  loop_interval_hours: 6

criteria:
  trades_IS_min: 40
  trades_OOS_min: 50
  PF_IS_min: 1.3
  PF_OOS_min: 1.25
  WR_IS_min: 0.52
  WR_OOS_min: 0.50
  MaxDD_OOS_max: 0.15
  consistency_ratio_min: 0.7
  robustness_PF_min: 1.15

alerts:
  enabled: true
  webhook: https://hooks.slack.com/...
  on_success: true
  on_error: true
```

---

## 9. Anti-Overfitting Guarantees

**The Brain MUST NOT:**
- ❌ Optimize parameters based on OOS
- ❌ Change IS/OOS split per strategy
- ❌ Cherry-pick time periods
- ❌ Infinite parameter search
- ❌ Hide negative results

**The Brain MUST:**
- ✅ Use fixed parameter grid
- ✅ Test all grid combinations
- ✅ Log all results (pass & fail)
- ✅ Separate IS/OOS strictly  
- ✅ Run robustness tests
- ✅ Report honestly (even 0 candidates)

---

## 10. Next Steps After Finding Candidate

**If status = L2_CANDIDATE:**

1. Create detailed spec in `STRATEGY_SPECS/`
2. Run extended validation:
   - Walk-forward test (6 windows)
   - Cross-asset validation
   - Stress test (volatile periods)
3. If passes extended → status = `PAPER_TRADING_READY`
4. Deploy to paper trading with beta mode (50% sizing)

**If 0 candidates after full grid:**
- Honest assessment: "This edge doesn't exist in current data"
- Don't force it by relaxing criteria
- Move to next strategy (F002, F003)

---

**Ready for implementation: `auto_search_loop.py` + `backtest_F001.py`** 🚀
