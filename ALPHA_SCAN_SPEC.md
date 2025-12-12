# Alpha Scan v1 — Specification

## Overview

**Goal:** Systematic screening of 54 experiments to find viable alpha candidates OR definitively conclude no edge exists in this search space.

**Timeline:** 2025-12-08 onwards (1-2 weeks max)

---

## Experiment Grid Summary

**Total Experiments:** 54

**Dimensions:**
- **Symbols (3):** BTC/USDT, ETH/USDT, SOL/USDT
- **Timeframes (2):** 15m, 1h (NOT 5m — killed by costs in Stage 8b)
- **Target Types (3):** Momentum, Reversal, Vol Expansion
- **Horizons (2-3 per target):** Short, Medium (varies by TF)
- **Feature Sets (2):** base, extended
- **Model (1):** LightGBM only (P0)

**Distribution:**
- 18 experiments per symbol
- 27 experiments per timeframe
- 18 experiments per target type
- 36 with extended features, 18 with base (to test feature value)

---

## Logged Metrics

### ML Metrics
- `auc` — ROC AUC on validation set

### Pre-Cost Trading Metrics
- `win_rate_pre_cost` — Fraction of winning trades
- `sharpe_pre_cost` — Annualized Sharpe ratio
- `profit_factor_pre_cost` — Gross profit / Gross loss
- `total_return_pre_cost` — Cumulative return
- `max_drawdown_pre_cost` — Maximum peak-to-valley drawdown

### Post-Cost Trading Metrics
- `win_rate_post_cost` — After 0.3% transaction costs
- `sharpe_post_cost` — After costs
- `profit_factor_post_cost` — After costs
- `total_return_post_cost` — After costs
- `max_drawdown_post_cost` — After costs

### Volume Metrics
- `n_trades` — Total number of trades
- `trades_per_day` — Average trades per day
- `trades_per_month` — n_trades × 30 / days_in_test_period

### Derived Metrics
- `pf_delta` = profit_factor_post_cost - profit_factor_pre_cost
- `sharpe_delta` = sharpe_post_cost - sharpe_pre_cost
- `cost_impact_pct` = (total_return_pre - total_return_post) / abs(total_return_pre) × 100

---

## Classification Thresholds

### CANDIDATE (meets ALL criteria)

```python
CANDIDATE_CRITERIA = {
    'pf_post_cost': 1.15,        # Minimum profit factor after costs
    'sharpe_post_cost': 1.5,     # Minimum Sharpe after costs
    'trades_per_month': 1000,    # Maximum trade frequency
    'max_drawdown_post_cost': -0.25,  # Max 25% drawdown
}
```

**Description:** Experiments that meet all thresholds are **viable alpha candidates** worthy of robustness testing.

---

### BORDERLINE (shows some promise)

```python
BORDERLINE_CRITERIA = {
    'pf_post_cost': 1.05,        # Shows some edge
    'sharpe_post_cost': 1.0,     # Positive risk-adjusted return
    'trades_per_month': 2000,    # Moderate frequency
}
```

**Description:** Experiments that don't meet candidate criteria but show some signal. May warrant investigation but unlikely to become production candidates.

---

### REJECTED (no viable edge)

**Any experiment not meeting borderline criteria is REJECTED.**

Common rejection reasons:
- `pf_post_cost ≤ 1.02` — Edge too thin or doesn't exist
- `sharpe_post_cost ≤ 1.0` — Poor risk-adjusted returns
- `trades_per_month > 3000` — Excessive trading (costs will dominate)
- `max_drawdown_post_cost < -0.30` — Unacceptable risk

---

## Composite Scoring

For ranking experiments within each category:

```python
def composite_score(result):
    """
    Composite score balancing performance vs trade frequency.
    Higher is better.
    """
    sharpe = max(result['sharpe_post_cost'], 0.1)  # Floor at 0.1
    pf = max(result['pf_post_cost'], 1.0)
    trades_per_day = result['n_trades'] / result['test_period_days']
    
    # Penalize high trade frequency (log scale)
    frequency_penalty = np.log(1 + trades_per_day)
    
    # Balance Sharpe and excess PF, penalize frequency
    score = sharpe * (pf - 1.0) / frequency_penalty
    
    return score
```

**Usage:** Sort candidates by composite_score to prioritize:
- High Sharpe
- High profit factor
- Low trade frequency

---

## Transaction Costs

**All experiments use realistic costs:**

- **Commission:** 10 bps (0.10%) per trade
- **Slippage:** 5 bps (0.05%) per direction (0.10% total)
- **Total per round-trip:** 0.30%

**Rationale:** Industry-standard costs for 5m-1h crypto trading on major exchanges.

---

## Test Period

**Train:** 2024-01-01 to 2024-03-31 (90 days, ~3 months)  
**Test:** 2024-04-01 to 2024-04-30 (30 days, 1 month)

**Why April 2024?**
- Recent enough to be relevant
- Different regime than training (check for overfitting)
- Sufficient liquidity on all symbols

---

## Expected Workflow

### Phase 1 (Current)
1. ✅ Define grid (`alpha_scan_grid.yaml`)
2. ✅ Define spec (this document)
3. Define candidate/borderline/rejected criteria

### Phase 2
1. Implement `run_alpha_grid.py` (batch runner)
2. Implement `summarize_alpha_scan.py` (aggregator + classifier)
3. Run all 54 experiments (~2-3 hours total @ 2-3 min each)
4. Auto-classify results
5. Generate summary report

### Phase 3
1. For each candidate: Define 2 additional test periods
2. Re-run on new periods
3. Check survival criteria
4. Final verdict

---

## Survival Criteria (Phase 3)

**For a candidate to be considered "robust":**

On **base period** (April 2024):
- Must meet all CANDIDATE criteria

On **alt periods** (e.g., May 2024, Feb 2024):
- `pf_post_cost ≥ 1.05` (relaxed threshold)
- `sharpe_post_cost > 1.0`
- `trades_per_month ≤ 1.5 × base_period_trades`

**If ANY alt period fails → mark as period-specific, NOT robust**

---

## Success / Failure Outcomes

### Success (Scenario A)

**Found 2–5 robust candidates:**

Example:
```
scan_027 (ETH/15m/vol_exp/ext): PF [1.18, 1.12, 1.09], Sharpe [1.7, 1.5, 1.4]
scan_041 (SOL/1h/reversal/ext): PF [1.16, 1.10, 1.13], Sharpe [1.6, 1.3, 1.5]
scan_033 (BTC/1h/momentum/ext): PF [1.15, 1.08, 1.11], Sharpe [1.5, 1.2, 1.4]
```

**Decision:** Proceed to Stage 10 (Paper-Trading Validation)

---

### Failure (Scenario B)

**Zero candidates pass robustness:**

Example:
```
54 experiments screened
→ 3 candidates from Phase 2
→ 0 passed multi-period robustness check

All failed due to:
- PF drops to ~1.0 on alt periods
- Sharpe becomes negative
- Trade count explodes
```

**Conclusion:**

> No sustainable edge found in:
> - Symbols: BTC, ETH, SOL
> - Timeframes: 15m, 1h
> - Targets: Momentum, Reversal, Vol Expansion
> - Features: base, extended
> - Realistic costs: 0.3% per round-trip

**Decision:** **ARCHIVE alpha research (Stages 7-8-8b-9), PIVOT TO AGI-BRAIN**

---

## Kill-Switch

**Trigger Point:** After Phase 3 complete

**Condition:** If **zero candidates** pass multi-period robustness check

**Action:**
1. Document null result in `alpha_scan_v1_report.md`
2. Update `alpha_research_report.md` with final conclusion
3. Archive full alpha research track
4. **PIVOT** to AGI-Brain / other revenue streams
5. **DO NOT** pursue further alpha research without new data sources or methods

---

## Notes

- This is the **last systematic attempt** at alpha discovery with current infrastructure
- Null result is VALUABLE — it definitively closes the question
- Either way, provides clarity for strategic direction
- Timebox: 1-2 weeks MAX (hard limit)

---

**Status:** Phase 1 ACTIVE  
**Created:** 2025-12-08  
**Grid Size:** 54 experiments  
**Next:** Implement Phase 2 batch machinery
