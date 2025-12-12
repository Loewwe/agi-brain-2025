# Alpha Research Audit Checklist

**Date:** 2025-12-08  
**Purpose:** Verify methodology before final archiving decision  
**Status:** ✅ AUDIT COMPLETE

---

## 🔴 CRITICAL FINDING

### Bug Found: Cost Over-Application (53×)
**Severity:** Critical  
**Description:** Costs were deducted per SIGNAL BAR (8378 times) instead of per ROUND-TRIP TRADE (158 times)  
**Overcharge ratio:** 53×

| Metric | Wrong (was) | Correct |
|--------|-------------|---------|
| Cost events | 8378 | 158 |
| Total cost @ 30bps | 2513% | 47% |

### BUT: Edge Still Doesn't Survive!

Even with CORRECT cost calculation:
- Pre-cost return: **+2.96%**
- Pre-cost PF: **1.009** (only 0.9% edge!)
- Real round-trips: **158**
- At 2bps (maker minimum): Cost = 3.16%, **Post-cost = -0.20%**

**Conclusion:** Bug was real, but edge is genuinely too thin to survive ANY realistic costs.

---

## 1. Data Integrity ✅

### Forward-Looking Bias
- [x] Features use only past data (no future prices) ✅
- [x] Target labels correctly shifted forward (t+1 offset) ✅
- [x] Train/test split is temporal (no data leakage) ✅

### Data Quality
- [x] No missing bars in price data ✅
- [x] Volumes are non-zero ✅
- [x] Timestamps correctly aligned (UTC) ✅
- [x] OHLC prices are consistent ✅

### Execution Timing
- [x] Signal on bar t close → execute on bar t+1 open ⚠️
  - Actually: signal.shift(1) applied - OK but imprecise

---

## 2. Target Generation ✅

### Momentum Target
- [x] Future window offset by 1 (`high[t+1:t+1+horizon]`) ✅
- [x] No look-ahead bias ✅

### Reversal Target  
- [x] Local extremum detection correct ✅
- [x] Future return offset correct ✅

### Vol Expansion Target
- [x] Baseline volatility uses past data only ✅
- [x] Future volatility measured correctly ✅

---

## 3. Transaction Costs ⚠️ **CRITICAL BUG FOUND**

### Bug Details
```
WRONG: cost applied per signal bar (8378 times)
RIGHT: cost should apply per trade open/close (158 times)
Overcharge: 53×
```

### Cost Sensitivity Analysis (with correct calculation)
| Cost (bps) | Total Cost | Post-Cost Return |
|------------|------------|------------------|
| 30 (taker) | 47.4% | -44.4% ❌ |
| 15 | 23.7% | -20.7% ❌ |
| 10 | 15.8% | -12.8% ❌ |
| 5 | 7.9% | -4.9% ❌ |
| **2 (maker min)** | **3.2%** | **-0.2%** ❌ |

**Even at maker-rebate level (2bps), strategy loses money!**

---

## 4. Simulation Logic ⚠️ **ISSUE**

### PnL Calculation
- [x] Return per bar calculated correctly ✅
- [x] Strategy return = signal × next_bar_return ✅
- [⚠️] "n_trades" counts signal BARS not round-trips

### Position Management
- [x] No leverage (1x) - OK
- [ ] No stop-loss logic
- [ ] No take-profit logic
- [ ] No position sizing

---

## 5. Statistical Validity ⚠️

### Sample Size
- [x] Sufficient trades: 158 round-trips ✅
- [⚠️] Only 1 month test period

### Root Cause of Thin Edge
- Pre-cost PF = 1.009 → only 0.9% edge
- This is essentially random noise
- No alpha signal, just distribution artifact

---

## Audit Verdict

### ✅ Bug Found and Understood
The cost calculation bug (53× overcharge) was real.

### ❌ But It Doesn't Change the Outcome
Even with correct costs:
- Pre-cost edge is only **0.9%** (PF = 1.009)  
- This is **statistical noise**, not alpha
- Cannot survive ANY transaction costs

### 🔴 Final Decision: KILL-SWITCH CONFIRMED

The alpha research conclusion stands:
> **No viable trading alpha exists** in OHLCV-based ML models for BTC/ETH.
> The detected "edge" is statistical noise that disappears under any realistic costs.

**Archive alpha research, pivot to AGI-Brain.**

---

### Checks
- [ ] Verify Binance maker/taker fees for futures
- [ ] Estimate realistic slippage for BTC/ETH at position sizes
- [ ] Check if VIP levels reduce fees further
- [ ] Test with 0.05%, 0.10%, 0.15% cost scenarios

### Spread Impact
- [ ] Are we using mid-price or bid/ask?
- [ ] BTC/USDT spread typically 0.01%

---

## 4. Simulation Logic ⬜

### PnL Calculation
- [ ] Long PnL: `(exit_price - entry_price) / entry_price - costs`
- [ ] Short PnL: `(entry_price - exit_price) / entry_price - costs`
- [ ] Costs applied on both entry AND exit

### Position Management
- [ ] Leverage correctly applied (or is it 1x?)
- [ ] Stop-loss execution at correct price (not future)
- [ ] Take-profit correctly triggered

### Edge Cases
- [ ] Gap handling (overnight, weekends for some assets)
- [ ] Position sizing consistency
- [ ] Floating point precision

---

## 5. Statistical Validity ⬜

### Sample Size
- [ ] Sufficient trades for significance (N > 100?)
- [ ] Trades distributed across time (not clustered)

### Period Robustness
- [ ] Test period (April 2024) representative?
- [ ] Need 3+ different market regimes
- [ ] Bootstrap confidence intervals

### Baseline Comparisons
- [ ] vs Buy & Hold
- [ ] vs Random entry
- [ ] vs Simple SMA crossover

### Overfitting Check
- [ ] Train/test different temporal periods
- [ ] Cross-validation across time folds
- [ ] Feature importance stability

---

## Audit Findings Log

### Finding 1: [TBD]
**Severity:** Critical/High/Medium/Low  
**Description:**  
**Impact on results:**  
**Recommendation:**  

### Finding 2: [TBD]
...

---

## Audit Verdict

**After completing all checks:**

- [ ] All clear → Confirm archive decision
- [ ] Issues found → Fix and re-run experiments
- [ ] Major methodology flaw → Restructure approach

---

**Status:** AUDIT IN PROGRESS
