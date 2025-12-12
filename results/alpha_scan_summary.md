# Alpha Scan v1 — Summary Report

**Generated:** 2025-12-08T01:14:58

---

## Overview

**Total Experiments:** 27

| Category | Count | % |
|----------|-------|---|
| 🟢 Candidates | 0 | 0.0% |
| 🟡 Borderline | 0 | 0.0% |
| ⚪ Rejected | 27 | 100.0% |
| ❌ Errors | 0 | 0.0% |

---

## Thresholds

**Candidate (must meet ALL):**
- PF_post_cost ≥ 1.15
- Sharpe_post_cost ≥ 1.5
- trades/month ≤ 1000
- MaxDD ≥ -25%

**Borderline (some promise):**
- PF_post_cost ≥ 1.05
- Sharpe_post_cost ≥ 1.0
- trades/month ≤ 2000

---

## 🟢 Candidates

**No candidates found.**

---

## 🟡 Borderline

**No borderline experiments.**

---

## ⚪ Rejected

**27 experiments rejected** (not meeting borderline criteria)

Common rejection reasons:
- PF_post_cost ≤ 1.02 (thin or no edge)
- Sharpe_post_cost ≤ 1.0 (poor risk-adjusted returns)
- Too many trades (costs dominate)

### Top 5 Rejected (best of the rejected):

| ID | Symbol | TF | Target | Feat | PF_pre | PF_post | Sharpe_pre | Sharpe_post | Trades | MaxDD | AUC | Score |
|-----|--------|-----|--------|------|--------|---------|------------|-------------|--------|-------|------|-------|
| scan_001_btc_5m_momentum_short_ext | BTC/USDT | 5m | momentum | ext | 0.992 | 0.011 | -0.69 | -521.66 | 7384 | -100.0% | 0.726 | -0.000 |
| scan_002_btc_5m_momentum_med_ext | BTC/USDT | 5m | momentum | ext | 1.039 | 0.014 | 3.29 | -498.27 | 7286 | -100.0% | 0.692 | -0.000 |
| scan_003_btc_5m_reversal_short_ext | BTC/USDT | 5m | reversal | ext | 1.008 | 0.021 | 0.75 | -450.01 | 8612 | -100.0% | 0.771 | -0.000 |
| scan_004_btc_5m_reversal_med_ext | BTC/USDT | 5m | reversal | ext | 1.020 | 0.022 | 1.79 | -444.56 | 8606 | -100.0% | 0.814 | -0.000 |
| scan_005_btc_5m_volexp_short_ext | BTC/USDT | 5m | vol_expansion | ext | 1.028 | 0.025 | 2.45 | -425.49 | 8628 | -100.0% | 0.697 | -0.000 |

---

## Verdict

🛑 **NULL RESULT:** No candidates or meaningful borderline experiments.

**Conclusion:**
> No sustainable edge found in current search space:
> - Symbols: BTC, ETH, SOL
> - Timeframes: 15m, 1h
> - Targets: Momentum, Reversal, Vol Expansion
> - Realistic costs: 0.3% per round-trip

**Decision:** Archive alpha research, **PIVOT TO AGI-BRAIN**
