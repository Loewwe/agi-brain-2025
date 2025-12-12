# Stage 9 Quick Reference — Alpha Scan v1

## Status Line
```
🟡 Stage 9 — Alpha Scan v1 (P0): APPROVED & READY TO START

Goal: Mass screening (50-200 experiments) → find 2-5 viable candidates OR honest null result
Focus: 15m/1h timeframes (NOT 5m), realistic costs (0.3%), multi-period robustness
Timebox: 1-2 weeks MAX
Kill-Switch: If zero candidates survive Phase 3 → Archive & Pivot to AGI
Decision: Last scientific attempt before full AGI pivot
```

---

## Why This Exists

**Stage 8b Result:** Edge destroyed by transaction costs at 5m timeframe  
**Stage 9 Hypothesis:** Longer timeframes (15m, 1h) → fewer trades → edge might survive

**Alternative Conclusion:** After 50–200 experiments, if still no edge → **definitively close alpha search**

---

## 3 Phases Overview

| Phase | What | Output | Kill-Switch |
|-------|------|--------|-------------|
| **1. Experiment Space** | Define 30-60 configs (symbols × TF × targets × features) | `alpha_scan_grid.yaml` | — |
| **2. Mass Execution** | Run all + auto-filter (candidates/borderline/rejected) | 2-5 candidates OR "all rejected" | If 0 candidates → skip Phase 3 |
| **3. Robustness** | Test candidates on 2 more periods | Final list of robust candidates | If 0 robust → **ARCHIVE & PIVOT** |

---

## Key Differences from Stage 8

| Aspect | Stage 8 | Stage 9 Alpha Scan v1 |
|--------|---------|------------------------|
| **Timeframes** | 5m (killed by costs) | **15m, 1h** (reduce trade freq) |
| **Volume** | 6 manual experiments | **50-200 automated experiments** |
| **Filtering** | Manual analysis | **Auto-classification** (candidate/borderline/reject) |
| **Robustness** | Single period | **Multi-period** (2-3 test windows) |
| **Decision** | "Thin edge, need validation" | **"Viable" OR "Null result, pivot to AGI"** |

---

## P0 Candidate Thresholds

**Must meet ALL:**
- `PF_post_cost ≥ 1.15`
- `Sharpe_post_cost ≥ 1.5`
- `trades_per_month ≤ 1000` (better: ≤500)
- `max_drawdown ≤ 25%`
- **Robust across 2+ periods**

**Borderline (some promise):**
- `PF_post_cost ≥ 1.05`
- `Sharpe_post_cost ≥ 1.0`
- `trades_per_month ≤ 2000`

**Rejected:** Everything else

---

## Success / Failure Scenarios

### ✅ Success (Scenario A)

**Found 2–5 robust candidates:**
```
scan_027 (SOL/1h/momentum): PF 1.18, Sharpe 1.7, 420 trades/mo
scan_041 (BTC/15m/vol_exp): PF 1.16, Sharpe 1.6, 680 trades/mo
scan_033 (ETH/1h/reversal): PF 1.15, Sharpe 1.5, 580 trades/mo
```

**Next:** Stage 10 (Paper-Trading Validation)

### 🛑 Failure (Scenario B)

**Zero survivors after robustness check:**
```
54 experiments → 3 candidates → 0 passed multi-period test
```

**Conclusion:**
> No sustainable edge found in:
> - Symbols: BTC, ETH, SOL
> - Timeframes: 15m, 1h
> - Targets: Momentum, Reversal, Vol Expansion
> - Realistic costs: 0.3% per round-trip

**Next:** **ARCHIVE alpha research, PIVOT TO AGI-BRAIN**

---

## Tools Created

**Phase 1:**
- `alpha_scan_grid.yaml` — experiment matrix
- `ALPHA_SCAN_SPEC.md` — metrics & thresholds

**Phase 2:**
- `scripts/run_alpha_grid.py` — batch launcher
- `scripts/summarize_alpha_scan.py` — aggregator + ranker
- `results/alpha_scan_summary.md` — auto-generated report

**Phase 3:**
- Multi-period configs
- `alpha_scan_v1_report.md` — final verdict

---

## Time Estimate

- Phase 1: 2-3 days (grid definition)
- Phase 2: 3-5 days (batch machinery + ~50 experiments @ 2-3 min each)
- Phase 3: 3-4 days (multi-period for  candidates)
- **Total: 1-2 weeks**

---

## Risk Assessment

**High Risk:** All experiments fail → honest null result (this is VALUABLE, not a failure)  
**Medium Risk:** Borderline candidates (PF 1.05-1.10) — hard to decide  
**Low Risk:** Technical implementation (Stage 8 infra is solid)

---

## For STATUS.md

```markdown
🔵 **Stage 9 — Alpha Scan v1 (P0):** NOT STARTED (pending decision)
├─ ⬜ Phase 1: Experiment Space (30-60 configs, 15m/1h)
├─ ⬜ Phase 2: Mass Execution (batch + auto-filter)
└─ ⬜ Phase 3: Multi-Period Robustness
🧨 **Kill-Switch:** Zero robust candidates → Archive & Pivot to AGI
📅 **Timebox:** 1-2 weeks
🎯 **Goal:** Find 2-5 viable candidates OR honest null result
```
