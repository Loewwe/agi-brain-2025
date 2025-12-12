# Stage 8b Quick Reference

## Status Line
```
🟡 Stage 8b — Proof-of-Edge (P0): APPROVED & ACTIVE

Goal: Cost-aware validation (PF ≥ 1.10, Sharpe ≥ 1.5 post-cost, <1k trades/month, 2+ periods)
Status: ⬜ Phase 1 · ⬜ Phase 2 · ⬜ Phase 3
Timebox: 7-10 days MAX (no scope expansion)
Kill-Switch: After Phase 1+2, if no config with PF_post_cost ≥ 1.05-1.10 → ARCHIVE & PIVOT TO AGI
```

---

## 🧨 Kill-Switch Rule (Critical)

**After Phase 1 (Costs) + Phase 2 (Trade Thinning):**

**IF:** No configuration exists with:
- `PF_post_cost ≥ 1.05–1.10` AND
- `n_trades` reduced by at least 2× from baseline

**THEN:**
1. Document in `alpha_research_report.md`: "Edge ≈ 1.0, not viable for Stage 9"
2. **ARCHIVE** AlphaFamily v0 as R&D result (not for production/funding)
3. **PIVOT** to AGI-Brain / Stage 9+10 architecture
4. **DO NOT** continue to Phase 3 (waste of time)

**ELSE IF:** 1-2 configs meet criteria:
1. Mark as **VolExp/Momentum v1 (candidate)**
2. Continue to Phase 3 (robustness check)
3. Decide later: Stage 9 (paper-trading) or defer

---

## Quick Summary

**What:** Validate VolExp/Momentum edge with realistic transaction costs
**Why:** Current PF 1.02-1.03 is pre-cost; may disappear with 0.15-0.2% fees+slippage
**When:** After Stage 8 v0 (complete)
**How:** 3 phases (costs, thinning, robustness)

## Phases Overview

| Phase | Tasks | Tests | DoD |
|-------|-------|-------|-----|
| **1. Costs** | Add commission/slippage to eval | 3 tests | Before/after table, 6 experiments re-run |
| **2. Thinning** | Confidence thresholds, trade reduction | 2 tests | Threshold sweep, trades ↓2-4× |
| **3. Robustness** | 2+ test periods, cross-validation | 2 tests | Period summary, stable PF/Sharpe |

## Success / Failure Outcomes

**Success (Variant B):**
- PF_post_cost ≥ 1.15, Sharpe ≥ 1.5 on 2+ periods, <1000 trades/month
- **Action:** Promote VolExp/Momentum v1 to Stage 9 (paper-trading)

**Failure (Variant A):**
- PF_post_cost ≈ 1.0-1.03, unstable across periods
- **Action:** Archive AlphaFamily v0 as R&D, not for production/funding

## Dependencies

**Requires:**
- ✅ Stage 8 complete (28/28 tests, 6 experiments)
- ✅ `src/research/eval.py` pipeline
- ✅ Experiment configs in `experiments/`

**Provides to Stage 9:**
- Validated alpha configs (if success)
- OR honest null result (if failure)

## Time Estimate

- Phase 1: 2-3 days (eval extensions + tests)
- Phase 2: 2-3 days (filtering + sweep)
- Phase 3: 3-4 days (multi-period + analysis)
- **Total:** ~7-10 days (assuming no blockers)

## Risk Assessment

**High Risk:**
- Edge disappears post-cost (most likely given thin PF)
- Robustness fails on different periods

**Medium Risk:**
- Trade thinning reduces trades but also reduces edge
- Data quality issues on historical periods

**Low Risk:**
- Technical implementation (pipeline is solid)
