# Stage 8b Status — For ROADMAP / STATUS.md

## Compact Status Line

```markdown
🟡 **Stage 8b — Proof-of-Edge (P0):** APPROVED & ACTIVE  
├─ ⬜ Phase 1: Cost-Aware Re-Eval (fees/slippage)  
├─ ⬜ Phase 2: Trade Thinning (confidence thresholds)  
└─ ⬜ Phase 3: Robustness Check (2+ periods)  
🧨 **Kill-Switch:** After Phase 1+2, if PF < 1.05 → Archive & Pivot to AGI  
📅 **Timebox:** 7-10 days MAX
```

---

## Full Status Block (for detailed docs)

### Stage 8b — Proof-of-Edge (P0)

**Status:** 🟡 APPROVED & ACTIVE (2025-12-08)

**Objective:**  
Honest answer: Does VolExp/Momentum edge survive real transaction costs (fees + slippage) at reasonable trade volume?

**Current State:**
- Stage 8 complete: Edge detected (VolExp BTC/ETH: Sharpe +2.0-2.5, PF 1.02-1.03 pre-cost)
- Edge is thin — may disappear with 0.15-0.2% transaction costs per round-trip
- Need validation: cost-aware metrics, trade volume reduction, multi-period robustness

**Phases:**
1. ⬜ **Phase 1: Cost-Aware Re-Eval** (3 tests)
   - Add commission_bps (10) + slippage_bps (5) to eval pipeline
   - Recalculate all 6 experiments with post-cost metrics
   - Before/after costs comparison table
   
2. ⬜ **Phase 2: Trade Thinning** (2 tests)
   - Implement min_confidence threshold filtering
   - Run threshold sweep (0.55 / 0.6 / 0.65 / 0.7)
   - Target: Reduce trades 2-4× while preserving/improving PF/Sharpe
   
3. ⬜ **Phase 3: Robustness Check** (2 tests)
   - Define 2+ additional test periods (May 2024, Jan 2024)
   - Cross-period validation
   - Period stability analysis

**🧨 Kill-Switch Decision Point:**

After completing Phase 1 + Phase 2:
- **IF** no config shows `PF_post_cost ≥ 1.05-1.10` with trades reduced:
  - **STOP** immediately (skip Phase 3)
  - Document: "Edge ≈ 1.0, not viable"
  - Archive AlphaFamily v0 as R&D result
  - **PIVOT** to AGI-Brain / Stage 9+10 architecture
  
- **ELSE** 1-2 configs meet criteria:
  - Continue to Phase 3 (robustness)
  - Mark as VolExp/Momentum v1 candidate
  - Later decide: Stage 9 (paper) or defer

**Success Definition:**
- PF_post_cost ≥ 1.10 (better: 1.15+)
- Sharpe_post_cost ≥ 1.5
- n_trades ≤ 1000/month/symbol
- MaxDD ≤ 20-25%
- Stable across 2+ test periods

**Failure Definition:**
- PF_post_cost ≈ 1.0-1.03 (edge disappears with costs)
- Unstable across periods
- Too many trades even with filtering

**Timebox:** 7-10 days MAX  
**Scope:** No new models, no new features, use existing Stage 8 infrastructure only

**Dependencies:**
- ✅ Stage 8 complete (28/28 tests)
- ✅ 6 experiments analyzed
- ✅ Pipeline validated

**Deliverables:**
- `alpha_research_report.md` updated with Stage 8b summary
- Before/after costs table
- Threshold sweep results
- Cross-period analysis (if Phase 3 reached)
- **Final decision:** Promote to Stage 9 OR Archive

---

## Risk Assessment

**High Probability:** Edge disappears post-cost (most likely outcome given thin PF 1.02-1.03)  
**Medium Probability:** Edge survives but only marginally (PF 1.05-1.08)  
**Low Probability:** Strong edge emerges (PF 1.15+, Sharpe 2.0+)

**Mitigation:** Kill-switch prevents wasted time if edge doesn't exist

---

## Next Immediate Action

Start **Phase 1: Cost-Aware Re-Evaluation**
1. Extend `eval.py` with commission_bps and slippage_bps params
2. Create `test_costs_eval.py` with 3 tests
3. Re-run exp_003, exp_004, exp_002 with costs
4. Generate before/after comparison

**ETA Phase 1:** 2-3 days
