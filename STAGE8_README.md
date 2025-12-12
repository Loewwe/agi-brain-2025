# Stage 8 Alpha Research — Quick Start

## Setup Complete ✅

**Status:** 28/28 tests passing | 5/5 phases complete

---

## Running Experiments

### Option 1: Run All (Batch)

```bash
./scripts/run_all_experiments.sh
```

This runs all 6 experiments (2-3 min each):
- `exp_001` — Momentum, BTC
- `exp_002` — Momentum, ETH
- `exp_003` — Vol Expansion, BTC
- `exp_004` — Vol Expansion, ETH
- `exp_005` — Reversal, BTC (control)
- `exp_006` — Reversal, ETH (control)

Results saved to `results/exp_*.json`

### Option 2: Run Individual

```bash
python scripts/run_alpha_experiment.py \
    --config experiments/exp_001_momentum_btc_5m.yaml \
    --output results/exp_001.json
```

---

## Analyzing Results

Open `research/05_alpha_scan.ipynb` (create if needed):

```python
import json
import pandas as pd
from pathlib import Path

# Load all results
results = []
for p in Path("results").glob("exp_*.json"):
    with open(p) as f:
        data = json.load(f)
        data['exp_id'] = p.stem
        results.append(data)

df = pd.DataFrame(results)

# View summary
print(df[['exp_id', 'auc', 'win_rate', 'sharpe', 'profit_factor', 'n_trades']])
```

---

## Next Steps

1. **Run batch:** `./scripts/run_all_experiments.sh`
2. **Open notebook:** Create `research/05_alpha_scan.ipynb`
3. **Analyze:** Feature importances, distributions, train vs test
4. **Update report:** `alpha_research_report.md`
5. **Decide:** Case A (edge found) → deep dive | Case B (no edge) → expand hypotheses

---

## Experiment Parameters

All experiments use:
- **Train:** 2024-01-01 to 2024-03-31 (3 months)
- **Test:** 2024-04-01 to 2024-04-30 (1 month)
- **Features:** Extended (OBV, Multi-TF, Microstructure)
- **Model:** LightGBM (tabular baseline)
- **Seed:** 42 (deterministic)

Targets:
- **Momentum:** 12-bar horizon, 0.5% min move
- **Vol Expansion:** 48-bar window, 1.5× factor
- **Reversal:** 15-bar horizon, 0.8% move, 3-bar extremum

---

## File Structure

```
experiments/          # YAML configs
  exp_001_*.yaml
  ...

results/              # JSON outputs
  exp_001.json
  ...

alpha_research_report.md  # Summary report

research/             # Analysis notebooks
  05_alpha_scan.ipynb (TBD)
```
