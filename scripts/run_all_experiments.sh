#!/bin/bash
# Run all Stage 8 Alpha Research experiments

echo "🧪 Running Stage 8 Alpha Research Experiment Batch"
echo "=================================================="
echo ""

# Create results directory
mkdir -p results

# Run experiments
for exp in experiments/exp_00*.yaml; do
    exp_name=$(basename "$exp" .yaml)
    echo "▶ Running $exp_name..."
    python scripts/run_alpha_experiment.py \
        --config "$exp" \
        --output "results/${exp_name}.json"
    echo ""
done

echo "✅ All experiments complete!"
echo "Results saved to results/"
echo ""
echo "Next: Open research/05_alpha_scan.ipynb to analyze results"
