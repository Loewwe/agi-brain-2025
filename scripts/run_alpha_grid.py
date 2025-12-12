#!/usr/bin/env python3
"""
Alpha Scan Grid Runner (Stage 9 Phase 2)

Batch execution of alpha experiments from grid configuration.
Runs all experiments, saves results, skips already completed ones.

Usage:
    python scripts/run_alpha_grid.py --grid alpha_scan_grid.yaml --output-dir results/alpha_scan
    
    # Resume from where we left off
    python scripts/run_alpha_grid.py --grid alpha_scan_grid.yaml --output-dir results/alpha_scan --resume
    
    # Run specific experiment
    python scripts/run_alpha_grid.py --grid alpha_scan_grid.yaml --id scan_001
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.eval import (
    run_experiment,
    ExperimentConfig,
    DateRange,
    FeatureSet,
    ModelType,
)
from src.research.targets import TargetConfig, TargetType


def parse_target_config(target_dict: dict) -> TargetConfig:
    """Parse target configuration from YAML dict."""
    target_type_map = {
        "momentum": TargetType.MOMENTUM,
        "reversal": TargetType.REVERSAL,
        "vol_expansion": TargetType.VOL_EXPANSION,
    }
    
    target_type = target_type_map.get(target_dict["type"])
    if target_type is None:
        raise ValueError(f"Unknown target type: {target_dict['type']}")
    
    return TargetConfig(
        type=target_type,
        horizon_bars=target_dict.get("horizon_bars", 12),
        min_move_pct=target_dict.get("min_move_pct", 0.005),
        extremum_window=target_dict.get("extremum_window", 3),
        vol_window=target_dict.get("vol_window", 24),
        vol_factor=target_dict.get("vol_factor", 1.5),
    )


def parse_experiment_config(exp_dict: dict) -> ExperimentConfig:
    """Parse experiment configuration from YAML dict."""
    feature_set_map = {
        "base": FeatureSet.BASE,
        "extended": FeatureSet.EXTENDED,
    }
    
    model_type_map = {
        "lightgbm": ModelType.LIGHTGBM,
        "seq": ModelType.SEQ,
    }
    
    return ExperimentConfig(
        symbol=exp_dict["symbol"],
        timeframe=exp_dict["timeframe"],
        target=parse_target_config(exp_dict["target"]),
        feature_set=feature_set_map.get(exp_dict.get("feature_set", "extended"), FeatureSet.EXTENDED),
        model_type=model_type_map.get(exp_dict.get("model_type", "lightgbm"), ModelType.LIGHTGBM),
        train_period=DateRange(
            start=date.fromisoformat(str(exp_dict["train_period"]["start"])),
            end=date.fromisoformat(str(exp_dict["train_period"]["end"])),
        ),
        test_period=DateRange(
            start=date.fromisoformat(str(exp_dict["test_period"]["start"])),
            end=date.fromisoformat(str(exp_dict["test_period"]["end"])),
        ),
        random_state=exp_dict.get("random_state", 42),
        commission_bps=exp_dict.get("commission_bps", 10),
        slippage_bps=exp_dict.get("slippage_bps", 5),
    )


def load_grid(grid_path: str) -> list[dict]:
    """Load experiment grid from YAML file."""
    with open(grid_path, "r") as f:
        grid = yaml.safe_load(f)
    return grid.get("experiments", [])


def get_result_path(output_dir: str, exp_id: str) -> Path:
    """Get path for experiment result file."""
    return Path(output_dir) / f"{exp_id}.json"


def run_single_experiment(exp_dict: dict, output_dir: str, force: bool = False) -> dict:
    """Run a single experiment and save results."""
    exp_id = exp_dict["id"]
    result_path = get_result_path(output_dir, exp_id)
    
    # Skip if already exists and not forcing
    if result_path.exists() and not force:
        print(f"⏭️  {exp_id}: Already exists, skipping")
        with open(result_path, "r") as f:
            return json.load(f)
    
    print(f"🔬 {exp_id}: Running...")
    start_time = time.time()
    
    try:
        config = parse_experiment_config(exp_dict)
        result = run_experiment(config)
        
        # Convert result to dict with additional metadata
        result_dict = result.model_dump()
        result_dict["_meta"] = {
            "id": exp_id,
            "symbol": exp_dict["symbol"],
            "timeframe": exp_dict["timeframe"],
            "target_type": exp_dict["target"]["type"],
            "horizon_bars": exp_dict["target"].get("horizon_bars", 12),
            "feature_set": exp_dict.get("feature_set", "extended"),
            "model_type": exp_dict.get("model_type", "lightgbm"),
            "train_start": str(exp_dict["train_period"]["start"]),
            "train_end": str(exp_dict["train_period"]["end"]),
            "test_start": str(exp_dict["test_period"]["start"]),
            "test_end": str(exp_dict["test_period"]["end"]),
            "commission_bps": exp_dict.get("commission_bps", 10),
            "slippage_bps": exp_dict.get("slippage_bps", 5),
            "elapsed_seconds": time.time() - start_time,
        }
        
        # Calculate derived metrics
        test_days = 30  # April 2024
        if result_dict["n_trades"] > 0:
            result_dict["trades_per_day"] = result_dict["n_trades"] / test_days
            result_dict["trades_per_month"] = result_dict["n_trades"]  # Already per month
        else:
            result_dict["trades_per_day"] = 0
            result_dict["trades_per_month"] = 0
        
        # Save result
        os.makedirs(output_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        pf = result_dict.get("profit_factor_post_cost")
        sharpe = result_dict.get("sharpe_post_cost")
        n_trades = result_dict.get("n_trades", 0)
        
        status = "✅" if pf and pf > 1.05 else "⚪"
        pf_str = f"{pf:.3f}" if pf is not None else "N/A"
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
        print(f"{status} {exp_id}: PF_post={pf_str}, Sharpe_post={sharpe_str}, trades={n_trades}, ({elapsed:.1f}s)")
        
        return result_dict
        
    except Exception as e:
        print(f"❌ {exp_id}: Error - {str(e)}")
        error_result = {
            "_meta": {"id": exp_id, "error": str(e)},
            "notes": f"Error: {str(e)}",
        }
        with open(result_path, "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result


def run_grid(
    grid_path: str,
    output_dir: str,
    resume: bool = True,
    specific_id: str | None = None,
) -> list[dict]:
    """Run all experiments in grid."""
    experiments = load_grid(grid_path)
    
    if specific_id:
        experiments = [e for e in experiments if e["id"] == specific_id]
        if not experiments:
            print(f"❌ No experiment found with id: {specific_id}")
            return []
    
    total = len(experiments)
    print(f"\n{'='*60}")
    print(f"ALPHA SCAN v1 — Running {total} experiments")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{total}] ", end="")
        result = run_single_experiment(exp, output_dir, force=not resume)
        results.append(result)
    
    # Print summary
    completed = len([r for r in results if "_meta" in r and "error" not in r.get("_meta", {})])
    errors = len([r for r in results if "_meta" in r and "error" in r.get("_meta", {})])
    
    print(f"\n{'='*60}")
    print(f"DONE: {completed}/{total} completed, {errors} errors")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run alpha scan experiment grid")
    parser.add_argument("--grid", required=True, help="Path to grid YAML file")
    parser.add_argument("--output-dir", default="results/alpha_scan", help="Output directory for results")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed experiments")
    parser.add_argument("--force", action="store_true", help="Force re-run all experiments")
    parser.add_argument("--id", dest="specific_id", help="Run only specific experiment ID")
    
    args = parser.parse_args()
    
    resume = not args.force
    run_grid(args.grid, args.output_dir, resume=resume, specific_id=args.specific_id)


if __name__ == "__main__":
    main()
