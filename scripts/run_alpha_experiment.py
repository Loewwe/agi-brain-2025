#!/usr/bin/env python3
"""
Alpha Research Experiment Runner

CLI tool for running alpha research experiments.
"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import date

from src.research.eval import run_experiment, ExperimentConfig, DateRange
from src.research.targets import TargetConfig, TargetType


def load_config(config_path: Path) -> ExperimentConfig:
    """Load experiment config from YAML or JSON."""
    with open(config_path) as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    # Convert date strings to date objects
    if 'train_period' in data:
        data['train_period'] = DateRange(
            start=date.fromisoformat(data['train_period']['start']),
            end=date.fromisoformat(data['train_period']['end']),
        )
    
    if 'test_period' in data:
        data['test_period'] = DateRange(
            start=date.fromisoformat(data['test_period']['start']),
            end=date.fromisoformat(data['test_period']['end']),
        )
    
    # Convert target config
    if 'target' in data:
        data['target'] = TargetConfig(**data['target'])
    
    return ExperimentConfig(**data)


def main():
    parser = argparse.ArgumentParser(
        description='Run alpha research experiment'
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to experiment config (YAML or JSON)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Path to save results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Run experiment
    print(f"\nRunning experiment:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Target: {config.target.type}")
    print(f"  Model: {config.model_type}")
    print(f"  Train: {config.train_period.start} to {config.train_period.end}")
    print(f"  Test: {config.test_period.start} to {config.test_period.end}")
    print()
    
    result = run_experiment(config)
    
    # Display results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"AUC:            {result.auc:.4f}" if result.auc else "AUC:            N/A")
    print(f"Trades:         {result.n_trades}")
    
    if result.n_trades > 0:
        print(f"Win Rate:       {result.win_rate:.2%}" if result.win_rate else "Win Rate:       N/A")
        print(f"Total Return:   {result.total_return:.2%}" if result.total_return else "Total Return:   N/A")
        print(f"Sharpe Ratio:   {result.sharpe:.2f}" if result.sharpe else "Sharpe Ratio:   N/A")
        print(f"Profit Factor:  {result.profit_factor:.2f}" if result.profit_factor else "Profit Factor:  N/A")
        print(f"Max Drawdown:   {result.max_drawdown:.2%}" if result.max_drawdown else "Max Drawdown:   N/A")
    
    if result.notes:
        print(f"\nNotes: {result.notes}")
    
    print("="*60)
    
    # Save results
    if args.output:
        output_data = result.model_dump()
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
