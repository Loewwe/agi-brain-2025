#!/usr/bin/env python3
"""
Stage7 Experiment CLI.

Usage:
    # Create and run experiment
    python scripts/run_experiment.py \
        --name "Stage6 90-day test" \
        --symbols BTC,ETH,SOL \
        --days 90
    
    # List experiments
    python scripts/run_experiment.py --list
    
    # View experiment
    python scripts/run_experiment.py --view <experiment_id>
    
    # Run existing experiment
    python scripts/run_experiment.py --run <experiment_id>
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment import (
    Experiment,
    ExperimentRunner,
    ExperimentStatus,
    SimulatorConfig,
)
from src.experiment.experiment import create_experiment_runner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage7 Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Actions
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all experiments",
    )
    parser.add_argument(
        "--view", "-v",
        type=str,
        metavar="ID",
        help="View experiment details",
    )
    parser.add_argument(
        "--run", "-r",
        type=str,
        metavar="ID",
        help="Run existing experiment",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show experiment statistics",
    )
    
    # Create new experiment
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--goal", "-g",
        type=str,
        default="Backtest Stage6 strategy",
        help="Experiment goal",
    )
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        default="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT",
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=90,
        help="Number of days to backtest",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD), defaults to --days ago",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD), defaults to today",
    )
    
    # Configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Don't fetch missing data",
    )
    
    return parser.parse_args()


def list_experiments(runner: ExperimentRunner):
    """List all experiments."""
    experiments = runner.list_experiments(limit=100)
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\n{'ID':<10} {'Name':<30} {'Status':<12} {'Trades':<8} {'Win Rate':<10} {'Return':<10}")
    print("-" * 90)
    
    for exp in experiments:
        trades = "-"
        win_rate = "-"
        ret = "-"
        
        if exp.result:
            trades = str(exp.result.total_trades)
            win_rate = f"{exp.result.win_rate * 100:.1f}%"
            ret = f"{exp.result.total_return_pct:.2f}%"
        
        status = exp.status.value
        if exp.status == ExperimentStatus.COMPLETED:
            status = "✅ " + status
        elif exp.status == ExperimentStatus.FAILED:
            status = "❌ " + status
        elif exp.status == ExperimentStatus.RUNNING:
            status = "🔄 " + status
        
        print(f"{exp.experiment_id:<10} {exp.name[:28]:<30} {status:<12} {trades:<8} {win_rate:<10} {ret:<10}")
    
    print(f"\nTotal: {len(experiments)} experiments")


def view_experiment(runner: ExperimentRunner, experiment_id: str):
    """View experiment details."""
    exp = runner.get_experiment(experiment_id)
    
    if not exp:
        print(f"Experiment '{experiment_id}' not found.")
        return
    
    print(f"\n{'=' * 60}")
    print(f"Experiment: {exp.name}")
    print(f"{'=' * 60}")
    
    print(f"\nID:          {exp.experiment_id}")
    print(f"Goal:        {exp.goal}")
    print(f"Status:      {exp.status.value}")
    print(f"Symbols:     {', '.join(exp.symbols)}")
    print(f"Date Range:  {exp.date_from} to {exp.date_to}")
    print(f"Created:     {exp.created_at}")
    
    if exp.started_at:
        print(f"Started:     {exp.started_at}")
    if exp.completed_at:
        print(f"Completed:   {exp.completed_at}")
    
    if exp.error:
        print(f"\n❌ Error: {exp.error}")
    
    if exp.result:
        print(f"\n{'=' * 60}")
        print("Results")
        print(f"{'=' * 60}")
        
        r = exp.result
        print(f"\nTotal Trades:    {r.total_trades}")
        print(f"Win Rate:        {r.win_rate * 100:.2f}%")
        print(f"Profit Factor:   {r.profit_factor:.2f}")
        print(f"Avg R:           {r.avg_r_multiple:.3f}")
        print(f"Expectancy:      {r.expectancy:.3f}R")
        
        print(f"\nTotal Return:    {r.total_return_pct:.2f}%")
        print(f"Avg Daily P&L:   {r.avg_daily_pnl_pct:.3f}%")
        print(f"Max Drawdown:    {r.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio:    {r.sharpe_ratio:.3f}")
        
        if r.exits_by_reason:
            print(f"\nExits by Reason:")
            for reason, count in sorted(r.exits_by_reason.items()):
                pct = count / r.total_trades * 100 if r.total_trades > 0 else 0
                print(f"  {reason:<15} {count:>5} ({pct:.1f}%)")


def show_stats(runner: ExperimentRunner):
    """Show experiment statistics."""
    stats = runner.get_stats()
    
    print(f"\n{'=' * 40}")
    print("Experiment Statistics")
    print(f"{'=' * 40}")
    
    print(f"\nTotal experiments: {stats['total']}")
    
    print("\nBy status:")
    for status, count in stats['by_status'].items():
        print(f"  {status:<12} {count}")
    
    if stats['completed_count'] > 0:
        print(f"\nCompleted experiments: {stats['completed_count']}")
        print(f"  Avg Win Rate: {stats['avg_win_rate'] * 100:.1f}%")
        print(f"  Avg Return:   {stats['avg_return_pct']:.2f}%")


async def create_and_run_experiment(
    runner: ExperimentRunner,
    name: str,
    goal: str,
    symbols: list[str],
    date_from: date,
    date_to: date,
    fetch_data: bool = True,
):
    """Create and run a new experiment."""
    print(f"\n🧪 Creating experiment: {name}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Period:  {date_from} to {date_to}")
    
    # Create experiment
    exp = runner.create_experiment(
        name=name,
        goal=goal,
        symbols=symbols,
        date_from=date_from,
        date_to=date_to,
    )
    
    print(f"   ID: {exp.experiment_id}")
    print(f"\n🚀 Running experiment...")
    
    # Run experiment
    try:
        exp = await runner.run(exp, fetch_missing_data=fetch_data)
        
        print(f"\n✅ Experiment completed!")
        view_experiment(runner, exp.experiment_id)
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise


async def run_existing_experiment(
    runner: ExperimentRunner,
    experiment_id: str,
    fetch_data: bool = True,
):
    """Run an existing experiment."""
    exp = runner.get_experiment(experiment_id)
    
    if not exp:
        print(f"Experiment '{experiment_id}' not found.")
        return
    
    print(f"\n🚀 Running experiment: {exp.name}")
    
    try:
        exp = await runner.run(exp, fetch_missing_data=fetch_data)
        
        print(f"\n✅ Experiment completed!")
        view_experiment(runner, exp.experiment_id)
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize runner
    runner = create_experiment_runner(args.data_path)
    
    # Handle actions
    if args.list:
        list_experiments(runner)
        return
    
    if args.view:
        view_experiment(runner, args.view)
        return
    
    if args.stats:
        show_stats(runner)
        return
    
    if args.run:
        asyncio.run(run_existing_experiment(
            runner,
            args.run,
            fetch_data=not args.no_fetch,
        ))
        return
    
    # Create new experiment
    if args.name:
        # Parse symbols
        symbols = [s.strip() for s in args.symbols.split(",")]
        
        # Parse dates
        if args.end_date:
            date_to = date.fromisoformat(args.end_date)
        else:
            date_to = date.today()
        
        if args.start_date:
            date_from = date.fromisoformat(args.start_date)
        else:
            date_from = date_to - timedelta(days=args.days)
        
        asyncio.run(create_and_run_experiment(
            runner=runner,
            name=args.name,
            goal=args.goal,
            symbols=symbols,
            date_from=date_from,
            date_to=date_to,
            fetch_data=not args.no_fetch,
        ))
        return
    
    # No action specified - show help
    print("Stage7 Experiment CLI")
    print("\nUsage examples:")
    print("  python scripts/run_experiment.py --list")
    print("  python scripts/run_experiment.py --name 'Stage6 90d' --symbols BTC,ETH --days 90")
    print("  python scripts/run_experiment.py --view <ID>")
    print("  python scripts/run_experiment.py --run <ID>")
    print("\nRun with --help for all options.")


if __name__ == "__main__":
    main()
