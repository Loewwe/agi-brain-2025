"""
Comparison - Strategy comparison framework.

Provides tools for comparing multiple strategies on the same dataset.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from .simulator import Simulator, StrategyBase
from .models import SimulatorConfig, ExperimentResult

logger = structlog.get_logger()


@dataclass
class ComparisonResult:
    """Result of comparing multiple strategies."""
    
    # Strategy names to metrics
    metrics: dict[str, ExperimentResult] = field(default_factory=dict)
    
    # Strategy names to equity curves (Series with timestamp index)
    equity_curves: dict[str, pd.Series] = field(default_factory=dict)
    
    # Comparison metrics
    best_strategy: str = ""
    best_metric_value: float = 0.0
    comparison_metric: str = "total_return_pct"
    
    # Statistical comparison (optional)
    statistical_tests: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "best_strategy": self.best_strategy,
            "best_metric_value": self.best_metric_value,
            "comparison_metric": self.comparison_metric,
            "statistical_tests": self.statistical_tests,
        }
    
    def summary(self) -> str:
        """Generate text summary of comparison."""
        lines = ["Strategy Comparison Summary", "=" * 40]
        
        for name, m in self.metrics.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Total Return: {m.total_return_pct:.2f}%")
            lines.append(f"  Sharpe Ratio: {m.sharpe_ratio:.2f}")
            lines.append(f"  Win Rate: {m.win_rate:.1%}")
            lines.append(f"  Max Drawdown: {m.max_drawdown_pct:.2f}%")
            lines.append(f"  Trades: {m.total_trades}")
        
        lines.append(f"\n{'=' * 40}")
        lines.append(f"BEST: {self.best_strategy} ({self.comparison_metric}: {self.best_metric_value:.2f})")
        
        return "\n".join(lines)


class StrategyComparison:
    """
    Compare multiple strategies on the same dataset.
    
    Example:
        comparison = StrategyComparison(config)
        result = comparison.compare(
            dataset,
            strategies={"baseline": stage6_strategy, "alpha": alpha_strategy}
        )
        print(result.summary())
    """
    
    def __init__(self, config: SimulatorConfig | None = None):
        """
        Initialize comparison framework.
        
        Args:
            config: Simulator config (shared across strategies)
        """
        self.config = config or SimulatorConfig()
    
    def compare(
        self,
        dataset: pd.DataFrame,
        strategies: dict[str, StrategyBase],
        comparison_metric: str = "total_return_pct",
    ) -> ComparisonResult:
        """
        Run all strategies on the same dataset and compare.
        
        Args:
            dataset: OHLCV + features DataFrame
            strategies: Dict of strategy name -> strategy instance
            comparison_metric: Metric to use for determining "best"
            
        Returns:
            ComparisonResult with all metrics
        """
        result = ComparisonResult(comparison_metric=comparison_metric)
        
        for name, strategy in strategies.items():
            logger.info("comparison.running", strategy=name)
            
            # Run simulation
            simulator = Simulator(self.config)
            sim_result = simulator.run(dataset, strategy)
            
            # Store metrics
            result.metrics[name] = sim_result.metrics
            
            # Build equity curve from daily stats
            equity_data = {
                d.date: d.equity_close 
                for d in simulator.daily_stats
            }
            result.equity_curves[name] = pd.Series(equity_data)
            
            logger.info(
                "comparison.completed",
                strategy=name,
                return_pct=sim_result.metrics.total_return_pct,
                trades=sim_result.metrics.total_trades,
            )
        
        # Determine best strategy
        if result.metrics:
            best_name = max(
                result.metrics.keys(),
                key=lambda k: getattr(result.metrics[k], comparison_metric)
            )
            result.best_strategy = best_name
            result.best_metric_value = getattr(
                result.metrics[best_name], 
                comparison_metric
            )
        
        return result
    
    def compare_with_baseline(
        self,
        dataset: pd.DataFrame,
        main_strategy: StrategyBase,
        baseline_strategy: StrategyBase,
        main_name: str = "alpha",
        baseline_name: str = "baseline",
    ) -> dict:
        """
        Compare a main strategy against a baseline.
        
        Returns dict with:
        - is_better: bool (main is better than baseline)
        - return_diff: float (return difference in %)
        - sharpe_diff: float
        - metrics: dict of both strategies
        
        Args:
            dataset: Test dataset
            main_strategy: Strategy to evaluate
            baseline_strategy: Baseline to compare against
            main_name: Name for main strategy
            baseline_name: Name for baseline
            
        Returns:
            Comparison dict
        """
        result = self.compare(
            dataset,
            {main_name: main_strategy, baseline_name: baseline_strategy},
        )
        
        main_metrics = result.metrics[main_name]
        baseline_metrics = result.metrics[baseline_name]
        
        return {
            "is_better": main_metrics.total_return_pct > baseline_metrics.total_return_pct,
            "return_diff": main_metrics.total_return_pct - baseline_metrics.total_return_pct,
            "sharpe_diff": main_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio,
            "main_return": main_metrics.total_return_pct,
            "baseline_return": baseline_metrics.total_return_pct,
            "main_sharpe": main_metrics.sharpe_ratio,
            "baseline_sharpe": baseline_metrics.sharpe_ratio,
            "metrics": {
                main_name: main_metrics.to_dict(),
                baseline_name: baseline_metrics.to_dict(),
            },
        }


def run_multi_period_comparison(
    periods: list[tuple[str, pd.DataFrame]],
    strategies: dict[str, StrategyBase],
    config: SimulatorConfig | None = None,
) -> dict:
    """
    Run comparison across multiple time periods.
    
    Useful for testing across different market regimes.
    
    Args:
        periods: List of (period_name, dataset) tuples
        strategies: Dict of strategy name -> strategy
        config: Simulator config
        
    Returns:
        Dict with per-period and aggregate results
    """
    config = config or SimulatorConfig()
    comparison = StrategyComparison(config)
    
    all_results = {}
    aggregate_returns: dict[str, list[float]] = {s: [] for s in strategies}
    aggregate_sharpes: dict[str, list[float]] = {s: [] for s in strategies}
    
    for period_name, dataset in periods:
        logger.info("multi_period.running", period=period_name)
        
        result = comparison.compare(dataset, strategies)
        all_results[period_name] = result
        
        for strat_name, metrics in result.metrics.items():
            aggregate_returns[strat_name].append(metrics.total_return_pct)
            aggregate_sharpes[strat_name].append(metrics.sharpe_ratio)
    
    # Calculate aggregate stats
    aggregate = {}
    for strat_name in strategies:
        returns = aggregate_returns[strat_name]
        sharpes = aggregate_sharpes[strat_name]
        
        aggregate[strat_name] = {
            "avg_return": np.mean(returns),
            "total_return": sum(returns),
            "avg_sharpe": np.mean(sharpes),
            "positive_periods": sum(1 for r in returns if r > 0),
            "total_periods": len(returns),
            "consistency": sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
        }
    
    # Find overall best
    best_strategy = max(
        strategies.keys(),
        key=lambda s: aggregate[s]["avg_return"]
    )
    
    return {
        "periods": {p: r.to_dict() for p, r in all_results.items()},
        "aggregate": aggregate,
        "best_strategy": best_strategy,
        "period_count": len(periods),
    }
