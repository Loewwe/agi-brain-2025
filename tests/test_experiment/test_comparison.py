"""Tests for comparison module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from src.experiment.comparison import StrategyComparison, ComparisonResult
from src.experiment.simulator import StrategyBase, SimulatorResult, ExperimentResult
from src.experiment.models import SimulatorConfig


class MockStrategy(StrategyBase):
    """Mock strategy that does nothing."""
    def check_entry(self, bar, prev_bar, symbol, state, timestamp):
        return None
    
    def check_exit(self, position, bar, bar_idx):
        return None


class TestStrategyComparison:
    """Tests for StrategyComparison."""
    
    @pytest.fixture
    def dataset(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="5min")
        return pd.DataFrame({
            "close": np.random.uniform(50000, 51000, 100),
            "high": np.random.uniform(50000, 51000, 100),
            "low": np.random.uniform(50000, 51000, 100),
            "volume": np.random.uniform(100, 1000, 100),
            "symbol": ["BTC"] * 100,
        }, index=dates)
    
    def test_compare_runs_strategies(self, dataset):
        """Test running comparison."""
        config = SimulatorConfig()
        comparison = StrategyComparison(config)
        
        strat1 = MockStrategy(config)
        strat2 = MockStrategy(config)
        
        result = comparison.compare(
            dataset,
            {"s1": strat1, "s2": strat2}
        )
        
        assert isinstance(result, ComparisonResult)
        assert "s1" in result.metrics
        assert "s2" in result.metrics
        assert "s1" in result.equity_curves
    
    def test_compare_with_baseline(self, dataset):
        """Test baseline comparison."""
        config = SimulatorConfig()
        comparison = StrategyComparison(config)
        
        main = MockStrategy(config)
        baseline = MockStrategy(config)
        
        res = comparison.compare_with_baseline(
            dataset,
            main,
            baseline
        )
        
        assert "is_better" in res
        assert "return_diff" in res
        assert "metrics" in res
