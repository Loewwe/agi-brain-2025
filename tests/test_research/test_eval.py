
import pytest
import json
from pathlib import Path
from datetime import date

from src.research.eval import (
    run_experiment,
    ExperimentConfig,
    EvalResult,
    DateRange,
    ModelType,
    FeatureSet,
)
from src.research.targets import TargetConfig, TargetType


def test_eval_result_fields_present():
    """Test that EvalResult contains all required fields."""
    # Create minimal experiment config
    config = ExperimentConfig(
        symbol="BTC/USDT",
        timeframe="5m",
        target=TargetConfig(type=TargetType.MOMENTUM, horizon_bars=12),
        feature_set=FeatureSet.BASE,
        model_type=ModelType.LIGHTGBM,
        train_period=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 15)),
        test_period=DateRange(start=date(2024, 1, 16), end=date(2024, 1, 20)),
        random_state=42,
    )
    
    # Run experiment
    result = run_experiment(config)
    
    # Check that result is EvalResult
    assert isinstance(result, EvalResult)
    
    # Check required fields exist
    assert hasattr(result, 'n_trades')
    assert hasattr(result, 'win_rate')
    assert hasattr(result, 'sharpe')
    assert hasattr(result, 'profit_factor')
    assert hasattr(result, 'auc')
    
    # Check n_trades is a number
    assert isinstance(result.n_trades, int)
    assert result.n_trades >= 0
    
    # If there are trades, check metrics are not nonsensical
    if result.n_trades > 0 and result.win_rate is not None:
        assert 0 <= result.win_rate <= 1, "Win rate should be in [0, 1]"


def test_experiment_deterministic():
    """Test that same config yields same result (determinism)."""
    config = ExperimentConfig(
        symbol="BTC/USDT",
        timeframe="5m",
        target=TargetConfig(type=TargetType.MOMENTUM, horizon_bars=10),
        feature_set=FeatureSet.BASE,
        model_type=ModelType.LIGHTGBM,
        train_period=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 10)),
        test_period=DateRange(start=date(2024, 1, 11), end=date(2024, 1, 15)),
        random_state=42,
    )
    
    # Run twice
    result1 = run_experiment(config)
    result2 = run_experiment(config)
    
    # Check same n_trades
    assert result1.n_trades == result2.n_trades, "Number of trades should be deterministic"
    
    # Check same AUC (if available)
    if result1.auc is not None and result2.auc is not None:
        assert abs(result1.auc - result2.auc) < 1e-6, "AUC should be deterministic"


def test_experiment_handles_no_trades():
    """Test that experiment handles scenarios with no trades gracefully."""
    # Use a config that will likely produce neutral predictions
    config = ExperimentConfig(
        symbol="BTC/USDT",
        timeframe="5m",
        target=TargetConfig(
            type=TargetType.VOL_EXPANSION,
            horizon_bars=10,
            vol_window=50,
            vol_factor=5.0,  # Very high threshold, unlikely to trigger
        ),
        feature_set=FeatureSet.BASE,
        model_type=ModelType.LIGHTGBM,
        train_period=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 10)),
        test_period=DateRange(start=date(2024, 1, 11), end=date(2024, 1, 12)),
        random_state=42,
    )
    
    result = run_experiment(config)
    
    # Should not raise an exception
    assert isinstance(result, EvalResult)
    
    # If no trades, metrics should be None or 0
    if result.n_trades == 0:
        # This is acceptable - no trades means no trading metrics
        assert result.win_rate is None or result.win_rate == 0
        assert result.profit_factor is None

