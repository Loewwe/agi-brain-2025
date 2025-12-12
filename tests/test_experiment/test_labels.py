"""Tests for Labels module - target labeling for ML."""

import pytest
from datetime import datetime
import pandas as pd
import numpy as np

from src.experiment.labels import (
    LabelConfig,
    LabelsBuilder,
    create_ml_dataset,
)


class TestLabelConfig:
    """Tests for LabelConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LabelConfig()
        
        assert config.horizon_bars == 3
        assert config.target_type == "direction"
        assert config.up_threshold_pct == 0.3
        assert config.down_threshold_pct == -0.3
    
    def test_invalid_horizon(self):
        """Test that zero/negative horizon raises error."""
        with pytest.raises(ValueError):
            LabelConfig(horizon_bars=0)
        
        with pytest.raises(ValueError):
            LabelConfig(horizon_bars=-1)
    
    def test_invalid_thresholds(self):
        """Test that invalid thresholds raise error."""
        with pytest.raises(ValueError):
            LabelConfig(up_threshold_pct=-0.1)
        
        with pytest.raises(ValueError):
            LabelConfig(down_threshold_pct=0.1)


class TestLabelsBuilder:
    """Tests for LabelsBuilder."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        
        # Create price series with known movement
        prices = 50000.0 + np.cumsum(np.random.randn(100) * 50)
        
        return pd.DataFrame({
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.random.uniform(1000, 5000, 100),
        }, index=dates)
    
    def test_add_labels_creates_columns(self, sample_df):
        """Test that add_labels creates expected columns."""
        config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(config)
        
        result = builder.add_labels(sample_df)
        
        assert "future_return" in result.columns
        assert "target_direction" in result.columns
        assert "target_threshold" in result.columns
        assert "target" in result.columns
    
    def test_future_return_calculation(self, sample_df):
        """Test future return is calculated correctly."""
        config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(config)
        
        result = builder.add_labels(sample_df)
        
        # Check first few rows manually
        for i in range(len(sample_df) - 3):
            expected = ((sample_df.iloc[i + 3]["close"] - sample_df.iloc[i]["close"]) 
                       / sample_df.iloc[i]["close"]) * 100
            actual = result.iloc[i]["future_return"]
            
            assert abs(actual - expected) < 1e-10
    
    def test_direction_target(self, sample_df):
        """Test direction target is binary 0/1."""
        config = LabelConfig(horizon_bars=3, target_type="direction")
        builder = LabelsBuilder(config)
        
        result = builder.add_labels(sample_df)
        
        # Check all direction values are 0 or 1
        valid_values = result["target_direction"].dropna()
        assert all(v in [0, 1] for v in valid_values)
        
        # Check direction matches future_return sign
        for i in range(len(result) - 3):
            if result.iloc[i]["future_return"] > 0:
                assert result.iloc[i]["target_direction"] == 1
            else:
                assert result.iloc[i]["target_direction"] == 0
    
    def test_threshold_target(self, sample_df):
        """Test threshold target is -1/0/1."""
        config = LabelConfig(
            horizon_bars=3,
            target_type="threshold",
            up_threshold_pct=0.1,
            down_threshold_pct=-0.1,
        )
        builder = LabelsBuilder(config)
        
        result = builder.add_labels(sample_df)
        
        # Check all threshold values are -1, 0, or 1
        valid_values = result["target_threshold"].dropna()
        assert all(v in [-1, 0, 1] for v in valid_values)
    
    def test_last_rows_have_nan(self, sample_df):
        """Test last horizon_bars rows have NaN future_return."""
        config = LabelConfig(horizon_bars=5)
        builder = LabelsBuilder(config)
        
        result = builder.add_labels(sample_df)
        
        # Last 5 rows should have NaN future_return
        assert result["future_return"].iloc[-5:].isna().all()


class TestLabelsDeterminism:
    """Test that labeling is deterministic."""
    
    def test_same_input_same_output(self):
        """Test that same input produces same labels."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=50, freq="5min")
        prices = 50000.0 + np.cumsum(np.random.randn(50) * 50)
        
        df = pd.DataFrame({
            "close": prices,
        }, index=dates)
        
        config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(config)
        
        # Label twice
        result1 = builder.add_labels(df)
        result2 = builder.add_labels(df)
        
        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_deterministic_across_runs(self):
        """Test labels are same across multiple runs."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=50, freq="5min")
        prices = 50000.0 + np.cumsum(np.random.randn(50) * 50)
        
        df = pd.DataFrame({
            "close": prices,
        }, index=dates)
        
        config = LabelConfig(horizon_bars=3, target_type="threshold")
        
        # Create multiple builders
        results = []
        for _ in range(3):
            builder = LabelsBuilder(config)
            results.append(builder.add_labels(df))
        
        # All should be identical
        for r in results[1:]:
            pd.testing.assert_frame_equal(results[0], r)


class TestTrainTestSplit:
    """Test time-based train/val/test split."""
    
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        return pd.DataFrame({
            "close": np.random.uniform(50000, 51000, 100),
        }, index=dates)
    
    def test_split_sizes(self, sample_df):
        """Test split produces correct sizes."""
        config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(config)
        
        train_idx, val_idx, test_idx = builder.get_train_test_split_indices(
            sample_df, train_ratio=0.7, val_ratio=0.15
        )
        
        # Check sizes roughly match ratios
        n = len(sample_df) - 3  # Valid rows
        assert len(train_idx) == 70
        assert len(val_idx) > 0
        assert len(test_idx) > 0
    
    def test_no_overlap(self, sample_df):
        """Test no overlap between train/val/test."""
        config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(config)
        
        train_idx, val_idx, test_idx = builder.get_train_test_split_indices(
            sample_df
        )
        
        # No overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0


class TestCreateMLDataset:
    """Test create_ml_dataset helper."""
    
    def test_creates_x_y(self):
        """Test creates feature matrix and target."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        
        df = pd.DataFrame({
            "close": np.random.uniform(50000, 51000, 100),
            "rsi": np.random.uniform(30, 70, 100),
            "atr": np.random.uniform(100, 500, 100),
            "volume": np.random.uniform(1000, 5000, 100),
        }, index=dates)
        
        config = LabelConfig(horizon_bars=3)
        X, y = create_ml_dataset(df, config, feature_columns=["rsi", "atr", "volume"])
        
        assert len(X) == len(y)
        assert len(X) == 97  # 100 - 3 NaN rows removed
        assert "rsi" in X.columns
        assert "close" not in X.columns
        assert "target" not in X.columns
