"""Tests for AlphaEngine module."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment.alpha_engine import (
    AlphaConfig,
    AlphaEngine,
    TrainResult,
    evaluate_vs_baseline,
)
from src.experiment.labels import LabelConfig, LabelsBuilder


class TestAlphaConfig:
    """Tests for AlphaConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AlphaConfig()
        
        assert config.model_type == "lightgbm"
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.random_seed == 42
    
    def test_serialization(self):
        """Test config serialization."""
        config = AlphaConfig(
            model_type="linear",
            train_ratio=0.8,
        )
        
        data = config.to_dict()
        restored = AlphaConfig.from_dict(data)
        
        assert restored.model_type == "linear"
        assert restored.train_ratio == 0.8


class TestAlphaEngine:
    """Tests for AlphaEngine."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample labeled dataset."""
        np.random.seed(42)
        n = 500
        
        # Create features with some signal
        df = pd.DataFrame({
            "close": np.random.uniform(50000, 51000, n),
            "rsi": np.random.uniform(20, 80, n),
            "atr": np.random.uniform(100, 500, n),
            "ema200": np.random.uniform(49000, 50000, n),
            "volume_surge": np.random.uniform(0.5, 2.0, n),
        }, index=pd.date_range(start="2024-01-01", periods=n, freq="5min"))
        
        # Add labels
        label_config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(label_config)
        df = builder.add_labels(df)
        
        # Drop NaN rows
        df = df.dropna(subset=["future_return"])
        
        return df
    
    def test_train_lightgbm(self, sample_dataset):
        """Test training with LightGBM."""
        config = AlphaConfig(model_type="lightgbm")
        engine = AlphaEngine(config)
        
        result = engine.train(sample_dataset)
        
        assert engine.is_trained
        assert result.train_size > 0
        assert result.test_size > 0
        assert "roc_auc" in result.test_metrics
    
    def test_train_linear(self, sample_dataset):
        """Test training with linear model."""
        config = AlphaConfig(model_type="linear")
        engine = AlphaEngine(config)
        
        result = engine.train(sample_dataset)
        
        assert engine.is_trained
        assert "accuracy" in result.test_metrics
    
    def test_predict_returns_scores(self, sample_dataset):
        """Test prediction returns expected columns."""
        config = AlphaConfig(model_type="lightgbm")
        engine = AlphaEngine(config)
        engine.train(sample_dataset)
        
        predictions = engine.predict(sample_dataset)
        
        assert "alpha_score" in predictions.columns
        assert "alpha_direction" in predictions.columns
        assert "alpha_confidence" in predictions.columns
        
        # Scores should be in [0, 1]
        assert predictions["alpha_score"].min() >= 0
        assert predictions["alpha_score"].max() <= 1
    
    def test_predict_single(self, sample_dataset):
        """Test single row prediction."""
        config = AlphaConfig(model_type="lightgbm")
        engine = AlphaEngine(config)
        engine.train(sample_dataset)
        
        features = {
            "rsi": 45.0,
            "atr": 300.0,
            "ema200": 49500.0,
            "volume_surge": 1.5,
        }
        
        result = engine.predict_single(features)
        
        assert "alpha_score" in result
        assert "alpha_direction" in result
        assert 0 <= result["alpha_score"] <= 1


class TestAlphaEngineDeterminism:
    """Test AlphaEngine determinism."""
    
    @pytest.fixture
    def sample_dataset(self):
        np.random.seed(42)
        n = 300
        
        df = pd.DataFrame({
            "close": np.random.uniform(50000, 51000, n),
            "rsi": np.random.uniform(20, 80, n),
            "atr": np.random.uniform(100, 500, n),
        }, index=pd.date_range(start="2024-01-01", periods=n, freq="5min"))
        
        label_config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(label_config)
        df = builder.add_labels(df)
        df = df.dropna(subset=["future_return"])
        
        return df
    
    def test_same_seed_same_result(self, sample_dataset):
        """Test that same seed produces same results."""
        config = AlphaConfig(model_type="lightgbm", random_seed=42)
        
        # Train twice
        engine1 = AlphaEngine(config)
        result1 = engine1.train(sample_dataset)
        
        engine2 = AlphaEngine(config)
        result2 = engine2.train(sample_dataset)
        
        # Metrics should be identical
        assert result1.test_metrics["roc_auc"] == result2.test_metrics["roc_auc"]
        assert result1.test_metrics["accuracy"] == result2.test_metrics["accuracy"]
    
    def test_predictions_reproducible(self, sample_dataset):
        """Test that predictions are reproducible."""
        config = AlphaConfig(model_type="lightgbm", random_seed=42)
        
        engine1 = AlphaEngine(config)
        engine1.train(sample_dataset)
        pred1 = engine1.predict(sample_dataset)
        
        engine2 = AlphaEngine(config)
        engine2.train(sample_dataset)
        pred2 = engine2.predict(sample_dataset)
        
        # Predictions should be identical
        pd.testing.assert_series_equal(
            pred1["alpha_score"], 
            pred2["alpha_score"]
        )


class TestAlphaEngineSaveLoad:
    """Test AlphaEngine save/load."""
    
    @pytest.fixture
    def sample_dataset(self):
        np.random.seed(42)
        n = 200
        
        df = pd.DataFrame({
            "close": np.random.uniform(50000, 51000, n),
            "rsi": np.random.uniform(20, 80, n),
            "atr": np.random.uniform(100, 500, n),
        }, index=pd.date_range(start="2024-01-01", periods=n, freq="5min"))
        
        label_config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(label_config)
        df = builder.add_labels(df)
        df = df.dropna(subset=["future_return"])
        
        return df
    
    def test_save_and_load(self, sample_dataset):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            # Train and save
            config = AlphaConfig(model_type="lightgbm")
            engine = AlphaEngine(config)
            result = engine.train(sample_dataset)
            engine.save(save_path)
            
            # Load
            loaded = AlphaEngine.load(save_path)
            
            assert loaded.is_trained
            assert loaded.config.model_type == "lightgbm"
            assert loaded.feature_columns == engine.feature_columns
    
    def test_loaded_model_predicts_same(self, sample_dataset):
        """Test loaded model produces same predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            config = AlphaConfig(model_type="lightgbm", random_seed=42)
            engine = AlphaEngine(config)
            engine.train(sample_dataset)
            
            # Get predictions before save
            pred1 = engine.predict(sample_dataset)
            
            # Save and load
            engine.save(save_path)
            loaded = AlphaEngine.load(save_path)
            
            # Get predictions after load
            pred2 = loaded.predict(sample_dataset)
            
            # Should be identical
            pd.testing.assert_series_equal(
                pred1["alpha_score"],
                pred2["alpha_score"]
            )


class TestEvaluateVsBaseline:
    """Test baseline comparison."""
    
    @pytest.fixture
    def sample_dataset(self):
        np.random.seed(42)
        n = 200
        
        df = pd.DataFrame({
            "close": np.random.uniform(50000, 51000, n),
            "rsi": np.random.uniform(20, 80, n),
            "atr": np.random.uniform(100, 500, n),
        }, index=pd.date_range(start="2024-01-01", periods=n, freq="5min"))
        
        label_config = LabelConfig(horizon_bars=3)
        builder = LabelsBuilder(label_config)
        df = builder.add_labels(df)
        df = df.dropna(subset=["future_return"])
        
        return df
    
    def test_baseline_comparison(self, sample_dataset):
        """Test comparing with baselines."""
        config = AlphaConfig(model_type="lightgbm")
        engine = AlphaEngine(config)
        engine.train(sample_dataset)
        
        comparison = evaluate_vs_baseline(engine, sample_dataset)
        
        assert "alpha_accuracy" in comparison
        assert "always_up_accuracy" in comparison
        assert "random_accuracy" in comparison
        assert "alpha_vs_baseline" in comparison
