
import pytest
import numpy as np
from pathlib import Path

from src.research.models_lab import LightGBMAlphaModel, SeqAlphaModel, TrainResult


def test_lightgbm_model_train_and_predict():
    """Test LightGBM model training and prediction on synthetic data."""
    np.random.seed(42)
    
    # Synthetic data: one feature perfectly predicts target
    n_train = 1000
    n_test = 200
    n_features = 5
    
    X_train = np.random.randn(n_train, n_features)
    # Target depends on first feature
    y_train = (X_train[:, 0] > 0).astype(int)
    
    X_test = np.random.randn(n_test, n_features)
    y_test = (X_test[:, 0] > 0).astype(int)
    
    # Train model
    model = LightGBMAlphaModel(random_state=42)
    result = model.fit(X_train, y_train, X_test, y_test)
    
    # Check result
    assert isinstance(result, TrainResult)
    assert result.auc is not None
    assert result.auc > 0.8, f"AUC should be >0.8 on simple synthetic data, got {result.auc}"
    
    # Check predictions
    y_pred = model.predict_proba(X_test)
    
    assert y_pred.shape == (n_test,), f"Expected shape ({n_test},), got {y_pred.shape}"
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Probabilities should be in [0, 1]"


def test_seq_model_train_and_predict_shape():
    """Test sequence model training and prediction shape."""
    np.random.seed(42)
    
    # Synthetic sequence data
    # Target: if last value in window > 0, then 1
    window_size = 16
    n_samples = 500
    n_features = 3
    
    # Generate random sequence
    X = np.random.randn(n_samples, n_features)
    
    # Target: last feature value > 0
    y = (X[:, 0] > 0).astype(int)
    
    # Split
    n_train = 400
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    # Train model
    model = SeqAlphaModel(
        window_size=window_size,
        hidden_size=32,
        num_layers=1,
        random_state=42,
    )
    
    result = model.fit(X_train, y_train, X_test, y_test)
    
    # Check result
    assert isinstance(result, TrainResult)
    assert result.auc is not None
    assert result.auc > 0.6, f"AUC should be >0.6 on synthetic data, got {result.auc}"
    
    # Check predictions
    y_pred = model.predict_proba(X_test)
    
    # Shape should account for window
    expected_len = len(X_test) - window_size + 1
    assert len(y_pred) == expected_len, f"Expected {expected_len} predictions, got {len(y_pred)}"
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Probabilities should be in [0, 1]"


def test_models_deterministic_with_seed():
    """Test that models are deterministic with fixed seed."""
    np.random.seed(42)
    
    # Synthetic data
    n = 500
    X = np.random.randn(n, 3)
    y = (X[:, 0] > 0).astype(int)
    
    X_train = X[:400]
    y_train = y[:400]
    X_test = X[400:]
    
    # Train two models with same seed
    model1 = LightGBMAlphaModel(random_state=42)
    model1.fit(X_train, y_train)
    pred1 = model1.predict_proba(X_test)
    
    model2 = LightGBMAlphaModel(random_state=42)
    model2.fit(X_train, y_train)
    pred2 = model2.predict_proba(X_test)
    
    # Predictions should be identical (or very close)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-5, atol=1e-6,
                                err_msg="Predictions should be deterministic with same seed")


def test_save_and_load_roundtrip(tmp_path):
    """Test model save and load."""
    np.random.seed(42)
    
    # Synthetic data
    n = 500
    X = np.random.randn(n, 3)
    y = (X[:, 0] > 0).astype(int)
    
    X_train = X[:400]
    y_train = y[:400]
    X_test = X[400:]
    
    # Train and save
    model = LightGBMAlphaModel(random_state=42)
    model.fit(X_train, y_train)
    pred_original = model.predict_proba(X_test)
    
    save_path = tmp_path / "model"
    model.save(save_path)
    
    # Load and predict
    model_loaded = LightGBMAlphaModel.load(save_path)
    pred_loaded = model_loaded.predict_proba(X_test)
    
    # Predictions should match
    np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-5, atol=1e-6,
                                err_msg="Loaded model should give same predictions")

