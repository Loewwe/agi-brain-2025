"""
AlphaEngine - ML-based alpha signal generation.

Provides:
- AlphaEngine: Trainable model for generating alpha signals
- AlphaConfig: Configuration for model training
- TrainResult: Training metrics and feature importance

Supports:
- LightGBM (primary)
- Linear (baseline)
- Save/load for reproducibility
"""

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlphaConfig:
    """Configuration for AlphaEngine."""
    
    # Model type
    model_type: Literal["lightgbm", "linear"] = "lightgbm"
    
    # Features
    feature_columns: list[str] | None = None  # None = auto-detect
    target_column: str = "target"
    
    # Data split (time-based)
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test = 1 - train - val
    
    # LightGBM parameters
    lgb_params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 100,
        "random_state": 42,
    })
    
    # Linear parameters
    linear_params: dict[str, Any] = field(default_factory=lambda: {
        "C": 1.0,
        "random_state": 42,
        "max_iter": 1000,
    })
    
    # Training
    random_seed: int = 42
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "lgb_params": self.lgb_params,
            "linear_params": self.linear_params,
            "random_seed": self.random_seed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AlphaConfig":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# TRAIN RESULT
# =============================================================================

@dataclass
class TrainResult:
    """Result of AlphaEngine training."""
    
    # Metrics
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    
    # Feature importance
    feature_importance: dict[str, float] = field(default_factory=dict)
    
    # Data info
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    
    # Training info
    training_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    config_hash: str = ""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "feature_importance": self.feature_importance,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "training_time_seconds": self.training_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "config_hash": self.config_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainResult":
        """Deserialize from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RollingTrainResult:
    """Result of rolling window training."""
    folds: list[dict]  # List of metrics per fold
    avg_metrics: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "folds": self.folds,
            "avg_metrics": self.avg_metrics,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# ALPHA ENGINE
# =============================================================================

class AlphaEngine:
    """
    ML-based alpha signal generation engine.
    
    Workflow:
    1. Create engine with config
    2. Train on labeled dataset
    3. Predict alpha scores on new data
    4. Save/load for reproducibility
    
    Example:
        config = AlphaConfig(model_type="lightgbm")
        engine = AlphaEngine(config)
        result = engine.train(df_labeled)
        predictions = engine.predict(df_new)
    """
    
    def __init__(self, config: AlphaConfig | None = None):
        """
        Initialize AlphaEngine.
        
        Args:
            config: AlphaEngine configuration (default if None)
        """
        self.config = config or AlphaConfig()
        self.model = None
        self.feature_columns: list[str] = []
        self.is_trained = False
        self._train_result: TrainResult | None = None
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        logger.info(
            "alpha_engine.initialized",
            model_type=self.config.model_type,
        )
    
    def train(self, df: pd.DataFrame) -> TrainResult:
        """
        Train the alpha model.
        
        Args:
            df: DataFrame with features and target column
            
        Returns:
            TrainResult with metrics
        """
        from time import time
        start_time = time()
        
        # Validate input
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found")
        
        # Determine feature columns
        if self.config.feature_columns:
            self.feature_columns = self.config.feature_columns
        else:
            self.feature_columns = self._auto_detect_features(df)
        
        # Prepare data
        X = df[self.feature_columns].copy()
        y = df[self.config.target_column].copy()
        
        # Handle NaN in features
        X = X.fillna(X.median())
        
        # Time-based split (no shuffling)
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        logger.info(
            "alpha_engine.data_split",
            train=len(X_train),
            val=len(X_val),
            test=len(X_test),
            features=len(self.feature_columns),
        )
        
        # Train model
        if self.config.model_type == "lightgbm":
            self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.config.model_type == "linear":
            self._train_linear(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.is_trained = True
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(X_train, y_train, "train")
        val_metrics = self._calculate_metrics(X_val, y_val, "val")
        test_metrics = self._calculate_metrics(X_test, y_test, "test")
        
        # Feature importance
        feature_importance = self._get_feature_importance()
        
        # Create config hash for reproducibility
        config_hash = hashlib.md5(
            json.dumps(self.config.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
        
        self._train_result = TrainResult(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
            training_time_seconds=time() - start_time,
            config_hash=config_hash,
        )
        
        logger.info(
            "alpha_engine.trained",
            test_auc=test_metrics.get("roc_auc", 0),
            test_accuracy=test_metrics.get("accuracy", 0),
            time_seconds=self._train_result.training_time_seconds,
        )
        
        return self._train_result

    def train_rolling(
        self,
        df: pd.DataFrame,
        window_months: int = 3,
        step_months: int = 1,
    ) -> RollingTrainResult:
        """
        Perform rolling window training (Walk-Forward Validation).
        
        Args:
            df: DataFrame with features and target
            window_months: Training window size in months
            step_months: Test window size (and step size) in months
            
        Returns:
            RollingTrainResult with aggregated metrics
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        start_date = df.index.min()
        end_date = df.index.max()
        
        current_start = start_date
        folds = []
        
        logger.info(
            "alpha_engine.rolling_train_start",
            start=start_date,
            end=end_date,
            window=window_months,
            step=step_months,
        )
        
        print(f"DEBUG: Rolling Train Start. Start={start_date}, End={end_date}, Window={window_months}, Step={step_months}")
        
        while True:
            # Define windows
            train_end = current_start + pd.DateOffset(months=window_months)
            test_end = train_end + pd.DateOffset(months=step_months)
            
            print(f"DEBUG: Loop Iteration. CurrentStart={current_start}, TrainEnd={train_end}, TestEnd={test_end}")
            
            if test_end > end_date:
                print(f"DEBUG: Breaking because TestEnd ({test_end}) > EndDate ({end_date})")
                break
                
            # Slice data
            train_mask = (df.index >= current_start) & (df.index < train_end)
            test_mask = (df.index >= train_end) & (df.index < test_end)
            
            df_train = df[train_mask]
            df_test = df[test_mask]
            
            print(f"DEBUG: TrainLen={len(df_train)}, TestLen={len(df_test)}")
            
            if len(df_train) < 100 or len(df_test) < 100:
                logger.warning("alpha_engine.rolling_skip_small_fold", train_len=len(df_train), test_len=len(df_test))
                current_start += pd.DateOffset(months=step_months)
                continue
                
            # Train on this fold
            # We create a temporary engine to avoid overwriting self.model state if we want to keep the final one?
            # Actually, usually we want to evaluate the strategy performance.
            # But here we just want to evaluate Model Metrics stability.
            
            # Reset model
            self.is_trained = False
            self.model = None
            
            # Use standard train method but with custom split (train=100%, val=0%, test=0% effectively, or we split train internally)
            # To use `train` method, we need to respect its split ratios.
            # But here we have explicit train/test sets.
            # So we should use internal methods or adjust config.
            
            # Let's adjust config temporarily to use last 20% of train window as validation
            # and then evaluate on test window manually.
            
            # 1. Train on df_train (split into train/val)
            # We can use `train` method on df_train, but it will split it.
            # That's fine, we need validation for early stopping anyway.
            
            print("DEBUG: Calling self.train(df_train)")
            fold_result = self.train(df_train)
            print("DEBUG: self.train returned")
            
            # 2. Evaluate on df_test
            X_test = df_test[self.feature_columns].fillna(0) # Handle NaNs
            y_test = df_test[self.config.target_column]
            
            test_metrics = self._calculate_metrics(X_test, y_test, "rolling_test")
            
            fold_info = {
                "train_start": current_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": train_end.isoformat(),
                "test_end": test_end.isoformat(),
                "metrics": test_metrics,
            }
            folds.append(fold_info)
            print(f"DEBUG: Appended fold. Total folds: {len(folds)}")
            
            logger.info(
                "alpha_engine.rolling_fold",
                test_start=train_end.date(),
                auc=test_metrics.get("roc_auc", 0),
                acc=test_metrics.get("accuracy", 0),
            )
            
            # Move forward
            current_start += pd.DateOffset(months=step_months)
            
        # Aggregate metrics
        avg_metrics = {}
        if folds:
            metric_keys = folds[0]["metrics"].keys()
            for key in metric_keys:
                values = [f["metrics"][key] for f in folds]
                avg_metrics[key] = float(np.mean(values))
                
        return RollingTrainResult(folds=folds, avg_metrics=avg_metrics)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate alpha predictions.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            DataFrame with added columns:
            - alpha_score: Probability/confidence (0-1)
            - alpha_direction: 1 (long) or 0 (short)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())
        
        # Get predictions
        if self.config.model_type == "lightgbm":
            proba = self.model.predict_proba(X)[:, 1]
        elif self.config.model_type == "linear":
            proba = self.model.predict_proba(X)[:, 1]
        else:
            proba = np.zeros(len(X))
        
        # Create output DataFrame
        result = df.copy()
        result["alpha_score"] = proba
        result["alpha_direction"] = (proba > 0.5).astype(int)
        result["alpha_confidence"] = np.abs(proba - 0.5) * 2  # 0-1 scale
        
        return result
    
    def predict_single(self, features: dict[str, float]) -> dict:
        """
        Predict for a single row.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Dictionary with alpha_score, alpha_direction, alpha_confidence
        """
        df = pd.DataFrame([features])
        result = self.predict(df)
        
        return {
            "alpha_score": float(result["alpha_score"].iloc[0]),
            "alpha_direction": int(result["alpha_direction"].iloc[0]),
            "alpha_confidence": float(result["alpha_confidence"].iloc[0]),
        }
    
    def save(self, path: Path | str) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Directory path for saving
            
        Returns:
            Path to saved model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            "config": self.config.to_dict(),
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
            "train_result": self._train_result.to_dict() if self._train_result else None,
        }
        
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("alpha_engine.saved", path=str(path))
        return path
    
    @classmethod
    def load(cls, path: Path | str) -> "AlphaEngine":
        """
        Load model from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded AlphaEngine
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create engine
        config = AlphaConfig.from_dict(metadata["config"])
        engine = cls(config)
        
        # Load model
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            engine.model = pickle.load(f)
        
        engine.feature_columns = metadata["feature_columns"]
        engine.is_trained = metadata["is_trained"]
        
        if metadata.get("train_result"):
            engine._train_result = TrainResult.from_dict(metadata["train_result"])
        
        logger.info("alpha_engine.loaded", path=str(path))
        return engine
    
    def get_train_result(self) -> TrainResult | None:
        """Get training result."""
        return self._train_result
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _auto_detect_features(self, df: pd.DataFrame) -> list[str]:
        """Auto-detect feature columns."""
        exclude = [
            self.config.target_column,
            "target", "target_direction", "target_threshold", "future_return",
            "symbol", "timestamp", "open", "high", "low", "close", "volume",
        ]
        
        features = []
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, bool]:
                # Check if column has low cardinality (might be categorical)
                if df[col].nunique() > 2 or col.startswith(("rsi", "atr", "ema", "volume")):
                    features.append(col)
        
        return features
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        
        params = self.config.lgb_params.copy()
        
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
    
    def _train_linear(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Train linear model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        params = self.config.linear_params.copy()
        
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])
        self.model.fit(X_train, y_train)
    
    def _calculate_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_name: str,
    ) -> dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        
        if len(X) == 0:
            return {}
        
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }
        
        # ROC-AUC requires both classes present
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.5
        
        logger.debug(
            f"alpha_engine.metrics.{split_name}",
            **metrics,
        )
        
        return metrics
    
    def _get_feature_importance(self) -> dict[str, float]:
        """Get feature importance."""
        if self.config.model_type == "lightgbm":
            importance = self.model.feature_importances_
        elif self.config.model_type == "linear":
            # Get coefficients from pipeline
            clf = self.model.named_steps["clf"]
            importance = np.abs(clf.coef_[0])
        else:
            importance = np.zeros(len(self.feature_columns))
        
        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        return {
            col: float(imp)
            for col, imp in zip(self.feature_columns, importance)
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def evaluate_vs_baseline(
    engine: AlphaEngine,
    df: pd.DataFrame,
) -> dict:
    """
    Evaluate alpha engine vs trivial baselines.
    
    Args:
        engine: Trained AlphaEngine
        df: Test dataset with target
        
    Returns:
        Dictionary with comparison metrics
    """
    target = df[engine.config.target_column]
    predictions = engine.predict(df)
    
    # Alpha engine predictions
    alpha_pred = predictions["alpha_direction"]
    
    # Baseline: always predict 1 (up)
    always_up = pd.Series([1] * len(target), index=target.index)
    
    # Baseline: random
    np.random.seed(42)
    random_pred = pd.Series(np.random.randint(0, 2, len(target)), index=target.index)
    
    from sklearn.metrics import accuracy_score
    
    return {
        "alpha_accuracy": float(accuracy_score(target, alpha_pred)),
        "always_up_accuracy": float(accuracy_score(target, always_up)),
        "random_accuracy": float(accuracy_score(target, random_pred)),
        "alpha_vs_baseline": float(accuracy_score(target, alpha_pred)) - 0.5,
    }
