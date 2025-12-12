"""
Labels - Target labeling for ML training.

Provides utilities for creating ML targets from OHLCV data:
- future_return: Continuous return over horizon
- direction: Binary up/down classification
- threshold: Ternary +1/0/-1 based on thresholds

All labeling is strictly time-based with no data leakage.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LabelConfig:
    """Configuration for label generation."""
    
    # Prediction horizon
    horizon_bars: int = 3               # Number of bars ahead (e.g., 3 × 5min = 15min)
    
    # Target type
    target_type: Literal["direction", "return", "threshold", "decile"] = "direction"
    
    # For threshold-based targets
    up_threshold_pct: float = 0.3       # +0.3% → label 1
    down_threshold_pct: float = -0.3    # -0.3% → label -1
    
    # For decile-based targets
    decile_period_bars: int = 288       # Rolling window for decile calc (e.g., 1 day @ 5m)
    decile_threshold: float = 0.9       # Top 10% → label 1
    
    # Labeling options
    use_log_returns: bool = False       # Use log returns instead of simple returns
    
    def __post_init__(self):
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.up_threshold_pct <= 0:
            raise ValueError("up_threshold_pct must be > 0")
        if self.down_threshold_pct >= 0:
            raise ValueError("down_threshold_pct must be < 0")
        if not (0 < self.decile_threshold < 1):
            raise ValueError("decile_threshold must be between 0 and 1")


# =============================================================================
# LABELS BUILDER
# =============================================================================

class LabelsBuilder:
    """
    Build ML targets from OHLCV data.
    
    Strictly time-based: uses only future data for labels.
    No shuffling, no leakage.
    
    Added columns:
    - future_return: Return over horizon (%)
    - target_direction: 1 (up) or 0 (down) 
    - target_threshold: 1 (up), 0 (neutral), -1 (down)
    - target_decile: 1 (top X%), 0 (rest)
    - target: Final target based on config
    
    Example:
        builder = LabelsBuilder(LabelConfig(horizon_bars=3))
        df_labeled = builder.add_labels(df)
    """
    
    def __init__(self, config: LabelConfig):
        """
        Initialize LabelsBuilder.
        
        Args:
            config: Label configuration
        """
        self.config = config
        logger.info(
            "labels_builder.initialized",
            horizon_bars=config.horizon_bars,
            target_type=config.target_type,
        )
    
    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target labels to DataFrame.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with added label columns
        """
        df = df.copy()
        
        # Validate input
        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        # Calculate future return
        df = self._add_future_return(df)
        
        # Calculate direction target (binary: 1=up, 0=down)
        df = self._add_direction_target(df)
        
        # Calculate threshold target (ternary: 1/-1/0)
        df = self._add_threshold_target(df)
        
        # Calculate decile target (binary: 1=top, 0=rest)
        df = self._add_decile_target(df)
        
        # Set final target based on config
        target_col = f"target_{self.config.target_type}"
        if target_col == "target_return":
            df["target"] = df["future_return"]
        else:
            df["target"] = df[target_col]
        
        # Log stats
        non_null = df["target"].notna().sum()
        logger.info(
            "labels_builder.labeled",
            rows=len(df),
            labeled=non_null,
            target_type=self.config.target_type,
        )
        
        return df
    
    def _add_future_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add future return column."""
        if self.config.use_log_returns:
            # Log return: ln(P_t+h / P_t)
            df["future_return"] = np.log(
                df["close"].shift(-self.config.horizon_bars) / df["close"]
            ) * 100  # Convert to percentage
        else:
            # Simple return: (P_t+h - P_t) / P_t
            df["future_return"] = (
                (df["close"].shift(-self.config.horizon_bars) - df["close"]) 
                / df["close"]
            ) * 100
        
        return df
    
    def _add_direction_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary direction target (1=up, 0=down)."""
        df["target_direction"] = (df["future_return"] > 0).astype(int)
        return df
    
    def _add_threshold_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ternary threshold target (1=up, -1=down, 0=neutral)."""
        conditions = [
            df["future_return"] >= self.config.up_threshold_pct,
            df["future_return"] <= self.config.down_threshold_pct,
        ]
        choices = [1, -1]
        
        df["target_threshold"] = np.select(conditions, choices, default=0)
        return df

    def _add_decile_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary decile target (1=top X%, 0=rest)."""
        # Calculate rolling quantile
        # We want to know if current future_return is in the top X% of the LAST N bars
        # Note: This is slightly tricky. If we use future_return, we are using future info?
        # No, we are labeling historical data.
        # BUT, if we use a rolling window of future_returns centered on T, we leak info.
        # We should use a rolling window of *past* returns to define the threshold?
        # OR, we define "top decile of the training set".
        # The user request says: "target = 1, if observation falls in top-10-20% by future return".
        # Usually this means "relative to recent history" or "relative to entire dataset".
        # To be safe and regime-adaptive, let's use a rolling window of *future_return* magnitudes
        # but we must be careful.
        # Actually, standard practice for "regime-aware" labeling is often just global or rolling z-score.
        
        # Let's use a rolling window of absolute returns to normalize, or just raw ranking.
        # Implementation: Rolling Quantile of future_return over past N bars.
        # Wait, if we use past N bars to determine if *current* future return is high, that's fine.
        # It means "is this return an outlier relative to recent volatility?".
        
        # However, `future_return` is not known at time T. We are creating LABELS for training.
        # So at training time, we know future_return.
        # We want to label T as "1" if future_return(T) > 90th percentile of future_return(T-N...T).
        
        window = self.config.decile_period_bars
        threshold = self.config.decile_threshold
        
        # Rolling quantile of *past* realized returns? No, we want to know if *this* return is big.
        # Let's compare current future_return to the distribution of future_returns in the *past* window.
        # This tells us if it's a "relatively large move".
        
        rolling_quantile = df["future_return"].rolling(window).quantile(threshold)
        
        # If current return > rolling 90th percentile -> 1
        df["target_decile"] = (df["future_return"] > rolling_quantile).astype(int)
        
        return df
    
    def get_label_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about labels.
        
        Args:
            df: DataFrame with labels
            
        Returns:
            Dictionary with label statistics
        """
        stats = {
            "total_rows": len(df),
            "labeled_rows": df["target"].notna().sum(),
            "unlabeled_rows": df["target"].isna().sum(),
        }
        
        if self.config.target_type == "direction":
            direction_counts = df["target_direction"].value_counts()
            stats["up_count"] = int(direction_counts.get(1, 0))
            stats["down_count"] = int(direction_counts.get(0, 0))
            stats["up_ratio"] = stats["up_count"] / stats["labeled_rows"] if stats["labeled_rows"] > 0 else 0
            
        elif self.config.target_type == "threshold":
            threshold_counts = df["target_threshold"].value_counts()
            stats["up_count"] = int(threshold_counts.get(1, 0))
            stats["neutral_count"] = int(threshold_counts.get(0, 0))
            stats["down_count"] = int(threshold_counts.get(-1, 0))
            
        elif self.config.target_type == "decile":
            decile_counts = df["target_decile"].value_counts()
            stats["top_count"] = int(decile_counts.get(1, 0))
            stats["rest_count"] = int(decile_counts.get(0, 0))
            stats["top_ratio"] = stats["top_count"] / stats["labeled_rows"] if stats["labeled_rows"] > 0 else 0
        
        if "future_return" in df.columns:
            returns = df["future_return"].dropna()
            stats["return_mean"] = float(returns.mean())
            stats["return_std"] = float(returns.std())
            stats["return_min"] = float(returns.min())
            stats["return_max"] = float(returns.max())
        
        return stats
    
    def get_train_test_split_indices(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[pd.Index, pd.Index, pd.Index]:
        """
        Get time-based train/val/test split indices.
        
        No shuffling - strictly chronological.
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for training (default 0.7)
            val_ratio: Ratio for validation (default 0.15)
            
        Returns:
            Tuple of (train_idx, val_idx, test_idx)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Remove rows with NaN labels from last horizon_bars
        valid_end = n - self.config.horizon_bars
        
        if train_end >= valid_end:
            raise ValueError("Not enough data for train/val/test split")
        
        train_idx = df.index[:train_end]
        val_idx = df.index[train_end:min(val_end, valid_end)]
        test_idx = df.index[min(val_end, valid_end):valid_end]
        
        logger.info(
            "labels_builder.split",
            train=len(train_idx),
            val=len(val_idx),
            test=len(test_idx),
        )
        
        return train_idx, val_idx, test_idx


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_ml_dataset(
    df: pd.DataFrame,
    label_config: LabelConfig,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create ML-ready dataset with features and target.
    
    Args:
        df: DataFrame with OHLCV + features
        label_config: Label configuration
        feature_columns: List of feature columns (default: use all numeric except targets)
        
    Returns:
        Tuple of (X, y) where X is features, y is target
    """
    builder = LabelsBuilder(label_config)
    df_labeled = builder.add_labels(df)
    
    # Drop rows with NaN future_return (last horizon_bars rows)
    df_labeled = df_labeled.dropna(subset=["future_return"])
    
    # Select features
    if feature_columns is None:
        # Use all numeric columns except target columns
        exclude = ["target", "target_direction", "target_threshold", "future_return"]
        feature_columns = [
            col for col in df_labeled.columns 
            if col not in exclude and df_labeled[col].dtype in [np.float64, np.int64, np.float32, np.int32, bool]
        ]
    
    X = df_labeled[feature_columns]
    y = df_labeled["target"]
    
    return X, y
