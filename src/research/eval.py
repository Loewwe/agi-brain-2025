"""
Evaluation Pipeline for Research

End-to-end experiment execution:
1. Load data
2. Build targets
3. Add features
4. Train model
5. Evaluate performance
"""

from enum import Enum
from typing import Optional
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel

from .data_loader import load_joined_dataset
from .targets import build_target, TargetConfig, TargetType
from .features_ext import add_extended_features, FeaturesConfig
from .models_lab import LightGBMAlphaModel, SeqAlphaModel, TrainResult
from ..experiment.models import DatasetConfig


class ModelType(str, Enum):
    LIGHTGBM = "lightgbm"
    SEQ = "seq"


class FeatureSet(str, Enum):
    BASE = "base"
    EXTENDED = "extended"


class DateRange(BaseModel):
    """Date range for train/test periods."""
    start: date
    end: date


class ExperimentConfig(BaseModel):
    """Configuration for alpha research experiment."""
    symbol: str
    timeframe: str = "5m"
    
    # Target configuration
    target: TargetConfig
    
    # Feature configuration
    feature_set: FeatureSet = FeatureSet.EXTENDED
    
    # Model configuration
    model_type: ModelType = ModelType.LIGHTGBM
    window_size: Optional[int] = None  # For sequence models
    
    # Data periods
    train_period: DateRange
    test_period: DateRange
    
    # Random seed
    random_state: int = 42
    
    # Transaction costs (Stage 8b)
    commission_bps: int = 10  # 0.10% commission per trade
    slippage_bps: int = 5     # 0.05% slippage per direction (0.1% total)


class EvalResult(BaseModel):
    """Evaluation results from experiment."""
    # Trading metrics
    win_rate: Optional[float] = None
    sharpe: Optional[float] = None
    profit_factor: Optional[float] = None
    n_trades: int = 0
    
    # ML metrics
    auc: Optional[float] = None
    
    # Metadata
    notes: Optional[str] = None
    
    # Additional stats
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Feature analysis
    feature_importances: Optional[dict[str, float]] = None
    
    # Post-cost metrics (Stage 8b)
    win_rate_post_cost: Optional[float] = None
    sharpe_post_cost: Optional[float] = None
    profit_factor_post_cost: Optional[float] = None
    total_return_post_cost: Optional[float] = None
    max_drawdown_post_cost: Optional[float] = None


def run_experiment(config: ExperimentConfig) -> EvalResult:
    """
    Run end-to-end alpha research experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Evaluation results
    """
    # 1. Load data
    dataset_config = DatasetConfig(
        symbols=[config.symbol],
        date_from=config.train_period.start,
        date_to=config.test_period.end,
        timeframe=config.timeframe,
    )
    
    df = load_joined_dataset(dataset_config)
    
    if df.empty:
        return EvalResult(notes="No data available")
    
    # 2. Add features
    if config.feature_set == FeatureSet.EXTENDED:
        df = add_extended_features(df)
    
    # 3. Build target
    target_series = build_target(df, config.target)
    df['target'] = target_series
    
    # Drop rows with NaN target
    df = df.dropna(subset=['target'])
    
    if len(df) < 100:
        return EvalResult(notes="Insufficient data after target filtering")
    
    # 4. Split train/test by date
    df['timestamp'] = pd.to_datetime(df.index)
    
    train_mask = (
        (df['timestamp'].dt.date >= config.train_period.start) &
        (df['timestamp'].dt.date <= config.train_period.end)
    )
    test_mask = (
        (df['timestamp'].dt.date >= config.test_period.start) &
        (df['timestamp'].dt.date <= config.test_period.end)
    )
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    if len(df_train) < 50 or len(df_test) < 20:
        return EvalResult(notes="Insufficient train or test samples")
    
    # 5. Prepare features (exclude target, timestamp, price columns, symbol)
    exclude_cols = ['target', 'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target'].to_numpy().astype(int)  # Convert to numpy
    X_test = df_test[feature_cols]
    y_test = df_test['target'].to_numpy().astype(int)  # Convert to numpy
    
    # 6. Train model
    if config.model_type == ModelType.LIGHTGBM:
        model = LightGBMAlphaModel(random_state=config.random_state)
    else:  # SEQ
        if config.window_size is None:
            config.window_size = 32
        model = SeqAlphaModel(
            window_size=config.window_size,
            random_state=config.random_state,
        )
    
    train_result = model.fit(X_train, y_train, X_test, y_test)
    
    # 7. Predict on test
    y_pred_proba = model.predict_proba(X_test)
    
    # For sequence models, adjust test data length
    if config.model_type == ModelType.SEQ:
        y_test = y_test[config.window_size-1:]
        df_test = df_test.iloc[config.window_size-1:].copy()
    
    # 8. Calculate trading metrics (simple backtest)
    # Entry: if prob > 0.6, go long; if prob < 0.4, go short
    signals = np.where(y_pred_proba > 0.6, 1, np.where(y_pred_proba < 0.4, -1, 0))
    
    # Simulate trades
    df_test = df_test.iloc[:len(signals)].copy()
    df_test['signal'] = signals
    df_test['returns'] = df_test['close'].pct_change()
    df_test['strategy_returns'] = df_test['signal'].shift(1) * df_test['returns']
    
    trades = df_test[df_test['signal'] != 0]
    n_trades = len(trades)
    
    if n_trades == 0:
        return EvalResult(
            auc=train_result.auc,
            n_trades=0,
            notes="No trades generated (all predictions neutral)",
        )
    
    # Calculate metrics
    trade_returns = trades['strategy_returns'].dropna()
    
    if len(trade_returns) == 0:
        win_rate = None
        sharpe = None
        profit_factor = None
        total_return = None
    else:
        wins = (trade_returns > 0).sum()
        win_rate = wins / len(trade_returns) if len(trade_returns) > 0 else None
        
        # Sharpe (annualized, assuming 5m bars)
        mean_ret = trade_returns.mean()
        std_ret = trade_returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24 * 12) if std_ret > 0 else None
        
        # Profit factor
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        
        # Total return
        total_return = (1 + trade_returns).prod() - 1
    
    # Max drawdown
    cumulative = (1 + df_test['strategy_returns'].fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else None
    
    # ========== POST-COST CALCULATIONS (Stage 8b) ==========
    # Apply transaction costs to each trade
    win_rate_post_cost = None
    sharpe_post_cost = None
    profit_factor_post_cost = None
    total_return_post_cost = None
    max_drawdown_post_cost = None
    
    if len(trade_returns) > 0:
        # Calculate total cost per round-trip trade
        # Commission: entry + exit = 2× commission_bps
        # Slippage: entry + exit = 2× slippage_bps
        total_cost_bps = 2 * (config.commission_bps + config.slippage_bps)
        cost_per_trade = total_cost_bps / 10000.0  # Convert bps to decimal
        
        # Apply cost to each trade return
        trade_returns_post_cost = trade_returns - cost_per_trade
        
        # Recalculate metrics with costs
        wins_post_cost = (trade_returns_post_cost > 0).sum()
        win_rate_post_cost = wins_post_cost / len(trade_returns_post_cost)
        
        # Sharpe post-cost
        mean_ret_post_cost = trade_returns_post_cost.mean()
        std_ret_post_cost = trade_returns_post_cost.std()
        sharpe_post_cost = (mean_ret_post_cost / std_ret_post_cost) * np.sqrt(252 * 24 * 12) if std_ret_post_cost > 0 else None
        
        # Profit factor post-cost
        gross_profit_post_cost = trade_returns_post_cost[trade_returns_post_cost > 0].sum()
        gross_loss_post_cost = abs(trade_returns_post_cost[trade_returns_post_cost < 0].sum())
        profit_factor_post_cost = gross_profit_post_cost / gross_loss_post_cost if gross_loss_post_cost > 0 else None
        
        # Total return post-cost
        total_return_post_cost = (1 + trade_returns_post_cost).prod() - 1
        
        # Max drawdown post-cost (recalculate full equity curve)
        df_test_cost = df_test.copy()
        df_test_cost['strategy_returns_post_cost'] = df_test_cost['strategy_returns'].fillna(0)
        
        # Apply cost only to actual trades (where signal != 0)
        trade_mask = df_test_cost['signal'] != 0
        df_test_cost.loc[trade_mask, 'strategy_returns_post_cost'] -= cost_per_trade
        
        cumulative_post_cost = (1 + df_test_cost['strategy_returns_post_cost']).cumprod()
        running_max_post_cost = cumulative_post_cost.expanding().max()
        drawdown_post_cost = (cumulative_post_cost - running_max_post_cost) / running_max_post_cost
        max_drawdown_post_cost = drawdown_post_cost.min()
    
    return EvalResult(
        win_rate=win_rate,
        sharpe=sharpe,
        profit_factor=profit_factor,
        n_trades=n_trades,
        auc=train_result.auc,
        total_return=total_return,
        max_drawdown=max_drawdown,
        feature_importances=train_result.feature_importances,
        # Post-cost metrics
        win_rate_post_cost=win_rate_post_cost,
        sharpe_post_cost=sharpe_post_cost,
        profit_factor_post_cost=profit_factor_post_cost,
        total_return_post_cost=total_return_post_cost,
        max_drawdown_post_cost=max_drawdown_post_cost,
    )

