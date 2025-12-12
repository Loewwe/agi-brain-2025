#!/usr/bin/env python3
"""
Audit Script: Run real experiment and analyze actual trade patterns
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def count_real_trades(signals: np.ndarray) -> dict:
    """Count actual position opens/closes/reversals."""
    opens = 0
    closes = 0
    reversals = 0
    bars_in_position = 0
    
    prev_signal = 0
    for s in signals:
        if prev_signal == 0 and s != 0:
            opens += 1
        elif prev_signal != 0 and s == 0:
            closes += 1
        elif prev_signal != 0 and s != 0 and prev_signal != s:
            reversals += 1
        
        if s != 0:
            bars_in_position += 1
            
        prev_signal = s
    
    # Close any open position at end
    if prev_signal != 0:
        closes += 1
    
    total_round_trips = opens
    avg_hold = bars_in_position / max(total_round_trips, 1)
    
    return {
        'signal_bars': int(bars_in_position),
        'opens': opens,
        'closes': closes,
        'reversals': reversals,
        'round_trips': total_round_trips,
        'avg_hold_bars': avg_hold,
    }


def run_audit():
    """Run a real experiment and analyze trading patterns."""
    
    from src.research.eval import (
        run_experiment,
        ExperimentConfig,
        DateRange,
        FeatureSet,
        ModelType,
    )
    from src.research.targets import TargetConfig, TargetType
    from src.research.data_loader import load_joined_dataset
    from src.experiment.models import DatasetConfig
    from src.research.features_ext import add_extended_features
    from src.research.targets import build_target
    from src.research.models_lab import LightGBMAlphaModel
    
    print("=" * 60)
    print("REAL EXPERIMENT AUDIT")
    print("=" * 60)
    
    # Replicate exp_003
    config = ExperimentConfig(
        symbol="BTC/USDT",
        timeframe="5m",
        target=TargetConfig(
            type=TargetType.VOL_EXPANSION,
            horizon_bars=12,
            vol_window=24,
            vol_factor=1.5,
        ),
        feature_set=FeatureSet.EXTENDED,
        model_type=ModelType.LIGHTGBM,
        train_period=DateRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        test_period=DateRange(start=date(2024, 4, 1), end=date(2024, 4, 30)),
        random_state=42,
        commission_bps=10,
        slippage_bps=5,
    )
    
    print(f"\nLoading data for {config.symbol}...")
    
    dataset_config = DatasetConfig(
        symbols=[config.symbol],
        date_from=config.train_period.start,
        date_to=config.test_period.end,
        timeframe=config.timeframe,
    )
    
    df = load_joined_dataset(dataset_config)
    print(f"Loaded {len(df)} bars")
    
    # Add features
    df = add_extended_features(df)
    
    # Build target
    target_series = build_target(df, config.target)
    df['target'] = target_series
    df = df.dropna(subset=['target'])
    
    # Split
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
    
    print(f"Train: {len(df_train)} bars, Test: {len(df_test)} bars")
    
    # Prepare features
    exclude_cols = ['target', 'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target'].to_numpy().astype(int)
    X_test = df_test[feature_cols]
    y_test = df_test['target'].to_numpy().astype(int)
    
    # Train model
    print("\nTraining model...")
    model = LightGBMAlphaModel(random_state=42)
    train_result = model.fit(X_train, y_train, X_test, y_test)
    
    print(f"AUC: {train_result.auc:.3f}")
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)
    
    # Generate signals (same logic as eval.py)
    signals = np.where(y_pred_proba > 0.6, 1, np.where(y_pred_proba < 0.4, -1, 0))
    
    print("\n" + "=" * 60)
    print("SIGNAL ANALYSIS")
    print("=" * 60)
    
    trade_info = count_real_trades(signals)
    
    print(f"Test bars: {len(signals)}")
    print(f"Signal bars (current 'n_trades'): {trade_info['signal_bars']}")
    print(f"Actual position opens: {trade_info['opens']}")
    print(f"Actual round-trips: {trade_info['round_trips']}")
    print(f"Avg bars per position: {trade_info['avg_hold_bars']:.1f}")
    print(f"Overcharge ratio: {trade_info['signal_bars'] / max(trade_info['round_trips'], 1):.1f}x")
    
    # Calculate returns
    df_test = df_test.iloc[:len(signals)].copy()
    df_test['signal'] = signals
    df_test['returns'] = df_test['close'].pct_change()
    df_test['strategy_returns'] = df_test['signal'].shift(1) * df_test['returns']
    
    # Get returns on signal bars only
    trades = df_test[df_test['signal'] != 0]
    trade_returns = trades['strategy_returns'].dropna()
    
    print("\n" + "=" * 60)
    print("PnL ANALYSIS")
    print("=" * 60)
    
    # Pre-cost metrics
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    total_return_pre = (1 + trade_returns).prod() - 1
    pf_pre = gross_profit / gross_loss if gross_loss > 0 else 0
    
    print(f"Pre-cost total return: {total_return_pre*100:+.2f}%")
    print(f"Pre-cost profit factor: {pf_pre:.3f}")
    
    # WRONG way (current implementation)
    print("\n--- WRONG: Cost per signal bar ---")
    cost_per_bar = 0.003  # 30 bps
    trade_returns_wrong = trade_returns - cost_per_bar
    total_return_wrong = (1 + trade_returns_wrong).prod() - 1
    print(f"Total cost: {trade_info['signal_bars']} × 0.3% = {trade_info['signal_bars'] * 0.3:.1f}%")
    print(f"Post-cost return: {total_return_wrong*100:+.2f}%")
    
    # CORRECT way: cost per round-trip only
    print("\n--- CORRECT: Cost per round-trip ---")
    n_round_trips = trade_info['round_trips']
    
    for cost_bps in [30, 15, 10, 5, 2]:
        cost_pct = cost_bps / 100 / 100  # Convert bps
        total_cost = n_round_trips * cost_pct
        
        # Apply cost on position changes only
        # Simple approximation: spread cost across all returns proportionally
        avg_cost_per_bar = total_cost / max(len(trade_returns), 1)
        trade_returns_correct = trade_returns - avg_cost_per_bar
        
        # Or simpler: just subtract total cost from total return
        total_return_correct = total_return_pre - total_cost
        
        status = "✅" if total_return_correct > 0 else "❌"
        print(f"  {cost_bps} bps: Return = {total_return_correct*100:+.2f}% (cost = {total_cost*100:.2f}%) {status}")


if __name__ == "__main__":
    run_audit()
