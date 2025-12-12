"""
Target Generation for Research

Three target types:
1. Momentum: Will price continue moving up significantly?
2. Reversal: Will there be a reversal after an extremum?
3. Volatility Expansion: Will future volatility exceed baseline?
"""

from enum import Enum
from typing import Literal
import pandas as pd
import numpy as np
from pydantic import BaseModel


class TargetType(str, Enum):
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOL_EXPANSION = "vol_expansion"


class TargetConfig(BaseModel):
    """Configuration for target generation."""
    type: TargetType
    
    # General parameters
    horizon_bars: int = 12  # e.g., 12×5m ≈ 1 hour
    min_move_pct: float = 0.003  # 0.3% minimum move
    
    # Volatility expansion specific
    vol_window: int = 48  # baseline volatility window
    vol_factor: float = 1.5  # how much stronger than baseline
    
    # Reversal specific
    extremum_window: int = 3  # bars to check for local min/max


def build_target(df: pd.DataFrame, config: TargetConfig) -> pd.Series:
    """
    Build target labels based on config.
    
    Args:
        df: DataFrame with OHLCV columns
        config: Target configuration
        
    Returns:
        Series with {0, 1} labels, NaN where insufficient data
    """
    if config.type == TargetType.MOMENTUM:
        return _build_momentum_target(df, config)
    elif config.type == TargetType.REVERSAL:
        return _build_reversal_target(df, config)
    elif config.type == TargetType.VOL_EXPANSION:
        return _build_vol_expansion_target(df, config)
    else:
        raise ValueError(f"Unknown target type: {config.type}")


def _build_momentum_target(df: pd.DataFrame, config: TargetConfig) -> pd.Series:
    """
    Momentum target: Will there be significant upward movement?
    
    Algorithm:
    1. For each bar t, look at future window [t+1 : t+1+horizon]
    2. Find max(high) in that window
    3. Calculate return: (max_future - close[t]) / close[t]
    4. Target = 1 if return >= min_move_pct, else 0
    """
    n = len(df)
    targets = np.full(n, np.nan, dtype=float)
    
    close = df['close'].values
    high = df['high'].values
    
    for t in range(n - config.horizon_bars):
        # Future window (no leakage: starts at t+1)
        future_high = high[t+1 : t+1+config.horizon_bars]
        
        if len(future_high) == 0:
            continue
            
        max_future = future_high.max()
        future_ret = (max_future - close[t]) / close[t]
        
        targets[t] = 1 if future_ret >= config.min_move_pct else 0
    
    return pd.Series(targets, index=df.index, dtype='Int8')


def _build_reversal_target(df: pd.DataFrame, config: TargetConfig) -> pd.Series:
    """
    Reversal target: Will there be a reversal after an extremum?
    
    Algorithm:
    1. Detect local min/max using extremum_window
    2. If local max: check if price drops >= min_move_pct
    3. If local min: check if price rises >= min_move_pct
    4. Non-extremums get target = 0
    """
    n = len(df)
    targets = np.full(n, np.nan, dtype=float)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    k = config.extremum_window
    
    for t in range(k, n - max(k, config.horizon_bars)):
        # Check if local maximum
        window_high = high[t-k : t+k+1]
        is_local_max = high[t] == window_high.max()
        
        # Check if local minimum
        window_low = low[t-k : t+k+1]
        is_local_min = low[t] == window_low.min()
        
        if is_local_max:
            # Look for reversal down
            future_low = low[t+1 : t+1+config.horizon_bars]
            if len(future_low) > 0:
                min_future = future_low.min()
                drawdown = (min_future - close[t]) / close[t]
                targets[t] = 1 if drawdown <= -config.min_move_pct else 0
        elif is_local_min:
            # Look for reversal up
            future_high = high[t+1 : t+1+config.horizon_bars]
            if len(future_high) > 0:
                max_future = future_high.max()
                bounce = (max_future - close[t]) / close[t]
                targets[t] = 1 if bounce >= config.min_move_pct else 0
        else:
            # Not an extremum
            targets[t] = 0
    
    return pd.Series(targets, index=df.index, dtype='Int8')


def _build_vol_expansion_target(df: pd.DataFrame, config: TargetConfig) -> pd.Series:
    """
    Volatility Expansion target: Will future volatility exceed baseline?
    
    Algorithm:
    1. Calculate log returns
    2. Compute baseline volatility (rolling std over vol_window)
    3. For each bar, compute future volatility (std of next horizon_bars)
    4. Target = 1 if future_vol >= baseline_vol * vol_factor
    """
    # Calculate log returns
    close = df['close']
    log_ret = np.log(close / close.shift(1))
    
    # Baseline volatility (historical)
    baseline_vol = log_ret.rolling(config.vol_window).std()
    
    n = len(df)
    targets = np.full(n, np.nan, dtype=float)
    
    for t in range(config.vol_window, n - config.horizon_bars):
        # Check if we have valid baseline
        if pd.isna(baseline_vol.iloc[t]):
            continue
            
        # Future volatility
        future_ret = log_ret.iloc[t+1 : t+1+config.horizon_bars]
        
        if len(future_ret) < 2:  # Need at least 2 points for std
            continue
            
        future_vol = future_ret.std()
        
        # Target = 1 if expansion exceeds factor
        targets[t] = 1 if future_vol >= baseline_vol.iloc[t] * config.vol_factor else 0
    
    return pd.Series(targets, index=df.index, dtype='Int8')
