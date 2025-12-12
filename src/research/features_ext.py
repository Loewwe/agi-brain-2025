"""
Extended Features for Research

Three feature blocks:
1. Volume/OBV: On-Balance Volume, volume-price divergence
2. Multi-Timeframe: EMA/RSI from different timeframes/periods
3. Microstructure: Choppiness, run lengths, body-to-range ratio
"""

import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional


class FeaturesConfig(BaseModel):
    """Configuration for extended features."""
    # Volume features
    obv_norm_window: int = 50
    vol_ma_window: int = 20
    
    # Multi-timeframe (simulated via periods)
    ema_fast_period: int = 20
    ema_slow_period: int = 200
    rsi_period: int = 14
    
    # Microstructure
    run_length_window: int = 20
    sign_change_window: int = 20
    body_range_window: int = 10


def add_extended_features(
    df: pd.DataFrame,
    config: Optional[FeaturesConfig] = None
) -> pd.DataFrame:
    """
    Add extended features to dataframe.
    
    Args:
        df: DataFrame with OHLCV columns
        config: Feature configuration (uses defaults if None)
        
    Returns:
        DataFrame with added feature columns
    """
    if config is None:
        config = FeaturesConfig()
    
    df = df.copy()
    
    # Block 1: Volume/OBV Features
    df = _add_volume_features(df, config)
    
    # Block 2: Multi-Timeframe Features
    df = _add_multitimeframe_features(df, config)
    
    # Block 3: Microstructure Features
    df = _add_microstructure_features(df, config)
    
    return df


def _add_volume_features(df: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Add OBV and volume-price divergence features."""
    
    # OBV (On-Balance Volume)
    close = df['close']
    volume = df['volume']
    
    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    
    df['obv'] = obv
    
    # OBV normalized (z-score over rolling window)
    obv_series = pd.Series(obv, index=df.index)
    obv_mean = obv_series.rolling(config.obv_norm_window).mean()
    obv_std = obv_series.rolling(config.obv_norm_window).std()
    df['obv_norm'] = (obv_series - obv_mean) / (obv_std + 1e-8)
    
    # Volume-Price Divergence
    # Compare volume change vs price change
    vol_ma = volume.rolling(config.vol_ma_window).mean()
    price_ma = close.rolling(config.vol_ma_window).mean()
    
    vol_signal = np.sign(volume - vol_ma).fillna(0)
    price_signal = np.sign(close - price_ma).fillna(0)
    
    # Divergence: positive when both agree, negative when diverge
    df['vol_price_div'] = vol_signal * price_signal
    
    # Volume ratio (current vs MA)
    df['vol_ratio'] = volume / (vol_ma + 1e-8)
    
    return df


def _add_multitimeframe_features(df: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Add multi-timeframe features (simulated via different periods)."""
    
    close = df['close']
    
    # Fast and Slow EMAs
    ema_fast = close.ewm(span=config.ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=config.ema_slow_period, adjust=False).mean()
    
    df['ema_fast'] = ema_fast
    df['ema_slow'] = ema_slow
    
    # Distance to slow EMA (normalized)
    df['distance_to_slow_ema'] = (close - ema_slow) / (ema_slow + 1e-8)
    
    # EMA slopes (approximated by difference)
    df['ema_fast_slope'] = ema_fast.diff()
    df['ema_slow_slope'] = ema_slow.diff()
    
    # Multi-TF Trend Alignment
    # 1 if both EMAs trending same direction, 0 otherwise
    fast_sign = np.sign(df['ema_fast_slope']).fillna(0)
    slow_sign = np.sign(df['ema_slow_slope']).fillna(0)
    df['mtf_trend_alignment'] = (fast_sign == slow_sign).astype(int)
    
    # RSI on different period (simulated higher TF)
    # Standard RSI calculation
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(config.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(config.rsi_period).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # "Higher TF" RSI (using longer period)
    rsi_period_long = config.rsi_period * 4
    gain_long = (delta.where(delta > 0, 0)).rolling(rsi_period_long).mean()
    loss_long = (-delta.where(delta < 0, 0)).rolling(rsi_period_long).mean()
    rs_long = gain_long / (loss_long + 1e-8)
    df['rsi_mtf'] = 100 - (100 / (1 + rs_long))
    
    return df


def _add_microstructure_features(df: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Add microstructure/choppiness features."""
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    
    # Price direction (up/down bars)
    direction = np.sign(close.diff()).fillna(0)
    
    # Run lengths (consecutive up/down bars)
    run_up_len = np.zeros(len(df))
    run_down_len = np.zeros(len(df))
    
    current_up = 0
    current_down = 0
    
    for i in range(1, len(df)):
        if direction.iloc[i] > 0:
            current_up += 1
            current_down = 0
        elif direction.iloc[i] < 0:
            current_down += 1
            current_up = 0
        else:
            # Flat bar
            pass
        
        run_up_len[i] = current_up
        run_down_len[i] = current_down
    
    df['run_up_len'] = run_up_len
    df['run_down_len'] = run_down_len
    
    # Number of sign changes in window
    sign_changes = (direction.diff() != 0).astype(int)
    df['n_sign_changes'] = sign_changes.rolling(config.sign_change_window).sum()
    
    # Body-to-Range Ratio
    candle_body = np.abs(close - open_)
    candle_range = high - low
    df['body_to_range_ratio'] = candle_body / (candle_range + 1e-8)
    
    # Average body-to-range over window
    df['body_to_range_avg'] = df['body_to_range_ratio'].rolling(config.body_range_window).mean()
    
    # Wick ratios (optional, for completeness)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    
    df['wick_ratio_up'] = upper_wick / (candle_range + 1e-8)
    df['wick_ratio_down'] = lower_wick / (candle_range + 1e-8)
    
    return df
