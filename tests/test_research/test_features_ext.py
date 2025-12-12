
import pytest
import pandas as pd
import numpy as np

from src.research.features_ext import add_extended_features, FeaturesConfig


# ============================================================================
# VOLUME/OBV TESTS
# ============================================================================

def test_obv_monotonic_behavior():
    """Test OBV increases on up bars, decreases on down bars."""
    n = 20
    
    # Up trend: prices increasing
    close_up = np.linspace(100, 110, n)
    volume = np.full(n, 1000.0)
    
    df = pd.DataFrame({
        'open': close_up * 0.99,
        'high': close_up * 1.01,
        'low': close_up * 0.98,
        'close': close_up,
        'volume': volume,
    })
    
    result = add_extended_features(df)
    
    # OBV should increase (prices going up)
    obv = result['obv'].values
    assert obv[-1] > obv[5], "OBV should increase in uptrend"
    
    # Down trend: prices decreasing
    close_down = np.linspace(110, 100, n)
    df2 = pd.DataFrame({
        'open': close_down * 1.01,
        'high': close_down * 1.02,
        'low': close_down * 0.99,
        'close': close_down,
        'volume': volume,
    })
    
    result2 = add_extended_features(df2)
    obv2 = result2['obv'].values
    
    # OBV should decrease (prices going down)
    assert obv2[-1] < obv2[5], "OBV should decrease in downtrend"


def test_obv_no_nans_after_warmup():
    """Check no NaNs/Infs in OBV features after warmup."""
    n = 100
    close = np.random.randn(n).cumsum() + 100
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995
    volume = np.random.rand(n) * 1000 + 500
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    result = add_extended_features(df)
    
    # After warmup (skip first 60 bars for all rolling windows)
    warmup = 60
    
    # Check OBV
    obv_valid = result['obv'].iloc[warmup:]
    assert not obv_valid.isna().any(), "OBV should not have NaNs after warmup"
    assert not np.isinf(obv_valid).any(), "OBV should not have Infs"
    
    # Check normalized OBV
    obv_norm_valid = result['obv_norm'].iloc[warmup:]
    assert not np.isinf(obv_norm_valid).any(), "OBV_norm should not have Infs"


def test_vol_price_div_reacts_to_divergence():
    """Test vol-price divergence reacts to different scenarios."""
    n = 30
    
    # Scenario 1: Price up, Volume up (bullish confirmation)
    close_up = np.linspace(100, 110, n)
    vol_up = np.linspace(500, 1500, n)
    
    df1 = pd.DataFrame({
        'open': close_up * 0.99,
        'high': close_up * 1.01,
        'low': close_up * 0.98,
        'close': close_up,
        'volume': vol_up,
    })
    
    result1 = add_extended_features(df1)
    
    # Scenario 2: Price down, Volume up (bearish with high volume)
    close_down = np.linspace(110, 100, n)
    vol_up2 = np.linspace(500, 1500, n)
    
    df2 = pd.DataFrame({
        'open': close_down * 1.01,
        'high': close_down * 1.02,
        'low': close_down * 0.99,
        'close': close_down,
        'volume': vol_up2,
    })
    
    result2 = add_extended_features(df2)
    
    # The divergence feature should differ between the two scenarios
    div1_avg = result1['vol_price_div'].iloc[-10:].mean()
    div2_avg = result2['vol_price_div'].iloc[-10:].mean()
    
    # Not strict equality, but they should trend differently
    # (one positive trend, one negative)
    assert abs(div1_avg - div2_avg) > 0.1, "Divergence should differ between scenarios"


# ============================================================================
# MULTI-TIMEFRAME TESTS
# ============================================================================

def test_mtf_features_consistent_with_trend():
    """Test MTF features track the trend correctly."""
    n = 100
    
    # Uptrend
    close = np.linspace(100, 120, n)
    high = close * 1.005
    low = close * 0.995
    open_ = close * 0.998
    volume = np.random.rand(n) * 1000 + 500
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    result = add_extended_features(df)
    
    # EMA fast and slow should both increase
    ema_fast_change = result['ema_fast'].iloc[-1] - result['ema_fast'].iloc[20]
    ema_slow_change = result['ema_slow'].iloc[-1] - result['ema_slow'].iloc[20]
    
    assert ema_fast_change > 0, "EMA fast should increase in uptrend"
    assert ema_slow_change > 0, "EMA slow should increase in uptrend"
    
    # Distance to slow EMA should be positive on average
    dist_avg = result['distance_to_slow_ema'].iloc[-20:].mean()
    assert dist_avg > 0, "Distance to slow EMA should be positive in uptrend"


def test_mtf_alignment_flag():
    """Test trend alignment flag detects aligned/misaligned trends."""
    n = 50
    
    # Aligned uptrend (both EMAs rising)
    close_aligned = np.linspace(100, 110, n)
    
    df_aligned = pd.DataFrame({
        'open': close_aligned * 0.99,
        'high': close_aligned * 1.01,
        'low': close_aligned * 0.98,
        'close': close_aligned,
        'volume': np.full(n, 1000.0),
    })
    
    result_aligned = add_extended_features(df_aligned)
    
    # Check alignment in the middle section (after warmup)
    alignment_mid = result_aligned['mtf_trend_alignment'].iloc[25:40]
    aligned_rate = alignment_mid.sum() / len(alignment_mid)
    
    assert aligned_rate > 0.7, "Alignment should be high in consistent uptrend"


def test_mtf_no_future_leakage():
    """Verify MTF features use only past data."""
    n = 60
    close = np.random.randn(n).cumsum() + 100
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995
    volume = np.random.rand(n) * 1000 + 500
    
    df1 = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    df2 = df1.copy()
    
    # Corrupt future (after bar 40)
    df2.loc[40:, 'close'] = df2.loc[40:, 'close'] + 50
    
    result1 = add_extended_features(df1)
    result2 = add_extended_features(df2)
    
    # Features at bar 35 should be identical (corruption is after)
    assert result1['ema_fast'].iloc[35] == result2['ema_fast'].iloc[35], \
        "EMA should not use future data"


# ============================================================================
# MICROSTRUCTURE TESTS
# ============================================================================

def test_run_lengths_increasing():
    """Test run lengths increase with consecutive same-direction bars."""
    n = 20
    
    # 10 consecutive up bars
    close = np.array([100 + i for i in range(n)])
    
    df = pd.DataFrame({
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'close': close,
        'volume': np.full(n, 1000.0),
    })
    
    result = add_extended_features(df)
    
    # Run up lengths should increase: 1, 2, 3, ...
    run_up = result['run_up_len'].values
    
    # Check that run lengths are monotonically increasing
    assert run_up[5] < run_up[10], "Run up length should increase"
    assert run_up[10] < run_up[15], "Run up length should keep increasing"
    
    # Run down should be 0 (no down bars)
    assert result['run_down_len'].iloc[-1] == 0, "No down runs in uptrend"


def test_n_sign_changes_high_in_choppy_segment():
    """Test sign changes detect choppy vs smooth segments."""
    n = 40
    
    # Smooth trend: all up
    close_smooth = np.linspace(100, 110, n)
    
    # Choppy: alternating up/down
    close_choppy = 100 + np.array([i % 2 for i in range(n)])
    
    df_smooth = pd.DataFrame({
        'open': close_smooth * 0.99,
        'high': close_smooth * 1.01,
        'low': close_smooth * 0.98,
        'close': close_smooth,
        'volume': np.full(n, 1000.0),
    })
    
    df_choppy = pd.DataFrame({
        'open': close_choppy * 0.99,
        'high': close_choppy * 1.01,
        'low': close_choppy * 0.98,
        'close': close_choppy,
        'volume': np.full(n, 1000.0),
    })
    
    result_smooth = add_extended_features(df_smooth)
    result_choppy = add_extended_features(df_choppy)
    
    # Sign changes should be much higher in choppy segment
    changes_smooth = result_smooth['n_sign_changes'].iloc[-1]
    changes_choppy = result_choppy['n_sign_changes'].iloc[-1]
    
    assert changes_choppy > changes_smooth * 5, \
        f"Choppy segment should have more sign changes ({changes_choppy} vs {changes_smooth})"


def test_body_to_range_ratio_limits():
    """Test body-to-range ratio is in [0, 1] and handles edge cases."""
    n = 30
    
    # Normal bars
    close = np.random.randn(n).cumsum() + 100
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    open_ = close + (np.random.rand(n) - 0.5)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.full(n, 1000.0),
    })
    
    result = add_extended_features(df)
    
    # Check bounds
    ratio = result['body_to_range_ratio']
    
    assert (ratio >= 0).all(), "Body-to-range ratio should be >= 0"
    assert (ratio <= 1.1).all(), "Body-to-range ratio should be <= 1 (with small epsilon)"
    
    # Check no NaN/Inf (epsilon in denominator prevents this)
    assert not ratio.isna().any(), "No NaNs"
    assert not np.isinf(ratio).any(), "No Infs"
