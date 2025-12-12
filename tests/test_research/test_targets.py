
import pytest
import pandas as pd
import numpy as np

from src.research.targets import build_target, TargetType, TargetConfig


# ============================================================================
# MOMENTUM TESTS
# ============================================================================

def test_momentum_target_rising_trend():
    """Test momentum target on clear uptrend."""
    # Synthetic: linear rise 10% over 100 bars
    n = 100
    close = np.linspace(100, 110, n)
    high = close * 1.001  # slightly higher
    low = close * 0.999
    
    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
    })
    
    config = TargetConfig(
        type=TargetType.MOMENTUM,
        horizon_bars=5,
        min_move_pct=0.005,  # 0.5%
    )
    
    target = build_target(df, config)
    
    # Most bars in the middle should be 1 (uptrend continues)
    middle = target.iloc[10:80]
    positive_rate = middle.sum() / len(middle)
    
    assert positive_rate > 0.7, f"Expected >70% positive in uptrend, got {positive_rate:.2%}"


def test_momentum_target_flat_or_downtrend():
    """Test momentum target on flat/down market."""
    n = 100
    close = np.full(n, 100.0)  # flat
    high = close * 1.0005
    low = close * 0.9995
    
    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
    })
    
    config = TargetConfig(
        type=TargetType.MOMENTUM,
        horizon_bars=5,
        min_move_pct=0.01,  # 1%
    )
    
    target = build_target(df, config)
    
    # Should be mostly 0 (no momentum)
    valid = target.dropna()
    positive_rate = valid.sum() / len(valid) if len(valid) > 0 else 0
    
    assert positive_rate < 0.1, f"Expected <10% positive in flat market, got {positive_rate:.2%}"


def test_momentum_target_no_leakage():
    """Verify momentum target only uses future data."""
    n = 50
    close = np.linspace(100, 105, n)
    high = close * 1.01
    low = close * 0.99
    
    df1 = pd.DataFrame({'close': close, 'high': high, 'low': low})
    df2 = df1.copy()
    
    # Corrupt future within the window that bar 15 uses (16-26)
    # Bar 15 looks at bars 16-25 (horizon=10)
    df2.loc[16:25, 'high'] = 90
    
    config = TargetConfig(type=TargetType.MOMENTUM, horizon_bars=10)
    
    target1 = build_target(df1, config)
    target2 = build_target(df2, config)
    
    # Target at bar 15 should differ (corrupted future affects it)
    assert target1.iloc[15] != target2.iloc[15], "Targets should change when future is corrupted"


# ============================================================================
# REVERSAL TESTS
# ============================================================================

def test_reversal_target_on_synthetic_peak_valley():
    """Test reversal detection on clear peak and valley."""
    # Create: rise → peak → fall → valley → rise
    rise1 = np.linspace(100, 110, 20)
    fall = np.linspace(110, 95, 20)
    rise2 = np.linspace(95, 105, 20)
    
    close = np.concatenate([rise1, fall, rise2])
    high = close * 1.005
    low = close * 0.995
    
    df = pd.DataFrame({'close': close, 'high': high, 'low': low})
    
    config = TargetConfig(
        type=TargetType.REVERSAL,
        horizon_bars=15,
        min_move_pct=0.05,  # 5% reversal
        extremum_window=3,
    )
    
    target = build_target(df, config)
    
    # Peak around index 20, valley around index 40
    # Should have target=1 near those points
    assert target.iloc[18:22].sum() > 0, "Should detect reversal near peak"
    assert target.iloc[38:42].sum() > 0, "Should detect reversal near valley"


def test_reversal_target_non_extremes_zero():
    """Test that non-extremum bars get target=0."""
    # Pure trend, no reversals
    close = np.linspace(100, 120, 50)
    high = close * 1.001
    low = close * 0.999
    
    df = pd.DataFrame({'close': close, 'high': high, 'low': low})
    
    config = TargetConfig(
        type=TargetType.REVERSAL,
        horizon_bars=10,
        extremum_window=3,
    )
    
    target = build_target(df, config)
    
    # Most should be 0 (no extremums in smooth trend)
    valid = target.dropna()
    positive_rate = valid.sum() / len(valid) if len(valid) > 0 else 0
    
    assert positive_rate < 0.2, f"Expected <20% reversals in smooth trend, got {positive_rate:.2%}"


def test_reversal_target_window_edges():
    """Test edge handling (no IndexError)."""
    n = 30
    close = np.random.randn(n).cumsum() + 100
    high = close * 1.01
    low = close * 0.99
    
    df = pd.DataFrame({'close': close, 'high': high, 'low': low})
    
    config = TargetConfig(type=TargetType.REVERSAL, horizon_bars=10)
    
    # Should not raise
    target = build_target(df, config)
    
    # First and last bars should be NaN or 0
    assert pd.isna(target.iloc[0]) or target.iloc[0] == 0
    assert pd.isna(target.iloc[-1]) or target.iloc[-1] == 0


# ============================================================================
# VOLATILITY EXPANSION TESTS
# ============================================================================

def test_vol_expansion_on_spike_segment():
    """Test vol expansion on low→high volatility transition."""
    # Low vol segment
    low_vol = np.random.randn(100) * 0.5 + 100
    
    # High vol segment
    high_vol = np.random.randn(50) * 3 + 100
    
    close = np.concatenate([low_vol, high_vol])
    high = close * 1.005
    low = close * 0.995
    
    df = pd.DataFrame({'close': close, 'high': high, 'low': low})
    
    config = TargetConfig(
        type=TargetType.VOL_EXPANSION,
        vol_window=50,
        horizon_bars=20,
        vol_factor=2.0,
    )
    
    target = build_target(df, config)
    
    # Near the transition (around index 90-100), should detect spike
    transition = target.iloc[85:105]
    positive_count = transition.sum()
    
    assert positive_count > 0, "Should detect volatility expansion at transition"


def test_vol_expansion_baseline_stability():
    """Test that stable low-vol gives mostly target=0."""
    # Low, stable volatility
    close = np.random.randn(100) * 0.3 + 100
    high = close * 1.002
    low = close * 0.998
    
    df = pd.DataFrame({'close': close, 'high': high, 'low': low})
    
    config = TargetConfig(
        type=TargetType.VOL_EXPANSION,
        vol_window=30,
        horizon_bars=10,
        vol_factor=1.8,
    )
    
    target = build_target(df, config)
    
    valid = target.dropna()
    positive_rate = valid.sum() / len(valid) if len(valid) > 0 else 0
    
    assert positive_rate < 0.15, f"Expected <15% expansion in stable vol, got {positive_rate:.2%}"


def test_vol_expansion_no_leakage():
    """Verify vol expansion only uses future data."""
    n = 80
    close = np.random.randn(n).cumsum() + 100
    high = close * 1.01
    low = close * 0.99
    
    df1 = pd.DataFrame({'close': close, 'high': high, 'low': low})
    df2 = df1.copy()
    
    # Corrupt future (make it very volatile)
    df2.loc[50:, 'close'] = df2.loc[50:, 'close'] + np.random.randn(n-50) * 10
    
    config = TargetConfig(type=TargetType.VOL_EXPANSION, vol_window=30, horizon_bars=15)
    
    target1 = build_target(df1, config)
    target2 = build_target(df2, config)
    
    # Targets before corruption should differ
    if not pd.isna(target1.iloc[45]) and not pd.isna(target2.iloc[45]):
        assert target1.iloc[45] != target2.iloc[45], "Targets should change when future volatility changes"

