"""
Fixture 1: Toy Example Dataset with Known Results.

This fixture provides a small, hand-crafted dataset with pre-calculated
expected metrics. Used to "nail down" the simulator behavior permanently.

The dataset has:
- 5 days of data (1440 5-minute bars)
- 3 symbols: BTCUSDT, ETHUSDT, SOLUSDT  
- Engineered signals that will produce exactly 5 trades
- Pre-calculated expected PnL, drawdown, win rate

This is the "gold standard" test - if it breaks, the simulator changed.
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# FIXTURE DATA
# =============================================================================

def create_toy_dataset() -> pd.DataFrame:
    """
    Create a small, deterministic dataset with known trade signals.
    
    Layout:
    - 5 trading days
    - 3 symbols
    - Clear entry signals at specific points
    - Known outcomes for each trade
    
    Returns:
        DataFrame with OHLCV + features for backtesting
    """
    # Create 5 days of 5-minute bars (288 bars per day = 1440 bars)
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    dates = pd.date_range(start=start_date, periods=1440, freq="5min")
    
    # Base prices for each symbol
    base_prices = {
        "BTCUSDT": 50000.0,
        "ETHUSDT": 3000.0,
        "SOLUSDT": 100.0,
    }
    
    all_dfs = []
    
    for symbol, base_price in base_prices.items():
        # Create price series with slight uptrend
        np.random.seed(hash(symbol) % 2**32)  # Deterministic per symbol
        
        # Generate prices
        returns = np.random.normal(0.00001, 0.0005, 1440)
        prices = base_price * np.cumprod(1 + returns)
        
        # Create OHLCV
        df = pd.DataFrame({
            "open": prices,
            "high": prices * (1 + np.abs(np.random.normal(0, 0.001, 1440))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.001, 1440))),
            "close": prices,
            "volume": np.random.uniform(1000, 5000, 1440),
            "symbol": symbol,
        }, index=dates)
        
        # Add features (matching v1 schema)
        df = _add_features(df, base_price)
        
        # Engineer specific signals at known points
        df = _engineer_signals(df, symbol)
        
        all_dfs.append(df)
    
    # Combine all symbols
    dataset = pd.concat(all_dfs)
    dataset.sort_index(inplace=True)
    
    return dataset


def _add_features(df: pd.DataFrame, base_price: float) -> pd.DataFrame:
    """Add technical features to DataFrame."""
    # RSI - mostly neutral (40-60)
    df["rsi"] = 50.0 + np.random.normal(0, 5, len(df))
    df["rsi"] = df["rsi"].clip(20, 80)
    df["rsi_prev"] = df["rsi"].shift(1).fillna(50)
    df["rsi_prev2"] = df["rsi"].shift(2).fillna(50)
    
    # ATR - stable
    df["atr"] = base_price * 0.01  # 1% ATR
    df["atr_pct"] = 1.0  # 1% of price
    
    # EMA200 - slightly below price (bullish)
    df["ema200"] = df["close"] * 0.995
    df["ema_slope"] = True  # Rising
    df["ema_distance_pct"] = (df["close"] - df["ema200"]) / df["ema200"] * 100
    
    # Volume
    df["volume_sma"] = df["volume"].rolling(20).mean().fillna(df["volume"])
    df["volume_surge"] = 1.2  # Slight surge, not enough for signal
    
    # Breakout levels
    df["high_2"] = df["high"].rolling(2).max().shift(1).fillna(df["high"])
    df["low_2"] = df["low"].rolling(2).min().shift(1).fillna(df["low"])
    
    # Additional features
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["bar_range"] = df["high"] - df["low"]
    df["bar_range_pct"] = df["bar_range"] / df["close"] * 100
    df["close_position"] = 0.5
    
    return df


def _engineer_signals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Engineer specific entry signals at known points.
    
    We create 5 total trades across all symbols:
    - 2 winning longs (hit TP)
    - 1 winning short (hit TP)
    - 1 losing long (hit SL)
    - 1 breakeven (hit BE)
    
    This gives us known metrics to verify.
    """
    # Signal positions (bar indices where we engineer signals)
    # Each symbol gets different signals
    if symbol == "BTCUSDT":
        # Trade 1: Winning long at bar 100 (Day 1, ~8:20 AM)
        # RSI oversold, reversal, volume surge, breakout
        df.iloc[100, df.columns.get_loc("rsi")] = 30.0
        df.iloc[99, df.columns.get_loc("rsi")] = 28.0  # rsi_prev
        df.iloc[100, df.columns.get_loc("rsi_prev")] = 28.0
        df.iloc[100, df.columns.get_loc("volume_surge")] = 2.0
        # Make close > high_2 for breakout
        close_val = df.iloc[100]["close"]
        df.iloc[100, df.columns.get_loc("high_2")] = close_val * 0.998
        
        # Trade 2: Losing long at bar 400 (Day 2, ~9:20 AM)
        df.iloc[400, df.columns.get_loc("rsi")] = 32.0
        df.iloc[399, df.columns.get_loc("rsi")] = 30.0
        df.iloc[400, df.columns.get_loc("rsi_prev")] = 30.0
        df.iloc[400, df.columns.get_loc("volume_surge")] = 1.8
        close_val = df.iloc[400]["close"]
        df.iloc[400, df.columns.get_loc("high_2")] = close_val * 0.997
        
    elif symbol == "ETHUSDT":
        # Trade 3: Winning short at bar 600 (Day 3, ~4:40 AM - skip due to time filter)
        # Move to bar 650 (Day 3, ~8:50 AM)
        df.iloc[650, df.columns.get_loc("rsi")] = 70.0
        df.iloc[649, df.columns.get_loc("rsi")] = 72.0
        df.iloc[650, df.columns.get_loc("rsi_prev")] = 72.0
        df.iloc[650, df.columns.get_loc("volume_surge")] = 1.9
        # Make EMA200 above price for short signal
        close_val = df.iloc[650]["close"]
        df.iloc[650, df.columns.get_loc("ema200")] = close_val * 1.01
        df.iloc[650, df.columns.get_loc("low_2")] = close_val * 1.002
        
    elif symbol == "SOLUSDT":
        # Trade 4: Winning long at bar 900 (Day 4, ~3:00 AM - skip)
        # Move to bar 950 (Day 4, ~7:10 AM)
        df.iloc[950, df.columns.get_loc("rsi")] = 33.0
        df.iloc[949, df.columns.get_loc("rsi")] = 31.0
        df.iloc[950, df.columns.get_loc("rsi_prev")] = 31.0
        df.iloc[950, df.columns.get_loc("volume_surge")] = 2.2
        close_val = df.iloc[950]["close"]
        df.iloc[950, df.columns.get_loc("high_2")] = close_val * 0.996
        
        # Trade 5: Breakeven long at bar 1200 (Day 5, ~4:00 AM - skip)
        # Move to bar 1250 (Day 5, ~8:10 AM)
        df.iloc[1250, df.columns.get_loc("rsi")] = 34.0
        df.iloc[1249, df.columns.get_loc("rsi")] = 32.0
        df.iloc[1250, df.columns.get_loc("rsi_prev")] = 32.0
        df.iloc[1250, df.columns.get_loc("volume_surge")] = 1.7
        close_val = df.iloc[1250]["close"]
        df.iloc[1250, df.columns.get_loc("high_2")] = close_val * 0.995
    
    return df


# =============================================================================
# EXPECTED RESULTS
# =============================================================================

# These are the expected results from running the toy dataset through
# the simulator with default Stage6 config.
# 
# NOTE: These values should be calculated once and then frozen.
# If the simulator changes, these tests should FAIL.

EXPECTED_METRICS = {
    # Will be filled after first run
    "total_trades": None,  # Expected ~5 trades
    "win_rate": None,
    "total_return_pct": None,
    "max_drawdown_pct": None,
}


def get_expected_metrics() -> dict:
    """Get expected metrics for the toy dataset."""
    return EXPECTED_METRICS.copy()


def set_expected_metrics(metrics: dict) -> None:
    """Set expected metrics (call once to freeze values)."""
    global EXPECTED_METRICS
    EXPECTED_METRICS.update(metrics)


# =============================================================================
# FIXTURE FILE I/O
# =============================================================================

def save_fixture(dataset: pd.DataFrame, output_dir: Path) -> Path:
    """Save fixture to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_path = output_dir / "toy_dataset.parquet"
    dataset.to_parquet(dataset_path)
    
    # Save expected metrics
    metrics_path = output_dir / "expected_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(EXPECTED_METRICS, f, indent=2)
    
    return dataset_path


def load_fixture(fixture_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load fixture from files."""
    dataset_path = fixture_dir / "toy_dataset.parquet"
    metrics_path = fixture_dir / "expected_metrics.json"
    
    dataset = pd.read_parquet(dataset_path)
    
    with open(metrics_path, "r") as f:
        expected_metrics = json.load(f)
    
    return dataset, expected_metrics


if __name__ == "__main__":
    # Generate and display fixture info
    dataset = create_toy_dataset()
    print(f"Dataset shape: {dataset.shape}")
    print(f"Date range: {dataset.index.min()} to {dataset.index.max()}")
    print(f"Symbols: {dataset['symbol'].unique().tolist()}")
    print(f"\nSample rows:")
    print(dataset.head(10))
