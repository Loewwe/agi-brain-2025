
import pytest
import pandas as pd
from datetime import date
from pathlib import Path

from src.research.data_loader import load_ohlcv, load_joined_dataset
from src.experiment.models import DatasetConfig
from src.experiment.market_log import MarketLog


@pytest.mark.asyncio
async def test_load_ohlcv_shape():
    """Checks that dataframe is not empty and has required columns."""
    # Use a known symbol and small date range
    df = await load_ohlcv("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2))
    
    # Check shape
    assert not df.empty, "DataFrame should not be empty"
    
    # Check required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.asyncio
async def test_load_joined_dataset_no_nans():
    """Checks for no NaNs in key columns after warmup period."""
    config = DatasetConfig(
        symbols=["BTC/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 1, 3),
        timeframe="5m",
    )
    
    df = load_joined_dataset(config)
    
    # Skip first N rows for warmup (indicators need history)
    warmup = 200
    df_after_warmup = df.iloc[warmup:]
    
    # Check key columns for NaNs
    key_cols = ["open", "high", "low", "close", "volume"]
    for col in key_cols:
        nan_count = df_after_warmup[col].isna().sum()
        assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"


def test_load_joined_dataset_deterministic(tmp_path):
    """Checks that same config yields same dataframe hash."""
    config = DatasetConfig(
        symbols=["BTC/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 1, 2),
        timeframe="5m",
    )
    
    # Load twice
    df1 = load_joined_dataset(config)
    df2 = load_joined_dataset(config)
    
    # Compare shapes
    assert df1.shape == df2.shape, "Shapes should match"
    
    # Compare values (use hash for efficiency)
    hash1 = pd.util.hash_pandas_object(df1).sum()
    hash2 = pd.util.hash_pandas_object(df2).sum()
    
    assert hash1 == hash2, "DataFrames should be identical"
