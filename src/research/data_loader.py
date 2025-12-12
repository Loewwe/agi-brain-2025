"""
Data Loader for Research Environment

Lightweight wrapper over MarketLog/DatasetBuilder for convenient data access.
"""

from datetime import date
from pathlib import Path
import pandas as pd
import asyncio

from ..experiment.market_log import MarketLog
from ..experiment.dataset_builder import DatasetBuilder
from ..experiment.models import DatasetConfig


async def load_ohlcv(
    symbol: str,
    start: date,
    end: date,
    timeframe: str = "5m",
    market_log: MarketLog | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV data for a single symbol.
    
    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        start: Start date
        end: End date
        timeframe: Timeframe (default: "5m")
        market_log: Optional MarketLog instance (will create if None)
        
    Returns:
        DataFrame with OHLCV columns
    """
    if market_log is None:
        market_log = MarketLog("data/market")
    
    # Fetch data if needed
    await market_log.ensure_data([symbol], timeframe, start, end)
    return market_log.get_bars(symbol, timeframe, start, end)


def load_joined_dataset(
    config: DatasetConfig,
    market_log: MarketLog | None = None,
) -> pd.DataFrame:
    """
    Load and build a complete dataset with features.
    
    Args:
        config: Dataset configuration
        market_log: Optional MarketLog instance
        
    Returns:
        DataFrame with OHLCV + features
    """
    if market_log is None:
        market_log = MarketLog("data/market")
    
    builder = DatasetBuilder(market_log)
    
    # Build dataset (don't save for research)
    dataset, metadata = builder.build(config, save=False)
    
    return dataset
