
import sys
from pathlib import Path
import pandas as pd
import structlog
from datetime import date
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment.market_log import MarketLog
from src.experiment.dataset_builder import DatasetBuilder
from src.experiment.models import DatasetConfig
from src.experiment.regime_detector import MarketRegimeDetector, RegimeConfig, Regime

logger = structlog.get_logger()

async def segment_history():
    print("\n--- Segmenting History by Regime ---")
    
    # 1. Load Data
    print("1. Loading Data (Jan-May 2024)...")
    market_log = MarketLog("data/market")
    # Ensure data exists (should be cached from previous runs)
    await market_log.ensure_data(["BTC/USDT"], "5m", date(2024, 1, 1), date(2024, 5, 31))
    
    config = DatasetConfig(
        symbols=["BTC/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 5, 31),
        timeframe="5m",
    )
    
    builder = DatasetBuilder(market_log)
    dataset, _ = builder.build(config, save=False)
    print(f"Dataset: {len(dataset)} rows")
    
    # 2. Detect Regimes
    print("2. Detecting Regimes...")
    detector = MarketRegimeDetector(RegimeConfig(adx_threshold=40.0))
    
    # Apply detector
    dataset["regime"] = dataset.apply(lambda row: detector.detect(row).value, axis=1)
    
    # 3. Analysis
    print("\n--- Regime Distribution (Overall) ---")
    counts = dataset["regime"].value_counts(normalize=True) * 100
    print(counts)
    
    # Calculate returns per regime
    dataset["future_return"] = dataset["close"].shift(-12) / dataset["close"] - 1 # 1h return
    
    print("\n--- Avg 1h Return per Regime ---")
    avg_returns = dataset.groupby("regime")["future_return"].mean() * 100
    print(avg_returns)
    
    # 4. Monthly Breakdown
    print("\n--- Monthly Breakdown ---")
    dataset["month"] = dataset.index.to_period("M")
    
    monthly_counts = dataset.groupby(["month", "regime"]).size().unstack(fill_value=0)
    monthly_pct = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100
    
    print(monthly_pct)
    
    # 5. Specific Period Analysis
    print("\n--- Jan-Apr vs May ---")
    jan_apr = dataset[dataset.index < "2024-05-01"]
    may = dataset[dataset.index >= "2024-05-01"]
    
    print("Jan-Apr Distribution:")
    print(jan_apr["regime"].value_counts(normalize=True) * 100)
    
    print("\nMay Distribution:")
    print(may["regime"].value_counts(normalize=True) * 100)
    
    # Check ADX stats for May
    print("\nMay ADX Stats:")
    print(may["adx"].describe())

if __name__ == "__main__":
    asyncio.run(segment_history())
