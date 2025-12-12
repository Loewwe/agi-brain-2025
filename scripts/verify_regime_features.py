
import sys
from pathlib import Path
import pandas as pd
import structlog
from datetime import date

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment.market_log import MarketLog
from src.experiment.dataset_builder import DatasetBuilder
from src.experiment.models import DatasetConfig

logger = structlog.get_logger()

async def verify_features():
    print("\n--- Verifying Regime Features ---")
    
    # Use existing market data
    market_log = MarketLog("data/market")
    await market_log.ensure_data(["BTC/USDT"], "1h", date(2024, 1, 1), date(2024, 1, 5))
    
    config = DatasetConfig(
        symbols=["BTC/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 1, 5),
        timeframe="1h",
    )
    
    builder = DatasetBuilder(market_log)
    dataset, _ = builder.build(config, save=False)
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")
    
    # Check for new features
    new_features = ["dist_to_ema200", "volatility_ratio", "candle_amplitude", "adx"]
    
    for feature in new_features:
        if feature not in dataset.columns:
            print(f"❌ Missing feature: {feature}")
            sys.exit(1)
        
        # Check for NaNs (some expected at start due to rolling windows)
        nan_count = dataset[feature].isna().sum()
        print(f"✅ Found {feature}. NaNs: {nan_count}/{len(dataset)}")
        
        # Check values range
        min_val = dataset[feature].min()
        max_val = dataset[feature].max()
        mean_val = dataset[feature].mean()
        print(f"   Range: {min_val:.4f} to {max_val:.4f} (Mean: {mean_val:.4f})")

    print("\nAll regime features verified successfully.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(verify_features())
