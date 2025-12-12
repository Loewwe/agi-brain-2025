
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
from src.experiment.labels import LabelConfig, LabelsBuilder

logger = structlog.get_logger()

async def prepare_regime_datasets():
    print("\n--- Preparing Regime Datasets ---")
    
    # 1. Load Data
    print("1. Loading Data (Jan-May 2024)...")
    market_log = MarketLog("data/market")
    await market_log.ensure_data(["BTC/USDT"], "5m", date(2024, 1, 1), date(2024, 5, 31))
    
    config = DatasetConfig(
        symbols=["BTC/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 5, 31),
        timeframe="5m",
    )
    
    builder = DatasetBuilder(market_log)
    dataset, _ = builder.build(config, save=False)
    
    # 2. Labeling (Direction)
    print("2. Labeling (Direction)...")
    label_config = LabelConfig(
        horizon_bars=12, # 1 hour
        target_type="direction",
    )
    labels_builder = LabelsBuilder(label_config)
    dataset = labels_builder.add_labels(dataset)
    
    # Drop NaNs
    dataset.dropna(inplace=True)
    print(f"Labeled Dataset: {len(dataset)} rows")
    
    # 3. Detect Regimes
    print("3. Detecting Regimes...")
    detector = MarketRegimeDetector(RegimeConfig(adx_threshold=25.0))
    dataset["regime"] = dataset.apply(lambda row: detector.detect(row).value, axis=1)
    
    # 4. Split and Save
    print("4. Splitting and Saving...")
    output_dir = Path("data/datasets/regimes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for regime in Regime:
        regime_df = dataset[dataset["regime"] == regime.value].copy()
        
        # Save
        path = output_dir / f"dataset_{regime.value}.parquet"
        regime_df.to_parquet(path)
        
        print(f"Saved {regime.value}: {len(regime_df)} rows -> {path}")
        
        # Verify Split
        train_len = len(regime_df[regime_df.index < "2024-05-01"])
        test_len = len(regime_df[regime_df.index >= "2024-05-01"])
        print(f"  Train (Jan-Apr): {train_len}")
        print(f"  Test (May):      {test_len}")

if __name__ == "__main__":
    asyncio.run(prepare_regime_datasets())
