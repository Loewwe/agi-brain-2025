
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
from src.experiment.labels import LabelConfig, create_ml_dataset
from src.experiment.alpha_engine import AlphaEngine, AlphaConfig

logger = structlog.get_logger()

async def test_rolling_train():
    print("\n--- Testing Rolling Training ---")
    
    # 1. Get Data (Jan - May 2024)
    market_log = MarketLog("data/market")
    await market_log.ensure_data(["BTC/USDT"], "1h", date(2024, 1, 1), date(2024, 5, 31))
    
    config = DatasetConfig(
        symbols=["BTC/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 5, 31),
        timeframe="1h",
    )
    
    builder = DatasetBuilder(market_log)
    dataset, _ = builder.build(config, save=False)
    print(f"Dataset: {len(dataset)} rows")
    print(f"Index Start: {dataset.index.min()}")
    print(f"Index End:   {dataset.index.max()}")
    
    # 2. Add Labels
    label_config = LabelConfig(
        horizon_bars=3,
        target_type="direction", # Simple target for test
    )
    X, y = create_ml_dataset(dataset, label_config)
    df_labeled = pd.concat([X, y], axis=1)
    
    # 3. Rolling Train
    # Window = 2 months, Step = 1 month
    # Fold 1: Train Jan-Feb, Test Mar
    # Fold 2: Train Feb-Mar, Test Apr
    # Fold 3: Train Mar-Apr, Test May
    
    alpha_config = AlphaConfig(model_type="lightgbm")
    engine = AlphaEngine(alpha_config)
    
    print("Starting rolling training...")
    result = engine.train_rolling(
        df_labeled,
        window_months=2,
        step_months=1,
    )
    
    print("\n--- Results ---")
    print(f"Total Folds: {len(result.folds)}")
    print("Average Metrics:", result.avg_metrics)
    
    for i, fold in enumerate(result.folds):
        print(f"\nFold {i+1}:")
        print(f"  Train: {fold['train_start']} -> {fold['train_end']}")
        print(f"  Test:  {fold['test_start']} -> {fold['test_end']}")
        print(f"  AUC:   {fold['metrics'].get('roc_auc', 0):.4f}")
        print(f"  Acc:   {fold['metrics'].get('accuracy', 0):.4f}")
    
    assert len(result.folds) >= 2
    print("\n✅ Rolling Training Verified")

if __name__ == "__main__":
    asyncio.run(test_rolling_train())
