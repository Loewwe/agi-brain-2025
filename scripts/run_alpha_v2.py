
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
from src.experiment.simulator import Simulator, SimulatorConfig
from src.experiment.strategies.single_agent import SingleAgentStrategy, SingleAgentConfig
from src.experiment.strategies.alpha_strategy import AlphaStrategy

logger = structlog.get_logger()

async def run_alpha_v2():
    print("\n--- Running Alpha v2 Experiment ---")
    
    # 1. Data Preparation
    print("1. Preparing Data...")
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
    print(f"Dataset: {len(dataset)} rows")
    
    # 2. Labeling (Direction Target)
    print("2. Labeling (Direction Target)...")
    label_config = LabelConfig(
        horizon_bars=12, # 1 hour
        target_type="direction",
    )
    X, y = create_ml_dataset(dataset, label_config)
    df_labeled = pd.concat([X, y], axis=1)
    
    # Target is already binary (0/1) for direction
    
    # Update config
    alpha_config = AlphaConfig(
        model_type="lightgbm",
        target_column="target", # Default is 'target'
        train_ratio=0.8, # 80% Train
        val_ratio=0.2,   # 20% Validation
    )
    
    # 3. Training & Simulation (Rolling)
    print("3. Training & Simulation (Rolling)...")
    
    # We need to simulate iteratively.
    # But AlphaEngine.train_rolling returns metrics, not a model.
    # It trains and tests on folds.
    # We want to capture the PREDICTIONS for the test fold (May).
    
    # Actually, train_rolling does not return predictions.
    # It returns metrics.
    # We need to modify train_rolling or use a loop here.
    
    # Let's use a manual rolling loop here to integrate with Simulator.
    
    window_months = 3
    step_months = 1
    
    # We want to test May (2024-05).
    # So Train Window: Feb, Mar, Apr. (3 months)
    # Test Window: May.
    
    test_start_date = date(2024, 5, 1)
    test_end_date = date(2024, 5, 31)
    
    train_start_date = date(2024, 2, 1)
    train_end_date = date(2024, 4, 30)
    
    print(f"Train: {train_start_date} to {train_end_date}")
    print(f"Test:  {test_start_date} to {test_end_date}")
    
    train_mask = (df_labeled.index.date >= train_start_date) & (df_labeled.index.date <= train_end_date)
    test_mask = (df_labeled.index.date >= test_start_date) & (df_labeled.index.date <= test_end_date)
    
    df_train = df_labeled[train_mask]
    df_test = df_labeled[test_mask]
    
    print(f"Train Rows: {len(df_train)}")
    print(f"Test Rows:  {len(df_test)}")
    
    engine = AlphaEngine(alpha_config)
    engine.train(df_train)
    
    # Predict on May
    df_test_pred = engine.predict(df_test)
    
    # Setup Strategy
    agent_config = SingleAgentConfig(
        min_confidence=0.6, # Threshold
        max_daily_trades=5,
        trend_filter_type="NONE",
    )
    
    strategy = SingleAgentStrategy(agent_config, engine)
    
    sim_config = SimulatorConfig(
        start_balance=10000.0,
        atr_multiplier=1.5,
        tp_multiplier=2.0,
        timeout_bars=12,
    )
    
    simulator = Simulator(sim_config)
    result = simulator.run(df_test_pred, strategy)
    
    print("\n--- Simulation Results (May 2024) ---")
    print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print(f"Trades: {result.metrics.total_trades}")
    print(f"Win Rate: {result.metrics.win_rate:.2f}%")
    
    # Check DoD
    if result.metrics.total_return_pct >= 0:
        print("✅ PnL >= 0")
    else:
        print("❌ PnL < 0")
        
    if result.metrics.sharpe_ratio >= 0.5:
        print("✅ Sharpe >= 0.5")
    else:
        print("❌ Sharpe < 0.5")

if __name__ == "__main__":
    asyncio.run(run_alpha_v2())
