
import sys
from pathlib import Path
import pandas as pd
import structlog
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment.alpha_engine import AlphaEngine, AlphaConfig
from src.experiment.strategies.single_agent import SingleAgentStrategy, SingleAgentConfig
from src.experiment.simulator import Simulator, SimulatorConfig
from src.experiment.regime_detector import Regime

logger = structlog.get_logger()

def evaluate_regime_model(regime_name: str, dataset_path: Path):
    print(f"\n=== Evaluating Regime: {regime_name} ===")
    
    # 1. Load Data
    df = pd.read_parquet(dataset_path)
    print(f"Total Rows: {len(df)}")
    
    # 2. Split Train (Jan-Apr) / Test (May)
    train_mask = df.index < "2024-05-01"
    test_mask = df.index >= "2024-05-01"
    
    df_train = df[train_mask]
    df_test = df[test_mask]
    
    print(f"Train (Jan-Apr): {len(df_train)}")
    print(f"Test (May):      {len(df_test)}")
    
    if len(df_test) < 100:
        print("Not enough test data. Skipping.")
        return
    
    # 3. Train Model
    print("Training AlphaEngine...")
    alpha_config = AlphaConfig(
        model_type="lightgbm",
        target_column="target", # Direction
        train_ratio=0.8,
        val_ratio=0.2,
    )
    engine = AlphaEngine(alpha_config)
    train_result = engine.train(df_train)
    
    # Save Model
    model_dir = Path("data/models/regimes")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{regime_name}.pkl"
    engine.save(model_path)
    print(f"Saved model to {model_path}")
    
    print(f"Train AUC: {train_result.train_metrics.get('roc_auc'):.4f}")
    print(f"Val AUC:   {train_result.val_metrics.get('roc_auc'):.4f}")
    
    # 4. Predict on Test
    print("Predicting on May...")
    df_test_pred = engine.predict(df_test)
    
    # 5. Simulate
    print("Simulating...")
    agent_config = SingleAgentConfig(
        min_confidence=0.55, # Slightly lower threshold for regime-specific?
        max_daily_trades=10, # Allow more trades
        trend_filter_type="NONE", # Regime is already filtered
    )
    strategy = SingleAgentStrategy(agent_config, engine)
    
    sim_config = SimulatorConfig(
        start_balance=10000.0,
        atr_multiplier=2.0,
        tp_multiplier=3.0, # Try wider stops/targets
        timeout_bars=12,
    )
    
    simulator = Simulator(sim_config)
    result = simulator.run(df_test_pred, strategy)
    
    metrics = result.metrics
    print(f"--- Results for {regime_name} (May) ---")
    print(f"Total Return: {metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"Trades:       {metrics.total_trades}")
    print(f"Win Rate:     {metrics.win_rate * 100:.2f}%")
    
    return metrics

def main():
    regime_dir = Path("data/datasets/regimes")
    
    results = {}
    
    for regime in Regime:
        path = regime_dir / f"dataset_{regime.value}.parquet"
        if path.exists():
            metrics = evaluate_regime_model(regime.value, path)
            results[regime.value] = metrics
        else:
            print(f"Dataset not found: {path}")

if __name__ == "__main__":
    main()
