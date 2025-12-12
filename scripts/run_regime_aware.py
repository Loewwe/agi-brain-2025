
import sys
from pathlib import Path
import pandas as pd
import structlog
import asyncio
from datetime import date

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment.market_log import MarketLog
from src.experiment.dataset_builder import DatasetBuilder
from src.experiment.models import DatasetConfig
from src.experiment.regime_alpha_engine import RegimeAwareAlphaEngine
from src.experiment.strategies.single_agent import SingleAgentStrategy, SingleAgentConfig
from src.experiment.simulator import Simulator, SimulatorConfig
from src.experiment.labels import LabelConfig, LabelsBuilder

logger = structlog.get_logger()

async def run_regime_aware_simulation():
    print("\n=== Running Regime-Aware Simulation (May 2024) ===")
    
    # 1. Load Data (May Only)
    print("1. Loading Data...")
    market_log = MarketLog("data/market")
    await market_log.ensure_data(["BTC/USDT"], "5m", date(2024, 5, 1), date(2024, 5, 31))
    
    config = DatasetConfig(
        symbols=["BTC/USDT"],
        date_from=date(2024, 5, 1),
        date_to=date(2024, 5, 31),
        timeframe="5m",
    )
    
    builder = DatasetBuilder(market_log)
    dataset, _ = builder.build(config, save=False)
    print(f"Dataset: {len(dataset)} rows")
    
    # 2. Initialize Engine
    print("2. Initializing RegimeAwareAlphaEngine (Sideways Only)...")
    engine = RegimeAwareAlphaEngine(
        models_dir="data/models/regimes",
        allowed_regimes=["sideways"],
    )
    
    # 3. Predict
    print("3. Generating Predictions...")
    predictions = engine.predict(dataset)
    
    # Check for NaNs
    print(f"Predictions NaNs: {predictions['alpha_score'].isna().sum()}")
    
    # 4. Simulate
    print("4. Simulating...")
    agent_config = SingleAgentConfig(
        min_confidence=0.55,
        max_daily_trades=10,
        trend_filter_type="NONE",
    )
    
    # We need to wrap RegimeAwareAlphaEngine to look like AlphaEngine for Strategy?
    # SingleAgentStrategy expects `engine.predict_single` or just uses `engine` passed to it?
    # SingleAgentStrategy calls `self.alpha_engine.predict_single(features)`.
    # RegimeAwareAlphaEngine implements `predict_single`.
    # So it should work.
    
    strategy = SingleAgentStrategy(agent_config, engine)
    
    sim_config = SimulatorConfig(
        start_balance=10000.0,
        atr_multiplier=2.0,
        tp_multiplier=3.0,
        timeout_bars=12,
    )
    
    simulator = Simulator(sim_config)
    result = simulator.run(predictions, strategy)
    
    metrics = result.metrics
    print("\n--- Simulation Results (Regime Aware) ---")
    print(f"Total Return: {metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"Trades:       {metrics.total_trades}")
    print(f"Win Rate:     {metrics.win_rate * 100:.2f}%")
    
    # Regime Breakdown in Simulation?
    # Simulator doesn't track regime.
    # But we can infer from predictions["regime"].
    
    print("\n--- Regime Breakdown ---")
    regime_counts = predictions["regime"].value_counts(normalize=True) * 100
    print(regime_counts)

if __name__ == "__main__":
    asyncio.run(run_regime_aware_simulation())
