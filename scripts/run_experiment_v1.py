"""
Experiment V1 Execution Script.

Runs the full pipeline for Stage 7 Step 3:
1. Data Generation (BTC/USDT, ETH/USDT)
2. Model Training (AlphaEngine)
3. Backtest Execution (SingleAgent vs Baseline)
4. Comparison Report
"""

import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment.market_log import MarketLog
from src.experiment.dataset_builder import DatasetBuilder
from src.experiment.models import DatasetConfig, SimulatorConfig
from src.experiment.alpha_engine import AlphaEngine, AlphaConfig
from src.experiment.strategies import SingleAgentStrategy, SingleAgentConfig
from src.experiment.simulator import Stage6Strategy
from src.experiment.comparison import StrategyComparison
from src.experiment.labels import LabelConfig, LabelsBuilder

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


async def ensure_data():
    """Ensure market data exists."""
    market_log = MarketLog("data/market")
    
    symbols = ["BTC/USDT", "ETH/USDT"]
    date_from = date(2024, 1, 1)
    date_to = date(2024, 6, 1)
    
    logger.info("experiment.data_check", symbols=symbols)
    
    await market_log.ensure_data(
        symbols=symbols,
        timeframe="5m",
        date_from=date_from,
        date_to=date_to,
        fetch_missing=True,
    )
    
    return market_log


def run_experiment():
    """Run the full experiment."""
    # 1. Data Generation
    market_log = asyncio.run(ensure_data())
    
    builder = DatasetBuilder(market_log, output_path="data/datasets")
    
    config = DatasetConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 6, 1),
        timeframe="5m",
    )
    
    logger.info("experiment.building_dataset")
    dataset, metadata = builder.build(config)
    
    # 2. Labeling
    logger.info("experiment.labeling_dataset")
    label_config = LabelConfig(
        horizon_bars=12,  # 1 hour
        target_type="direction",
    )
    labels_builder = LabelsBuilder(label_config)
    dataset_labeled = labels_builder.add_labels(dataset)
    
    # Drop rows with NaN target (last horizon_bars rows)
    dataset_labeled = dataset_labeled.dropna(subset=["target"])
    
    # 3. Model Training
    logger.info("experiment.training_model")
    
    alpha_config = AlphaConfig(
        model_type="lightgbm",
        target_column="target",
    )
    
    engine = AlphaEngine(alpha_config)
    train_result = engine.train(dataset_labeled)
    
    logger.info(
        "experiment.training_completed",
        accuracy=train_result.test_metrics["accuracy"],
        roc_auc=train_result.test_metrics["roc_auc"],
    )
    
    # Save model
    model_path = "data/models/alpha_v1"
    engine.save(model_path)
    
    # 4. Backtest & Comparison
    logger.info("experiment.running_backtest")
    
    # Split data for testing (use the test split from engine)
    # The engine splits by time, so we need to identify the test period
    # For simplicity, we'll use the last 15% of the dataset as the test set
    test_start_idx = int(len(dataset_labeled) * 0.85)
    test_dataset = dataset_labeled.iloc[test_start_idx:]
    
    logger.info(
        "experiment.test_set",
        start=test_dataset.index.min().isoformat(),
        end=test_dataset.index.max().isoformat(),
        rows=len(test_dataset),
    )
    
    sim_config = SimulatorConfig(
        start_balance=1000.0,
        commission=0.0004,  # Binance Futures
    )
    
    # Strategies
    agent_config = SingleAgentConfig(
        min_confidence=0.6,
        max_daily_trades=10,
    )
    single_agent = SingleAgentStrategy(engine, agent_config, sim_config)
    
    baseline = Stage6Strategy(sim_config)
    
    # Run comparison
    comparison = StrategyComparison(sim_config)
    result = comparison.compare(
        test_dataset,
        {
            "SingleAgent": single_agent,
            "Baseline (Stage6)": baseline,
        }
    )
    
    # 4. Report
    print("\n" + result.summary())
    
    # Verify DoD Metrics
    agent_metrics = result.metrics["SingleAgent"]
    baseline_metrics = result.metrics["Baseline (Stage6)"]
    
    sharpe_ok = agent_metrics.sharpe_ratio >= 1.0
    pnl_ok = agent_metrics.total_return_pct > 0
    better_than_baseline = agent_metrics.total_return_pct > baseline_metrics.total_return_pct
    
    logger.info(
        "experiment.dod_verification",
        sharpe_ok=sharpe_ok,
        pnl_ok=pnl_ok,
        better_than_baseline=better_than_baseline,
        agent_sharpe=agent_metrics.sharpe_ratio,
        agent_return=agent_metrics.total_return_pct,
        baseline_return=baseline_metrics.total_return_pct,
    )
    
    if sharpe_ok and pnl_ok and better_than_baseline:
        print("\n✅ DoD Criteria MET!")
    else:
        print("\n❌ DoD Criteria NOT MET")


if __name__ == "__main__":
    run_experiment()
