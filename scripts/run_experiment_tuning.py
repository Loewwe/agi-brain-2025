"""
Experiment Tuning Script.

Implements the 4-step tuning process:
1. Pure Alpha Check (No SL/TP)
2. SL/TP Impact Analysis
3. Threshold Sweep
4. Multi-period Verification
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
from src.experiment.comparison import StrategyComparison
from src.experiment.labels import LabelConfig, LabelsBuilder
from src.experiment.simulator import Simulator

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
    
    await market_log.ensure_data(
        symbols=symbols,
        timeframe="5m",
        date_from=date_from,
        date_to=date_to,
        fetch_missing=True,
    )
    
    return market_log


def get_dataset():
    """Get or build dataset."""
    market_log = asyncio.run(ensure_data())
    builder = DatasetBuilder(market_log, output_path="data/datasets")
    
    config = DatasetConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 6, 1),
        timeframe="5m",
    )
    
    dataset, _ = builder.build(config, save=False)
    
    # Labeling
    label_config = LabelConfig(
        horizon_bars=3,  # Shorten to 3 bars (15m)
        target_type="direction",
        use_log_returns=True,
    )
    labels_builder = LabelsBuilder(label_config)
    dataset_labeled = labels_builder.add_labels(dataset)
    dataset = dataset_labeled.dropna(subset=["future_return"])
    
    print("\n--- Dataset Stats ---")
    if "adx" in dataset.columns:
        print(dataset["adx"].describe())
    else:
        print("WARNING: 'adx' column missing from dataset!")
    print("---------------------\n")
    
    return dataset_labeled


def train_model(dataset):
    """Train AlphaEngine."""
    alpha_config = AlphaConfig(
        model_type="lightgbm",
        target_column="target",
        train_ratio=0.7,
        val_ratio=0.15,
    )
    
    engine = AlphaEngine(alpha_config)
    engine.train(dataset)
    return engine


def run_tuning():
    """Run tuning experiments."""
    print("\n=== Starting Tuning Process ===\n")
    
    # 1. Prepare Data & Model
    print("Loading data and training model...")
    dataset = get_dataset()
    engine = train_model(dataset)
    
    # Use test set (last 15%)
    test_start_idx = int(len(dataset) * 0.85)
    test_dataset = dataset.iloc[test_start_idx:]
    
    print(f"Test set: {test_dataset.index.min()} to {test_dataset.index.max()} ({len(test_dataset)} rows)")
    
    # Base configs
    base_sim_config = SimulatorConfig(
        start_balance=1000.0,
        commission=0.0004,
        slippage_pct=0.0002,
    )
    
    base_agent_config = SingleAgentConfig(
        min_confidence=0.6,
        max_daily_trades=100,  # High limit for pure alpha check
    )
    
    comparison = StrategyComparison(base_sim_config)
    
    # =========================================================================
    # Step 1: Pure Alpha Check
    # =========================================================================
    # print("\n--- Step 1: Pure Alpha Check ---")
    
    # # Disable SL/TP, rely on timeout (horizon)
    # pure_sim_config = SimulatorConfig(
    #     start_balance=1000.0,
    #     commission=0.0004,
    #     sl_max_pct=1.0,       # 100% (effectively disabled)
    #     tp_multiplier=100.0,  # Huge (effectively disabled)
    #     timeout_bars=12,      # Exit after 1h (horizon)
    # )
    
    # pure_agent_config = SingleAgentConfig(
    #     min_confidence=0.6,
    #     max_daily_trades=100,
    #     use_trailing=False,
    # )
    
    # pure_strategy = SingleAgentStrategy(engine, pure_agent_config, pure_sim_config)
    
    # res_pure = comparison.compare(test_dataset, {"PureAlpha": pure_strategy})
    # print(res_pure.summary())
    
    # =========================================================================
    # Step 2: SL/TP Impact Analysis
    # =========================================================================
    # --- Step 2: SL/TP Impact Analysis ---
    # print("\n--- Step 2: SL/TP Impact Analysis ---")
    
    # strategies_step2 = {}
    
    # # Variant A: TP-only
    # config_tp = SimulatorConfig(
    #     start_balance=1000.0,
    #     commission=0.0004,
    #     sl_max_pct=1.0,       # No SL
    #     tp_multiplier=2.0,    # 2 ATR TP
    #     timeout_bars=12,
    # )
    # strategies_step2["TP_Only"] = SingleAgentStrategy(engine, base_agent_config, config_tp)
    
    # # Variant B: SL-only
    # config_sl = SimulatorConfig(
    #     start_balance=1000.0,
    #     commission=0.0004,
    #     sl_max_pct=0.01,      # 1% SL
    #     atr_multiplier=1.5,   # 1.5 ATR SL
    #     tp_multiplier=100.0,  # No TP
    #     timeout_bars=12,
    # )
    # strategies_step2["SL_Only"] = SingleAgentStrategy(engine, base_agent_config, config_sl)
    
    # # Variant C: Soft SL/TP
    # config_soft = SimulatorConfig(
    #     start_balance=1000.0,
    #     commission=0.0004,
    #     sl_max_pct=0.02,      # 2% SL (wider)
    #     atr_multiplier=2.0,   # 2 ATR SL (wider)
    #     tp_multiplier=3.0,    # 3 ATR TP
    #     timeout_bars=12,
    # )
    # strategies_step2["Soft_SL_TP"] = SingleAgentStrategy(engine, base_agent_config, config_soft)
    
    # res_step2 = comparison.compare(test_dataset, strategies_step2)
    # print(res_step2.summary())
    
    # =========================================================================
    # Step 3: Threshold Sweep
    # =========================================================================
    # --- Step 3: Threshold Sweep ---
    # print("\n--- Step 3: Threshold Sweep ---")
    
    # thresholds = [0.55, 0.60, 0.65, 0.70]
    # strategies_step3 = {}
    
    # # Use best config from Step 2 (assuming Soft SL/TP for now, or Pure if others fail)
    # # Let's use Soft SL/TP as base
    # sweep_sim_config = config_soft
    
    # for th in thresholds:
    #     cfg = SingleAgentConfig(
    #         min_confidence=th,
    #         max_daily_trades=20,
    #     )
    #     strategies_step3[f"Th_{th}"] = SingleAgentStrategy(engine, cfg, sweep_sim_config)
    
    # res_step3 = comparison.compare(test_dataset, strategies_step3)
    # print(res_step3.summary())
    
    # =========================================================================
    # Step 4: Multi-period Verification
    # =========================================================================
    # --- Step 4: Multi-period Verification ---
    # print("\n--- Step 4: Multi-period Verification ---")
    
    # # Define periods
    # # Q1: Jan-Mar (Bullish/Choppy)
    # # Q2: Apr-May (Test set)
    
    # # We need to split the full dataset
    # # Jan-Mar
    # period1_mask = (dataset.index >= "2024-01-01") & (dataset.index < "2024-04-01")
    # period1_data = dataset[period1_mask]
    
    # # Apr-May (Test set used above)
    # period2_data = test_dataset
    
    # # Best strategy from Step 3 (let's pick Th_0.60 as candidate)
    # best_agent_config = SingleAgentConfig(min_confidence=0.60, max_daily_trades=20)
    # # best_sim_config = config_soft
    
    # # best_strategy = SingleAgentStrategy(engine, best_agent_config, best_sim_config)
    
    # print(f"Testing Best Config on Period 1 (Jan-Mar): {len(period1_data)} rows")
    # res_p1 = comparison.compare(period1_data, {"Best_P1": best_strategy})
    # print(res_p1.summary())
    
    # print("\n--- Step 4: Multi-period Verification ---")
    
    # # Define periods
    # # Q1: Jan-Mar (Bullish/Choppy)
    # # Q2: Apr-May (Test set)
    
    # # We need to split the full dataset
    # # Jan-Mar
    # period1_mask = (dataset.index >= "2024-01-01") & (dataset.index < "2024-04-01")
    # period1_data = dataset[period1_mask]
    
    # # Apr-May (Test set used above)
    # period2_data = test_dataset
    
    # # Best strategy from Step 3 (let's pick Th_0.60 as candidate)
    # best_agent_config = SingleAgentConfig(min_confidence=0.60, max_daily_trades=20)
    # best_sim_config = config_soft
    
    # best_strategy = SingleAgentStrategy(engine, best_agent_config, best_sim_config)
    
    # print(f"Testing Best Config on Period 1 (Jan-Mar): {len(period1_data)} rows")
    # res_p1 = comparison.compare(period1_data, {"Best_P1": best_strategy})
    # print(res_p1.summary())
    
    # print(f"Testing Best Config on Period 2 (Apr-May): {len(period2_data)} rows")
    # res_p2 = comparison.compare(period2_data, {"Best_P2": best_strategy})
    # print(res_p2.summary())


    # Step 3b: Regime Filter Verification (EMA Trend)
    print("\n--- Step 3b: Regime Filter Verification (EMA Trend) ---")
    
    sim_config = SimulatorConfig(
        start_balance=1000.0,
        commission=0.0004,
        slippage_pct=0.0002,
        timeout_bars=6,
    )
    
    STEP_3B_CONFIGS = [
        # 1. Baseline (No Filter)
        SingleAgentConfig(
            min_confidence=0.65,
            max_daily_trades=20,
            trend_filter_type="NONE"
        ),
        # 2. EMA Trend Filter
        SingleAgentConfig(
            min_confidence=0.65,
            max_daily_trades=20,
            trend_filter_type="EMA"
        ),
    ]
    
    results_3b = []
    for agent_config in STEP_3B_CONFIGS:
        # Create strategy
        strategy = SingleAgentStrategy(agent_config, engine)
        
        # Run simulation
        simulator = Simulator(sim_config)
        result = simulator.run(test_dataset, strategy)
        
        # Store result
        name = f"Filter_{agent_config.trend_filter_type}"
        results_3b.append({
            "strategy": name,
            "return_pct": result.metrics.total_return_pct,
            "trades": result.metrics.total_trades,
            "sharpe": result.metrics.sharpe_ratio,
            "max_drawdown": result.metrics.max_drawdown_pct,
            "win_rate": result.metrics.win_rate
        })
        logger.info("comparison.completed", strategy=name, return_pct=result.metrics.total_return_pct, trades=result.metrics.total_trades)

    # Print Summary
    print("\nStrategy Comparison Summary")
    print("=" * 40)
    for res in results_3b:
        print(f"\n{res['strategy']}:")
        print(f"  Total Return: {res['return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {res['sharpe']:.2f}")
        print(f"  Win Rate: {res['win_rate']*100:.1f}%")
        print(f"  Max Drawdown: {res['max_drawdown']:.2f}%")
        print(f"  Trades: {res['trades']}")
    print("\n" + "=" * 40)
    
    # Identify best
    best_res = max(results_3b, key=lambda x: x["return_pct"])
    print(f"BEST: {best_res['strategy']} (total_return_pct: {best_res['return_pct']:.2f})")
    
    # reversal_config = SingleAgentConfig(
    #     min_confidence=0.65,
    #     exit_on_reversal=True,
    #     reversal_threshold=0.5,
    
    # --- Step 5: Reversal Exit Verification ---
    # print("\n--- Step 5: Reversal Exit Verification ---")
    # # The original snippet had `f.write` and `run_simulation` which are not defined.
    # # Assuming the intent was to print and then run a simulation using the existing comparison object.
    # # Also, `alpha_engine` was used instead of `engine`. Correcting these.
    
    # reversal_config = SingleAgentConfig(
    #     min_confidence=0.65,
    #     exit_on_reversal=True,
    #     reversal_threshold=0.5,
    #     max_daily_trades=20
    # )
    
    # # Assuming a simulation function or direct comparison call is intended here.
    # # For now, let's integrate it with the existing comparison object.
    # # If `run_simulation` is a custom function, it needs to be defined elsewhere.
    # # For this edit, I'll adapt it to use the existing `comparison` object.
    
    # # Define a simulator config for this step. Using a base one for now.
    # reversal_sim_config = SimulatorConfig(
    #     start_balance=1000.0,
    #     commission=0.0004,
    #     slippage_pct=0.0002,
    #     timeout_bars=6,  # Shorten timeout to match 3-bar horizon (approx 2x)
    # )
    
    # reversal_strategy = SingleAgentStrategy(engine, reversal_config, reversal_sim_config)
    
    # print(f"Testing Reversal Exit Strategy on Test Dataset: {len(test_dataset)} rows")
    # res_reversal = comparison.compare(test_dataset, {"Reversal_Exit_0.65": reversal_strategy})
    # print(res_reversal.summary())


if __name__ == "__main__":
    run_tuning()

