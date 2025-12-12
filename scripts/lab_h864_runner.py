
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import sys
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hypothesis_mass import H864_Scalper
from scripts.mass_screening_runner import calculate_metrics, load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_lab_h864():
    logger.info("Starting H_#864_SCALPER Lab Mode...")
    
    # Load Data
    data_map = load_data()
    if not data_map:
        logger.error("No data loaded.")
        return

    # Select Data (5m for scalper)
    tf = "5m"
    run_data = {}
    for sym, tfs in data_map.items():
        if tf in tfs:
            run_data[sym] = tfs[tf]
            
    if not run_data:
        logger.error(f"No {tf} data found for Lab.")
        return
        
    logger.info(f"Running on {len(run_data)} symbols: {list(run_data.keys())}")
    for sym, df in run_data.items():
        logger.info(f"  {sym}: {len(df)} rows (Start: {df.index[0]}, End: {df.index[-1]})")
    
    # Run H864
    strategy = H864_Scalper()
    events = strategy.find_triggers(run_data)
    
    # Metrics
    start_date = date(2025, 1, 1)
    end_date = date(2025, 12, 12)
    metrics = calculate_metrics(events, start_date, end_date)
    
    logger.info("="*40)
    logger.info("LAB RESULTS: H_#864_SCALPER")
    logger.info("="*40)
    logger.info(f"Total Trades: {metrics['total_events']}")
    logger.info(f"Win Rate:     {metrics['win_rate']:.1%}")
    logger.info(f"Profit Factor:{metrics['profit_factor']:.2f}")
    logger.info(f"Avg Return:   {metrics['avg_return']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Daily Freq:   {metrics['daily_frequency']:.1f}")
    logger.info("="*40)
    
    # Save Lab Report
    with open("results/mass_screening_2025/H864_LAB_REPORT.txt", "w") as f:
        f.write(str(metrics))

if __name__ == "__main__":
    run_lab_h864()
