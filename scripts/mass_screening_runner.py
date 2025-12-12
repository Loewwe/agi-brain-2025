

print("DEBUG: Top of file")
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import sys
import importlib
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import hypothesis base and implementations
# We will dynamically import from scripts.hypothesis_mass
from scripts.hypothesis_stage1 import Hypothesis, HypothesisEvent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path(__file__).parent.parent / "data" / "backtest_cache" / "mass_screening_2025"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "mass_screening_2025"
CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "all_hypotheses.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "raw_metrics").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "aggregated").mkdir(parents=True, exist_ok=True)

def load_catalog():
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)

def load_data():
    data = {}
    if not DATA_DIR.exists():
        logger.error(f"Data dir {DATA_DIR} does not exist.")
        return {}
        
    files = list(DATA_DIR.glob("*.pkl"))
    logger.info(f"Found {len(files)} data files in {DATA_DIR}")
    
    for f in files:
        # Filename format: SYMBOL_TF_2025.pkl (e.g. BTC_USDT_5m_2025.pkl)
        # We want to map it to "BTC/USDT" -> "5m" -> df
        parts = f.stem.split('_')
        # Assuming standard format: BTC_USDT_5m_2025
        # Symbol is parts[0] + "/" + parts[1]
        # TF is parts[2]
        
        if len(parts) < 4:
            logger.warning(f"Skipping weird file: {f.name}")
            continue
            
        symbol = f"{parts[0]}/{parts[1]}"
        tf = parts[2]
        
        if symbol not in data:
            data[symbol] = {}
            
        with open(f, "rb") as pf:
            df = pickle.load(pf)
            data[symbol][tf] = df
            
    return data

def calculate_metrics(events, start_date, end_date):
    if not events:
        return {
            "total_events": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "cagr": 0.0,
            "daily_frequency": 0.0
        }
        
    # Sort
    events.sort(key=lambda x: x.timestamp)
    
    # Simulation
    capital = 1000.0
    equity = capital
    peak_equity = capital
    max_dd = 0.0
    equity_curve = [capital]
    
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    # Costs
    FEES = 0.0014 # 0.14% round trip
    SLIPPAGE = 0.0003 # 0.03% round trip
    TOTAL_COST = FEES + SLIPPAGE
    
    for e in events:
        # Fixed position size 100% equity (compounding) or Fixed Amount?
        # TZ says "position_sizing: fixed" -> let's assume fixed fraction 1.0 for raw alpha test
        # Or better: Fixed $1000 per trade to measure raw pnl?
        # Let's use 100% equity compounding to capture CAGR properly, 
        # but for "raw alpha" fixed size is often better.
        # Let's use Compounding to match Stage 2 logic.
        
        pos_size = equity # Full allocation
        
        raw_ret = e.rew_pct / 100.0
        net_ret = raw_ret - TOTAL_COST
        
        pnl = pos_size * net_ret
        equity += pnl
        equity_curve.append(equity)
        
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)
            
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_dd:
            max_dd = dd
            
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
    
    # CAGR
    days = (end_date - start_date).days
    years = max(days / 365.0, 0.01)
    cagr = (equity / capital) ** (1 / years) - 1 if equity > 0 else -1.0
    
    # Sharpe
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(events)) # Approx trade-based sharpe
        
    return {
        "total_events": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": (equity - capital) / capital / total_trades if total_trades > 0 else 0,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "daily_frequency": total_trades / days if days > 0 else 0
    }

def run_screening_programmatic():
    """Run screening and return results list."""
    print("DEBUG: Starting run_screening_programmatic...")
    logger.info("Starting Mass Screening 2025 (Programmatic)...")
    
    # 1. Load Catalog
    catalog = load_catalog()
    logger.info(f"Loaded {len(catalog)} hypotheses from catalog.")
    
    # 2. Load Data
    data_map = load_data() # {Symbol: {TF: df}}
    if not data_map:
        logger.error("No data loaded. Run fetch_mass_screening_data.py first.")
        return []
        
    # 3. Import Implementations
    try:
        import scripts.hypothesis_mass as hyp_module
    except ImportError:
        logger.error("Could not import scripts.hypothesis_mass")
        return []

    results = []
    
    for item in catalog:
        hyp_id = item["id"]
        name = item["name"]
        
        # Try to find class by ID prefix
        cls = None
        for attr_name in dir(hyp_module):
            if attr_name.startswith(hyp_id + "_") or attr_name == hyp_id:
                cls = getattr(hyp_module, attr_name)
                break
        
        if not cls:
            continue
            
        try:
            hyp_instance = cls()
            
            # Select Data (Default 15m)
            tf = "15m"
            run_data = {}
            for sym, tfs in data_map.items():
                if tf in tfs:
                    run_data[sym] = tfs[tf]
            
            if not run_data:
                continue
                
            # Run
            events = hyp_instance.find_triggers(run_data)
            
            # Calculate Metrics
            start_date = date(2025, 1, 1)
            end_date = date(2025, 12, 12)
            
            metrics = calculate_metrics(events, start_date, end_date)
            
            # Save Raw Metrics
            metrics["id"] = hyp_id
            metrics["name"] = name
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"  Error running {hyp_id}: {e}")
            
    return results

def run_screening():
    """CLI Entry point."""
    results = run_screening_programmatic()
    
    # Save Summary
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_DIR / "aggregated" / "summary.csv", index=False)
        logger.info(f"Saved summary to {RESULTS_DIR / 'aggregated' / 'summary.csv'}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    run_screening()
