
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import sys
import pickle
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hypothesis_mass import H864_Scalper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "backtest_cache" / "mass_screening_2025"

def load_data_local():
    print("DEBUG: Loading data from", DATA_DIR)
    data = {}
    if not DATA_DIR.exists():
        print("ERROR: Data dir does not exist.")
        return {}
        
    files = list(DATA_DIR.glob("*.pkl"))
    print(f"DEBUG: Found {len(files)} files.")
    
    for f in files:
        parts = f.stem.split('_')
        if len(parts) < 4: continue
        
        symbol = f"{parts[0]}/{parts[1]}"
        tf = parts[2]
        
        if symbol not in data:
            data[symbol] = {}
            
        with open(f, "rb") as pf:
            df = pickle.load(pf)
            data[symbol][tf] = df
            
    return data

def calculate_metrics_local(events, start_date, end_date):
    if not events:
        return {"total_events": 0}
        
    events.sort(key=lambda x: x.timestamp)
    
    capital = 1000.0
    equity = capital
    peak_equity = capital
    max_dd = 0.0
    equity_curve = [capital]
    
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    FEES = 0.0014 # 0.14% round trip
    
    for e in events:
        pos_size = equity 
        raw_ret = e.rew_pct / 100.0
        net_ret = raw_ret - FEES
        
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
    
    days = (end_date - start_date).days
    years = max(days / 365.0, 0.01)
    cagr = (equity / capital) ** (1 / years) - 1 if equity > 0 else -1.0
    
    return {
        "total_events": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": (equity - capital) / capital / total_trades if total_trades > 0 else 0,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "daily_frequency": total_trades / days if days > 0 else 0
    }

def run_lab_h864():
    print("Starting H_#864_SCALPER Lab Mode (Self-Contained)...")
    
    data_map = load_data_local()
    
    # Select Data (5m)
    tf = "5m"
    run_data = {}
    for sym, tfs in data_map.items():
        if tf in tfs:
            run_data[sym] = tfs[tf]
            
    if not run_data:
        print(f"ERROR: No {tf} data found.")
        return
        
    print(f"Running on {len(run_data)} symbols: {list(run_data.keys())}")
    for sym, df in run_data.items():
        print(f"  {sym}: {len(df)} rows")
    
    # Run H864
    strategy = H864_Scalper()
    events = strategy.find_triggers(run_data)
    
    print(f"Generated {len(events)} events.")
    
    # Metrics
    start_date = date(2025, 1, 1)
    end_date = date(2025, 12, 12)
    metrics = calculate_metrics_local(events, start_date, end_date)
    
    print("="*40)
    print("LAB RESULTS: H_#864_SCALPER")
    print("="*40)
    print(f"Total Trades: {metrics['total_events']}")
    if metrics['total_events'] > 0:
        print(f"Win Rate:     {metrics['win_rate']:.1%}")
        print(f"Profit Factor:{metrics['profit_factor']:.2f}")
        print(f"Avg Return:   {metrics['avg_return']:.2%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Daily Freq:   {metrics['daily_frequency']:.1f}")
    print("="*40)
    
    with open("results/mass_screening_2025/H864_LAB_REPORT.txt", "w") as f:
        f.write(str(metrics))

if __name__ == "__main__":
    run_lab_h864()
