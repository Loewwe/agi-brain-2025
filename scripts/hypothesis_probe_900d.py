#!/usr/bin/env python3
"""
Alpha Research v4.0 - Stage 2: Probe Run (900 Days)

Tests H001, H002, H003, H009 on BTC and ETH for the period 2022-01-01 to 2024-09-12.
Implements strict "Kill-Switch" to fail early if strategies are not robust.

Usage:
  python scripts/hypothesis_probe_900d.py
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import ccxt
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
PROBE_START = date(2022, 1, 1)
PROBE_END = date(2024, 9, 12)  # Before Stage 1 Train period
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

CAPITAL = 1000.0
RISK_PCT = 0.015  # 1.5% risk per trade
MAX_LEVERAGE = 3.0
MAX_POSITIONS = 3

# Costs (Per side)
COMMISSION_RATE = 0.0004  # 0.04%
SLIPPAGE_RATE = 0.0003    # 0.03%
TOTAL_FEE_PCT = (COMMISSION_RATE + SLIPPAGE_RATE) * 2  # Round trip ~0.14%

# ... (ProbeResult and ProbeRunner classes remain the same)

def fetch_data() -> Dict[str, pd.DataFrame]:
    """Fetch 900 days of 1h data for BTC and ETH sequentially (Synchronous)."""
    logger.info(f"Fetching data from {PROBE_START} to {PROBE_END}...")
    
    data = {}
    
    since = int(datetime.combine(PROBE_START, datetime.min.time()).timestamp() * 1000)
    end_ts = int(datetime.combine(PROBE_END, datetime.min.time()).timestamp() * 1000)
    
    for symbol in SYMBOLS:
        logger.info(f"Fetching {symbol}...")
        exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
        
        try:
            all_ohlcv = []
            curr_since = since
            
            while curr_since < end_ts:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=curr_since, limit=1000)
                except Exception as e:
                    logger.warning(f"  Fetch error: {e}, retrying...")
                    time.sleep(5)
                    continue

                if not ohlcv:
                    break
                
                # Filter out candles beyond end_ts
                ohlcv = [x for x in ohlcv if x[0] <= end_ts]
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                curr_since = ohlcv[-1][0] + 1
                
                logger.info(f"  Fetched {len(ohlcv)} candles, total: {len(all_ohlcv)}")
                
                if len(ohlcv) < 1000:
                    break
                
                # Safety
                if len(all_ohlcv) > 30000:
                    break
                    
                time.sleep(1) # Polite delay
            
            if not all_ohlcv:
                logger.warning(f"No data for {symbol}")
                continue
                
            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df.index.name = symbol # Store symbol in index name for retrieval
            
            # Indicators
            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # Vol MA
            df["vol_ma"] = df["volume"].rolling(20).mean()
            
            data[symbol] = df
            logger.info(f"  ✅ Loaded {len(df)} candles for {symbol}")
            
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
        # finally:
        #     exchange.close() # Sync ccxt doesn't need close()
            
    return data

def main():
    # 1. Load Data
    data = fetch_data()
    if not data:
        logger.error("Failed to load data. Aborting.")
        return

    # 2. Hypotheses
    hypotheses = [
        H001_AsianPump(),
        H002_EuropeanOpenBreakout(),
        H003_US_PreMarket_Dip(),
        H009_RSI_Divergence_Bull()
    ]
    
    all_results = []
    
    # 3. Run Probe
    logger.info("\nStarting Probe Run...")
    for hyp in hypotheses:
        runner = ProbeRunner(hyp)
        results = runner.run(data)
        for res in results:
            res.symbol = res.symbol # Ensure symbol is set
            all_results.append(res)
            logger.info(f"  {res.hyp_id} on {res.symbol}: PF={res.profit_factor:.2f}, CAGR={res.cagr:.1%}, DD={res.max_dd:.1%}, Trades={res.trades}")

    # 4. Kill-Switch Analysis
    logger.info("\n--- KILL-SWITCH ANALYSIS ---")
    
    # Condition 1: All 4 hypotheses (on both symbols) have PF < 1.0 and CAGR <= 0
    all_failed = all(r.profit_factor < 1.0 and r.cagr <= 0 for r in all_results)
    
    # Condition 2: Best PF < 1.2 and CAGR < 10%
    best_pf = max(r.profit_factor for r in all_results) if all_results else 0
    best_cagr = max(r.cagr for r in all_results) if all_results else -1.0
    best_failed = best_pf < 1.2 and best_cagr < 0.10
    
    # Condition 3: Avg PF < 1.0
    avg_pf = np.mean([r.profit_factor for r in all_results]) if all_results else 0
    avg_failed = avg_pf < 1.0
    
    verdict = "PASSED"
    reasons = []
    
    if all_failed:
        verdict = "FAILED"
        reasons.append("All strategies failed (PF < 1.0)")
    elif best_failed:
        verdict = "FAILED"
        reasons.append(f"Best strategy too weak (PF={best_pf:.2f}, CAGR={best_cagr:.1%})")
    elif avg_failed:
        verdict = "FAILED"
        reasons.append(f"Average PF too low ({avg_pf:.2f})")
        
    logger.info(f"VERDICT: {verdict}")
    if reasons:
        logger.info(f"Reasons: {', '.join(reasons)}")
        
    # 5. Save Report
    report_path = Path("lab/results/alpha_research_v4/probe_900d_results.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Probe Run Results (900 Days)\n\n")
        f.write(f"**Period:** {PROBE_START} to {PROBE_END}\n")
        f.write(f"**Verdict:** {verdict}\n\n")
        
        f.write("| Hypothesis | Symbol | PF | CAGR | MaxDD | WinRate | Trades |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in all_results:
            f.write(f"| {r.hyp_id} | {r.symbol} | {r.profit_factor:.2f} | {r.cagr:.1%} | {r.max_dd:.1%} | {r.winrate:.1%} | {r.trades} |\n")
            
        if reasons:
            f.write(f"\n**Failure Reasons:**\n")
            for reason in reasons:
                f.write(f"- {reason}\n")
                
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
