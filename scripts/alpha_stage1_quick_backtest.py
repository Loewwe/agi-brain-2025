#!/usr/bin/env python3
"""
Alpha Search v2.0 - Stage 1: Quick Backtest

Processes 16,974 configs from Stage 0.
Quick backtest: 60 days, 3-4 symbols, simplified costs.
Output: ~500-800 survivors for Stage 2.

Usage:
  python alpha_stage1_quick_backtest.py --workers 2 --max-configs 1000
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_stage6 import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Quick symbols for Stage 1
QUICK_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "APT/USDT:USDT",
    "OP/USDT:USDT",
]


@dataclass
class QuickBacktestResult:
    """Quick backtest metrics."""
    config_id: int
    total_return_pct: float
    avg_daily_return_pct: float
    max_dd_pct: float
    num_trades: int
    trades_per_day: float
    winrate_trades: float
    profit_factor: float
    avg_trade_pnl_pct: float
    passed_stage1: bool
    rejection_reason: str


def quick_backtest_single(config: dict, data: Dict[str, pd.DataFrame]) -> QuickBacktestResult:
    """
    Run quick backtest for single config.
    Simplified version without full Stage6 complexity.
    """
    config_id = config['id']
    
    # Extract params
    leverage = config['leverage']
    sl_pct = config['sl_pct']
    tp_pct = config['tp_pct']
    rsi_period = config['rsi_period']
    rsi_oversold = config['rsi_oversold']
    rsi_overbought = config['rsi_overbought']
    volume_surge_min = config['volume_surge_min']
    
    # Costs
    commission_per_side = 0.0005  # 0.05%
    slippage_per_side = 0.0003    # 0.03%
    total_cost_per_side = commission_per_side + slippage_per_side
    
    # State
    equity = 10000.0
    start_equity = equity
    positions = {}
    trades = []
    
    # Get all timestamps
    all_bars = []
    for symbol, df in data.items():
        # Add indicators
        df = df.copy()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_surge'] = df['volume'] / df['vol_ma']
        
        for timestamp, row in df.iterrows():
            all_bars.append((timestamp, symbol, row))
    
    all_bars.sort(key=lambda x: x[0])
    
    # Process bars
    for timestamp, symbol, row in all_bars:
        close = row['close']
        high = row['high']
        low = row['low']
        rsi = row.get('rsi', 50)
        vol_surge = row.get('vol_surge', 1.0)
        
        # Check exits
        if symbol in positions:
            pos = positions[symbol]
            exit_price = None
            exit_reason = None
            
            if pos['side'] == 'LONG':
                if low <= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'
                elif high >= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'
            else:  # SHORT
                if high >= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'
                elif low <= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'
            
            if exit_price:
                # Calculate PnL
                if pos['side'] == 'LONG':
                    price_change = (exit_price - pos['entry']) / pos['entry']
                else:
                    price_change = (pos['entry'] - exit_price) / pos['entry']
                
                pnl_gross = pos['size'] * price_change * leverage
                
                # Costs
                notional = pos['size'] * leverage
                entry_cost = notional * total_cost_per_side
                exit_cost = notional * total_cost_per_side
                pnl_net = pnl_gross - entry_cost - exit_cost
                
                equity += pnl_net
                
                trades.append({
                    'pnl': pnl_net,
                    'pnl_pct': pnl_net / pos['size'] * 100,
                })
                
                del positions[symbol]
        
        # Check entries (max 3 positions)
        if symbol not in positions and len(positions) < 3:
            signal = None
            
            # Simple momentum logic
            if rsi < rsi_oversold and vol_surge >= volume_surge_min:
                signal = 'LONG'
            elif rsi > rsi_overbought and vol_surge >= volume_surge_min:
                signal = 'SHORT'
            
            if signal:
                size = equity * 0.33  # 33% per position
                
                if signal == 'LONG':
                    sl = close * (1 - sl_pct / 100)
                    tp = close * (1 + tp_pct / 100)
                else:
                    sl = close * (1 + sl_pct / 100)
                    tp = close * (1 - tp_pct / 100)
                
                positions[symbol] = {
                    'side': signal,
                    'entry': close,
                    'size': size,
                    'sl': sl,
                    'tp': tp,
                }
    
    # Calculate metrics
    total_return = equity - start_equity
    total_return_pct = total_return / start_equity * 100
    
    num_trades = len(trades)
    if num_trades == 0:
        return QuickBacktestResult(
            config_id=config_id,
            total_return_pct=total_return_pct,
            avg_daily_return_pct=0,
            max_dd_pct=0,
            num_trades=0,
            trades_per_day=0,
            winrate_trades=0,
            profit_factor=0,
            avg_trade_pnl_pct=0,
            passed_stage1=False,
            rejection_reason="no_trades"
        )
    
    # More metrics
    wins = sum(1 for t in trades if t['pnl'] > 0)
    winrate = wins / num_trades * 100
    
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_trade_pnl_pct = np.mean([t['pnl_pct'] for t in trades])
    
    # Estimate days (30 days)
    days = 30
    trades_per_day = num_trades / days
    avg_daily_return = total_return_pct / days
    
    # Rough DD estimate
    equity_curve = [start_equity]
    running_equity = start_equity
    for t in trades:
        running_equity += t['pnl']
        equity_curve.append(running_equity)
    
    peak = start_equity
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (eq - peak) / peak * 100
        max_dd = min(max_dd, dd)
    
    # Stage 1 filter
    passed = True
    reason = "passed"
    
    if num_trades < 30:
        passed = False
        reason = "too_few_trades"
    elif trades_per_day > 15 and profit_factor < 1.1:
        passed = False
        reason = "overtrading_low_pf"
    elif total_return_pct < -10:
        passed = False
        reason = "high_loss"
    elif max_dd < -30:
        passed = False
        reason = "high_dd"
    elif winrate < 40:
        passed = False
        reason = "low_winrate"
    
    return QuickBacktestResult(
        config_id=config_id,
        total_return_pct=round(total_return_pct, 2),
        avg_daily_return_pct=round(avg_daily_return, 3),
        max_dd_pct=round(max_dd, 2),
        num_trades=num_trades,
        trades_per_day=round(trades_per_day, 2),
        winrate_trades=round(winrate, 1),
        profit_factor=round(profit_factor, 2),
        avg_trade_pnl_pct=round(avg_trade_pnl_pct, 2),
        passed_stage1=passed,
        rejection_reason=reason
    )


async def fetch_data_once(days: int = 30) -> Dict[str, pd.DataFrame]:
    """Fetch data once for all backtests."""
    logger.info(f"Fetching {days} days of data for {len(QUICK_SYMBOLS)} symbols...")
    
    root = Path(__file__).parent.parent
    cache_dir = root / "data" / "backtest_cache"
    
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(QUICK_SYMBOLS, days=days, timeframe="5m")
    
    logger.info(f"✅ Data loaded: {len(data)} symbols")
    return data


def process_config_wrapper(args):
    """Wrapper for multiprocessing."""
    config, data = args
    try:
        result = quick_backtest_single(config, data)
        return result
    except Exception as e:
        logger.error(f"Error processing config {config['id']}: {e}")
        return None


async def run_stage1(
    max_configs: Optional[int] = None,
    max_workers: int = 2
):
    """Run Stage 1 on all configs."""
    
    # Load configs
    root = Path(__file__).parent.parent
    input_dir = root / "reports" / "alpha_search_v2"
    configs_df = pd.read_csv(input_dir / "stage0_passed.csv")
    
    if max_configs:
        configs_df = configs_df.head(max_configs)
    
    configs = configs_df.to_dict('records')
    logger.info(f"Loaded {len(configs)} configs from Stage 0")
    
    # Fetch data once
    data = await fetch_data_once(days=30)
    
    # Run backtests in parallel
    logger.info(f"Starting Stage 1 with {max_workers} workers...")
    start_time = time.time()
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for config in configs:
            futures.append(executor.submit(process_config_wrapper, (config, data)))
        
        for i, future in enumerate(futures):
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i+1}/{len(configs)}...")
            
            result = future.result()
            if result:
                results.append(result)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Stage 1 complete in {elapsed/60:.1f} minutes")
    
    # Save results
    results_df = pd.DataFrame([vars(r) for r in results])
    output_path = input_dir / "stage1_results.csv"
    results_df.to_csv(output_path, index=False)
    
    # Stats
    passed = results_df[results_df['passed_stage1'] == True]
    
    print("\n" + "="*60)
    print("ALPHA SEARCH v2.0 - STAGE 1 COMPLETE")
    print("="*60)
    print(f"Processed: {len(results)} configs")
    print(f"Passed:    {len(passed)} ({len(passed)/len(results)*100:.1f}%)")
    print(f"Time:      {elapsed/60:.1f} minutes")
    print(f"\nOutput: {output_path}")
    
    # Rejection breakdown
    print("\nRejection reasons:")
    rejection_counts = results_df[results_df['passed_stage1'] == False]['rejection_reason'].value_counts()
    for reason, count in rejection_counts.items():
        print(f"  {reason}: {count}")
    
    # Top performers
    print("\nTop 10 by total_return_pct:")
    top10 = results_df.nlargest(10, 'total_return_pct')[['config_id', 'total_return_pct', 'winrate_trades', 'profit_factor', 'num_trades']]
    print(top10.to_string(index=False))
    
    print("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Alpha Search v2.0 - Stage 1")
    parser.add_argument("--max-configs", type=int, default=None, help="Max configs to process (for testing)")
    parser.add_argument("--workers", type=int, default=2, help="Max workers")
    args = parser.parse_args()
    
    asyncio.run(run_stage1(
        max_configs=args.max_configs,
        max_workers=args.workers
    ))


if __name__ == "__main__":
    main()
