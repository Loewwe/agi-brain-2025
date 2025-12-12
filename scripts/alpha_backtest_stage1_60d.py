#!/usr/bin/env python3
"""
Alpha Search v2.0 - Stage 1: 60-Day Ranking Backtest

Fast 60-day screening of all Stage 0 configs with ranking by score.
Outputs ranked list for Stage 2 selection.

Usage:
  python alpha_backtest_stage1_60d.py --input stage0_passed.csv
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List
import argparse

import pandas as pd
import numpy as np
import pytz

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest_stage6 import DataFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Time window
KZT = pytz.timezone('Asia/Almaty')
UTC = pytz.UTC
TRADING_WINDOW_KZT_START = time(16, 0)
TRADING_WINDOW_KZT_END = time(23, 59)

# Trading params
STARTING_CAPITAL = 1000.0
MAX_LEVERAGE = 3.0
RISK_PER_TRADE = 0.015
MAX_POSITION_NOTIONAL = 300.0
MAX_OPEN_POSITIONS = 3

# Costs
COMMISSION_PER_SIDE = 0.0004
SLIPPAGE_PER_SIDE = 0.0003
TOTAL_COST_PER_SIDE = COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE

# Symbols
SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "OP/USDT:USDT",
    "NEAR/USDT:USDT",
    "APT/USDT:USDT",
    "DOT/USDT:USDT",
]


@dataclass
class Stage1Metrics:
    """Metrics for Stage 1 backtest."""
    alpha_id: int
    total_return_pct_60d: float
    avg_daily_return_pct: float
    winrate_trades: float
    profit_factor: float
    max_drawdown_pct: float
    trades_count: int
    sharpe_approx: float
    score_stage1: float
    is_garbage: bool
    is_candidate: bool
    rank_stage1: int


def is_in_trading_window(timestamp: pd.Timestamp) -> bool:
    """Check if timestamp is within trading window."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    kzt_time = timestamp.astimezone(KZT)
    current_time = kzt_time.time()
    return TRADING_WINDOW_KZT_START <= current_time <= TRADING_WINDOW_KZT_END


async def load_60d_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load 60 days of data."""
    logger.info(f"Loading 60 days of data for {len(symbols)} symbols...")
    cache_dir = Path(__file__).parent.parent / "data" / "backtest_cache"
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(symbols, days=60, timeframe="5m")
    logger.info(f"✅ Loaded {len(data)} symbols")
    return data


def run_stage1_backtest(config: dict, data: Dict[str, pd.DataFrame]) -> Stage1Metrics:
    """Run 60-day backtest for single alpha."""
    alpha_id = config['id']
    
    # Extract params
    leverage = min(config.get('leverage', 2.0), MAX_LEVERAGE)
    sl_pct = config['sl_pct']
    tp_pct = config['tp_pct']
    rsi_period = config['rsi_period']
    rsi_oversold = config['rsi_oversold']
    rsi_overbought = config['rsi_overbought']
    volume_surge_min = config.get('volume_surge_min', 1.2)
    
    # State
    equity = STARTING_CAPITAL
    peak_equity = equity
    positions = {}
    trades = []
    daily_results = {}
    
    # Process bars
    all_bars = []
    for symbol, df in data.items():
        df = df.copy()
        
        # Indicators
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
    
    if not all_bars:
        return _empty_metrics(alpha_id)
    
    # Track daily
    current_day = None
    day_start_equity = equity
    
    for timestamp, symbol, row in all_bars:
        # Daily tracking
        day = timestamp.date()
        if current_day is None or day != current_day:
            if current_day is not None:
                daily_pnl_pct = (equity - day_start_equity) / day_start_equity
                daily_results[str(current_day)] = {
                    'equity': equity,
                    'pnl_pct': daily_pnl_pct
                }
            current_day = day
            day_start_equity = equity
        
        # Time window filter
        if not is_in_trading_window(timestamp):
            continue
        
        close = row['close']
        high = row['high']
        low = row['low']
        rsi = row.get('rsi', 50)
        vol_surge = row.get('vol_surge', 1.0)
        
        # Exits
        if symbol in positions:
            pos = positions[symbol]
            exit_price = None
            
            if pos['side'] == 'LONG':
                if low <= pos['sl']:
                    exit_price = pos['sl']
                elif high >= pos['tp']:
                    exit_price = pos['tp']
            else:
                if high >= pos['sl']:
                    exit_price = pos['sl']
                elif low <= pos['tp']:
                    exit_price = pos['tp']
            
            if exit_price:
                if pos['side'] == 'LONG':
                    price_change = (exit_price - pos['entry']) / pos['entry']
                else:
                    price_change = (pos['entry'] - exit_price) / pos['entry']
                
                pnl_gross = pos['size'] * price_change * pos['leverage']
                notional = pos['size'] * pos['leverage']
                commission = notional * TOTAL_COST_PER_SIDE * 2
                pnl_net = pnl_gross - commission
                
                equity += pnl_net
                peak_equity = max(peak_equity, equity)
                trades.append({'pnl': pnl_net, 'timestamp': timestamp})
                del positions[symbol]
        
        # Entries
        if symbol not in positions and len(positions) < MAX_OPEN_POSITIONS:
            signal = None
            
            if rsi < rsi_oversold and vol_surge >= volume_surge_min:
                signal = 'LONG'
            elif rsi > rsi_overbought and vol_surge >= volume_surge_min:
                signal = 'SHORT'
            
            if signal:
                size = min(equity * RISK_PER_TRADE, equity * 0.33, MAX_POSITION_NOTIONAL / leverage)
                
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
                    'leverage': leverage,
                    'sl': sl,
                    'tp': tp,
                }
    
    # Final day
    if current_day is not None:
        daily_pnl_pct = (equity - day_start_equity) / day_start_equity
        daily_results[str(current_day)] = {'equity': equity, 'pnl_pct': daily_pnl_pct}
    
    # Calculate metrics
    if not trades:
        return _empty_metrics(alpha_id)
    
    total_return = equity - STARTING_CAPITAL
    total_return_pct = total_return / STARTING_CAPITAL * 100
    avg_daily_return_pct = total_return_pct / 60
    
    wins = sum(1 for t in trades if t['pnl'] > 0)
    winrate = wins / len(trades) * 100
    
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    
    # DD
    peak = STARTING_CAPITAL
    max_dd = 0
    for day_data in daily_results.values():
        eq = day_data['equity']
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)
    max_dd_pct = abs(max_dd) * 100
    
    # Sharpe
    daily_returns = [d['pnl_pct'] for d in daily_results.values()]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    
    # Score
    score = (
        total_return_pct
        * max(pf, 0)
        * max(0, 1 - max_dd_pct / 100)
        * math.log(1 + len(trades))
    )
    
    # Flags
    is_garbage = (total_return_pct < -30 and pf < 0.8)
    is_candidate = (not is_garbage and len(trades) >= 50)
    
    return Stage1Metrics(
        alpha_id=alpha_id,
        total_return_pct_60d=round(total_return_pct, 2),
        avg_daily_return_pct=round(avg_daily_return_pct, 3),
        winrate_trades=round(winrate, 1),
        profit_factor=round(pf, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        trades_count=len(trades),
        sharpe_approx=round(sharpe, 2),
        score_stage1=round(score, 2),
        is_garbage=is_garbage,
        is_candidate=is_candidate,
        rank_stage1=0  # Will be assigned after sorting
    )


def _empty_metrics(alpha_id: int) -> Stage1Metrics:
    """Return empty metrics for failed backtest."""
    return Stage1Metrics(
        alpha_id=alpha_id,
        total_return_pct_60d=0,
        avg_daily_return_pct=0,
        winrate_trades=0,
        profit_factor=0,
        max_drawdown_pct=0,
        trades_count=0,
        sharpe_approx=0,
        score_stage1=0,
        is_garbage=True,
        is_candidate=False,
        rank_stage1=0
    )


async def run_stage1(input_csv: Path, output_csv: Path):
    """Run Stage 1 on all configs."""
    logger.info("\n" + "="*60)
    logger.info("ALPHA SEARCH v2.0 - STAGE 1: 60-DAY RANKING")
    logger.info("="*60)
    
    # Load configs
    configs_df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(configs_df)} configs from Stage 0")
    
    configs = configs_df.to_dict('records')
    
    # Load data
    data = await load_60d_data(SYMBOLS)
    
    # Run backtests
    logger.info(f"Testing {len(configs)} configs...")
    results = []
    
    for i, config in enumerate(configs):
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i+1}/{len(configs)}...")
        
        try:
            metrics = run_stage1_backtest(config, data)
            results.append(asdict(metrics))
        except Exception as e:
            logger.error(f"Error processing config {config['id']}: {e}")
            results.append(asdict(_empty_metrics(config['id'])))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original configs
    full_df = configs_df.merge(results_df, left_on='id', right_on='alpha_id', how='left')
    
    # Rank candidates
    candidates = full_df[full_df['is_candidate'] == True].copy()
    candidates = candidates.sort_values('score_stage1', ascending=False)
    candidates['rank_stage1'] = range(1, len(candidates) + 1)
    
    # Update ranks in full df
    full_df.loc[full_df['is_candidate'] == True, 'rank_stage1'] = candidates['rank_stage1'].values
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_csv, index=False)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("STAGE 1 COMPLETE")
    logger.info("="*60)
    logger.info(f"Total configs: {len(full_df)}")
    logger.info(f"Candidates: {len(candidates)}")
    logger.info(f"Garbage: {sum(full_df['is_garbage'])}")
    logger.info(f"\nOutput: {output_csv}")
    
    if len(candidates) > 0:
        logger.info("\nTop 10 by score:")
        top10 = candidates.head(10)[['alpha_id', 'total_return_pct_60d', 'winrate_trades', 'profit_factor', 'score_stage1']]
        logger.info("\n" + top10.to_string(index=False))
    
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Alpha Search Stage 1 - 60D Ranking")
    parser.add_argument("--input", default="reports/alpha_search_v2/stage0_passed.csv")
    parser.add_argument("--output", default="reports/alpha_search_v2/stage1_results_ranked.csv")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    
    asyncio.run(run_stage1(input_path, output_path))


if __name__ == "__main__":
    main()
