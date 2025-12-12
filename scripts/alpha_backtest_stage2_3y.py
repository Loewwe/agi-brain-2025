#!/usr/bin/env python3
"""
Alpha Search v2.0 - Stage 2: 3-Year Realistic Backtest

Tests Stage 1 survivors on 3-year period with realistic conditions:
- Period: 2022-01-01 to 2024-12-31
- Time window: 16:00-00:00 KZT (11:00-19:00 UTC)
- Realistic costs: 0.14% round-trip
- Risk management: daily stop -8%, max DD -35%
- Criteria: CAGR ≥15-20%, WR ≥45%, PF ≥1.3

Usage:
  python alpha_backtest_stage2_3y.py --input stage1_survivors.csv
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, time, timedelta
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


# Time window configuration
KZT = pytz.timezone('Asia/Almaty')
UTC = pytz.UTC
TRADING_WINDOW_KZT_START = time(16, 0)  # 16:00 KZT
TRADING_WINDOW_KZT_END = time(23, 59)   # 00:00 KZT (midnight)

# Risk parameters
STARTING_CAPITAL = 1000.0  # USDT
MAX_LEVERAGE = 3.0
RISK_PER_TRADE = 0.015  # 1.5%
MAX_POSITION_NOTIONAL = 300.0  # $
MAX_OPEN_POSITIONS = 3
MAX_DAILY_LOSS_PCT = 0.08  # -8%
MAX_TOTAL_DD_PCT = 0.35    # -35%

# Costs
COMMISSION_PER_SIDE = 0.0004  # 0.04%
SLIPPAGE_PER_SIDE = 0.0003    # 0.03%
TOTAL_COST_PER_SIDE = COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE  # 0.07%
ROUND_TRIP_COST = TOTAL_COST_PER_SIDE * 2  # 0.14%

# Symbols (liquid futures)
SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "OP/USDT:USDT",
    "NEAR/USDT:USDT",
    "APT/USDT:USDT",
    "DOT/USDT:USDT",
]


@dataclass
class Stage2Metrics:
    """Metrics for Stage 2 backtest."""
    alpha_id: int
    period_start: str
    period_end: str
    starting_capital: float
    final_capital: float
    total_return_pct: float
    cagr: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    num_trades: int
    num_winning_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    total_commission: float
    killed_by_daily_stop: bool
    killed_by_total_dd: bool
    pass_stage2: bool


def is_in_trading_window(timestamp: pd.Timestamp) -> bool:
    """Check if timestamp is within trading window (16:00-00:00 KZT)."""
    if timestamp.tzinfo is None:
        # Assume UTC if naive
        timestamp = timestamp.tz_localize(UTC)
    
    # Convert to KZT
    kzt_time = timestamp.astimezone(KZT)
    current_time = kzt_time.time()
    
    # 16:00 - 23:59:59 KZT
    return TRADING_WINDOW_KZT_START <= current_time <= TRADING_WINDOW_KZT_END


async def load_3y_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load 3 years of data for backtesting."""
    logger.info(f"Loading 3 years of data for {len(symbols)} symbols...")
    
    cache_dir = Path(__file__).parent.parent / "data" / "backtest_cache"
    fetcher = DataFetcher(cache_dir)
    
    # 3 years = ~1095 days
    data = await fetcher.fetch_all(symbols, days=1095, timeframe="5m")
    
    logger.info(f"✅ Loaded {len(data)} symbols")
    return data


def run_realistic_backtest_3y(config: dict, data: Dict[str, pd.DataFrame]) -> Stage2Metrics:
    """
    Run 3-year realistic backtest for single alpha configuration.
    
    Args:
        config: Alpha configuration from Stage 1
        data: 3-year OHLCV data
    
    Returns:
        Stage2Metrics with all results
    """
    alpha_id = config['id']
    
    # Extract params
    leverage = min(config.get('leverage', 2.0), MAX_LEVERAGE)
    sl_pct = config['sl_pct']
    tp_pct = config['tp_pct']
    rsi_period = config['rsi_period']
    rsi_oversold = config['rsi_oversold']
    rsi_overbought = config['rsi_overbought']
    volume_surge_min = config['volume_surge_min']
    
    # State
    equity = STARTING_CAPITAL
    peak_equity = equity
    positions = {}
    trades = []
    daily_results = {}
    total_commission = 0.0
    
    killed_by_daily_stop = False
    killed_by_total_dd = False
    
    # Process all bars by timestamp
    all_bars = []
    for symbol, df in data.items():
        df = df.copy()
        
        # Add indicators
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
        logger.warning(f"No data for alpha {alpha_id}")
        return Stage2Metrics(
            alpha_id=alpha_id,
            period_start="",
            period_end="",
            starting_capital=STARTING_CAPITAL,
            final_capital=equity,
            total_return_pct=0,
            cagr=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            profit_factor=0,
            win_rate=0,
            num_trades=0,
            num_winning_trades=0,
            avg_trade_pnl=0,
            avg_win=0,
            avg_loss=0,
            max_consecutive_losses=0,
            total_commission=0,
            killed_by_daily_stop=False,
            killed_by_total_dd=False,
            pass_stage2=False
        )
    
    period_start = all_bars[0][0]
    period_end = all_bars[-1][0]
    
    # Track daily equity
    current_day = None
    day_start_equity = equity
    
    # Process bars
    for timestamp, symbol, row in all_bars:
        # Check kill conditions
        if killed_by_daily_stop or killed_by_total_dd:
            break
        
        # Daily tracking
        day = timestamp.date()
        if current_day is None or day != current_day:
            if current_day is not None:
                # End of day
                daily_pnl_pct = (equity - day_start_equity) / day_start_equity
                daily_results[str(current_day)] = {
                    'equity': equity,
                    'pnl_pct': daily_pnl_pct
                }
                
                # Check daily stop
                if daily_pnl_pct < -MAX_DAILY_LOSS_PCT:
                    logger.info(f"Alpha {alpha_id} killed by daily stop on {current_day}: {daily_pnl_pct*100:.2f}%")
                    killed_by_daily_stop = True
                    break
            
            current_day = day
            day_start_equity = equity
        
        # Check total DD
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity
        if dd < -MAX_TOTAL_DD_PCT:
            logger.info(f"Alpha {alpha_id} killed by total DD: {dd*100:.2f}%")
            killed_by_total_dd = True
            break
        
        # Time window filter
        if not is_in_trading_window(timestamp):
            continue
        
        close = row['close']
        high = row['high']
        low = row['low']
        rsi = row.get('rsi', 50)
        vol_surge = row.get('vol_surge', 1.0)
        
        # Check exits
        if symbol in positions:
            pos = positions[symbol]
            exit_price = None
            
            if pos['side'] == 'LONG':
                if low <= pos['sl']:
                    exit_price = pos['sl']
                elif high >= pos['tp']:
                    exit_price = pos['tp']
            else:  # SHORT
                if high >= pos['sl']:
                    exit_price = pos['sl']
                elif low <= pos['tp']:
                    exit_price = pos['tp']
            
            if exit_price:
                # Calculate PnL
                if pos['side'] == 'LONG':
                    price_change = (exit_price - pos['entry']) / pos['entry']
                else:
                    price_change = (pos['entry'] - exit_price) / pos['entry']
                
                pnl_gross = pos['size'] * price_change * pos['leverage']
                
                # Costs
                notional = pos['size'] * pos['leverage']
                commission = notional * ROUND_TRIP_COST
                pnl_net = pnl_gross - commission
                
                equity += pnl_net
                total_commission += commission
                
                trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': pos['side'],
                    'pnl': pnl_net,
                    'commission': commission
                })
                
                del positions[symbol]
        
        # Check entries
        if symbol not in positions and len(positions) < MAX_OPEN_POSITIONS:
            signal = None
            
            # Simple RSI + volume strategy
            if rsi < rsi_oversold and vol_surge >= volume_surge_min:
                signal = 'LONG'
            elif rsi > rsi_overbought and vol_surge >= volume_surge_min:
                signal = 'SHORT'
            
            if signal:
                # Position sizing
                risk_amount = equity * RISK_PER_TRADE
                size = min(risk_amount, equity * 0.33, MAX_POSITION_NOTIONAL / leverage)
                
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
    if current_day is not None and not (killed_by_daily_stop or killed_by_total_dd):
        daily_pnl_pct = (equity - day_start_equity) / day_start_equity
        daily_results[str(current_day)] = {
            'equity': equity,
            'pnl_pct': daily_pnl_pct
        }
    
    # Calculate metrics
    if not trades:
        logger.warning(f"Alpha {alpha_id} had no trades")
        return Stage2Metrics(
            alpha_id=alpha_id,
            period_start=str(period_start.date()),
            period_end=str(period_end.date()),
            starting_capital=STARTING_CAPITAL,
            final_capital=equity,
            total_return_pct=0,
            cagr=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            profit_factor=0,
            win_rate=0,
            num_trades=0,
            num_winning_trades=0,
            avg_trade_pnl=0,
            avg_win=0,
            avg_loss=0,
            max_consecutive_losses=0,
            total_commission=total_commission,
            killed_by_daily_stop=killed_by_daily_stop,
            killed_by_total_dd=killed_by_total_dd,
            pass_stage2=False
        )
    
    # Returns
    total_return = equity - STARTING_CAPITAL
    total_return_pct = total_return / STARTING_CAPITAL * 100
    
    # CAGR
    years = (period_end - period_start).days / 365.25
    cagr = ((equity / STARTING_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Drawdown
    max_dd = 0
    peak = STARTING_CAPITAL
    for day_data in daily_results.values():
        eq = day_data['equity']
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)
    
    max_dd_pct = abs(max_dd) * 100
    
    # Trade metrics
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    num_winning = len(winning_trades)
    win_rate = num_winning / len(trades) * 100
    
    gross_profit = sum(t['pnl'] for t in winning_trades)
    gross_loss = abs(sum(t['pnl'] for t in losing_trades))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    
    avg_trade_pnl = np.mean([t['pnl'] for t in trades])
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    # Consecutive losses
    current_streak = 0
    max_streak = 0
    for t in trades:
        if t['pnl'] < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    # Sharpe
    daily_returns = [d['pnl_pct'] for d in daily_results.values()]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    
    # Pass criteria
    pass_stage2 = (
        cagr >= 15.0 and
        max_dd_pct <= 30.0 and
        pf >= 1.3 and
        win_rate >= 45.0 and
        len(trades) >= 150 and
        not killed_by_daily_stop and
        not killed_by_total_dd
    )
    
    return Stage2Metrics(
        alpha_id=alpha_id,
        period_start=str(period_start.date()),
        period_end=str(period_end.date()),
        starting_capital=STARTING_CAPITAL,
        final_capital=round(equity, 2),
        total_return_pct=round(total_return_pct, 2),
        cagr=round(cagr, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        sharpe_ratio=round(sharpe, 2),
        profit_factor=round(pf, 2),
        win_rate=round(win_rate, 1),
        num_trades=len(trades),
        num_winning_trades=num_winning,
        avg_trade_pnl=round(avg_trade_pnl, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        max_consecutive_losses=max_streak,
        total_commission=round(total_commission, 2),
        killed_by_daily_stop=killed_by_daily_stop,
        killed_by_total_dd=killed_by_total_dd,
        pass_stage2=pass_stage2
    )


async def run_stage2(input_csv: Path, output_dir: Path, limit: int = None):
    """Run Stage 2 on all Stage 1 survivors."""
    logger.info("\n" + "="*60)
    logger.info("ALPHA SEARCH v2.0 - STAGE 2: 3-YEAR REALISTIC BACKTEST")
    logger.info("="*60)
    
    # Load Stage 1 survivors
    survivors_df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(survivors_df)} survivors from Stage 1")
    
    # Filter for Stage 2
    # Note: For testing with Stage 0 data, we skip metric filters
    if 'passed_stage1' in survivors_df.columns:
        # Real Stage 1 output
        stage2_candidates = survivors_df[
            (survivors_df['passed_stage1'] == True) &
            (survivors_df['total_return_pct'] > 0) &
            (survivors_df['winrate_trades'] >= 45) &
            (survivors_df['profit_factor'] >= 1.2) &
            (survivors_df['num_trades'] >= 30)
        ]
    else:
        # Testing with Stage 0 data - use all
        logger.warning("Using Stage 0 data - skipping metric filters")
        stage2_candidates = survivors_df
    
    logger.info(f"After Stage 2 pre-filter: {len(stage2_candidates)} candidates")
    
    # Limit for testing
    if limit:
        stage2_candidates = stage2_candidates.head(limit)
        logger.info(f"Limited to {len(stage2_candidates)} for testing")
    
    # Load 3-year data
    data = await load_3y_data(SYMBOLS)
    
    # Run backtests
    results = []
    for idx, row in stage2_candidates.iterrows():
        config = row.to_dict()
        
        logger.info(f"Testing alpha {config['id']} ({idx+1}/{len(stage2_candidates)})...")
        
        metrics = run_realistic_backtest_3y(config, data)
        results.append(asdict(metrics))
        
        if metrics.pass_stage2:
            logger.info(f"  ✅ PASS - CAGR: {metrics.cagr}%, WR: {metrics.win_rate}%")
        else:
            logger.info(f"  ❌ FAIL - CAGR: {metrics.cagr}%, MaxDD: {metrics.max_drawdown_pct}%")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    output_path = output_dir / "stage2_summary.csv"
    results_df.to_csv(output_path, index=False)
    
    # Summary
    passed = results_df[results_df['pass_stage2'] == True]
    
    logger.info("\n" + "="*60)
    logger.info("STAGE 2 COMPLETE")
    logger.info("="*60)
    logger.info(f"Tested: {len(results)} alphas")
    logger.info(f"Passed: {len(passed)} ({len(passed)/len(results)*100:.1f}%)")
    logger.info(f"\nOutput: {output_path}")
    
    if len(passed) > 0:
        logger.info("\nTop 5 by CAGR:")
        top5 = passed.nlargest(5, 'cagr')[['alpha_id', 'cagr', 'max_drawdown_pct', 'win_rate', 'profit_factor']]
        logger.info("\n" + top5.to_string(index=False))
    
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Alpha Search Stage 2 - 3Y Backtest")
    parser.add_argument("--input", default="reports/alpha_search_v2/stage1_results.csv", 
                       help="Stage 1 results CSV")
    parser.add_argument("--output-dir", default="reports/alpha_search_v2/stage2", 
                       help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of alphas for testing")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input
    output_dir = script_dir / args.output_dir
    
    asyncio.run(run_stage2(input_path, output_dir, limit=args.limit))


if __name__ == "__main__":
    main()
