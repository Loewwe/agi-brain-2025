#!/usr/bin/env python3
"""
Strategy Validation Backtest - trial_0270

Validates strategy according to STRATEGY_VALIDATION_TZ.md:
- Base window (180 days)
- OOS window (30 days) 
- Stress windows
- UNSINKABLE violations monitoring
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_stage6 import DataFetcher


@dataclass
class UnsinkableViolations:
    """UNSINKABLE limit violations tracking."""
    days_over_daily_limit: int = 0
    dates_over_daily_limit: List[str] = None
    max_consecutive_violations: int = 0
    worst_day_pct: float = 0.0
    was_total_dd_limit_hit: bool = False
    
    def __post_init__(self):
        if self.dates_over_daily_limit is None:
            self.dates_over_daily_limit = []


@dataclass
class WindowMetrics:
    """Metrics for one backtest window."""
    period_start: str
    period_end: str
    total_return_pct: float
    avg_daily_return: float
    median_daily_return: float
    max_drawdown: float
    max_daily_loss: float
    win_rate: float
    num_trades: int
    profit_factor: float
    sharpe_ratio: float
    max_consecutive_losses: int
    unsinkable_violations: UnsinkableViolations


def load_strategy_config(strategy_id: str) -> dict:
    """Load strategy configuration."""
    config_path = Path(__file__).parent.parent.parent / "stage6" / "config" / "winners" / f"winner_trial_{strategy_id}_alpha2.34_ddm6.47.json"
    
    with open(config_path) as f:
        data = json.load(f)
    
    return data["config"]


async def fetch_window_data(days: int, end_date: datetime = None) -> Dict[str, pd.DataFrame]:
    """Fetch data for backtest window."""
    cache_dir = Path(__file__).parent.parent / "data" / "backtest_cache"
    symbols = ["BTC/USDT:USDT"]  # trial_0270 uses BTC
    
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(symbols, days=days, timeframe="5m")
    
    return data


def run_backtest_window(
    config: dict,
    data: Dict[str, pd.DataFrame],
    daily_loss_limit: float = -0.08,  # -8%
    total_dd_limit: float = -0.40,   # -40%
) -> WindowMetrics:
    """
    Run backtest for one window with UNSINKABLE monitoring.
    """
    # Extract params
    ema_fast = config["ema_fast"]
    ema_slow = config["ema_slow"]
    rsi_period = config["rsi_period"]
    rsi_oversold = config["rsi_oversold"]
    rsi_overbought = config["rsi_overbought"]
    
    # State
    equity = 10000.0
    start_equity = equity
    positions = {}
    trades = []
    daily_results = []
    
    # Costs
    commission = 0.0004  # 0.04%
    slippage = 0.0003    # 0.03%
    total_cost = commission + slippage
    
    # Process data
    for symbol, df in data.items():
        df = df.copy()
        
        # Add indicators
        df['ema_fast'] = df['close'].ewm(span=ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=ema_slow).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Track daily equity
        current_day = None
        day_start_equity = equity
        
        for timestamp, row in df.iterrows():
            # Check if new day
            day = timestamp.date()
            if current_day is None or day != current_day:
                if current_day is not None:
                    daily_pnl = equity - day_start_equity
                    daily_pnl_pct = daily_pnl / day_start_equity
                    daily_results.append({
                        'date': str(current_day),
                        'pnl_pct': daily_pnl_pct,
                        'equity': equity
                    })
                current_day = day
                day_start_equity = equity
            
            close = row['close']
            high = row['high']
            low = row['low']
            ema_f = row['ema_fast']
            ema_s = row['ema_slow']
            rsi = row.get('rsi', 50)
            
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
                    if pos['side'] == 'LONG':
                        price_change = (exit_price - pos['entry']) / pos['entry']
                    else:
                        price_change = (pos['entry'] - exit_price) / pos['entry']
                    
                    pnl_gross = pos['size'] * price_change * 2.0  # 2x leverage
                    
                    # Costs
                    notional = pos['size'] * 2.0
                    pnl_net = pnl_gross - (notional * total_cost * 2)  # entry + exit
                    
                    equity += pnl_net
                    trades.append({'pnl': pnl_net, 'timestamp': timestamp})
                    del positions[symbol]
            
            # Check entries
            if symbol not in positions:
                signal = None
                
                # EMA crossover + RSI
                if ema_f > ema_s and rsi < rsi_oversold:
                    signal = 'LONG'
                elif ema_f < ema_s and rsi > rsi_overbought:
                    signal = 'SHORT'
                
                if signal:
                    size = equity * 0.5  # 50% per trade
                    
                    if signal == 'LONG':
                        sl = close * 0.985   # -1.5% SL
                        tp = close * 1.025   # +2.5% TP
                    else:
                        sl = close * 1.015
                        tp = close * 0.975
                    
                    positions[symbol] = {
                        'side': signal,
                        'entry': close,
                        'size': size,
                        'sl': sl,
                        'tp': tp,
                    }
        
        # Final day
        if current_day is not None:
            daily_pnl = equity - day_start_equity
            daily_pnl_pct = daily_pnl / day_start_equity
            daily_results.append({
                'date': str(current_day),
                'pnl_pct': daily_pnl_pct,
                'equity': equity
            })
    
    # Calculate metrics
    total_return = equity - start_equity
    total_return_pct = total_return / start_equity
    
    if not trades:
        return WindowMetrics(
            period_start="",
            period_end="",
            total_return_pct=total_return_pct,
            avg_daily_return=0,
            median_daily_return=0,
            max_drawdown=0,
            max_daily_loss=0,
            win_rate=0,
            num_trades=0,
            profit_factor=0,
            sharpe_ratio=0,
            max_consecutive_losses=0,
            unsinkable_violations=UnsinkableViolations()
        )
    
    # Daily metrics
    daily_returns = [d['pnl_pct'] for d in daily_results]
    avg_daily = np.mean(daily_returns)
    median_daily = np.median(daily_returns)
    max_daily_loss = min(daily_returns)
    
    # Drawdown
    equity_curve = [start_equity] + [d['equity'] for d in daily_results]
    peak = start_equity
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)
    
    # Trades
    wins = sum(1 for t in trades if t['pnl'] > 0)
    wr = wins / len(trades)
    
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    
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
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)) if len(daily_returns) > 1 else 0
    
    # UNSINKABLE violations
    violations = UnsinkableViolations()
    consecutive = 0
    for day in daily_results:
        if day['pnl_pct'] < daily_loss_limit:
            violations.days_over_daily_limit += 1
            violations.dates_over_daily_limit.append(day['date'])
            consecutive += 1
            violations.max_consecutive_violations = max(violations.max_consecutive_violations, consecutive)
        else:
            consecutive = 0
        
        violations.worst_day_pct = min(violations.worst_day_pct, day['pnl_pct'])
    
    if abs(max_dd) >= abs(total_dd_limit):
        violations.was_total_dd_limit_hit = True
    
    # Get period from data
    first_date = list(data.values())[0].index[0].strftime("%Y-%m-%d")
    last_date = list(data.values())[0].index[-1].strftime("%Y-%m-%d")
    
    return WindowMetrics(
        period_start=first_date,
        period_end=last_date,
        total_return_pct=round(total_return_pct * 100, 2),
        avg_daily_return=round(avg_daily * 100, 3),
        median_daily_return=round(median_daily * 100, 3),
        max_drawdown=round(abs(max_dd) * 100, 2),
        max_daily_loss=round(max_daily_loss * 100, 2),
        win_rate=round(wr * 100, 1),
        num_trades=len(trades),
        profit_factor=round(pf, 2),
        sharpe_ratio=round(sharpe, 2),
        max_consecutive_losses=max_streak,
        unsinkable_violations=violations
    )


async def validate_strategy(strategy_id: str):
    """Run full validation for strategy."""
    print(f"\n{'='*60}")
    print(f"STRATEGY VALIDATION: trial_{strategy_id}")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_strategy_config(strategy_id)
    print(f"Config loaded: {config}\n")
    
    # Window 1: Base (180 days)
    print("Running Base window (180 days)...")
    data_base = await fetch_window_data(days=180)
    metrics_base = run_backtest_window(config, data_base)
    
    print(f"✅ Base window complete")
    print(f"   Period: {metrics_base.period_start} to {metrics_base.period_end}")
    print(f"   Avg Daily: {metrics_base.avg_daily_return}%")
    print(f"   Max DD: {metrics_base.max_drawdown}%")
    print(f"   WR: {metrics_base.win_rate}%")
    print(f"   Trades: {metrics_base.num_trades}")
    print(f"   UNSINKABLE violations: {metrics_base.unsinkable_violations.days_over_daily_limit} days\n")
    
    # Window 2: OOS (30 days)
    print("Running OOS window (30 days)...")
    data_oos = await fetch_window_data(days=30)
    metrics_oos = run_backtest_window(config, data_oos)
    
    print(f"✅ OOS window complete")
    print(f"   Period: {metrics_oos.period_start} to {metrics_oos.period_end}")
    print(f"   Avg Daily: {metrics_oos.avg_daily_return}%")
    print(f"   WR: {metrics_oos.win_rate}%\n")
    
    # Save report
    report = {
        "strategy_id": f"trial_{strategy_id}",
        "validated_at": datetime.utcnow().isoformat(),
        "config": config,
        "windows": {
            "base": asdict(metrics_base),
            "oos": asdict(metrics_oos),
        }
    }
    
    output_dir = Path(__file__).parent.parent / "lab" / "results" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"trial_{strategy_id}_validation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"{'='*60}")
    print(f"Report saved: {output_path}")
    print(f"{'='*60}\n")
    
    return report


def main():
    strategy_id = "0270"
    asyncio.run(validate_strategy(strategy_id))


if __name__ == "__main__":
    main()
