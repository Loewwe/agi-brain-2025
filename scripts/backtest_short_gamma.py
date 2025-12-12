#!/usr/bin/env python3
"""
Short-Gamma Mean-Reversion Backtest

Strategy: Short after pump ≥ ENTRY_THRESHOLD% in 4 hours.
Expect mean-reversion to take price back down.

Scenarios:
  - base: 2.5x, 3.7% entry, 1.6% TP, 4.5% SL
  - conservative: 1.0x, 4.5% entry, 1.2% TP, 6.0% SL
  - aggressive: 3.0x, 3.3% entry, 2.0% TP, 4.0% SL
  - nosl: 1.0x, 3.7% entry, 1.6% TP, no SL

Usage:
  python backtest_short_gamma.py --scenario base --days 180
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Literal
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_stage6 import DataFetcher, SYMBOLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SCENARIOS
# =============================================================================

@dataclass
class ShortGammaConfig:
    """Configuration for Short-Gamma strategy."""
    name: str = "base"
    leverage: float = 2.5
    entry_threshold_pct: float = 3.7
    take_profit_pct: float = 1.6
    stop_loss_pct: float = 4.5
    max_positions: int = 7
    lookback_bars: int = 48  # 4 hours in 5m bars
    commission_pct: float = 0.04  # Taker fee
    start_balance: float = 10000.0


SCENARIOS = {
    "base": ShortGammaConfig(
        name="base",
        leverage=2.5,
        entry_threshold_pct=3.7,
        take_profit_pct=1.6,
        stop_loss_pct=4.5,
    ),
    "conservative": ShortGammaConfig(
        name="conservative",
        leverage=1.0,
        entry_threshold_pct=4.5,
        take_profit_pct=1.2,
        stop_loss_pct=6.0,
    ),
    "aggressive": ShortGammaConfig(
        name="aggressive",
        leverage=3.0,
        entry_threshold_pct=3.3,
        take_profit_pct=2.0,
        stop_loss_pct=4.0,
    ),
    "nosl": ShortGammaConfig(
        name="nosl",
        leverage=1.0,
        entry_threshold_pct=3.7,
        take_profit_pct=1.6,
        stop_loss_pct=999.0,  # Effectively no SL
    ),
}


# =============================================================================
# POSITION TRACKING
# =============================================================================

@dataclass
class Position:
    """Open position."""
    symbol: str
    entry_price: float
    entry_time: datetime
    size_usd: float  # Position size in USDT
    leverage: float
    tp_price: float
    sl_price: float
    
    @property
    def notional(self) -> float:
        return self.size_usd * self.leverage


@dataclass
class Trade:
    """Completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_usd: float
    leverage: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str  # TP, SL, TIMEOUT


@dataclass
class DayStats:
    """Daily statistics."""
    date: str
    equity: float
    pnl_usd: float
    pnl_pct: float
    trades: int
    wins: int


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class ShortGammaBacktester:
    """Backtester for Short-Gamma Mean-Reversion strategy."""
    
    def __init__(self, config: ShortGammaConfig):
        self.config = config
        self.equity = config.start_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_stats: List[DayStats] = []
        self.equity_curve: List[float] = [config.start_balance]
        self.peak_equity = config.start_balance
        self.max_dd_pct = 0.0
        
    def calculate_pump_pct(self, df: pd.DataFrame, bar_idx: int) -> float:
        """Calculate pump percentage over lookback period."""
        if bar_idx < self.config.lookback_bars:
            return 0.0
        
        current_price = df.iloc[bar_idx]["close"]
        price_4h_ago = df.iloc[bar_idx - self.config.lookback_bars]["close"]
        
        if price_4h_ago <= 0:
            return 0.0
        
        return (current_price - price_4h_ago) / price_4h_ago * 100
    
    def check_entry(self, symbol: str, pump_pct: float) -> bool:
        """Check if entry conditions are met."""
        # Already in position for this symbol?
        if symbol in self.positions:
            return False
        
        # Max positions reached?
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Pump threshold met?
        if pump_pct < self.config.entry_threshold_pct:
            return False
        
        return True
    
    def open_position(self, symbol: str, price: float, timestamp: datetime):
        """Open a SHORT position."""
        # Size: equal allocation across max positions
        size_per_position = self.equity / self.config.max_positions
        
        # Account for commission (for entry)
        commission = size_per_position * self.config.leverage * self.config.commission_pct / 100
        self.equity -= commission
        
        # Calculate TP/SL prices (inverted for short)
        tp_price = price * (1 - self.config.take_profit_pct / 100)
        sl_price = price * (1 + self.config.stop_loss_pct / 100)
        
        position = Position(
            symbol=symbol,
            entry_price=price,
            entry_time=timestamp,
            size_usd=size_per_position,
            leverage=self.config.leverage,
            tp_price=tp_price,
            sl_price=sl_price,
        )
        
        self.positions[symbol] = position
        logger.debug(f"OPEN SHORT {symbol} @ {price:.4f}, TP={tp_price:.4f}, SL={sl_price:.4f}")
    
    def check_exits(self, symbol: str, high: float, low: float, close: float, timestamp: datetime) -> Optional[Trade]:
        """Check if position should be closed."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        exit_price = None
        exit_reason = None
        
        # For SHORT: TP hit when price goes DOWN to tp_price
        if low <= pos.tp_price:
            exit_price = pos.tp_price
            exit_reason = "TP"
        # SL hit when price goes UP to sl_price
        elif high >= pos.sl_price:
            exit_price = pos.sl_price
            exit_reason = "SL"
        
        if exit_price is None:
            return None
        
        # Calculate PnL for SHORT
        price_change_pct = (pos.entry_price - exit_price) / pos.entry_price * 100
        pnl_pct = price_change_pct * pos.leverage
        pnl_usd = pos.size_usd * pnl_pct / 100
        
        # Account for commission (for exit)
        commission = pos.notional * self.config.commission_pct / 100
        pnl_usd -= commission
        
        # Update equity
        self.equity += pnl_usd
        
        trade = Trade(
            symbol=symbol,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            leverage=pos.leverage,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
        )
        
        del self.positions[symbol]
        self.trades.append(trade)
        
        logger.debug(f"CLOSE {symbol} @ {exit_price:.4f}, PnL={pnl_usd:.2f} ({pnl_pct:.2f}%), reason={exit_reason}")
        
        return trade
    
    def run(self, data: Dict[str, pd.DataFrame]) -> dict:
        """Run backtest on data."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Short-Gamma Backtest: {self.config.name}")
        logger.info(f"Leverage: {self.config.leverage}x, Entry: {self.config.entry_threshold_pct}%")
        logger.info(f"TP: {self.config.take_profit_pct}%, SL: {self.config.stop_loss_pct}%")
        logger.info(f"{'='*60}")
        
        # Get all dates and timestamps
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.date)
        dates = sorted(all_dates)
        
        logger.info(f"Backtesting {len(dates)} days: {dates[0]} to {dates[-1]}")
        
        # Process each day
        for day in dates:
            day_str = day.strftime("%Y-%m-%d")
            day_equity_start = self.equity
            day_trades = 0
            day_wins = 0
            
            # Collect all bars for this day across symbols
            day_bars = []
            for symbol, df in data.items():
                day_df = df[df.index.date == day]
                for idx, (timestamp, row) in enumerate(day_df.iterrows()):
                    bar_idx = df.index.get_loc(timestamp)
                    day_bars.append((timestamp, symbol, row, bar_idx, df))
            
            day_bars.sort(key=lambda x: x[0])
            
            # Process each bar
            for timestamp, symbol, row, bar_idx, df in day_bars:
                # Check exits first
                trade = self.check_exits(symbol, row["high"], row["low"], row["close"], timestamp)
                if trade:
                    day_trades += 1
                    if trade.pnl_usd > 0:
                        day_wins += 1
                
                # Check entry
                pump_pct = self.calculate_pump_pct(df, bar_idx)
                if self.check_entry(symbol, pump_pct):
                    self.open_position(symbol, row["close"], timestamp)
                
                # Track equity curve
                self.equity_curve.append(self.equity)
                
                # Update drawdown
                self.peak_equity = max(self.peak_equity, self.equity)
                current_dd = (self.equity - self.peak_equity) / self.peak_equity * 100
                self.max_dd_pct = min(self.max_dd_pct, current_dd)
            
            # End of day stats
            day_pnl = self.equity - day_equity_start
            day_pnl_pct = day_pnl / day_equity_start * 100 if day_equity_start > 0 else 0
            
            self.daily_stats.append(DayStats(
                date=day_str,
                equity=self.equity,
                pnl_usd=day_pnl,
                pnl_pct=day_pnl_pct,
                trades=day_trades,
                wins=day_wins,
            ))
            
            if day_trades > 0:
                wr = day_wins / day_trades * 100
                logger.info(f"  {day_str}: PnL={day_pnl:+.2f} ({day_pnl_pct:+.2f}%), Trades={day_trades}, WR={wr:.0f}%")
        
        # Calculate final metrics
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> dict:
        """Calculate final performance metrics."""
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        losses = total_trades - wins
        
        total_return = self.equity - self.config.start_balance
        total_return_pct = total_return / self.config.start_balance * 100
        
        winrate = wins / total_trades * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio
        daily_returns = [d.pnl_pct for d in self.daily_stats]
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Best/worst days
        best_day = max(self.daily_stats, key=lambda d: d.pnl_pct) if self.daily_stats else None
        worst_day = min(self.daily_stats, key=lambda d: d.pnl_pct) if self.daily_stats else None
        
        # Max losing streak
        losing_streak = 0
        max_losing_streak = 0
        for d in self.daily_stats:
            if d.pnl_pct < 0:
                losing_streak += 1
                max_losing_streak = max(max_losing_streak, losing_streak)
            else:
                losing_streak = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        # Days in trade
        avg_trade_hours = np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades]) if self.trades else 0
        
        result = {
            "scenario": self.config.name,
            "config": asdict(self.config),
            "summary": {
                "total_return_usd": round(total_return, 2),
                "total_return_pct": round(total_return_pct, 2),
                "max_drawdown_pct": round(self.max_dd_pct, 2),
                "winrate": round(winrate, 1),
                "profit_factor": round(profit_factor, 2),
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "sharpe_ratio": round(sharpe, 2),
                "max_losing_streak_days": max_losing_streak,
                "avg_trade_duration_hours": round(avg_trade_hours, 1),
            },
            "exit_reasons": exit_reasons,
            "best_day": {"date": best_day.date, "pnl_pct": round(best_day.pnl_pct, 2)} if best_day else None,
            "worst_day": {"date": worst_day.date, "pnl_pct": round(worst_day.pnl_pct, 2)} if worst_day else None,
            "pass_criteria": {
                "profit_factor_ge_1.5": profit_factor >= 1.5,
                "max_dd_le_15": self.max_dd_pct >= -15,
                "winrate_ge_65": winrate >= 65,
                "losing_streak_le_5": max_losing_streak <= 5,
            }
        }
        
        # Check overall pass
        result["PASSED"] = all(result["pass_criteria"].values())
        
        return result


# =============================================================================
# MAIN
# =============================================================================

async def run_backtest(scenario: str = "base", days: int = 180):
    """Run backtest for a scenario."""
    
    # Get config
    if scenario not in SCENARIOS:
        logger.error(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
        return
    
    config = SCENARIOS[scenario]
    
    # Paths
    root = Path(__file__).parent.parent
    cache_dir = root / "data" / "backtest_cache"
    output_dir = root / "reports" / "short_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    logger.info(f"Fetching {days} days of data...")
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(SYMBOLS[:15], days=days, timeframe="5m")
    
    if not data:
        logger.error("No data fetched!")
        return
    
    # Run backtest
    backtester = ShortGammaBacktester(config)
    result = backtester.run(data)
    
    # Save result
    result_path = output_dir / f"result_{scenario}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Print summary
    s = result["summary"]
    pc = result["pass_criteria"]
    
    print("\n" + "="*60)
    print(f"SHORT-GAMMA BACKTEST: {scenario.upper()}")
    print("="*60)
    print(f"Total Return:    ${s['total_return_usd']:+.2f} ({s['total_return_pct']:+.1f}%)")
    print(f"Max Drawdown:    {s['max_drawdown_pct']:.1f}%")
    print(f"Win Rate:        {s['winrate']:.1f}%")
    print(f"Profit Factor:   {s['profit_factor']:.2f}")
    print(f"Total Trades:    {s['total_trades']}")
    print(f"Sharpe Ratio:    {s['sharpe_ratio']:.2f}")
    print(f"Max Losing Days: {s['max_losing_streak_days']}")
    print("-"*60)
    print("PASS CRITERIA:")
    for k, v in pc.items():
        status = "✅ PASS" if v else "❌ FAIL"
        print(f"  {k}: {status}")
    print("="*60)
    print(f"OVERALL: {'✅ PASSED' if result['PASSED'] else '❌ FAILED'}")
    print("="*60)
    print(f"Report saved: {result_path}")
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Short-Gamma Mean-Reversion Backtest")
    parser.add_argument("--scenario", type=str, default="base", 
                       choices=list(SCENARIOS.keys()),
                       help="Scenario to test")
    parser.add_argument("--days", type=int, default=180, help="Days to backtest")
    args = parser.parse_args()
    
    asyncio.run(run_backtest(args.scenario, args.days))


if __name__ == "__main__":
    main()
