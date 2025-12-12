#!/usr/bin/env python3
"""
Realistic Backtest 864 + Stage6

3 Scenarios:
  A - Baseline: All filters ON (current Stage6)
  B - Relaxed: Time window + relaxed filters
  C - Max Unlock: Filters OFF, Risk only

Features:
  - Realistic commissions (0.04% per side)
  - Slippage (0.03% per side)
  - Risk management (max trades, daily loss limit)
  - Time window (16:00-00:00 KZT = 11:00-19:00 UTC)

Usage:
  python backtest_stage6_realistic.py --scenario A --days 90
  python backtest_stage6_realistic.py --all --days 90
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_stage6 import DataFetcher, SYMBOLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SCENARIOS CONFIG
# =============================================================================

class Scenario(Enum):
    A_BASELINE = "A_baseline"
    B_RELAXED = "B_relaxed"
    C_FILTERS_OFF = "C_filters_off"


@dataclass
class BacktestConfig:
    """Configuration for realistic backtest."""
    scenario: str = "A_baseline"
    
    # Time window
    trading_window_enabled: bool = False
    trading_window_start_hour_utc: int = 11  # 16:00 KZT
    trading_window_end_hour_utc: int = 19    # 00:00 KZT
    
    # Filters control
    filters_disabled: bool = False      # Scenario C
    relax_filters: bool = False         # Scenario B
    
    # Relaxed filter values (for Scenario B)
    volume_surge_threshold: float = 1.5  # Default: 1.5, Relaxed: 1.2
    rsi_oversold: int = 35               # Default: 35, Relaxed: 40
    ema_dead_zone_pct: float = 0.004     # Default: 0.4%, Relaxed: 0.8%
    enhanced_breakout_enabled: bool = True  # Default: True, Relaxed: False
    
    # Costs
    commission_pct_per_side: float = 0.0004  # 0.04%
    slippage_pct_per_side: float = 0.0003   # 0.03%
    
    # Risk management
    max_open_trades: int = 3
    daily_loss_limit_pct: float = -8.0  # -8%
    max_leverage: float = 2.0
    trading_budget_pct: float = 0.4     # 40% of equity
    
    # Strategy params (from winning_strategy_best.json)
    ema_fast: int = 15
    ema_slow: int = 60
    rsi_period: int = 14
    rsi_overbought: int = 75
    bb_period: int = 25
    bb_std: float = 2.0
    
    # Capital
    start_equity: float = 10000.0


def get_scenario_config(scenario: Scenario) -> BacktestConfig:
    """Get config for specific scenario."""
    if scenario == Scenario.A_BASELINE:
        return BacktestConfig(
            scenario="A_baseline",
            trading_window_enabled=False,  # 24/7
            filters_disabled=False,
            relax_filters=False,
        )
    elif scenario == Scenario.B_RELAXED:
        return BacktestConfig(
            scenario="B_relaxed",
            trading_window_enabled=True,  # 16:00-00:00 KZT
            filters_disabled=False,
            relax_filters=True,
            # Relaxed values
            volume_surge_threshold=1.2,
            rsi_oversold=40,
            ema_dead_zone_pct=0.008,  # 0.8%
            enhanced_breakout_enabled=False,
        )
    elif scenario == Scenario.C_FILTERS_OFF:
        return BacktestConfig(
            scenario="C_filters_off",
            trading_window_enabled=True,  # 16:00-00:00 KZT
            filters_disabled=True,
            relax_filters=False,
            max_open_trades=5,
            max_leverage=3.0,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# =============================================================================
# POSITION & TRADE TRACKING
# =============================================================================

@dataclass
class Position:
    symbol: str
    side: str  # LONG/SHORT
    entry_price: float
    entry_time: datetime
    size_usd: float
    leverage: float
    stop_loss: float
    take_profit: float


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_usd: float
    leverage: float
    pnl_gross: float
    fees: float
    pnl_net: float
    pnl_pct: float
    exit_reason: str


@dataclass
class DayResult:
    date: str
    equity_start: float
    equity_end: float
    pnl_usd: float
    pnl_pct: float
    trades: int
    wins: int
    kill_switch_triggered: bool


# =============================================================================
# REALISTIC BACKTESTER
# =============================================================================

class RealisticBacktester:
    """Backtester with realistic costs and risk management."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.equity = config.start_equity
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_results: List[DayResult] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Tracking
        self.peak_equity = config.start_equity
        self.max_dd_pct = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.kill_switch_days = 0
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators."""
        # EMAs
        df["ema_fast"] = df["close"].ewm(span=self.config.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.config.ema_slow, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(self.config.bb_period).mean()
        df["bb_std"] = df["close"].rolling(self.config.bb_period).std()
        df["bb_upper"] = df["bb_mid"] + self.config.bb_std * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - self.config.bb_std * df["bb_std"]
        
        # Volume
        df["vol_ma"] = df["volume"].rolling(20).mean()
        df["volume_surge"] = df["volume"] / df["vol_ma"]
        
        # High/Low for breakout
        df["high_2"] = df["high"].rolling(2).max()
        df["low_2"] = df["low"].rolling(2).min()
        
        return df
    
    def is_in_trading_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is within trading window."""
        if not self.config.trading_window_enabled:
            return True
        
        hour = timestamp.hour
        return self.config.trading_window_start_hour_utc <= hour < self.config.trading_window_end_hour_utc
    
    def check_entry_signal(self, row: pd.Series, df: pd.DataFrame, bar_idx: int) -> Tuple[bool, str]:
        """Check if entry signal is valid."""
        close = row["close"]
        ema_fast = row["ema_fast"]
        ema_slow = row["ema_slow"]
        rsi = row["rsi"]
        bb_lower = row.get("bb_lower", close)
        volume_surge = row.get("volume_surge", 1.0)
        
        # Scenario C: All filters disabled
        if self.config.filters_disabled:
            # Only basic direction check
            if ema_fast > ema_slow and rsi < 50:
                return True, "LONG"
            elif ema_fast < ema_slow and rsi > 50:
                return True, "SHORT"
            return False, ""
        
        # Get filter thresholds (relaxed or default)
        vol_threshold = self.config.volume_surge_threshold
        rsi_oversold = self.config.rsi_oversold
        rsi_overbought = self.config.rsi_overbought
        dead_zone = self.config.ema_dead_zone_pct
        breakout_check = self.config.enhanced_breakout_enabled
        
        # Check EMA dead zone
        ema_diff_pct = abs(ema_fast - ema_slow) / ema_slow
        if ema_diff_pct < dead_zone:
            return False, ""
        
        # Check volume surge
        if volume_surge < vol_threshold:
            return False, ""
        
        # LONG conditions
        if ema_fast > ema_slow and rsi < rsi_oversold:
            if breakout_check:
                if close <= bb_lower:
                    return True, "LONG"
            else:
                return True, "LONG"
        
        # SHORT conditions
        if ema_fast < ema_slow and rsi > rsi_overbought:
            if breakout_check:
                if close >= row.get("bb_upper", close):
                    return True, "SHORT"
            else:
                return True, "SHORT"
        
        return False, ""
    
    def calculate_fees(self, notional: float, is_entry: bool = True) -> float:
        """Calculate fees + slippage for a trade."""
        commission = notional * self.config.commission_pct_per_side
        slippage = notional * self.config.slippage_pct_per_side
        return commission + slippage
    
    def open_position(self, symbol: str, side: str, price: float, timestamp: datetime):
        """Open a position."""
        if symbol in self.positions:
            return
        if len(self.positions) >= self.config.max_open_trades:
            return
        
        # Calculate size
        budget = self.equity * self.config.trading_budget_pct
        size_per_trade = budget / self.config.max_open_trades
        leverage = self.config.max_leverage
        
        # Entry fees
        notional = size_per_trade * leverage
        entry_fee = self.calculate_fees(notional, is_entry=True)
        self.equity -= entry_fee
        
        # SL/TP: 1.5% SL, 2.5% TP
        sl_pct = 0.015
        tp_pct = 0.025
        
        if side == "LONG":
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        else:
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
        
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            entry_time=timestamp,
            size_usd=size_per_trade,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        self.positions[symbol] = position
    
    def check_exits(self, symbol: str, high: float, low: float, timestamp: datetime) -> Optional[Trade]:
        """Check if position should be closed."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        exit_price = None
        exit_reason = None
        
        if pos.side == "LONG":
            if low <= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = "SL"
            elif high >= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = "TP"
        else:  # SHORT
            if high >= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = "SL"
            elif low <= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = "TP"
        
        if exit_price is None:
            return None
        
        # Calculate PnL
        if pos.side == "LONG":
            price_change_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            price_change_pct = (pos.entry_price - exit_price) / pos.entry_price
        
        pnl_gross = pos.size_usd * pos.leverage * price_change_pct
        
        # Exit fees
        notional = pos.size_usd * pos.leverage
        exit_fee = self.calculate_fees(notional, is_entry=False)
        
        pnl_net = pnl_gross - exit_fee
        pnl_pct = pnl_net / pos.size_usd * 100
        
        # Update equity
        self.equity += pnl_net
        self.daily_pnl += pnl_net
        self.daily_trades += 1
        if pnl_net > 0:
            self.daily_wins += 1
        
        trade = Trade(
            symbol=symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            leverage=pos.leverage,
            pnl_gross=pnl_gross,
            fees=exit_fee,
            pnl_net=pnl_net,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
        )
        
        del self.positions[symbol]
        self.trades.append(trade)
        
        return trade
    
    def run(self, data: Dict[str, pd.DataFrame]) -> dict:
        """Run backtest."""
        logger.info(f"\n{'='*60}")
        logger.info(f"REALISTIC BACKTEST: {self.config.scenario}")
        logger.info(f"{'='*60}")
        logger.info(f"Time window: {'16:00-00:00 KZT' if self.config.trading_window_enabled else '24/7'}")
        logger.info(f"Filters: {'OFF' if self.config.filters_disabled else ('RELAXED' if self.config.relax_filters else 'ON')}")
        logger.info(f"Max trades: {self.config.max_open_trades}, Leverage: {self.config.max_leverage}x")
        logger.info(f"Costs: {(self.config.commission_pct_per_side + self.config.slippage_pct_per_side)*100:.2f}% per side")
        
        # Get dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.date)
        dates = sorted(all_dates)
        
        logger.info(f"Period: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        
        # Add indicators
        for symbol in data:
            data[symbol] = self.add_indicators(data[symbol])
        
        # Process each day
        for day in dates:
            day_str = day.strftime("%Y-%m-%d")
            day_equity_start = self.equity
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            kill_switch = False
            
            # Collect all bars for this day
            day_bars = []
            for symbol, df in data.items():
                day_df = df[df.index.date == day]
                for timestamp, row in day_df.iterrows():
                    bar_idx = df.index.get_loc(timestamp)
                    day_bars.append((timestamp, symbol, row, bar_idx, df))
            
            day_bars.sort(key=lambda x: x[0])
            
            # Process each bar
            for timestamp, symbol, row, bar_idx, df in day_bars:
                # Check daily loss limit
                if self.daily_pnl / day_equity_start * 100 <= self.config.daily_loss_limit_pct:
                    kill_switch = True
                    # Close all positions
                    for pos_symbol in list(self.positions.keys()):
                        self.check_exits(pos_symbol, row["high"], row["low"], timestamp)
                    break
                
                # Check exits
                self.check_exits(symbol, row["high"], row["low"], timestamp)
                
                # Check entries (only in window)
                if self.is_in_trading_window(timestamp):
                    if symbol not in self.positions:
                        has_signal, side = self.check_entry_signal(row, df, bar_idx)
                        if has_signal:
                            self.open_position(symbol, side, row["close"], timestamp)
                
                # Track equity
                self.equity_curve.append((timestamp, self.equity))
                self.peak_equity = max(self.peak_equity, self.equity)
                current_dd = (self.equity - self.peak_equity) / self.peak_equity * 100
                self.max_dd_pct = min(self.max_dd_pct, current_dd)
            
            # End of day stats
            day_pnl = self.equity - day_equity_start
            day_pnl_pct = day_pnl / day_equity_start * 100 if day_equity_start > 0 else 0
            
            if kill_switch:
                self.kill_switch_days += 1
            
            self.daily_results.append(DayResult(
                date=day_str,
                equity_start=day_equity_start,
                equity_end=self.equity,
                pnl_usd=day_pnl,
                pnl_pct=day_pnl_pct,
                trades=self.daily_trades,
                wins=self.daily_wins,
                kill_switch_triggered=kill_switch,
            ))
            
            if self.daily_trades > 0:
                wr = self.daily_wins / self.daily_trades * 100
                logger.info(f"  {day_str}: PnL=${day_pnl:+.2f} ({day_pnl_pct:+.2f}%), Trades={self.daily_trades}, WR={wr:.0f}%{' [KILL]' if kill_switch else ''}")
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> dict:
        """Calculate final metrics."""
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_net > 0)
        losses = total_trades - wins
        
        total_return = self.equity - self.config.start_equity
        total_return_pct = total_return / self.config.start_equity * 100
        
        winrate_trades = wins / total_trades * 100 if total_trades > 0 else 0
        
        # Daily stats
        green_days = sum(1 for d in self.daily_results if d.pnl_pct > 0)
        total_days = len(self.daily_results)
        winrate_days = green_days / total_days * 100 if total_days > 0 else 0
        
        daily_returns = [d.pnl_pct for d in self.daily_results]
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        
        # Sharpe
        sharpe = avg_daily_return / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl_net for t in self.trades if t.pnl_net > 0)
        gross_loss = abs(sum(t.pnl_net for t in self.trades if t.pnl_net < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total fees
        total_fees = sum(t.fees for t in self.trades)
        
        return {
            "scenario": self.config.scenario,
            "config": {
                "filters": "OFF" if self.config.filters_disabled else ("RELAXED" if self.config.relax_filters else "ON"),
                "time_window": self.config.trading_window_enabled,
                "max_trades": self.config.max_open_trades,
                "leverage": self.config.max_leverage,
            },
            "summary": {
                "total_return_usd": round(total_return, 2),
                "total_return_pct": round(total_return_pct, 2),
                "avg_daily_return_pct": round(avg_daily_return, 3),
                "max_drawdown_pct": round(self.max_dd_pct, 2),
                "winrate_trades": round(winrate_trades, 1),
                "winrate_days": round(winrate_days, 1),
                "profit_factor": round(profit_factor, 2),
                "sharpe_ratio": round(sharpe, 2),
                "total_trades": total_trades,
                "total_days": total_days,
                "green_days": green_days,
                "kill_switch_days": self.kill_switch_days,
                "total_fees_usd": round(total_fees, 2),
            },
        }


# =============================================================================
# MAIN
# =============================================================================

async def run_scenario(scenario: Scenario, days: int = 90) -> dict:
    """Run single scenario."""
    config = get_scenario_config(scenario)
    
    root = Path(__file__).parent.parent
    cache_dir = root / "data" / "backtest_cache"
    
    # Fetch data
    logger.info(f"Fetching {days} days of data...")
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(SYMBOLS[:15], days=days, timeframe="5m")
    
    if not data:
        logger.error("No data!")
        return {}
    
    # Run backtest
    backtester = RealisticBacktester(config)
    result = backtester.run(data)
    
    # Save result
    output_dir = root / "reports" / "realistic_864_stage6" / f"scenario_{config.scenario}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = output_dir / "report.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Print summary
    s = result["summary"]
    print("\n" + "="*60)
    print(f"SCENARIO {scenario.value.upper()}")
    print("="*60)
    print(f"Total Return: ${s['total_return_usd']:+.2f} ({s['total_return_pct']:+.1f}%)")
    print(f"Avg Daily:    {s['avg_daily_return_pct']:+.3f}%")
    print(f"Max DD:       {s['max_drawdown_pct']:.1f}%")
    print(f"Win Rate:     {s['winrate_trades']:.1f}% (trades), {s['winrate_days']:.1f}% (days)")
    print(f"Profit Factor: {s['profit_factor']:.2f}")
    print(f"Sharpe:       {s['sharpe_ratio']:.2f}")
    print(f"Trades:       {s['total_trades']}")
    print(f"Kill-switch:  {s['kill_switch_days']} days")
    print(f"Total Fees:   ${s['total_fees_usd']:.2f}")
    print("="*60)
    
    return result


async def run_all_scenarios(days: int = 90):
    """Run all 3 scenarios and compare."""
    results = {}
    
    for scenario in Scenario:
        result = await run_scenario(scenario, days)
        results[scenario.value] = result
    
    # Save comparison
    root = Path(__file__).parent.parent
    output_dir = root / "reports" / "realistic_864_stage6"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "days": days,
        "scenarios": results,
    }
    
    comp_path = output_dir / "comparison_realistic_864.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON: REALISTIC BACKTEST 864 + STAGE6")
    print("="*80)
    print(f"{'Metric':<25} {'A (Baseline)':<18} {'B (Relaxed)':<18} {'C (Filters Off)':<18}")
    print("-"*80)
    
    for metric in ["total_return_pct", "avg_daily_return_pct", "max_drawdown_pct", 
                   "winrate_trades", "profit_factor", "total_trades", "kill_switch_days"]:
        print(f"{metric:<25}", end="")
        for scenario in Scenario:
            val = results.get(scenario.value, {}).get("summary", {}).get(metric, "N/A")
            print(f"{val:<18}", end="")
        print()
    
    print("="*80)
    print(f"Report saved: {comp_path}")
    
    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Realistic Backtest 864 + Stage6")
    parser.add_argument("--scenario", type=str, choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--days", type=int, default=90, help="Days to backtest")
    args = parser.parse_args()
    
    if args.scenario == "all":
        asyncio.run(run_all_scenarios(args.days))
    else:
        scenario_map = {"A": Scenario.A_BASELINE, "B": Scenario.B_RELAXED, "C": Scenario.C_FILTERS_OFF}
        asyncio.run(run_scenario(scenario_map[args.scenario], args.days))


if __name__ == "__main__":
    main()
