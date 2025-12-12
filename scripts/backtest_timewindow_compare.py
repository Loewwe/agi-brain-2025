#!/usr/bin/env python3
"""
Time-Window Backtest Comparison

Compares Stage6 strategy performance between:
  Scenario A: Baseline (24/7 trading)
  Scenario B: Time-Window (16:00-00:00 KZT = 11:00-19:00 UTC)
  
Period: Uses cached data from backtest_stage6.py

Metrics compared:
  - Total PnL (USDT and %)
  - Max Drawdown
  - Win Rate
  - Trades count (total, in-window, out-of-window)
  - PnL by hour of day
  - Sharpe ratio

Usage:
  python scripts/backtest_timewindow_compare.py [--days 10]
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Import from main backtester
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from backtest_stage6 import (
    BacktestConfig, SYMBOLS, DataFetcher, ExchangeSimulator,
    Stage6Strategy, PositionManager, ReportGenerator, Trade, DayStats
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# TIME-WINDOW CONFIG
# =============================================================================

@dataclass
class TimeWindowConfig(BacktestConfig):
    """Extended config with time-window parameters."""
    # Time-window filter
    trading_window_enabled: bool = False
    trading_window_start_utc: int = 11  # 16:00 KZT = 11:00 UTC
    trading_window_end_utc: int = 19    # 00:00 KZT = 19:00 UTC
    force_flat_at_midnight: bool = False


# =============================================================================
# TIME-WINDOW STRATEGY
# =============================================================================

class TimeWindowStrategy(Stage6Strategy):
    """Stage6 strategy with time-window filter."""
    
    def __init__(self, config: TimeWindowConfig):
        super().__init__(config)
        self.tw_config = config
    
    def is_trading_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is within trading window (UTC hours)."""
        if not self.tw_config.trading_window_enabled:
            return True
        
        hour = timestamp.hour
        start = self.tw_config.trading_window_start_utc
        end = self.tw_config.trading_window_end_utc
        
        # Window is start <= hour < end
        if start <= end:
            return start <= hour < end
        else:  # Wraps midnight (e.g., 22:00 - 06:00)
            return hour >= start or hour < end
    
    def check_entry_signal(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        symbol: str,
        existing_positions: set,
        timestamp: datetime = None
    ) -> Optional[str]:
        """Check for entry signal with time-window filter."""
        # Time-window check (overrides excluded_hours when enabled)
        if timestamp is not None and self.tw_config.trading_window_enabled:
            if not self.is_trading_window(timestamp):
                return None
        
        # Delegate to parent for rest of logic
        return super().check_entry_signal(row, prev_row, symbol, existing_positions, timestamp)


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

@dataclass
class ScenarioResult:
    """Results from a backtest scenario."""
    scenario: str
    total_pnl_usd: float
    total_pnl_pct: float
    max_drawdown_pct: float
    trades_total: int
    trades_in_window: int
    trades_out_window: int
    win_rate: float
    avg_trade_pnl: float
    sharpe_ratio: float
    profit_factor: float
    pnl_by_hour: Dict[int, float]
    equity_curve: List[float]
    daily_stats: List[Dict]


async def run_backtest_scenario(
    config: TimeWindowConfig,
    data: Dict[str, pd.DataFrame],
    scenario_name: str
) -> ScenarioResult:
    """Run a single backtest scenario."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Scenario: {scenario_name}")
    logger.info(f"Time-window enabled: {config.trading_window_enabled}")
    if config.trading_window_enabled:
        logger.info(f"Window: {config.trading_window_start_utc}:00 - {config.trading_window_end_utc}:00 UTC")
    logger.info(f"{'='*60}")
    
    # Initialize components
    exchange = ExchangeSimulator(config)
    strategy = TimeWindowStrategy(config)
    pos_manager = PositionManager(config)
    
    # Track metrics
    pnl_by_hour = {h: 0.0 for h in range(24)}
    trades_in_window = 0
    trades_out_window = 0
    equity_curve = [config.start_balance]
    daily_stats: List[DayStats] = []
    
    # Get date range from data
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index.date)
    dates = sorted(all_dates)
    
    if not dates:
        logger.error("No data available!")
        return None
    
    logger.info(f"Backtesting {len(dates)} days: {dates[0]} to {dates[-1]}")
    
    # Main loop - by day
    for day in dates:
        date_str = day.strftime("%Y-%m-%d")
        exchange.reset_day(date_str)
        day_equity_open = exchange.equity
        day_peak = day_equity_open
        day_trades = 0
        day_wins = 0
        
        # Get all bars for this day, sorted by time
        day_bars = []
        for symbol, df in data.items():
            day_df = df[df.index.date == day]
            for idx, row in day_df.iterrows():
                day_bars.append((idx, symbol, row))
        
        day_bars.sort(key=lambda x: x[0])
        
        prev_row_by_symbol = {}
        
        for bar_idx, (timestamp, symbol, row) in enumerate(day_bars):
            # Get current prices for all positions
            current_prices = {s: data[s].loc[timestamp, "close"] 
                            for s in exchange.positions.keys() 
                            if timestamp in data[s].index}
            
            # Update equity curve
            curr_equity = exchange.current_equity(current_prices)
            equity_curve.append(curr_equity)
            day_peak = max(day_peak, curr_equity)
            
            # Check exits for existing positions
            positions_to_check = list(exchange.positions.items())
            for pos_symbol, position in positions_to_check:
                if pos_symbol not in data:
                    continue
                if timestamp not in data[pos_symbol].index:
                    continue
                    
                pos_row = data[pos_symbol].loc[timestamp]
                exit_result = pos_manager.check_exits(
                    position,
                    pos_row["high"],
                    pos_row["low"],
                    pos_row["close"],
                    bar_idx
                )
                
                reason, exit_price, partial = exit_result
                if reason is not None:
                    trade = exchange.close_position(pos_symbol, exit_price, reason, timestamp, partial)
                    if trade and partial >= 1.0:
                        day_trades += 1
                        if trade.pnl_abs > 0:
                            day_wins += 1
                        
                        # Track PnL by hour
                        hour = trade.datetime_open.hour
                        pnl_by_hour[hour] += trade.pnl_abs
                        
                        # Track in/out window
                        in_window = strategy.is_trading_window(trade.datetime_open)
                        if in_window:
                            trades_in_window += 1
                        else:
                            trades_out_window += 1
            
            # Check for new entries
            if exchange.daily_stop_active:
                continue
                
            prev_row = prev_row_by_symbol.get(symbol)
            signal = strategy.check_entry_signal(
                row, prev_row, symbol, 
                set(exchange.positions.keys()),
                timestamp
            )
            
            if signal:
                can_trade, reason = exchange.can_open_trade()
                if can_trade:
                    exchange.open_position(
                        symbol, signal, row["close"], row["atr"],
                        timestamp, bar_idx
                    )
            
            prev_row_by_symbol[symbol] = row
        
        # End of day stats
        day_equity_close = exchange.equity
        day_pnl = day_equity_close - day_equity_open
        day_pnl_pct = (day_pnl / day_equity_open * 100) if day_equity_open > 0 else 0
        max_dd = ((day_equity_close - day_peak) / day_peak * 100) if day_peak > 0 else 0
        
        day_stat = DayStats(
            date=date_str,
            equity_open=day_equity_open,
            equity_close=day_equity_close,
            pnl_abs=day_pnl,
            pnl_pct=day_pnl_pct,
            max_dd_pct=max_dd,
            trades=day_trades,
            wins=day_wins
        )
        daily_stats.append(day_stat)
        
        logger.info(f"  {date_str}: PnL={day_pnl:+.2f} ({day_pnl_pct:+.1f}%), Trades={day_trades}, WR={day_stat.winrate:.0f}%")
    
    # Calculate final metrics
    trades = exchange.trades
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl_abs > 0)
    
    total_pnl_usd = exchange.equity - config.start_balance
    total_pnl_pct = (total_pnl_usd / config.start_balance * 100) if config.start_balance > 0 else 0
    max_dd = min(d.max_dd_pct for d in daily_stats) if daily_stats else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = np.mean([t.pnl_abs for t in trades]) if trades else 0
    
    # Sharpe
    daily_returns = [d.pnl_pct for d in daily_stats]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) if np.std(daily_returns) > 0 else 0
    
    # Profit factor
    total_profit = sum(t.pnl_abs for t in trades if t.pnl_abs > 0)
    total_loss = abs(sum(t.pnl_abs for t in trades if t.pnl_abs < 0))
    pf = (total_profit / total_loss) if total_loss > 0 else float('inf')
    
    result = ScenarioResult(
        scenario=scenario_name,
        total_pnl_usd=total_pnl_usd,
        total_pnl_pct=total_pnl_pct,
        max_drawdown_pct=max_dd,
        trades_total=total_trades,
        trades_in_window=trades_in_window,
        trades_out_window=trades_out_window,
        win_rate=win_rate,
        avg_trade_pnl=avg_pnl,
        sharpe_ratio=sharpe,
        profit_factor=pf,
        pnl_by_hour=pnl_by_hour,
        equity_curve=equity_curve,
        daily_stats=[asdict(d) for d in daily_stats]
    )
    
    logger.info(f"\n=== {scenario_name} Summary ===")
    logger.info(f"Final Equity: ${exchange.equity:.2f}")
    logger.info(f"Total PnL: ${total_pnl_usd:.2f} ({total_pnl_pct:.1f}%)")
    logger.info(f"Max DD: {max_dd:.1f}%")
    logger.info(f"Trades: {total_trades} (in-window: {trades_in_window}, out: {trades_out_window})")
    logger.info(f"Win Rate: {win_rate:.1f}%")
    logger.info(f"Sharpe: {sharpe:.2f}")
    
    return result


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

async def run_comparison(days: int = 10):
    """Run both scenarios and generate comparison report."""
    
    # Setup paths
    root = Path(__file__).parent.parent
    cache_dir = root / "data" / "backtest_cache"
    output_dir = root / "reports" / "timewindow_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data (shared between scenarios)
    logger.info("Fetching data...")
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(SYMBOLS[:20], days=days, timeframe="5m")  # Top 20 symbols
    
    if not data:
        logger.error("No data fetched!")
        return
    
    # === Scenario A: Baseline (24/7) ===
    config_a = TimeWindowConfig(
        trading_window_enabled=False,
        lookback_days=days
    )
    result_a = await run_backtest_scenario(config_a, data, "A_Baseline_24x7")
    
    # === Scenario B: Time-Window (16:00-00:00 KZT) ===
    config_b = TimeWindowConfig(
        trading_window_enabled=True,
        trading_window_start_utc=11,  # 16:00 KZT
        trading_window_end_utc=19,    # 00:00 KZT
        lookback_days=days
    )
    result_b = await run_backtest_scenario(config_b, data, "B_TimeWindow_16-00_KZT")
    
    # === Generate Comparison Report ===
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "period_days": days,
        "scenarios": {
            "A_Baseline": asdict(result_a) if result_a else None,
            "B_TimeWindow": asdict(result_b) if result_b else None
        },
        "comparison": {}
    }
    
    if result_a and result_b:
        comparison["comparison"] = {
            "pnl_diff_usd": result_b.total_pnl_usd - result_a.total_pnl_usd,
            "pnl_diff_pct": result_b.total_pnl_pct - result_a.total_pnl_pct,
            "dd_improvement": result_a.max_drawdown_pct - result_b.max_drawdown_pct,
            "trade_reduction": result_a.trades_total - result_b.trades_total,
            "wr_diff": result_b.win_rate - result_a.win_rate,
            "verdict": "TIME_WINDOW_BETTER" if (
                result_b.total_pnl_pct >= result_a.total_pnl_pct * 0.9 and
                result_b.max_drawdown_pct > result_a.max_drawdown_pct
            ) else "BASELINE_BETTER" if result_a.total_pnl_pct > result_b.total_pnl_pct * 1.1 else "SIMILAR"
        }
        
        # PnL by hour analysis
        losing_hours_baseline = [h for h, pnl in result_a.pnl_by_hour.items() if pnl < 0]
        losing_hours_window = [h for h, pnl in result_b.pnl_by_hour.items() if pnl < 0]
        comparison["pnl_by_hour_analysis"] = {
            "baseline_losing_hours": losing_hours_baseline,
            "timewindow_losing_hours": losing_hours_window,
            "window_hours_utc": list(range(11, 19)),
            "baseline_pnl_in_window": sum(result_a.pnl_by_hour[h] for h in range(11, 19)),
            "baseline_pnl_out_window": sum(result_a.pnl_by_hour[h] for h in range(24) if h < 11 or h >= 19),
        }
    
    # Save report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARISON REPORT SAVED: {report_path}")
    logger.info(f"{'='*60}")
    
    # Print summary
    if result_a and result_b:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Metric':<25} {'Baseline':<15} {'TimeWindow':<15} {'Diff':<10}")
        print("-"*60)
        print(f"{'Total PnL (USD)':<25} {result_a.total_pnl_usd:<15.2f} {result_b.total_pnl_usd:<15.2f} {result_b.total_pnl_usd - result_a.total_pnl_usd:+.2f}")
        print(f"{'Total PnL (%)':<25} {result_a.total_pnl_pct:<15.1f} {result_b.total_pnl_pct:<15.1f} {result_b.total_pnl_pct - result_a.total_pnl_pct:+.1f}")
        print(f"{'Max DD (%)':<25} {result_a.max_drawdown_pct:<15.1f} {result_b.max_drawdown_pct:<15.1f} {result_b.max_drawdown_pct - result_a.max_drawdown_pct:+.1f}")
        print(f"{'Trades Total':<25} {result_a.trades_total:<15} {result_b.trades_total:<15} {result_b.trades_total - result_a.trades_total:+d}")
        print(f"{'Win Rate (%)':<25} {result_a.win_rate:<15.1f} {result_b.win_rate:<15.1f} {result_b.win_rate - result_a.win_rate:+.1f}")
        print(f"{'Sharpe':<25} {result_a.sharpe_ratio:<15.2f} {result_b.sharpe_ratio:<15.2f} {result_b.sharpe_ratio - result_a.sharpe_ratio:+.2f}")
        print("="*60)
        print(f"VERDICT: {comparison['comparison']['verdict']}")
        print("="*60)
    
    return comparison


def generate_charts(result_a: ScenarioResult, result_b: ScenarioResult, output_dir: Path):
    """Generate comparison charts (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        return
    
    # === PnL by Hour Chart ===
    fig, ax = plt.subplots(figsize=(12, 6))
    hours = list(range(24))
    pnl_a = [result_a.pnl_by_hour.get(h, 0) for h in hours]
    pnl_b = [result_b.pnl_by_hour.get(h, 0) for h in hours]
    
    x = np.arange(24)
    width = 0.35
    
    bars_a = ax.bar(x - width/2, pnl_a, width, label='Baseline (24/7)', color='#ff6b6b', alpha=0.8)
    bars_b = ax.bar(x + width/2, pnl_b, width, label='TimeWindow (16-00 KZT)', color='#4ecdc4', alpha=0.8)
    
    # Highlight trading window
    ax.axvspan(11, 19, alpha=0.2, color='green', label='Trading Window (11-19 UTC)')
    
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('PnL (USDT)')
    ax.set_title('PnL by Hour of Day')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h:02d}' for h in hours])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pnl_by_hour.png', dpi=150)
    plt.close()
    logger.info(f"  Chart saved: pnl_by_hour.png")
    
    # === Equity Curves Chart ===
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(result_a.equity_curve, label='Baseline (24/7)', color='#ff6b6b', linewidth=1.5)
    ax.plot(result_b.equity_curve, label='TimeWindow (16-00 KZT)', color='#4ecdc4', linewidth=1.5)
    
    ax.set_xlabel('Time (bars)')
    ax.set_ylabel('Equity (USDT)')
    ax.set_title('Equity Curves Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves.png', dpi=150)
    plt.close()
    logger.info(f"  Chart saved: equity_curves.png")


async def run_comparison(days: int = 10):
    """Run both scenarios and generate comparison report."""
    
    # Setup paths
    root = Path(__file__).parent.parent
    cache_dir = root / "data" / "backtest_cache"
    output_dir = root / "reports" / "timewindow_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data (shared between scenarios)
    logger.info("Fetching data...")
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(SYMBOLS[:20], days=days, timeframe="5m")  # Top 20 symbols
    
    if not data:
        logger.error("No data fetched!")
        return
    
    # === Scenario A: Baseline (24/7) ===
    config_a = TimeWindowConfig(
        trading_window_enabled=False,
        lookback_days=days
    )
    result_a = await run_backtest_scenario(config_a, data, "A_Baseline_24x7")
    
    # === Scenario B: Time-Window (16:00-00:00 KZT) ===
    config_b = TimeWindowConfig(
        trading_window_enabled=True,
        trading_window_start_utc=11,  # 16:00 KZT
        trading_window_end_utc=19,    # 00:00 KZT
        lookback_days=days
    )
    result_b = await run_backtest_scenario(config_b, data, "B_TimeWindow_16-00_KZT")
    
    # === Generate Comparison Report ===
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "period_days": days,
        "scenarios": {
            "A_Baseline": asdict(result_a) if result_a else None,
            "B_TimeWindow": asdict(result_b) if result_b else None
        },
        "comparison": {}
    }
    
    if result_a and result_b:
        comparison["comparison"] = {
            "pnl_diff_usd": result_b.total_pnl_usd - result_a.total_pnl_usd,
            "pnl_diff_pct": result_b.total_pnl_pct - result_a.total_pnl_pct,
            "dd_improvement": result_a.max_drawdown_pct - result_b.max_drawdown_pct,
            "trade_reduction": result_a.trades_total - result_b.trades_total,
            "wr_diff": result_b.win_rate - result_a.win_rate,
            "verdict": "TIME_WINDOW_BETTER" if (
                result_b.total_pnl_pct >= result_a.total_pnl_pct * 0.9 and
                result_b.max_drawdown_pct > result_a.max_drawdown_pct
            ) else "BASELINE_BETTER" if result_a.total_pnl_pct > result_b.total_pnl_pct * 1.1 else "SIMILAR"
        }
        
        # PnL by hour analysis
        losing_hours_baseline = [h for h, pnl in result_a.pnl_by_hour.items() if pnl < 0]
        losing_hours_window = [h for h, pnl in result_b.pnl_by_hour.items() if pnl < 0]
        comparison["pnl_by_hour_analysis"] = {
            "baseline_losing_hours": losing_hours_baseline,
            "timewindow_losing_hours": losing_hours_window,
            "window_hours_utc": list(range(11, 19)),
            "baseline_pnl_in_window": sum(result_a.pnl_by_hour[h] for h in range(11, 19)),
            "baseline_pnl_out_window": sum(result_a.pnl_by_hour[h] for h in range(24) if h < 11 or h >= 19),
        }
        
        # Generate charts
        generate_charts(result_a, result_b, output_dir)
    
    # Save report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARISON REPORT SAVED: {report_path}")
    logger.info(f"{'='*60}")
    
    # Print summary
    if result_a and result_b:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Metric':<25} {'Baseline':<15} {'TimeWindow':<15} {'Diff':<10}")
        print("-"*60)
        print(f"{'Total PnL (USD)':<25} {result_a.total_pnl_usd:<15.2f} {result_b.total_pnl_usd:<15.2f} {result_b.total_pnl_usd - result_a.total_pnl_usd:+.2f}")
        print(f"{'Total PnL (%)':<25} {result_a.total_pnl_pct:<15.1f} {result_b.total_pnl_pct:<15.1f} {result_b.total_pnl_pct - result_a.total_pnl_pct:+.1f}")
        print(f"{'Max DD (%)':<25} {result_a.max_drawdown_pct:<15.1f} {result_b.max_drawdown_pct:<15.1f} {result_b.max_drawdown_pct - result_a.max_drawdown_pct:+.1f}")
        print(f"{'Trades Total':<25} {result_a.trades_total:<15} {result_b.trades_total:<15} {result_b.trades_total - result_a.trades_total:+d}")
        print(f"{'Win Rate (%)':<25} {result_a.win_rate:<15.1f} {result_b.win_rate:<15.1f} {result_b.win_rate - result_a.win_rate:+.1f}")
        print(f"{'Sharpe':<25} {result_a.sharpe_ratio:<15.2f} {result_b.sharpe_ratio:<15.2f} {result_b.sharpe_ratio - result_a.sharpe_ratio:+.2f}")
        print("="*60)
        print(f"VERDICT: {comparison['comparison']['verdict']}")
        print("="*60)


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

EXPECTED_BEST_CONFIG = {
    "ema_fast": 15,
    "ema_slow": 60,
    "rsi_period": 14,
    "rsi_overbought": 75,
    "rsi_oversold": 25,
    "bb_period": 25,
    "bb_std": 2.0
}


def validate_config(config_path: Path) -> dict:
    """Validate that config matches expected best config parameters."""
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Extract indicators if nested
        indicators = config.get("indicators", config)
        
        # Validate each parameter
        mismatches = []
        for key, expected in EXPECTED_BEST_CONFIG.items():
            actual = indicators.get(key)
            if actual != expected:
                mismatches.append(f"{key}: expected {expected}, got {actual}")
        
        if mismatches:
            logger.warning(f"Config validation FAILED:\n  " + "\n  ".join(mismatches))
        else:
            logger.info(f"✅ BestConfig OK: {config_path.name}")
        
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Time-Window Backtest Comparison v2.0")
    parser.add_argument("--days", type=int, default=10, help="Days to backtest (default: 10, ignored if --start-date used)")
    parser.add_argument("--config-path", type=str, default="configs/winning_strategy_best.json",
                       help="Path to strategy config JSON")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Start date YYYY-MM-DD (default: based on --days)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    
    # Resolve config path
    root = Path(__file__).parent.parent
    config_path = root / args.config_path if not Path(args.config_path).is_absolute() else Path(args.config_path)
    
    # Validate config
    config = validate_config(config_path)
    
    # Calculate days from date range if provided
    days = args.days
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
        days = (end - start).days + 1
        logger.info(f"Period: {args.start_date} to {args.end_date} ({days} days)")
    elif args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.now()
        days = (end - start).days + 1
        logger.info(f"Period: {args.start_date} to today ({days} days)")
    
    # Run comparison
    asyncio.run(run_comparison(days))


if __name__ == "__main__":
    main()

