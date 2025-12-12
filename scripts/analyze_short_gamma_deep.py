#!/usr/bin/env python3
"""
Deep Analysis of Short-Gamma NoSL Strategy

Analyzes:
1. Holding time distribution (hours/days)
2. Max unrealized drawdown per trade
3. Profit distribution
4. Correlation with market conditions

Usage:
  python analyze_short_gamma_deep.py --days 180
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_stage6 import DataFetcher, SYMBOLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class NoSLConfig:
    """NoSL strategy configuration."""
    leverage: float = 1.0
    entry_threshold_pct: float = 3.7
    take_profit_pct: float = 1.6
    max_positions: int = 7
    lookback_bars: int = 48  # 4 hours in 5m bars
    commission_pct: float = 0.04
    start_balance: float = 10000.0


# =============================================================================
# ENHANCED TRADE TRACKING
# =============================================================================

@dataclass
class DetailedTrade:
    """Trade with detailed metrics."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    
    # Deep metrics
    holding_hours: float
    max_unrealized_dd_pct: float  # Worst unrealized drawdown
    max_adverse_price: float  # Price that caused max DD
    price_path: List[float] = field(default_factory=list)  # Prices during holding


@dataclass
class Position:
    """Open position with tracking."""
    symbol: str
    entry_price: float
    entry_time: datetime
    size_usd: float
    tp_price: float
    max_adverse_price: float = 0.0  # For tracking unrealized DD


# =============================================================================
# DEEP ANALYZER
# =============================================================================

class DeepAnalyzer:
    """Deep analysis of NoSL strategy."""
    
    def __init__(self, config: NoSLConfig):
        self.config = config
        self.equity = config.start_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[DetailedTrade] = []
        
    def calculate_pump_pct(self, df: pd.DataFrame, bar_idx: int) -> float:
        """Calculate pump percentage over lookback period."""
        if bar_idx < self.config.lookback_bars:
            return 0.0
        
        current_price = df.iloc[bar_idx]["close"]
        price_4h_ago = df.iloc[bar_idx - self.config.lookback_bars]["close"]
        
        if price_4h_ago <= 0:
            return 0.0
        
        return (current_price - price_4h_ago) / price_4h_ago * 100
    
    def run(self, data: Dict[str, pd.DataFrame]) -> dict:
        """Run deep analysis."""
        logger.info(f"\n{'='*60}")
        logger.info("DEEP ANALYSIS: NoSL Short-Gamma Strategy")
        logger.info(f"{'='*60}")
        
        # Get all dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.date)
        dates = sorted(all_dates)
        
        logger.info(f"Analyzing {len(dates)} days: {dates[0]} to {dates[-1]}")
        
        # Process each day
        for day in dates:
            day_bars = []
            for symbol, df in data.items():
                day_df = df[df.index.date == day]
                for idx, (timestamp, row) in enumerate(day_df.iterrows()):
                    bar_idx = df.index.get_loc(timestamp)
                    day_bars.append((timestamp, symbol, row, bar_idx, df))
            
            day_bars.sort(key=lambda x: x[0])
            
            for timestamp, symbol, row, bar_idx, df in day_bars:
                high = row["high"]
                low = row["low"]
                close = row["close"]
                
                # Check existing positions - track max adverse move
                for pos_symbol, pos in list(self.positions.items()):
                    if pos_symbol not in data:
                        continue
                    if timestamp not in data[pos_symbol].index:
                        continue
                    
                    pos_row = data[pos_symbol].loc[timestamp]
                    pos_high = pos_row["high"]
                    
                    # For SHORT: high is adverse (price going up)
                    if pos_high > pos.max_adverse_price:
                        pos.max_adverse_price = pos_high
                    
                    # Check TP (low touches TP for short)
                    if pos_row["low"] <= pos.tp_price:
                        # Close position
                        exit_price = pos.tp_price
                        price_change_pct = (pos.entry_price - exit_price) / pos.entry_price * 100
                        pnl_pct = price_change_pct * self.config.leverage
                        pnl_usd = pos.size_usd * pnl_pct / 100
                        
                        # Calculate unrealized DD
                        max_dd_pct = (pos.max_adverse_price - pos.entry_price) / pos.entry_price * 100
                        
                        # Holding time
                        holding_hours = (timestamp - pos.entry_time).total_seconds() / 3600
                        
                        trade = DetailedTrade(
                            symbol=pos_symbol,
                            entry_time=pos.entry_time,
                            exit_time=timestamp,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            size_usd=pos.size_usd,
                            pnl_usd=pnl_usd,
                            pnl_pct=pnl_pct,
                            holding_hours=holding_hours,
                            max_unrealized_dd_pct=max_dd_pct,
                            max_adverse_price=pos.max_adverse_price,
                        )
                        
                        self.equity += pnl_usd
                        self.trades.append(trade)
                        del self.positions[pos_symbol]
                
                # Check new entries
                if len(self.positions) < self.config.max_positions:
                    if symbol not in self.positions:
                        pump_pct = self.calculate_pump_pct(df, bar_idx)
                        if pump_pct >= self.config.entry_threshold_pct:
                            size = self.equity / self.config.max_positions
                            tp_price = close * (1 - self.config.take_profit_pct / 100)
                            
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                entry_price=close,
                                entry_time=timestamp,
                                size_usd=size,
                                tp_price=tp_price,
                                max_adverse_price=close,
                            )
        
        # Calculate metrics
        return self.calculate_deep_metrics()
    
    def calculate_deep_metrics(self) -> dict:
        """Calculate deep analysis metrics."""
        if not self.trades:
            return {"error": "No trades"}
        
        # Holding time stats
        holding_hours = [t.holding_hours for t in self.trades]
        avg_holding = np.mean(holding_hours)
        max_holding = max(holding_hours)
        min_holding = min(holding_hours)
        
        # Holding time buckets
        under_1h = sum(1 for h in holding_hours if h < 1)
        h1_4 = sum(1 for h in holding_hours if 1 <= h < 4)
        h4_12 = sum(1 for h in holding_hours if 4 <= h < 12)
        h12_24 = sum(1 for h in holding_hours if 12 <= h < 24)
        d1_3 = sum(1 for h in holding_hours if 24 <= h < 72)
        d3_plus = sum(1 for h in holding_hours if h >= 72)
        
        # Unrealized DD stats
        max_dds = [t.max_unrealized_dd_pct for t in self.trades]
        avg_dd = np.mean(max_dds)
        worst_dd = max(max_dds)
        worst_dd_trade = max(self.trades, key=lambda t: t.max_unrealized_dd_pct)
        
        # Profit distribution
        pnls = [t.pnl_pct for t in self.trades]
        avg_pnl = np.mean(pnls)
        
        # Find still-open positions (if any)
        open_positions = list(self.positions.values())
        
        result = {
            "summary": {
                "total_trades": len(self.trades),
                "total_return_pct": round((self.equity - self.config.start_balance) / self.config.start_balance * 100, 2),
                "final_equity": round(self.equity, 2),
            },
            "holding_time": {
                "avg_hours": round(avg_holding, 1),
                "max_hours": round(max_holding, 1),
                "max_days": round(max_holding / 24, 1),
                "min_hours": round(min_holding, 2),
                "distribution": {
                    "under_1h": under_1h,
                    "1-4h": h1_4,
                    "4-12h": h4_12,
                    "12-24h": h12_24,
                    "1-3d": d1_3,
                    "3d+": d3_plus,
                }
            },
            "unrealized_drawdown": {
                "avg_max_dd_pct": round(avg_dd, 2),
                "worst_max_dd_pct": round(worst_dd, 2),
                "worst_dd_trade": {
                    "symbol": worst_dd_trade.symbol,
                    "entry_time": str(worst_dd_trade.entry_time),
                    "entry_price": worst_dd_trade.entry_price,
                    "max_adverse_price": worst_dd_trade.max_adverse_price,
                    "holding_hours": round(worst_dd_trade.holding_hours, 1),
                }
            },
            "profit_distribution": {
                "avg_pnl_pct": round(avg_pnl, 2),
                "all_trades_positive": all(p > 0 for p in pnls),
            },
            "open_positions": len(open_positions),
            "risk_assessment": {
                "max_theoretical_loss_if_50pct_pump": round(50 * self.config.leverage, 1),
                "max_theoretical_loss_if_100pct_pump": round(100 * self.config.leverage, 1),
                "current_leverage": self.config.leverage,
            }
        }
        
        return result


# =============================================================================
# MAIN
# =============================================================================

async def run_deep_analysis(days: int = 180):
    """Run deep analysis."""
    
    root = Path(__file__).parent.parent
    cache_dir = root / "data" / "backtest_cache"
    output_dir = root / "reports" / "short_gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    logger.info(f"Fetching {days} days of data...")
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(SYMBOLS[:15], days=days, timeframe="5m")
    
    if not data:
        logger.error("No data!")
        return
    
    # Run analysis
    config = NoSLConfig()
    analyzer = DeepAnalyzer(config)
    result = analyzer.run(data)
    
    # Save
    result_path = output_dir / "deep_analysis_nosl.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Print
    print("\n" + "="*60)
    print("DEEP ANALYSIS: NoSL Short-Gamma")
    print("="*60)
    
    print("\n📊 HOLDING TIME:")
    ht = result["holding_time"]
    print(f"  Average: {ht['avg_hours']:.1f} hours ({ht['avg_hours']/24:.1f} days)")
    print(f"  Maximum: {ht['max_hours']:.1f} hours ({ht['max_days']:.1f} days)")
    print(f"  Minimum: {ht['min_hours']:.2f} hours")
    print("\n  Distribution:")
    for bucket, count in ht["distribution"].items():
        print(f"    {bucket}: {count} trades")
    
    print("\n⚠️ UNREALIZED DRAWDOWN:")
    dd = result["unrealized_drawdown"]
    print(f"  Average max DD: {dd['avg_max_dd_pct']:.2f}%")
    print(f"  WORST max DD: {dd['worst_max_dd_pct']:.2f}%")
    print(f"  Worst trade: {dd['worst_dd_trade']['symbol']}")
    print(f"    Entry: ${dd['worst_dd_trade']['entry_price']:.2f}")
    print(f"    Max adverse: ${dd['worst_dd_trade']['max_adverse_price']:.2f}")
    print(f"    Holding: {dd['worst_dd_trade']['holding_hours']:.1f}h")
    
    print("\n🎯 RISK ASSESSMENT:")
    risk = result["risk_assessment"]
    print(f"  If BTC pumps +50%: potential -{risk['max_theoretical_loss_if_50pct_pump']}%")
    print(f"  If BTC pumps +100%: potential -{risk['max_theoretical_loss_if_100pct_pump']}%")
    
    print("\n" + "="*60)
    print(f"Report saved: {result_path}")
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep Analysis of NoSL Strategy")
    parser.add_argument("--days", type=int, default=180, help="Days to analyze")
    args = parser.parse_args()
    
    asyncio.run(run_deep_analysis(args.days))


if __name__ == "__main__":
    main()
