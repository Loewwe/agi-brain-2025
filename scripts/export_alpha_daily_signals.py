#!/usr/bin/env python3
"""
Export Alpha Daily Signals

Generates daily alpha signals based on winning_strategy_best.json config.
For each (symbol, date), calculates:
  - Alpha direction (LONG/SHORT/FLAT)
  - Ideal entry/exit prices
  - Ideal PnL
  - Win/loss flag

Output: reports/alpha_signals_daily.csv

Usage:
  python scripts/export_alpha_daily_signals.py --days 90
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Literal
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_stage6 import DataFetcher, SYMBOLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class AlphaConfig:
    """Configuration from winning_strategy_best.json"""
    ema_fast: int = 15
    ema_slow: int = 60
    rsi_period: int = 14
    rsi_overbought: int = 75
    rsi_oversold: int = 25
    bb_period: int = 25
    bb_std: float = 2.0
    
    @classmethod
    def from_json(cls, path: Path) -> "AlphaConfig":
        with open(path) as f:
            data = json.load(f)
        indicators = data.get("indicators", data)
        return cls(
            ema_fast=indicators.get("ema_fast", 15),
            ema_slow=indicators.get("ema_slow", 60),
            rsi_period=indicators.get("rsi_period", 14),
            rsi_overbought=indicators.get("rsi_overbought", 75),
            rsi_oversold=indicators.get("rsi_oversold", 25),
            bb_period=indicators.get("bb_period", 25),
            bb_std=indicators.get("bb_std", 2.0),
        )


# =============================================================================
# ALPHA SIGNAL GENERATOR
# =============================================================================

class AlphaSignalGenerator:
    """Generates daily alpha signals based on technical indicators."""
    
    def __init__(self, config: AlphaConfig):
        self.config = config
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators for alpha calculation."""
        # EMA fast/slow
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
        
        return df
    
    def calculate_daily_signal(self, daily_df: pd.DataFrame) -> dict:
        """Calculate signal for a single day based on close values."""
        if len(daily_df) == 0:
            return None
        
        last = daily_df.iloc[-1]
        
        # Skip if missing data
        if pd.isna(last.get("ema_fast")) or pd.isna(last.get("ema_slow")) or pd.isna(last.get("rsi")):
            return None
        
        close = last["close"]
        ema_fast = last["ema_fast"]
        ema_slow = last["ema_slow"]
        rsi = last["rsi"]
        bb_lower = last.get("bb_lower", close)
        bb_upper = last.get("bb_upper", close)
        
        # === SIGNAL LOGIC (matching research alpha) ===
        # Score components
        score = 0.0
        
        # EMA trend: fast > slow = bullish
        if ema_fast > ema_slow:
            score += 0.4
        elif ema_fast < ema_slow:
            score -= 0.4
        
        # RSI: oversold = bullish, overbought = bearish
        if rsi < self.config.rsi_oversold:
            score += 0.3
        elif rsi > self.config.rsi_overbought:
            score -= 0.3
        elif rsi < 40:
            score += 0.1
        elif rsi > 60:
            score -= 0.1
        
        # Bollinger: near lower band = bullish
        if bb_lower > 0:
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            if bb_position < 0.2:
                score += 0.3
            elif bb_position > 0.8:
                score -= 0.3
        
        # Determine direction
        if score >= 0.3:
            direction = "LONG"
        elif score <= -0.3:
            direction = "SHORT"
        else:
            direction = "FLAT"
        
        return {
            "alpha_score": score,
            "alpha_direction": direction,
            "close": close,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi,
        }
    
    def calculate_ideal_pnl(
        self,
        direction: str,
        entry_price: float,
        exit_price: float
    ) -> tuple:
        """Calculate ideal PnL for a directional trade."""
        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        elif direction == "SHORT":
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        else:
            pnl_pct = 0.0
        
        win = 1 if pnl_pct > 0 else 0
        return pnl_pct, win


# =============================================================================
# MAIN EXPORT
# =============================================================================

async def export_alpha_signals(
    days: int = 90,
    config_path: Path = None,
    output_path: Path = None
):
    """Export daily alpha signals for all symbols."""
    
    # Setup paths
    root = Path(__file__).parent.parent
    if config_path is None:
        config_path = root / "configs" / "winning_strategy_best.json"
    if output_path is None:
        output_path = root / "reports" / "alpha_signals_daily.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    if config_path.exists():
        config = AlphaConfig.from_json(config_path)
        logger.info(f"✅ Loaded config: {config_path.name}")
    else:
        config = AlphaConfig()
        logger.warning(f"Config not found, using defaults")
    
    # Fetch 5m data and resample to daily
    cache_dir = root / "data" / "backtest_cache"
    fetcher = DataFetcher(cache_dir)
    
    logger.info(f"Fetching {days} days of data for {len(SYMBOLS[:20])} symbols...")
    data = await fetcher.fetch_all(SYMBOLS[:20], days=days, timeframe="5m")
    
    if not data:
        logger.error("No data fetched!")
        return
    
    # Generate signals
    generator = AlphaSignalGenerator(config)
    results = []
    
    for symbol, df in data.items():
        logger.info(f"  Processing {symbol}...")
        
        # Add indicators
        df = generator.add_indicators(df)
        
        # Group by date
        df["date"] = df.index.date
        dates = df["date"].unique()
        
        for date in sorted(dates):
            day_df = df[df["date"] == date]
            
            if len(day_df) < 50:  # Need enough bars for indicators
                continue
            
            # Calculate signal using end-of-day values
            signal = generator.calculate_daily_signal(day_df)
            if signal is None:
                continue
            
            # Ideal entry: open next day, exit: close of day
            entry_price = day_df.iloc[0]["open"]  # Simplified: use day open
            exit_price = day_df.iloc[-1]["close"]  # Use day close
            
            direction = signal["alpha_direction"]
            pnl_pct, win = generator.calculate_ideal_pnl(direction, entry_price, exit_price)
            
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "alpha_score": round(signal["alpha_score"], 3),
                "alpha_direction": direction,
                "entry_price_ideal": round(entry_price, 6),
                "exit_price_ideal": round(exit_price, 6),
                "pnl_ideal_pct": round(pnl_pct, 4),
                "win_ideal": win,
                "rsi": round(signal["rsi"], 2),
                "ema_trend": "UP" if signal["ema_fast"] > signal["ema_slow"] else "DOWN",
            })
    
    # Save to CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)
    
    # Summary
    total_signals = len(df_out)
    directional = df_out[df_out["alpha_direction"] != "FLAT"]
    wins = directional[directional["win_ideal"] == 1]
    
    winrate = len(wins) / len(directional) * 100 if len(directional) > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ALPHA SIGNALS EXPORTED: {output_path}")
    logger.info(f"{'='*60}")
    logger.info(f"Total signals: {total_signals}")
    logger.info(f"Directional (LONG/SHORT): {len(directional)}")
    logger.info(f"Ideal Wins: {len(wins)}")
    logger.info(f"Ideal Win Rate: {winrate:.1f}%")
    logger.info(f"{'='*60}")
    
    return df_out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export Alpha Daily Signals")
    parser.add_argument("--days", type=int, default=90, help="Days to analyze")
    parser.add_argument("--config-path", type=str, default=None, help="Path to strategy config")
    args = parser.parse_args()
    
    config_path = Path(args.config_path) if args.config_path else None
    asyncio.run(export_alpha_signals(args.days, config_path))


if __name__ == "__main__":
    main()
