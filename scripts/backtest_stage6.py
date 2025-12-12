#!/usr/bin/env python3
"""
Stage6 Backtester - Honest 30-day simulation of Stage6 strategy.

Goals:
- Validate Stage6 makes profit over 1 month
- Check WR ≥55%, R:R ≥1.8, daily DD ≤-1%
- Detect bugs (missing SL, excessive timeout, etc.)

Usage:
    python scripts/backtest_stage6.py
"""

import asyncio
import os
import pickle
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Literal
from enum import Enum

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Backtest configuration matching STRICT Stage6 spec."""
    # Account
    start_balance: float = 200.0
    leverage: int = 20
    
    # Risk
    risk_per_trade: float = 0.009  # 0.9%
    max_positions: int = 3
    max_trades_day: int = 12
    daily_stop_pct: float = -0.01  # -1%
    
    # Trade management (original Stage6 spec)
    sl_max_pct: float = 0.012      # 1.2%
    atr_multiplier: float = 1.0
    tp_multiplier: float = 2.1
    be_trigger_r: float = 1.0      # Original spec
    be_offset_r: float = 0.15
    partial_trigger_r: float = 1.5  # Original spec
    partial_pct: float = 0.5
    trailing_distance_r: float = 1.0
    timeout_bars: int = 40         # 40 × 5m
    
    # Risk limits
    max_open_risk_pct: float = 0.027  # 3 × 0.9% original
    
    # Simulation
    commission: float = 0.0004     # 0.04%
    slippage_pct: float = 0.0002   # 0.02%
    
    # Data
    timeframe: str = "5m"
    lookback_days: int = 90  # Extended to 90 days for better sample
    
    # Entry filters (RSI relaxed per user request)
    rsi_oversold: float = 35.0       # Relaxed from 32
    rsi_overbought: float = 65.0     # Relaxed from 68
    volume_surge_mult: float = 1.5   # STRICT: was 1.0
    atr_min_pct: float = 0.5         # STRICT: was 0.3
    atr_max_pct: float = 3.0         # STRICT: was 5.0
    ema_long_threshold: float = 1.004  # STRICT: was 1.002
    ema_short_threshold: float = 0.996 # STRICT: was 0.998
    ema_dead_zone_pct: float = 0.004   # ±0.4% dead zone around EMA200
    
    # Time filter - exclude low-quality hours (UTC)
    # Asian session 00:00-06:00 UTC is typically choppy
    excluded_hours: tuple = (0, 1, 2, 3, 4, 5)  # 00:00-05:59 UTC


# Top liquid futures symbols
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT", "SOL/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
    "MATIC/USDT:USDT", "LTC/USDT:USDT", "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
    "ARB/USDT:USDT", "OP/USDT:USDT", "FIL/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT",
    "TIA/USDT:USDT", "SEI/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT", "BONK/USDT:USDT",
    "RENDER/USDT:USDT", "FET/USDT:USDT", "RUNE/USDT:USDT", "STX/USDT:USDT", "IMX/USDT:USDT",
    "AAVE/USDT:USDT", "UNI/USDT:USDT", "MKR/USDT:USDT", "LDO/USDT:USDT", "CRV/USDT:USDT",
    "SAND/USDT:USDT", "MANA/USDT:USDT", "GALA/USDT:USDT", "AXS/USDT:USDT", "ENS/USDT:USDT",
]

class ExitReason(str, Enum):
    TP = "TP"
    SL = "SL"
    TRAILING = "trailing"
    TIMEOUT = "timeout"
    DAILY_STOP = "daily_stop"
    BE = "breakeven"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """Open position state."""
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    size: float           # In base currency
    sl_price: float
    tp_price: float
    sl_distance: float    # 1R in price terms
    entry_time: datetime
    entry_bar_idx: int
    breakeven_done: bool = False
    partial_done: bool = False
    trailing_active: bool = False
    current_sl: float = 0.0
    
    def __post_init__(self):
        self.current_sl = self.sl_price


@dataclass
class Trade:
    """Completed trade record."""
    datetime_open: datetime
    datetime_close: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    reason: str
    pnl_abs: float
    pnl_pct: float
    r_multiple: float
    hold_minutes: int


@dataclass
class DayStats:
    """Daily statistics."""
    date: str
    equity_open: float
    equity_close: float
    pnl_abs: float
    pnl_pct: float
    max_dd_pct: float
    trades: int
    wins: int
    
    @property
    def winrate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0


# =============================================================================
# DATA FETCHER
# =============================================================================

class DataFetcher:
    """Fetch and cache OHLCV data with indicators."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def fetch_all(self, symbols: List[str], days: int, timeframe: str = "5m") -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for all symbols, cache locally."""
        cache_file = self.cache_dir / f"ohlcv_{days}d_{timeframe}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Fetching {days} days of {timeframe} data for {len(symbols)} symbols...")
        
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        
        data = {}
        # 30 days = ~8640 bars at 5m. We need to paginate (Binance limit ~1500)
        bars_needed = days * 288  # 288 bars per day at 5m
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"  [{i+1}/{len(symbols)}] Fetching {symbol}...")
                
                all_ohlcv = []
                since = int((datetime.now() - timedelta(days=days + 5)).timestamp() * 1000)
                batch_size = 1000
                
                while len(all_ohlcv) < bars_needed:
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    
                    # Move since to after last fetched bar
                    since = ohlcv[-1][0] + 1
                    
                    await asyncio.sleep(0.05)  # Rate limit
                    
                    if len(ohlcv) < batch_size:
                        break  # No more data
                
                if len(all_ohlcv) < 200:
                    logger.warning(f"    Skipping {symbol}: insufficient data ({len(all_ohlcv)} bars)")
                    continue
                
                df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
                
                logger.info(f"    Got {len(df)} bars")
                
                # Calculate indicators
                df = self._add_indicators(df)
                data[symbol] = df
                
                await asyncio.sleep(0.1)  # Rate limit
                
            except Exception as e:
                logger.warning(f"    Error fetching {symbol}: {e}")
        
        await exchange.close()
        
        # Cache
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached {len(data)} symbols to {cache_file}")
        
        return data
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # RSI 14
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # ATR 14
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # EMA 200 (on hourly - we approximate by using 12-bar EMA on 5m)
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["ema_slope"] = df["ema200"].diff(3) > 0  # Rising over last 3 bars
        
        # Volume SMA
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_surge"] = df["volume"] / df["volume_sma"]
        
        # RSI 5m previous (for reversal)
        df["rsi_prev"] = df["rsi"].shift(1)
        df["rsi_prev2"] = df["rsi"].shift(2)
        
        # Breakout high/low of last 2 bars (relaxed from 5 bars)
        df["high_2"] = df["high"].rolling(2).max().shift(1)
        df["low_2"] = df["low"].rolling(2).min().shift(1)
        
        return df


# =============================================================================
# EXCHANGE SIMULATOR
# =============================================================================

class ExchangeSimulator:
    """Simulates Binance Futures with commissions and slippage."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.equity = config.start_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.current_date: Optional[str] = None
        self.daily_stop_active = False
        self.peak_equity = config.start_balance
        self.max_dd_today = 0.0
    
    def reset_day(self, date_str: str):
        """Reset daily counters."""
        self.current_date = date_str
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.daily_stop_active = False
        self.max_dd_today = 0.0
    
    def can_open_trade(self, new_sl_risk_pct: float = 0.009) -> tuple[bool, str]:
        """Check if we can open a new trade."""
        if self.daily_stop_active:
            return False, "daily_stop"
        if len(self.positions) >= self.config.max_positions:
            return False, "max_positions"
        if self.trades_today >= self.config.max_trades_day:
            return False, "max_trades_day"
        
        # Check max open risk (sum of all position risks + new trade)
        current_open_risk = self._calculate_open_risk()
        if current_open_risk + new_sl_risk_pct > self.config.max_open_risk_pct:
            return False, "max_open_risk"
        
        return True, "ok"
    
    def _calculate_open_risk(self) -> float:
        """Calculate total open risk as % of starting equity."""
        total_risk = 0.0
        for pos in self.positions.values():
            # Risk = SL distance * size / equity
            risk = (pos.sl_distance * pos.size) / self.config.start_balance
            total_risk += risk
        return total_risk
    
    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L."""
        pnl = 0.0
        for symbol, pos in self.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            if pos.side == "long":
                pnl += (current_price - pos.entry_price) * pos.size
            else:
                pnl += (pos.entry_price - current_price) * pos.size
        return pnl
    
    def current_equity(self, prices: Dict[str, float]) -> float:
        """Current equity including unrealized P&L."""
        return self.equity + self.unrealized_pnl(prices)
    
    def open_position(
        self,
        symbol: str,
        side: Literal["long", "short"],
        price: float,
        atr: float,
        timestamp: datetime,
        bar_idx: int
    ) -> Optional[Position]:
        """Open a new position."""
        # Apply slippage
        if side == "long":
            entry_price = price * (1 + self.config.slippage_pct)
        else:
            entry_price = price * (1 - self.config.slippage_pct)
        
        # Calculate SL distance: min(ATR, 1.2%)
        sl_from_atr = atr * self.config.atr_multiplier
        sl_from_pct = entry_price * self.config.sl_max_pct
        sl_distance = min(sl_from_atr, sl_from_pct)
        
        # SL and TP prices
        if side == "long":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + sl_distance * self.config.tp_multiplier
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - sl_distance * self.config.tp_multiplier
        
        # Position sizing based on risk
        risk_amount = self.equity * self.config.risk_per_trade
        size = risk_amount / sl_distance  # Size in base currency
        
        # Apply commission
        commission = entry_price * size * self.config.commission
        self.equity -= commission
        
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_distance=sl_distance,
            entry_time=timestamp,
            entry_bar_idx=bar_idx,
        )
        
        self.positions[symbol] = position
        self.trades_today += 1
        
        logger.debug(f"OPEN {side.upper()} {symbol} @ {entry_price:.4f}, SL={sl_price:.4f}, TP={tp_price:.4f}")
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: ExitReason,
        timestamp: datetime,
        partial_pct: float = 1.0
    ) -> Optional[Trade]:
        """Close a position (full or partial)."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Apply slippage for SL/TP
        if reason in [ExitReason.SL, ExitReason.BE]:
            if pos.side == "long":
                exit_price = exit_price * (1 - self.config.slippage_pct)
            else:
                exit_price = exit_price * (1 + self.config.slippage_pct)
        
        close_size = pos.size * partial_pct
        
        # Calculate P&L
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * close_size
        else:
            pnl = (pos.entry_price - exit_price) * close_size
        
        # Commission
        commission = exit_price * close_size * self.config.commission
        pnl -= commission
        
        # Update equity
        self.equity += pnl
        self.daily_pnl += pnl
        
        # Calculate R
        r_multiple = pnl / (pos.sl_distance * close_size) if pos.sl_distance > 0 else 0
        
        # Hold time
        hold_minutes = int((timestamp - pos.entry_time).total_seconds() / 60)
        
        trade = Trade(
            datetime_open=pos.entry_time,
            datetime_close=timestamp,
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            sl_price=pos.sl_price,
            tp_price=pos.tp_price,
            reason=reason.value,
            pnl_abs=pnl,
            pnl_pct=(pnl / self.equity) * 100,
            r_multiple=r_multiple,
            hold_minutes=hold_minutes,
        )
        self.trades.append(trade)
        
        logger.debug(f"CLOSE {pos.side.upper()} {symbol} @ {exit_price:.4f}, PnL={pnl:.2f}, R={r_multiple:.2f}, reason={reason.value}")
        
        # Remove or update position
        if partial_pct >= 1.0:
            del self.positions[symbol]
        else:
            pos.size *= (1 - partial_pct)
        
        # Check daily stop
        daily_pnl_pct = self.daily_pnl / (self.equity - self.daily_pnl) if self.equity != self.daily_pnl else 0
        if daily_pnl_pct <= self.config.daily_stop_pct:
            self.daily_stop_active = True
            logger.info(f"  DAILY STOP triggered: {daily_pnl_pct*100:.2f}%")
        
        return trade


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class Stage6Strategy:
    """Stage6 entry/exit logic."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def check_entry_signal(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        symbol: str,
        existing_positions: set,
        timestamp: 'datetime' = None  # Add timestamp for time filter
    ) -> Optional[Literal["long", "short"]]:
        """
        Check for entry signal per Stage6 spec with enhancements:
        - Time filter: exclude Asian session (00:00-05:59 UTC)
        - Enhanced breakout: close > high_2 (not just high)
        - Volume confirmation: current volume > prev volume
        """
        if symbol in existing_positions:
            return None
        
        # --- TIME FILTER: Exclude Asian session hours ---
        if timestamp is not None:
            hour = timestamp.hour
            if hour in self.config.excluded_hours:
                return None
        
        close = row["close"]
        high = row["high"]
        low = row["low"]
        ema200 = row["ema200"]
        rsi = row["rsi"]
        rsi_prev = row["rsi_prev"]
        volume_surge = row["volume_surge"]
        atr_pct = row["atr_pct"]
        high_2 = row.get("high_2", high)
        low_2 = row.get("low_2", low)
        prev_volume = prev_row.get("volume", 0) if prev_row is not None else 0
        curr_volume = row.get("volume", 0)
        
        # Skip if missing critical data
        if pd.isna(ema200) or pd.isna(rsi) or pd.isna(atr_pct):
            return None
        if pd.isna(rsi_prev) or pd.isna(volume_surge):
            return None
        
        # --- FILTER 1: ATR in range (volatility check) ---
        if not (self.config.atr_min_pct <= atr_pct <= self.config.atr_max_pct):
            return None
        
        # --- FILTER 2: Volume surge ---
        if volume_surge < self.config.volume_surge_mult:
            return None
        
        # --- FILTER 3: Dead zone check (±0.4% around EMA200) ---
        ema_upper_dead = ema200 * (1 + self.config.ema_dead_zone_pct)
        ema_lower_dead = ema200 * (1 - self.config.ema_dead_zone_pct)
        if ema_lower_dead <= close <= ema_upper_dead:
            return None  # In dead zone, no trades
        
        # --- FILTER 4: Volume rising (confirmation) ---
        volume_rising = curr_volume > prev_volume if prev_volume > 0 else True
        
        # === LONG SIGNAL ===
        if close > ema200 * self.config.ema_long_threshold:
            # RSI oversold
            if rsi <= self.config.rsi_oversold:
                # 1-candle RSI reversal: RSI_prev was below threshold and RSI rising
                if rsi_prev < self.config.rsi_oversold and rsi > rsi_prev:
                    # Enhanced breakout: CLOSE must be above 2-bar high (not just high)
                    if not pd.isna(high_2) and close > high_2:
                        # Volume confirmation
                        if volume_rising:
                            return "long"
        
        # === SHORT SIGNAL ===
        if close < ema200 * self.config.ema_short_threshold:
            # RSI overbought
            if rsi >= self.config.rsi_overbought:
                # 1-candle RSI reversal: RSI_prev was above threshold and RSI falling
                if rsi_prev > self.config.rsi_overbought and rsi < rsi_prev:
                    # Enhanced breakout: CLOSE must be below 2-bar low (not just low)
                    if not pd.isna(low_2) and close < low_2:
                        # Volume confirmation
                        if volume_rising:
                            return "short"
        
        return None


# =============================================================================
# POSITION MANAGER
# =============================================================================

class PositionManager:
    """Manages open positions: SL/TP/BE/Trailing/Timeout."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def check_exits(
        self,
        position: Position,
        high: float,
        low: float,
        close: float,
        bar_idx: int
    ) -> tuple[Optional[ExitReason], float, float]:
        """
        Check if position should exit.
        Returns: (reason, exit_price, partial_pct) or (None, 0, 0)
        """
        is_long = position.side == "long"
        
        # Calculate current profit in R
        if is_long:
            profit = close - position.entry_price
        else:
            profit = position.entry_price - close
        profit_r = profit / position.sl_distance if position.sl_distance > 0 else 0
        
        # --- TIMEOUT ---
        bars_held = bar_idx - position.entry_bar_idx
        if bars_held >= self.config.timeout_bars:
            return ExitReason.TIMEOUT, close, 1.0
        
        # --- SL/TP check (using high/low for simulation) ---
        if is_long:
            # SL hit?
            if low <= position.current_sl:
                return ExitReason.SL if not position.breakeven_done else ExitReason.BE, position.current_sl, 1.0
            # TP hit?
            if high >= position.tp_price:
                return ExitReason.TP, position.tp_price, 1.0
        else:
            # SL hit?
            if high >= position.current_sl:
                return ExitReason.SL if not position.breakeven_done else ExitReason.BE, position.current_sl, 1.0
            # TP hit?
            if low <= position.tp_price:
                return ExitReason.TP, position.tp_price, 1.0
        
        # --- BREAKEVEN at +1R ---
        if not position.breakeven_done and profit_r >= self.config.be_trigger_r:
            if is_long:
                position.current_sl = position.entry_price + position.sl_distance * self.config.be_offset_r
            else:
                position.current_sl = position.entry_price - position.sl_distance * self.config.be_offset_r
            position.breakeven_done = True
            logger.debug(f"  BE triggered for {position.symbol}, new SL={position.current_sl:.4f}")
        
        # --- PARTIAL CLOSE at +1.5R ---
        if not position.partial_done and profit_r >= self.config.partial_trigger_r:
            position.partial_done = True
            position.trailing_active = True
            return None, close, self.config.partial_pct  # Partial close 50%
        
        # --- TRAILING STOP after +1.5R ---
        if position.trailing_active and profit_r >= self.config.partial_trigger_r:
            trailing_r = profit_r - self.config.trailing_distance_r
            if is_long:
                new_sl = position.entry_price + position.sl_distance * trailing_r
                if new_sl > position.current_sl:
                    position.current_sl = new_sl
            else:
                new_sl = position.entry_price - position.sl_distance * trailing_r
                if new_sl < position.current_sl:
                    position.current_sl = new_sl
        
        return None, 0, 0


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate backtest reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        trades: List[Trade],
        daily_stats: List[DayStats],
        config: BacktestConfig,
        final_equity: float
    ):
        """Generate all 3 report files."""
        self._generate_trades_csv(trades)
        self._generate_daily_csv(daily_stats)
        self._generate_report_md(trades, daily_stats, config, final_equity)
    
    def _generate_trades_csv(self, trades: List[Trade]):
        """Generate trades.csv."""
        rows = []
        for t in trades:
            rows.append({
                "datetime_open": t.datetime_open.strftime("%Y-%m-%d %H:%M"),
                "datetime_close": t.datetime_close.strftime("%Y-%m-%d %H:%M"),
                "symbol": t.symbol,
                "side": t.side,
                "entry": f"{t.entry_price:.6f}",
                "exit": f"{t.exit_price:.6f}",
                "sl": f"{t.sl_price:.6f}",
                "tp": f"{t.tp_price:.6f}",
                "reason": t.reason,
                "pnl_abs": f"{t.pnl_abs:.4f}",
                "pnl_pct": f"{t.pnl_pct:.4f}",
                "R": f"{t.r_multiple:.2f}",
                "hold_min": t.hold_minutes,
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "trades.csv", index=False)
        logger.info(f"  Wrote {len(rows)} trades to trades.csv")
    
    def _generate_daily_csv(self, daily_stats: List[DayStats]):
        """Generate daily_stats.csv."""
        rows = []
        for d in daily_stats:
            rows.append({
                "date": d.date,
                "equity_open": f"{d.equity_open:.2f}",
                "equity_close": f"{d.equity_close:.2f}",
                "pnl_abs": f"{d.pnl_abs:.2f}",
                "pnl_pct": f"{d.pnl_pct:.2f}",
                "max_dd_pct": f"{d.max_dd_pct:.2f}",
                "trades": d.trades,
                "winrate": f"{d.winrate:.1f}",
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "daily_stats.csv", index=False)
        logger.info(f"  Wrote {len(rows)} days to daily_stats.csv")
    
    def _generate_report_md(
        self,
        trades: List[Trade],
        daily_stats: List[DayStats],
        config: BacktestConfig,
        final_equity: float,
        btc_return: float = 0.0  # Pass BTC return for alpha calculation
    ):
        """Generate report.md with advanced metrics and Go/No-Go decision."""
        # === Basic metrics ===
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl_abs > 0)
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        win_rate_dec = wins / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.pnl_abs for t in trades if t.pnl_abs > 0)
        total_loss = abs(sum(t.pnl_abs for t in trades if t.pnl_abs < 0))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0
        avg_hold = np.mean([t.hold_minutes for t in trades]) if trades else 0
        
        total_return = ((final_equity - config.start_balance) / config.start_balance) * 100
        avg_daily_return = total_return / len(daily_stats) if daily_stats else 0
        
        max_dd = min(d.max_dd_pct for d in daily_stats) if daily_stats else 0
        daily_stops = sum(1 for d in daily_stats if d.pnl_pct <= config.daily_stop_pct * 100)
        
        # === Advanced metrics ===
        
        # 1. Expectancy (R)
        win_rs = [t.r_multiple for t in trades if t.r_multiple > 0]
        loss_rs = [abs(t.r_multiple) for t in trades if t.r_multiple < 0]
        avg_win_r = np.mean(win_rs) if win_rs else 0
        avg_loss_r = np.mean(loss_rs) if loss_rs else 1
        expectancy = (win_rate_dec * avg_win_r) - ((1 - win_rate_dec) * avg_loss_r)
        
        # 2. Profit per Day (R/day)
        total_r = sum(t.r_multiple for t in trades)
        r_per_day = total_r / len(daily_stats) if daily_stats else 0
        
        # 3. Risk-Adjusted Return
        risk_adjusted = abs(total_return / max_dd) if max_dd != 0 else 0
        
        # 4. Win/Loss Streak
        streaks = self._calculate_streaks(trades)
        streak_ratio = streaks['max_win'] / streaks['max_loss'] if streaks['max_loss'] > 0 else float('inf')
        
        # 5. Sharpe Ratio (simplified)
        daily_returns = [d.pnl_pct for d in daily_stats]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) if np.std(daily_returns) > 0 else 0
        
        # 6. Alpha vs BTC
        alpha = total_return - btc_return
        
        # 7. Exit Efficiency (simplified - R achieved / max possible R)
        max_possible_r = 2.1  # TP multiplier
        exit_efficiency = avg_r / max_possible_r if max_possible_r > 0 else 0
        
        # 8. Stability (WR first half vs second half)
        mid = len(daily_stats) // 2
        first_half_trades = [t for t in trades if t.datetime_open.strftime("%Y-%m-%d") <= daily_stats[mid].date] if mid > 0 else trades[:len(trades)//2]
        second_half_trades = [t for t in trades if t.datetime_open.strftime("%Y-%m-%d") > daily_stats[mid].date] if mid > 0 else trades[len(trades)//2:]
        wr_first = sum(1 for t in first_half_trades if t.pnl_abs > 0) / len(first_half_trades) if first_half_trades else 0
        wr_second = sum(1 for t in second_half_trades if t.pnl_abs > 0) / len(second_half_trades) if second_half_trades else 0
        stability = wr_first / wr_second if wr_second > 0 else 1.0
        
        # === Go/No-Go Decision ===
        hard_conditions = {
            'Expectancy ≥0.25R': expectancy >= 0.25,
            'Risk-Adj ≥3.0': risk_adjusted >= 3.0,
            'Alpha ≥+5%': alpha >= 5.0,
        }
        soft_conditions = {
            'Streak Ratio ≥1.5': streak_ratio >= 1.5,
            'Exit Eff ≥0.6': exit_efficiency >= 0.6,
            'Sharpe ≥0.8': sharpe >= 0.8,
        }
        hard_pass = all(hard_conditions.values())
        soft_pass = sum(soft_conditions.values()) >= 2
        go_decision = hard_pass and soft_pass
        
        # Exit reason distribution
        reasons = {}
        for t in trades:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        
        # Top/worst symbols
        symbol_pnl = {}
        for t in trades:
            symbol_pnl[t.symbol] = symbol_pnl.get(t.symbol, 0) + t.pnl_abs
        sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
        
        # === Write report ===
        report = f"""# Stage6 Backtest Report

**Period**: {daily_stats[0].date if daily_stats else 'N/A'} to {daily_stats[-1].date if daily_stats else 'N/A'} ({len(daily_stats)} days)
**Start Balance**: ${config.start_balance:.2f}

---

## Results Summary

| Metric | Value |
|:-------|------:|
| **Final Balance** | ${final_equity:.2f} |
| **Total Return** | {total_return:+.2f}% |
| **Avg Daily Return** | {avg_daily_return:+.3f}% |
| **Max Drawdown** | {max_dd:.2f}% |
| **Total Trades** | {total_trades} |
| **Win Rate** | {win_rate:.1f}% |
| **Avg R** | {avg_r:.2f} |
| **Profit Factor** | {profit_factor:.2f} |
| **Avg Hold Time** | {avg_hold:.0f} min |
| **Daily Stop Days** | {daily_stops} |

---

## Advanced Metrics

| Metric | Value | Target | Status |
|:-------|------:|:-------|:------:|
| **Expectancy** | {expectancy:.3f}R | ≥0.25R | {'✅' if expectancy >= 0.25 else '❌'} |
| **R per Day** | {r_per_day:.2f}R | ≥0.5R | {'✅' if r_per_day >= 0.5 else '❌'} |
| **Risk-Adjusted** | {risk_adjusted:.2f} | ≥3.0 | {'✅' if risk_adjusted >= 3.0 else '❌'} |
| **Sharpe** | {sharpe:.2f} | ≥0.8 | {'✅' if sharpe >= 0.8 else '❌'} |
| **Win Streak Max** | {streaks['max_win']} | - | - |
| **Loss Streak Max** | {streaks['max_loss']} | - | - |
| **Streak Ratio** | {streak_ratio:.2f} | ≥1.5 | {'✅' if streak_ratio >= 1.5 else '❌'} |
| **Exit Efficiency** | {exit_efficiency:.2f} | ≥0.6 | {'✅' if exit_efficiency >= 0.6 else '❌'} |
| **Alpha vs BTC** | {alpha:+.1f}% | ≥+5% | {'✅' if alpha >= 5.0 else '❌'} |
| **Stability** | {stability:.2f} | 0.8-1.2 | {'✅' if 0.8 <= stability <= 1.2 else '❌'} |

---

## 🚦 Go/No-Go Decision

### Hard Conditions (ALL must pass)

| Condition | Status |
|:----------|:------:|
"""
        for cond, passed in hard_conditions.items():
            report += f"| {cond} | {'✅' if passed else '❌'} |\n"
        
        report += f"""
### Soft Conditions (≥2 of 3 must pass)

| Condition | Status |
|:----------|:------:|
"""
        for cond, passed in soft_conditions.items():
            report += f"| {cond} | {'✅' if passed else '❌'} |\n"
        
        if go_decision:
            report += f"\n> [!TIP]\n> **✅ GO FOR MICROLIVE** — All hard conditions passed, {sum(soft_conditions.values())}/3 soft conditions passed.\n"
        else:
            report += f"\n> [!CAUTION]\n> **❌ NOT READY FOR MICROLIVE** — Hard: {sum(hard_conditions.values())}/3, Soft: {sum(soft_conditions.values())}/3.\n"
        
        report += f"""
---

## Exit Reason Distribution

| Reason | Count | % |
|:-------|------:|--:|
"""
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_trades * 100 if total_trades > 0 else 0
            report += f"| {reason} | {count} | {pct:.1f}% |\n"
        
        report += f"""
---

## Top 5 Symbols

| Profitable | PnL | Losing | PnL |
|:-----------|----:|:-------|----:|
"""
        top5 = sorted_symbols[:5]
        bot5 = sorted_symbols[-5:][::-1]
        for i in range(5):
            top_sym = top5[i] if i < len(top5) else ("-", 0)
            bot_sym = bot5[i] if i < len(bot5) else ("-", 0)
            report += f"| {top_sym[0]} | ${top_sym[1]:+.2f} | {bot_sym[0]} | ${bot_sym[1]:+.2f} |\n"
        
        with open(self.output_dir / "report.md", "w") as f:
            f.write(report)
        logger.info(f"  Wrote report.md")
    
    def _calculate_streaks(self, trades: List[Trade]) -> dict:
        """Calculate max win and loss streaks."""
        max_win = max_loss = current_win = current_loss = 0
        for t in trades:
            if t.pnl_abs > 0:
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)
        return {'max_win': max_win, 'max_loss': max_loss}


# =============================================================================
# MAIN BACKTESTER
# =============================================================================

async def run_backtest():
    """Main backtest loop."""
    config = BacktestConfig()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / "data" / "backtest_cache"
    output_dir = base_dir / "backtest_results"
    
    # Fetch data
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(SYMBOLS, config.lookback_days, config.timeframe)
    
    if not data:
        logger.error("No data fetched, aborting backtest")
        return
    
    logger.info(f"Loaded data for {len(data)} symbols")
    
    # Get common date range
    min_len = min(len(df) for df in data.values())
    logger.info(f"Using {min_len} bars per symbol")
    
    # Initialize components
    exchange = ExchangeSimulator(config)
    strategy = Stage6Strategy(config)
    pos_manager = PositionManager(config)
    report_gen = ReportGenerator(output_dir)
    
    # Align all dataframes
    dfs = {}
    for symbol, df in data.items():
        dfs[symbol] = df.iloc[-min_len:].reset_index(drop=True)
    
    # Get reference index
    ref_df = list(dfs.values())[0]
    
    # Daily tracking
    daily_stats = []
    current_day = None
    day_start_equity = config.start_balance
    day_trades = []
    day_wins = 0
    peak_equity_day = config.start_balance
    
    logger.info("Starting backtest simulation...")
    
    # Main loop
    for bar_idx in range(200, min_len):  # Skip first 200 for indicators
        timestamp = ref_df.index[bar_idx] if hasattr(ref_df.index[bar_idx], 'strftime') else datetime.now()
        if isinstance(ref_df.index, pd.RangeIndex):
            timestamp = datetime.now() - timedelta(minutes=(min_len - bar_idx) * 5)
        
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # New day?
        if date_str != current_day:
            # Save previous day stats
            if current_day is not None:
                equity_now = exchange.equity
                daily_stats.append(DayStats(
                    date=current_day,
                    equity_open=day_start_equity,
                    equity_close=equity_now,
                    pnl_abs=equity_now - day_start_equity,
                    pnl_pct=(equity_now - day_start_equity) / day_start_equity * 100,
                    max_dd_pct=exchange.max_dd_today,
                    trades=len(day_trades),
                    wins=day_wins,
                ))
            
            # Reset for new day
            current_day = date_str
            day_start_equity = exchange.equity
            day_trades = []
            day_wins = 0
            peak_equity_day = exchange.equity
            exchange.reset_day(date_str)
        
        # Collect current prices
        current_prices = {}
        for symbol, df in dfs.items():
            if bar_idx < len(df):
                current_prices[symbol] = df.iloc[bar_idx]["close"]
        
        # Track max DD
        current_eq = exchange.current_equity(current_prices)
        if current_eq > peak_equity_day:
            peak_equity_day = current_eq
        dd = (current_eq - peak_equity_day) / peak_equity_day * 100 if peak_equity_day > 0 else 0
        if dd < exchange.max_dd_today:
            exchange.max_dd_today = dd
        
        # Check exits for open positions
        for symbol in list(exchange.positions.keys()):
            if symbol not in dfs or bar_idx >= len(dfs[symbol]):
                continue
            
            row = dfs[symbol].iloc[bar_idx]
            pos = exchange.positions[symbol]
            
            reason, exit_price, partial_pct = pos_manager.check_exits(
                pos, row["high"], row["low"], row["close"], bar_idx
            )
            
            if reason:
                trade = exchange.close_position(symbol, exit_price, reason, timestamp)
                if trade:
                    day_trades.append(trade)
                    if trade.pnl_abs > 0:
                        day_wins += 1
            elif partial_pct > 0 and partial_pct < 1:
                # Partial close
                trade = exchange.close_position(symbol, exit_price, ExitReason.TRAILING, timestamp, partial_pct)
                if trade:
                    day_trades.append(trade)
                    if trade.pnl_abs > 0:
                        day_wins += 1
        
        # Check for new entries
        can_trade, _ = exchange.can_open_trade()
        if can_trade:
            for symbol, df in dfs.items():
                if bar_idx >= len(df) or bar_idx < 1:
                    continue
                
                row = df.iloc[bar_idx]
                prev_row = df.iloc[bar_idx - 1]
                
                signal = strategy.check_entry_signal(
                    row, prev_row, symbol, set(exchange.positions.keys()), timestamp
                )
                
                if signal and exchange.can_open_trade()[0]:
                    atr = row["atr"]
                    if pd.notna(atr) and atr > 0:
                        exchange.open_position(
                            symbol, signal, row["close"], atr, timestamp, bar_idx
                        )
        
        # Progress
        if bar_idx % 1000 == 0:
            logger.info(f"  Bar {bar_idx}/{min_len}, Equity: ${exchange.equity:.2f}, Positions: {len(exchange.positions)}")
    
    # Final day stats
    if current_day:
        equity_final = exchange.equity
        daily_stats.append(DayStats(
            date=current_day,
            equity_open=day_start_equity,
            equity_close=equity_final,
            pnl_abs=equity_final - day_start_equity,
            pnl_pct=(equity_final - day_start_equity) / day_start_equity * 100,
            max_dd_pct=exchange.max_dd_today,
            trades=len(day_trades),
            wins=day_wins,
        ))
    
    # Generate reports
    logger.info("Generating reports...")
    report_gen.generate(exchange.trades, daily_stats, config, exchange.equity)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"BACKTEST COMPLETE")
    logger.info(f"  Final Equity: ${exchange.equity:.2f}")
    logger.info(f"  Total Return: {((exchange.equity - config.start_balance) / config.start_balance) * 100:+.2f}%")
    logger.info(f"  Total Trades: {len(exchange.trades)}")
    logger.info(f"  Reports saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_backtest())
