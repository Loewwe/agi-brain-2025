"""
Simulator - Deterministic backtest engine for Stage7.

Refactored from backtest_stage6.py with:
- Clean separation of concerns
- Abstract strategy interface
- Deterministic execution (same input → same output)
- No external API calls
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import structlog

from .models import (
    SimulatorConfig,
    SimulatorResult,
    ExperimentResult,
    ExitReason,
    calculate_checksum,
)

logger = structlog.get_logger()


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
        if self.current_sl == 0.0:
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


@dataclass
class Signal:
    """Entry signal from strategy."""
    side: Literal["long", "short"]
    symbol: str
    strength: float = 1.0


@dataclass
class ExitSignal:
    """Exit signal from strategy."""
    reason: ExitReason
    exit_price: float
    partial_pct: float = 1.0


@dataclass
class SimulatorState:
    """Current state of the simulator."""
    equity: float
    positions: dict[str, Position]
    trades_today: int
    daily_pnl: float
    daily_stop_active: bool
    peak_equity: float
    current_date: str


# =============================================================================
# STRATEGY BASE CLASS
# =============================================================================

class StrategyBase(ABC):
    """
    Abstract base class for strategies.
    
    Implement check_entry() and check_exit() for your strategy logic.
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
    
    @abstractmethod
    def check_entry(
        self,
        bar: pd.Series,
        prev_bar: pd.Series | None,
        symbol: str,
        state: SimulatorState,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Check for entry signal.
        
        Args:
            bar: Current bar data with features
            prev_bar: Previous bar data (for reversal detection)
            symbol: Trading symbol
            state: Current simulator state
            timestamp: Current timestamp
            
        Returns:
            Signal if entry conditions met, None otherwise
        """
        pass
    
    @abstractmethod
    def check_exit(
        self,
        position: Position,
        bar: pd.Series,
        bar_idx: int,
    ) -> ExitSignal | None:
        """
        Check for exit signal.
        
        Args:
            position: Current position
            bar: Current bar data
            bar_idx: Current bar index
            
        Returns:
            ExitSignal if exit conditions met, None otherwise
        """
        pass


# =============================================================================
# STAGE6 STRATEGY IMPLEMENTATION
# =============================================================================

class Stage6Strategy(StrategyBase):
    """
    Stage6 strategy implementation.
    
    Entry conditions:
    - Trend: EMA200 filter with dead zone
    - Momentum: RSI oversold/overbought with reversal
    - Volatility: ATR in range
    - Volume: Surge above SMA
    
    Exit conditions:
    - SL/TP hit
    - Breakeven at +1R
    - Partial close at +1.5R
    - Trailing stop after +1.5R
    - Timeout at 40 bars
    """
    
    def check_entry(
        self,
        bar: pd.Series,
        prev_bar: pd.Series | None,
        symbol: str,
        state: SimulatorState,
        timestamp: datetime,
    ) -> Signal | None:
        """Check for entry signal per Stage6 spec."""
        # Skip if already in position for this symbol
        if symbol in state.positions:
            return None
        
        # Time filter: exclude Asian session
        if timestamp.hour in self.config.excluded_hours:
            return None
        
        # Get values
        close = bar.get("close")
        ema200 = bar.get("ema200")
        rsi = bar.get("rsi")
        rsi_prev = bar.get("rsi_prev")
        volume_surge = bar.get("volume_surge")
        atr_pct = bar.get("atr_pct")
        high_2 = bar.get("high_2", bar.get("high"))
        low_2 = bar.get("low_2", bar.get("low"))
        
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
        
        # --- FILTER 4: Volume rising ---
        prev_volume = prev_bar.get("volume", 0) if prev_bar is not None else 0
        curr_volume = bar.get("volume", 0)
        volume_rising = curr_volume > prev_volume if prev_volume > 0 else True
        
        # === LONG SIGNAL ===
        if close > ema200 * self.config.ema_long_threshold:
            if rsi <= self.config.rsi_oversold:
                # 1-candle RSI reversal
                if rsi_prev < self.config.rsi_oversold and rsi > rsi_prev:
                    # Breakout check
                    if not pd.isna(high_2) and close > high_2:
                        if volume_rising:
                            return Signal(side="long", symbol=symbol)
        
        # === SHORT SIGNAL ===
        if close < ema200 * self.config.ema_short_threshold:
            if rsi >= self.config.rsi_overbought:
                # 1-candle RSI reversal
                if rsi_prev > self.config.rsi_overbought and rsi < rsi_prev:
                    # Breakout check
                    if not pd.isna(low_2) and close < low_2:
                        if volume_rising:
                            return Signal(side="short", symbol=symbol)
        
        return None
    
    def check_exit(
        self,
        position: Position,
        bar: pd.Series,
        bar_idx: int,
    ) -> ExitSignal | None:
        """Check for exit signal."""
        is_long = position.side == "long"
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        
        # Calculate current profit in R
        if is_long:
            profit = close - position.entry_price
        else:
            profit = position.entry_price - close
        profit_r = profit / position.sl_distance if position.sl_distance > 0 else 0
        
        # --- TIMEOUT ---
        bars_held = bar_idx - position.entry_bar_idx
        if bars_held >= self.config.timeout_bars:
            return ExitSignal(reason=ExitReason.TIMEOUT, exit_price=close)
        
        # --- SL/TP check (using high/low for simulation) ---
        if is_long:
            if low <= position.current_sl:
                reason = ExitReason.BREAKEVEN if position.breakeven_done else ExitReason.SL
                return ExitSignal(reason=reason, exit_price=position.current_sl)
            if high >= position.tp_price:
                return ExitSignal(reason=ExitReason.TP, exit_price=position.tp_price)
        else:
            if high >= position.current_sl:
                reason = ExitReason.BREAKEVEN if position.breakeven_done else ExitReason.SL
                return ExitSignal(reason=reason, exit_price=position.current_sl)
            if low <= position.tp_price:
                return ExitSignal(reason=ExitReason.TP, exit_price=position.tp_price)
        
        # --- BREAKEVEN at +1R ---
        if not position.breakeven_done and profit_r >= self.config.be_trigger_r:
            if is_long:
                position.current_sl = position.entry_price + position.sl_distance * self.config.be_offset_r
            else:
                position.current_sl = position.entry_price - position.sl_distance * self.config.be_offset_r
            position.breakeven_done = True
        
        # --- PARTIAL CLOSE at +1.5R ---
        if not position.partial_done and profit_r >= self.config.partial_trigger_r:
            position.partial_done = True
            position.trailing_active = True
            return ExitSignal(
                reason=ExitReason.TP,
                exit_price=close,
                partial_pct=self.config.partial_pct,
            )
        
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
        
        return None


# =============================================================================
# SIMULATOR
# =============================================================================

class Simulator:
    """
    Deterministic backtest engine.
    
    Features:
    - No external API calls
    - Same input → same output
    - Complete metrics calculation
    - Trade-by-trade logging
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize Simulator.
        
        Args:
            config: Simulator configuration
        """
        self.config = config
        
        # Set random seed for determinism
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # State
        self.equity = config.start_balance
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.daily_stats: list[DayStats] = []
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.current_date: str | None = None
        self.daily_stop_active = False
        self.peak_equity = config.start_balance
        self.max_dd_today = 0.0
        self.day_equity_open = config.start_balance
        self.day_wins = 0
        
        logger.info("simulator.initialized", config=config.to_dict())
    
    def reset(self) -> None:
        """Reset simulator state."""
        self.equity = self.config.start_balance
        self.positions = {}
        self.trades = []
        self.daily_stats = []
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.current_date = None
        self.daily_stop_active = False
        self.peak_equity = self.config.start_balance
        self.max_dd_today = 0.0
        self.day_equity_open = self.config.start_balance
        self.day_wins = 0
    
    def run(
        self,
        dataset: pd.DataFrame,
        strategy: StrategyBase,
    ) -> SimulatorResult:
        """
        Run backtest on dataset.
        
        Args:
            dataset: OHLCV + features DataFrame
            strategy: Strategy implementation
            
        Returns:
            Complete backtest results
        """
        self.reset()
        
        # Group by symbol for multi-asset backtest
        symbols = dataset["symbol"].unique().tolist() if "symbol" in dataset.columns else ["UNKNOWN"]
        
        logger.info(
            "simulator.starting",
            symbols=len(symbols),
            rows=len(dataset),
        )
        
        # Get unique timestamps
        if isinstance(dataset.index, pd.DatetimeIndex):
            timestamps = dataset.index.unique().sort_values()
        else:
            timestamps = pd.to_datetime(dataset.index.unique()).sort_values()
        
        prev_bars: dict[str, pd.Series] = {}
        
        for bar_idx, timestamp in enumerate(timestamps):
            # Get all bars for this timestamp
            bars_at_time = dataset.loc[timestamp]
            
            # Handle single row case
            if isinstance(bars_at_time, pd.Series):
                bars_at_time = pd.DataFrame([bars_at_time])
            
            # Check for day change
            date_str = timestamp.strftime("%Y-%m-%d")
            if self.current_date != date_str:
                self._handle_day_change(date_str)
            
            # Process each symbol
            for _, bar in bars_at_time.iterrows():
                symbol = bar.get("symbol", "UNKNOWN")
                
                # Get current prices for equity calculation
                current_prices = {symbol: bar["close"]}
                
                # Update max drawdown
                current_eq = self._current_equity(current_prices)
                if current_eq > self.peak_equity:
                    self.peak_equity = current_eq
                dd_pct = (current_eq - self.peak_equity) / self.peak_equity * 100
                if dd_pct < self.max_dd_today:
                    self.max_dd_today = dd_pct
                
                # Create state for strategy
                state = SimulatorState(
                    equity=self.equity,
                    positions=self.positions,
                    trades_today=self.trades_today,
                    daily_pnl=self.daily_pnl,
                    daily_stop_active=self.daily_stop_active,
                    peak_equity=self.peak_equity,
                    current_date=self.current_date or date_str,
                )
                
                # Check exits first
                if symbol in self.positions:
                    exit_signal = strategy.check_exit(
                        self.positions[symbol],
                        bar,
                        bar_idx,
                    )
                    if exit_signal:
                        self._close_position(
                            symbol,
                            exit_signal.exit_price,
                            exit_signal.reason,
                            timestamp,
                            exit_signal.partial_pct,
                        )
                
                # Check entries
                if not self.daily_stop_active:
                    prev_bar = prev_bars.get(symbol)
                    entry_signal = strategy.check_entry(
                        bar,
                        prev_bar,
                        symbol,
                        state,
                        timestamp,
                    )
                    
                    if entry_signal and self._can_open_trade():
                        atr = bar.get("atr", bar["close"] * 0.01)
                        self._open_position(
                            symbol,
                            entry_signal.side,
                            bar["close"],
                            atr,
                            timestamp,
                            bar_idx,
                        )
                
                # Update prev bar
                prev_bars[symbol] = bar
        
        # Finalize last day
        if self.current_date:
            self._finalize_day()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Create result
        result = SimulatorResult(
            metrics=metrics,
            config=self.config,
            trades_count=len(self.trades),
            days_count=len(self.daily_stats),
            checksum=calculate_checksum({
                "trades": len(self.trades),
                "final_equity": self.equity,
                "total_pnl": sum(t.pnl_abs for t in self.trades),
            }),
        )
        
        logger.info(
            "simulator.completed",
            trades=len(self.trades),
            final_equity=f"{self.equity:.2f}",
            total_return=f"{metrics.total_return_pct:.2f}%",
        )
        
        return result
    
    def _can_open_trade(self) -> bool:
        """Check if we can open a new trade."""
        if self.daily_stop_active:
            return False
        if len(self.positions) >= self.config.max_positions:
            return False
        if self.trades_today >= self.config.max_trades_day:
            return False
        
        # Check max open risk
        current_risk = self._calculate_open_risk()
        if current_risk + self.config.risk_per_trade > self.config.max_open_risk_pct:
            return False
        
        return True
    
    def _calculate_open_risk(self) -> float:
        """Calculate total open risk as % of starting equity."""
        total_risk = 0.0
        for pos in self.positions.values():
            risk = (pos.sl_distance * pos.size) / self.config.start_balance
            total_risk += risk
        return total_risk
    
    def _current_equity(self, prices: dict[str, float]) -> float:
        """Calculate current equity including unrealized P&L."""
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            if pos.side == "long":
                unrealized += (current_price - pos.entry_price) * pos.size
            else:
                unrealized += (pos.entry_price - current_price) * pos.size
        return self.equity + unrealized
    
    def _open_position(
        self,
        symbol: str,
        side: Literal["long", "short"],
        price: float,
        atr: float,
        timestamp: datetime,
        bar_idx: int,
    ) -> Position:
        """Open a new position."""
        # Apply slippage
        if side == "long":
            entry_price = price * (1 + self.config.slippage_pct)
        else:
            entry_price = price * (1 - self.config.slippage_pct)
        
        # Calculate SL distance
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
        
        # Position sizing
        risk_amount = self.equity * self.config.risk_per_trade
        size = risk_amount / sl_distance
        
        # Commission
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
        
        return position
    
    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: ExitReason,
        timestamp: datetime,
        partial_pct: float = 1.0,
    ) -> Trade | None:
        """Close a position."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Apply slippage for SL
        if reason in [ExitReason.SL, ExitReason.BREAKEVEN]:
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
        
        # Track wins
        if pnl > 0:
            self.day_wins += 1
        
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
        
        # Remove or update position
        if partial_pct >= 1.0:
            del self.positions[symbol]
        else:
            pos.size *= (1 - partial_pct)
        
        # Check daily stop
        daily_pnl_pct = self.daily_pnl / (self.equity - self.daily_pnl) if self.equity != self.daily_pnl else 0
        if daily_pnl_pct <= self.config.daily_stop_pct:
            self.daily_stop_active = True
        
        return trade
    
    def _handle_day_change(self, new_date: str) -> None:
        """Handle transition to a new day."""
        if self.current_date:
            self._finalize_day()
        
        self.current_date = new_date
        self.day_equity_open = self.equity
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.daily_stop_active = False
        self.max_dd_today = 0.0
        self.day_wins = 0
    
    def _finalize_day(self) -> None:
        """Finalize statistics for the current day."""
        pnl_pct = (self.daily_pnl / self.day_equity_open) * 100 if self.day_equity_open > 0 else 0
        
        day_trades = [t for t in self.trades if t.datetime_close.strftime("%Y-%m-%d") == self.current_date]
        
        stats = DayStats(
            date=self.current_date,
            equity_open=self.day_equity_open,
            equity_close=self.equity,
            pnl_abs=self.daily_pnl,
            pnl_pct=pnl_pct,
            max_dd_pct=self.max_dd_today,
            trades=len(day_trades),
            wins=self.day_wins,
        )
        self.daily_stats.append(stats)
    
    def _calculate_metrics(self) -> ExperimentResult:
        """Calculate all metrics from trades."""
        total_trades = len(self.trades)
        
        if total_trades == 0:
            return ExperimentResult(
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_r_multiple=0.0,
                expectancy=0.0,
                total_return_pct=0.0,
                avg_daily_pnl_pct=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                risk_adjusted_return=0.0,
            )
        
        # Basic metrics
        wins = sum(1 for t in self.trades if t.pnl_abs > 0)
        losses = total_trades - wins
        win_rate = wins / total_trades
        
        total_profit = sum(t.pnl_abs for t in self.trades if t.pnl_abs > 0)
        total_loss = abs(sum(t.pnl_abs for t in self.trades if t.pnl_abs < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_r = np.mean([t.r_multiple for t in self.trades])
        avg_hold = np.mean([t.hold_minutes for t in self.trades])
        
        # P&L metrics
        total_return = ((self.equity - self.config.start_balance) / self.config.start_balance) * 100
        avg_daily = total_return / len(self.daily_stats) if self.daily_stats else 0
        
        max_dd = min((d.max_dd_pct for d in self.daily_stats), default=0)
        
        # Expectancy
        win_rs = [t.r_multiple for t in self.trades if t.r_multiple > 0]
        loss_rs = [abs(t.r_multiple) for t in self.trades if t.r_multiple < 0]
        avg_win_r = np.mean(win_rs) if win_rs else 0
        avg_loss_r = np.mean(loss_rs) if loss_rs else 1
        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
        
        # Sharpe
        daily_returns = [d.pnl_pct for d in self.daily_stats]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) if np.std(daily_returns) > 0 else 0
        
        # Risk-adjusted return
        risk_adjusted = abs(total_return / max_dd) if max_dd != 0 else 0
        
        # Exit distribution
        exits_by_reason: dict[str, int] = {}
        for t in self.trades:
            exits_by_reason[t.reason] = exits_by_reason.get(t.reason, 0) + 1
        
        return ExperimentResult(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_r_multiple=avg_r,
            expectancy=expectancy,
            total_return_pct=total_return,
            avg_daily_pnl_pct=avg_daily,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            risk_adjusted_return=risk_adjusted,
            wins=wins,
            losses=losses,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            avg_hold_minutes=avg_hold,
            exits_by_reason=exits_by_reason,
        )
    
    def get_trades(self) -> list[Trade]:
        """Get list of all trades."""
        return self.trades.copy()
    
    def get_daily_stats(self) -> list[DayStats]:
        """Get daily statistics."""
        return self.daily_stats.copy()
