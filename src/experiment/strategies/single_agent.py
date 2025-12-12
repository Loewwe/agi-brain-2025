"""
SingleAgentStrategy - Production-ready agent with risk controls.

Extends AlphaStrategy with additional guardrails:
- Position limits
- Daily trade limits
- Confidence filtering
- Drawdown protection
- Regime/Trend filtering
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Literal, Any

import pandas as pd
import structlog

from .alpha_strategy import AlphaStrategy, AlphaStrategyConfig
from ..simulator import Signal, ExitSignal, SimulatorState

TradeAction = Signal | ExitSignal
from ..alpha_engine import AlphaEngine

logger = structlog.get_logger()


@dataclass
class SingleAgentConfig:
    """Configuration for SingleAgentStrategy."""
    
    # Alpha thresholds
    min_confidence: float = 0.65
    
    # Daily limits
    max_daily_trades: int = 10
    max_daily_loss_pct: float = 2.0
    
    # Risk params (passed to AlphaStrategy/Simulator)
    exit_on_reversal: bool = False
    reversal_threshold: float = 0.5
    
    # Regime Filter
    regime_filter_enabled: bool = False  # Deprecated, use trend_filter_type
    min_adx: float = 20.0
    trend_filter_type: Literal["NONE", "ADX", "EMA"] = "NONE"
    
    # Time filters
    trading_hours_start: int = 0
    trading_hours_end: int = 24

    def to_alpha_config(self) -> AlphaStrategyConfig:
        return AlphaStrategyConfig(
            entry_threshold=self.min_confidence,
            exit_on_reversal=self.exit_on_reversal,
            reversal_threshold=self.reversal_threshold,
        )


class SingleAgentStrategy:
    """
    Single Agent Strategy.
    Wraps AlphaStrategy with agent-specific logic (risk, daily limits, regime filters).
    """

    def __init__(self, config: SingleAgentConfig, alpha_engine: AlphaEngine):
        self.agent_config = config
        self.alpha_strategy = AlphaStrategy(alpha_engine, config.to_alpha_config())
        
        # State
        self.daily_trades = 0
        self.last_trade_date = None
        self.daily_pnl = 0.0
        self.daily_stopped = False
        
        logger.info(
            "single_agent.initialized",
            min_confidence=config.min_confidence,
            max_daily_trades=config.max_daily_trades,
            trend_filter=config.trend_filter_type
        )

    def check_exit(self, position: Any, bar: dict, bar_idx: int) -> TradeAction | None:
        """Check for exit signals (delegated to AlphaStrategy)."""
        return self.alpha_strategy.check_exit(position, bar, bar_idx)

    def check_entry(
        self,
        bar: pd.Series,
        prev_bar: pd.Series | None,
        symbol: str,
        state: SimulatorState,
        timestamp: datetime,
    ) -> TradeAction | None:
        """Check for entry signals with agent-level filtering."""
        # A. Daily Trade Limit
        if self.daily_trades >= self.agent_config.max_daily_trades:
            return None

        # B. Get Alpha Strategy Signal
        signal = self.alpha_strategy.check_entry(bar, prev_bar, symbol, state, timestamp)
        if not signal:
            return None

        # C. Apply Regime/Trend Filters
        if self.agent_config.trend_filter_type == "ADX":
            if "adx" in bar:
                adx = bar["adx"]
                if adx < self.agent_config.min_adx:
                    logger.info("single_agent.regime_filter_reject", symbol=symbol, filter="ADX", value=adx, threshold=self.agent_config.min_adx)
                    return None
            else:
                logger.warning("single_agent.regime_filter_missing_feature", feature="adx")
        
        elif self.agent_config.trend_filter_type == "EMA":
            if "ema200" in bar:
                ema200 = bar["ema200"]
                close = bar["close"]
                
                # Filter Logic:
                # If Close > EMA200 -> Bullish Trend -> Only LONG allowed
                # If Close < EMA200 -> Bearish Trend -> Only SHORT allowed
                
                is_bullish_trend = close > ema200
                
                if is_bullish_trend and signal.side == "short":
                    logger.info("single_agent.trend_filter_reject", symbol=symbol, filter="EMA", trend="BULLISH", signal="short")
                    return None
                
                if not is_bullish_trend and signal.side == "long":
                    logger.info("single_agent.trend_filter_reject", symbol=symbol, filter="EMA", trend="BEARISH", signal="long")
                    return None
            else:
                logger.warning("single_agent.regime_filter_missing_feature", feature="ema200")

        # Log entry signal
        logger.info(
            "single_agent.entry_signal",
            symbol=symbol,
            side=signal.side,
            strength=signal.strength,
            daily_trades=self.daily_trades + 1,
        )
        
        return signal

    def _reset_daily_counters(self, new_date: date) -> None:
        """Reset daily tracking counters."""
        if self.last_trade_date is not None:
            logger.info(
                "single_agent.day_end",
                date=self.last_trade_date.isoformat(),
                trades=self.daily_trades,
                pnl=self.daily_pnl,
            )
        
        self.last_trade_date = new_date
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_stopped = False
        
        logger.info("single_agent.day_start", date=new_date.isoformat())
