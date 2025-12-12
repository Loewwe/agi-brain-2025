"""
AlphaStrategy - Strategy using AlphaEngine predictions.

Integrates AlphaEngine with Simulator for alpha-based trading signals.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from ..simulator import StrategyBase, Signal, ExitSignal, SimulatorState, Position
from ..alpha_engine import AlphaEngine
from ..models import SimulatorConfig, ExitReason

logger = structlog.get_logger()


@dataclass
class AlphaStrategyConfig:
    """Configuration for AlphaStrategy."""
    
    # Alpha thresholds
    entry_threshold: float = 0.6         # Min alpha_score for entry
    exit_threshold: float = 0.4          # Exit if alpha drops below
    
    # Direction control
    allow_long: bool = True
    allow_short: bool = True
    
    # Risk management
    sl_atr_multiplier: float = 1.5       # SL = ATR * multiplier
    tp_atr_multiplier: float = 3.0       # TP = ATR * multiplier
    
    # Trade management (inherited from SimulatorConfig    # Exit logic
    use_trailing: bool = False
    trailing_activation_r: float = 1.5
    trailing_distance_r: float = 0.5
    
    # Signal Reversal Exit
    exit_on_reversal: bool = False
    reversal_threshold: float = 0.5  # Exit if alpha score drops below this (for long) or above 1-this (for short)


class AlphaStrategy(StrategyBase):
    """
    Strategy that uses AlphaEngine for entry/exit decisions.
    
    Workflow:
    1. On each bar, get alpha prediction from engine
    2. If alpha_score > threshold, generate entry signal
    3. Exit based on SL/TP hit or alpha reversal
    
    Example:
        engine = AlphaEngine.load("model_path")
        strategy = AlphaStrategy(engine, config)
        result = simulator.run(dataset, strategy)
    """
    
    def __init__(
        self,
        alpha_engine: AlphaEngine,
        config: AlphaStrategyConfig | None = None,
        sim_config: SimulatorConfig | None = None,
    ):
        """
        Initialize AlphaStrategy.
        
        Args:
            alpha_engine: Trained AlphaEngine
            config: Strategy configuration
            sim_config: Simulator configuration (for risk params)
        """
        self.alpha_engine = alpha_engine
        self.config = config or AlphaStrategyConfig()
        self.sim_config = sim_config or SimulatorConfig()
        
        # Cache for predictions
        self._predictions_cache: dict[str, float] = {}
        
        logger.info(
            "alpha_strategy.initialized",
            entry_threshold=self.config.entry_threshold,
        )
    
    def check_entry(
        self,
        bar: pd.Series,
        prev_bar: pd.Series,
        symbol: str,
        state: SimulatorState,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Check for entry signal based on alpha prediction.
        
        Args:
            bar: Current OHLCV bar with features
            prev_bar: Previous bar
            symbol: Trading symbol
            state: Current simulator state
            timestamp: Current timestamp
            
        Returns:
            Signal if entry conditions met, None otherwise
        """
        # Skip if already in position for this symbol
        if symbol in state.positions:
            return None
        
        # Skip if max positions reached
        if len(state.positions) >= self.sim_config.max_positions:
            return None
        
        # Get alpha prediction
        try:
            features = self._extract_features(bar)
            prediction = self.alpha_engine.predict_single(features)
        except Exception as e:
            logger.warning("alpha_strategy.prediction_error", error=str(e))
            return None
        
        alpha_score = prediction["alpha_score"]
        
        # Check if score meets threshold
        if alpha_score < self.config.entry_threshold and alpha_score > (1 - self.config.entry_threshold):
            return None  # Not confident enough
        
        # Determine direction
        if alpha_score >= self.config.entry_threshold:
            # Bullish signal
            if not self.config.allow_long:
                return None
            side = "long"
            strength = alpha_score
        else:
            # Bearish signal (alpha_score < 1 - threshold)
            if not self.config.allow_short:
                return None
            side = "short"
            strength = 1 - alpha_score
        
        logger.debug(
            "alpha_strategy.signal",
            symbol=symbol,
            side=side,
            alpha_score=alpha_score,
            strength=strength,
        )
        
        # Return Signal (Simulator calculates SL/TP based on ATR)
        return Signal(side=side, symbol=symbol, strength=strength)
    
    def check_exit(
        self,
        position: Position,
        bar: pd.Series,
        bar_idx: int,
    ) -> ExitSignal | None:
        """
        Check for exit signal.
        
        Uses standard SL/TP hit + alpha reversal for early exit.
        
        Args:
            position: Current position
            bar: Current bar
            bar_idx: Bar index
            
        Returns:
            ExitSignal if exit conditions met
        """
        # Check SL hit
        if position.side == "long":
            if bar["low"] <= position.current_sl:
                return ExitSignal(
                    reason=ExitReason.SL,
                    exit_price=position.current_sl,
                )
            if bar["high"] >= position.tp_price:
                return ExitSignal(
                    reason=ExitReason.TP,
                    exit_price=position.tp_price,
                )
        else:  # short
            if bar["high"] >= position.current_sl:
                return ExitSignal(
                    reason=ExitReason.SL,
                    exit_price=position.current_sl,
                )
            if bar["low"] <= position.tp_price:
                return ExitSignal(
                    reason=ExitReason.TP,
                    exit_price=position.tp_price,
                )
        
        # Check trailing stop (if activated)
        if self.config.use_trailing and position.trailing_active:
            if position.side == "long":
                if bar["low"] <= position.current_sl:
                    return ExitSignal(
                        reason=ExitReason.TRAILING,
                        exit_price=position.current_sl,
                    )
            else:
                if bar["high"] >= position.current_sl:
                    return ExitSignal(
                        reason=ExitReason.TRAILING,
                        exit_price=position.current_sl,
                    )
        
        # Check timeout
        hold_bars = bar_idx - position.entry_bar_idx
        max_bars = self.sim_config.timeout_bars  # Use config timeout_bars
        if hold_bars >= max_bars:
            return ExitSignal(
                reason=ExitReason.TIMEOUT,
                exit_price=bar["close"],
            )
            
        # Check Signal Reversal
        if self.config.exit_on_reversal:
            # Predict on current bar
            features = self._extract_features(bar)
            # We use predict_single which expects a dict of features
            # Note: predict_single might be slow if called every bar for every position.
            # But for backtesting it's fine.
            prediction = self.alpha_engine.predict_single(features)
            score = prediction["alpha_score"]
            
            if position.side == "long":
                # Exit if confidence drops below threshold
                if score < self.config.reversal_threshold:
                    return ExitSignal(
                        reason=ExitReason.SIGNAL_REVERSAL,
                        exit_price=bar["close"],
                    )
            else:
                # Exit if confidence for short drops (i.e. score rises above 1-threshold)
                # Short signal means score < 0.5. Strong short is close to 0.
                # Reversal means score goes back up towards 1.
                if score > (1.0 - self.config.reversal_threshold):
                    return ExitSignal(
                        reason=ExitReason.SIGNAL_REVERSAL,
                        exit_price=bar["close"],
                    )
        
        return None
    
    def update_position(
        self,
        position: Position,
        bar: pd.Series,
        bar_idx: int,
    ) -> Position:
        """
        Update position state (trailing stop, breakeven, etc.).
        
        Args:
            position: Current position
            bar: Current bar
            bar_idx: Bar index
            
        Returns:
            Updated position
        """
        # Calculate current R
        if position.side == "long":
            pnl = bar["close"] - position.entry_price
        else:
            pnl = position.entry_price - bar["close"]
        
        current_r = pnl / position.sl_distance if position.sl_distance > 0 else 0
        
        # Activate trailing if threshold reached
        if not position.trailing_active and current_r >= self.config.trailing_activation_r:
            position.trailing_active = True
            
            # Set trailing stop
            trail_distance = position.sl_distance * self.config.trailing_distance_r
            if position.side == "long":
                position.current_sl = bar["close"] - trail_distance
            else:
                position.current_sl = bar["close"] + trail_distance
                
            logger.debug(
                "alpha_strategy.trailing_activated",
                symbol=position.symbol,
                current_r=current_r,
                new_sl=position.current_sl,
            )
        
        # Update trailing stop
        elif position.trailing_active:
            trail_distance = position.sl_distance * self.config.trailing_distance_r
            
            if position.side == "long":
                new_sl = bar["close"] - trail_distance
                if new_sl > position.current_sl:
                    position.current_sl = new_sl
            else:
                new_sl = bar["close"] + trail_distance
                if new_sl < position.current_sl:
                    position.current_sl = new_sl
        
        return position
    
    def _extract_features(self, bar: pd.Series) -> dict[str, float]:
        """Extract feature dict from bar for prediction."""
        features = {}
        for col in self.alpha_engine.feature_columns:
            if col in bar.index:
                features[col] = float(bar[col])
            else:
                features[col] = 0.0
        return features
