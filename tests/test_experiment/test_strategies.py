"""Tests for strategies package."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.experiment.strategies import (
    AlphaStrategy,
    AlphaStrategyConfig,
    SingleAgentStrategy,
    SingleAgentConfig,
)
from src.experiment.alpha_engine import AlphaEngine
from src.experiment.simulator import SimulatorState, Position, Signal, ExitSignal
from src.experiment.models import SimulatorConfig, ExitReason


class MockAlphaEngine:
    """Mock AlphaEngine for testing."""
    def __init__(self, prediction=None):
        self.prediction = prediction or {
            "alpha_score": 0.8,
            "alpha_direction": 1,
            "alpha_confidence": 0.6,
        }
        self.feature_columns = ["rsi", "atr"]
    
    def predict_single(self, features):
        return self.prediction


class TestAlphaStrategy:
    """Tests for AlphaStrategy."""
    
    @pytest.fixture
    def strategy(self):
        engine = MockAlphaEngine()
        config = AlphaStrategyConfig(entry_threshold=0.7)
        return AlphaStrategy(engine, config)
    
    @pytest.fixture
    def bar(self):
        return pd.Series({
            "close": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "rsi": 60.0,
            "atr": 100.0,
        })
    
    @pytest.fixture
    def state(self):
        return SimulatorState(
            equity=1000.0,
            positions={},
            trades_today=0,
            daily_pnl=0.0,
            daily_stop_active=False,
            peak_equity=1000.0,
            current_date="2024-01-01",
        )
    
    def test_entry_signal_long(self, strategy, bar, state):
        """Test long entry signal generation."""
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Engine predicts 0.8 (bullish)
        strategy.alpha_engine.prediction = {"alpha_score": 0.8, "alpha_direction": 1}
        
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        
        assert signal is not None
        assert signal.side == "long"
        assert signal.strength == 0.8
    
    def test_entry_signal_short(self, strategy, bar, state):
        """Test short entry signal generation."""
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Engine predicts 0.2 (bearish, strength 0.8)
        strategy.alpha_engine.prediction = {"alpha_score": 0.2, "alpha_direction": 0}
        
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        
        assert signal is not None
        assert signal.side == "short"
        assert signal.strength == 0.8  # 1 - 0.2
    
    def test_no_entry_low_confidence(self, strategy, bar, state):
        """Test no entry when confidence is low."""
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Engine predicts 0.5 (neutral)
        strategy.alpha_engine.prediction = {"alpha_score": 0.5, "alpha_direction": 1}
        
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        
        assert signal is None
    
    def test_exit_signal_sl(self, strategy, bar):
        """Test SL exit."""
        position = Position(
            symbol="BTC",
            side="long",
            entry_price=50000.0,
            size=1.0,
            sl_price=49950.0,
            tp_price=50200.0,
            sl_distance=50.0,
            entry_time=datetime(2024, 1, 1, 10, 0),
            entry_bar_idx=0,
            current_sl=49950.0,
        )
        
        # Bar low hits SL
        bar["low"] = 49940.0
        
        signal = strategy.check_exit(position, bar, 10)
        
        assert signal is not None
        assert signal.reason == ExitReason.SL
        assert signal.exit_price == 49950.0


class TestSingleAgentStrategy:
    """Tests for SingleAgentStrategy."""
    
    @pytest.fixture
    def strategy(self):
        engine = MockAlphaEngine()
        config = SingleAgentConfig(
            min_confidence=0.8,
            max_daily_trades=2,
            max_daily_loss_pct=1.0,
        )
        return SingleAgentStrategy(engine, config)
    
    @pytest.fixture
    def bar(self):
        return pd.Series({
            "close": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "rsi": 60.0,
            "atr": 100.0,
        })
    
    @pytest.fixture
    def state(self):
        return SimulatorState(
            equity=1000.0,
            positions={},
            trades_today=0,
            daily_pnl=0.0,
            daily_stop_active=False,
            peak_equity=1000.0,
            current_date="2024-01-01",
        )
    
    def test_confidence_filter(self, strategy, bar, state):
        """Test stricter confidence filter."""
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Engine predicts 0.75 (good for AlphaStrategy but < 0.8 for SingleAgent)
        strategy.alpha_engine.prediction = {"alpha_score": 0.75, "alpha_direction": 1}
        
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        assert signal is None
        
        # Engine predicts 0.85
        strategy.alpha_engine.prediction = {"alpha_score": 0.85, "alpha_direction": 1}
        
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        assert signal is not None
    
    def test_daily_trade_limit(self, strategy, bar, state):
        """Test max daily trades limit."""
        timestamp = datetime(2024, 1, 1, 12, 0)
        strategy.alpha_engine.prediction = {"alpha_score": 0.9, "alpha_direction": 1}
        
        # Trade 1
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        assert signal is not None
        
        # Trade 2
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        assert signal is not None
        
        # Trade 3 (should be blocked)
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        assert signal is None
    
    def test_daily_loss_limit(self, strategy):
        """Test daily loss limit stops trading."""
        # Set current date
        strategy._current_date = datetime(2024, 1, 1).date()
        
        # Simulate loss
        strategy.on_trade_closed(-1.5, "BTC", "SL")  # -1.5% loss > 1.0% limit
        
        assert strategy._daily_stopped
        
        # Try to trade on SAME day
        bar = pd.Series({"close": 50000.0, "rsi": 50.0, "atr": 100.0})
        state = MagicMock()
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp)
        assert signal is None
        
        # Try to trade on NEXT day (should work)
        timestamp_next = datetime(2024, 1, 2, 12, 0)
        strategy.alpha_engine.prediction = {"alpha_score": 0.9, "alpha_direction": 1}
        
        signal = strategy.check_entry(bar, None, "BTC", state, timestamp_next)
        assert signal is not None
        assert not strategy._daily_stopped
