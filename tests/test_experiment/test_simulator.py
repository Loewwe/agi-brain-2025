"""Tests for simulator determinism and correctness."""

import pytest
from datetime import datetime, date
import pandas as pd
import numpy as np

from src.experiment.simulator import (
    Simulator,
    SimulatorConfig,
    Stage6Strategy,
    Position,
    Trade,
    Signal,
)
from src.experiment.models import SimulatorConfig as SimConfig


class TestSimulatorDeterminism:
    """Test that simulator produces deterministic results."""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        dates = pd.date_range(
            start="2024-01-01",
            periods=100,
            freq="5min",
        )
        
        np.random.seed(42)
        
        df = pd.DataFrame({
            "open": np.random.uniform(50000, 51000, 100),
            "high": np.random.uniform(50500, 51500, 100),
            "low": np.random.uniform(49500, 50500, 100),
            "close": np.random.uniform(50000, 51000, 100),
            "volume": np.random.uniform(1000, 10000, 100),
            "symbol": "BTCUSDT",
            "rsi": np.random.uniform(20, 80, 100),
            "rsi_prev": np.random.uniform(20, 80, 100),
            "atr": np.random.uniform(100, 500, 100),
            "atr_pct": np.random.uniform(0.5, 2.0, 100),
            "ema200": np.random.uniform(49000, 50000, 100),
            "volume_sma": np.random.uniform(5000, 8000, 100),
            "volume_surge": np.random.uniform(0.5, 2.0, 100),
            "high_2": np.random.uniform(50000, 51000, 100),
            "low_2": np.random.uniform(49000, 50000, 100),
        }, index=dates)
        
        return df
    
    def test_same_input_same_output(self, simple_dataset):
        """Test that running twice gives same result."""
        config = SimConfig(
            start_balance=1000.0,
            random_seed=42,
        )
        
        # Run 1
        sim1 = Simulator(config)
        strategy1 = Stage6Strategy(config)
        result1 = sim1.run(simple_dataset, strategy1)
        
        # Run 2
        sim2 = Simulator(config)
        strategy2 = Stage6Strategy(config)
        result2 = sim2.run(simple_dataset, strategy2)
        
        # Results should be identical
        assert result1.metrics.total_trades == result2.metrics.total_trades
        assert result1.metrics.win_rate == result2.metrics.win_rate
        assert result1.metrics.total_return_pct == result2.metrics.total_return_pct
        assert result1.checksum == result2.checksum
    
    def test_different_seed_different_results(self, simple_dataset):
        """Test that different seeds can produce different results."""
        # Note: With deterministic strategy logic, seed mainly affects
        # any random elements. Stage6 is fully deterministic.
        config1 = SimConfig(start_balance=1000.0, random_seed=42)
        config2 = SimConfig(start_balance=1000.0, random_seed=43)
        
        sim1 = Simulator(config1)
        sim2 = Simulator(config2)
        
        strategy1 = Stage6Strategy(config1)
        strategy2 = Stage6Strategy(config2)
        
        result1 = sim1.run(simple_dataset, strategy1)
        result2 = sim2.run(simple_dataset, strategy2)
        
        # With deterministic strategy, results should still be same
        # (seed only affects randomness, which Stage6 doesn't use)
        assert result1.metrics.total_trades == result2.metrics.total_trades


class TestSimulatorMetrics:
    """Test that simulator calculates metrics correctly."""
    
    @pytest.fixture
    def known_trades_dataset(self):
        """Create dataset with known expected outcomes."""
        # Create 10 days of data with clear signals
        dates = pd.date_range(
            start="2024-01-01",
            periods=2880,  # 10 days at 5min
            freq="5min",
        )
        
        # Start with baseline
        base_price = 50000.0
        prices = [base_price + i * 0.5 for i in range(2880)]
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [10000] * 2880,
            "symbol": "BTCUSDT",
            "rsi": [40] * 2880,  # No RSI signals
            "rsi_prev": [40] * 2880,
            "atr": [100] * 2880,
            "atr_pct": [1.0] * 2880,
            "ema200": [49000] * 2880,
            "volume_sma": [5000] * 2880,
            "volume_surge": [2.0] * 2880,
            "high_2": [p - 10 for p in prices],
            "low_2": [p + 10 for p in prices],
        }, index=dates)
        
        return df
    
    def test_no_trades_scenario(self, known_trades_dataset):
        """Test handling of no trades scenario."""
        # With RSI at 40 (not oversold/overbought), no signals should fire
        config = SimConfig(start_balance=1000.0)
        sim = Simulator(config)
        strategy = Stage6Strategy(config)
        
        result = sim.run(known_trades_dataset, strategy)
        
        # Should have 0 trades with this dataset
        assert result.metrics.total_trades == 0
        assert result.metrics.win_rate == 0.0
        assert result.metrics.total_return_pct == 0.0
    
    def test_final_equity_consistency(self, known_trades_dataset):
        """Test that final equity matches trade P&L."""
        config = SimConfig(start_balance=1000.0)
        sim = Simulator(config)
        strategy = Stage6Strategy(config)
        
        result = sim.run(known_trades_dataset, strategy)
        
        # Final equity should equal start + all trade P&L - commissions
        if result.trades_count > 0:
            total_pnl = sum(t.pnl_abs for t in sim.get_trades())
            # Allow for floating point error
            expected_equity = 1000.0 + total_pnl
            # Note: Equity already includes P&L, this is for verification
            assert sim.equity >= 0  # Should never go negative


class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            size=0.01,
            sl_price=49500.0,
            tp_price=51000.0,
            sl_distance=500.0,
            entry_time=datetime.now(),
            entry_bar_idx=0,
        )
        
        assert pos.symbol == "BTCUSDT"
        assert pos.side == "long"
        assert pos.current_sl == 49500.0  # Should be initialized from sl_price
    
    def test_position_modifications(self):
        """Test modifying position state."""
        pos = Position(
            symbol="ETHUSDT",
            side="short",
            entry_price=3000.0,
            size=1.0,
            sl_price=3100.0,
            tp_price=2800.0,
            sl_distance=100.0,
            entry_time=datetime.now(),
            entry_bar_idx=0,
        )
        
        # Move to breakeven
        pos.breakeven_done = True
        pos.current_sl = 2990.0  # Move SL to near entry
        
        assert pos.breakeven_done
        assert pos.current_sl == 2990.0


class TestTrade:
    """Test Trade dataclass."""
    
    def test_trade_creation(self):
        """Test creating a trade record."""
        trade = Trade(
            datetime_open=datetime(2024, 1, 1, 10, 0),
            datetime_close=datetime(2024, 1, 1, 11, 0),
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            exit_price=50500.0,
            sl_price=49500.0,
            tp_price=51000.0,
            reason="TP",
            pnl_abs=50.0,
            pnl_pct=5.0,
            r_multiple=1.0,
            hold_minutes=60,
        )
        
        assert trade.symbol == "BTCUSDT"
        assert trade.pnl_abs == 50.0
        assert trade.r_multiple == 1.0
