"""
Unit tests for SL/TP Manager - Stage6 Specification

Tests:
- ATR-based SL calculation: min(1×ATR, 1.2%)
- Breakeven trigger at +1R
- Partial close at +1.5R
- Trailing stop
- Timeout enforcement
"""

import pytest
from unittest.mock import MagicMock, patch
import time
import sys
from pathlib import Path

# Add parent path to find sl_tp_manager (it's in project root, not in agi_brain)
# Path: tests/test_sl_tp_manager.py -> agi_brain/tests -> agi_brain -> alien-agi (project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sl_tp_manager import SLTPManager, PositionState


class MockExchange:
    """Mock exchange client for testing."""
    
    def __init__(self):
        self.orders = []
        self.positions = []
        self.current_prices = {}
        self.algo_orders = {}
    
    def fetch_positions(self):
        return self.positions
    
    def fetch_ticker(self, symbol):
        return {"last": self.current_prices.get(symbol, 100.0)}
    
    def fapiPrivateGetOpenAlgoOrders(self, params):
        symbol = params.get("symbol", "")
        return self.algo_orders.get(symbol, [])
    
    def price_to_precision(self, symbol, price):
        return round(price, 6)
    
    def amount_to_precision(self, symbol, amount):
        return round(amount, 4)
    
    def create_order(self, **kwargs):
        order = {"id": f"order_{len(self.orders)}", **kwargs}
        self.orders.append(order)
        return order
    
    def cancel_order(self, order_id, symbol):
        return True


class TestSLCalculation:
    """Test SL distance calculation."""
    
    def test_sl_atr_based_when_atr_smaller(self):
        """SL should use ATR when ATR < 1.2%."""
        manager = SLTPManager(MockExchange())
        entry_price = 100.0
        atr = 0.8  # 0.8% of price
        
        sl_dist = manager._calculate_sl_distance(entry_price, atr)
        
        # ATR (0.8) should be used, not 1.2%
        expected = 0.8 * manager.ATR_MULTIPLIER
        assert sl_dist == expected
        assert sl_dist < entry_price * manager.SL_MAX_PCT
    
    def test_sl_capped_at_max_pct(self):
        """SL should be capped at 1.2% when ATR is larger."""
        manager = SLTPManager(MockExchange())
        entry_price = 100.0
        atr = 2.0  # 2% of price - too large
        
        sl_dist = manager._calculate_sl_distance(entry_price, atr)
        
        # Should be capped at 1.2%
        expected = entry_price * manager.SL_MAX_PCT
        assert sl_dist == expected
    
    def test_sl_fallback_when_no_atr(self):
        """SL should use 1.2% when ATR is None."""
        manager = SLTPManager(MockExchange())
        entry_price = 100.0
        
        sl_dist = manager._calculate_sl_distance(entry_price, atr=None)
        
        expected = entry_price * manager.SL_MAX_PCT
        assert sl_dist == expected


class TestTradeManagement:
    """Test trade management logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.exchange = MockExchange()
        self.manager = SLTPManager(self.exchange)
    
    def test_profit_r_calculation_long(self):
        """Test profit calculation in R multiples for long."""
        entry = 100.0
        sl_dist = 1.0  # 1R = $1
        
        # At +$2 profit, should be +2R
        profit_r = self.manager._calculate_profit_r(entry, 102.0, sl_dist, is_long=True)
        assert profit_r == 2.0
        
        # At -$0.5 loss, should be -0.5R
        profit_r = self.manager._calculate_profit_r(entry, 99.5, sl_dist, is_long=True)
        assert profit_r == -0.5
    
    def test_profit_r_calculation_short(self):
        """Test profit calculation in R multiples for short."""
        entry = 100.0
        sl_dist = 1.0  # 1R = $1
        
        # At -$2 move (profit for short), should be +2R
        profit_r = self.manager._calculate_profit_r(entry, 98.0, sl_dist, is_long=False)
        assert profit_r == 2.0
    
    def test_breakeven_trigger_at_1r(self):
        """Test breakeven is triggered at +1R profit."""
        # Setup position at +1.2R profit
        self.exchange.positions = [{
            "symbol": "BTCUSDT",
            "contracts": 0.1,
            "entryPrice": 100.0,
        }]
        self.exchange.current_prices["BTCUSDT"] = 101.2  # +1.2% = +1R if SL is 1.2%
        self.exchange.algo_orders["BTCUSDT"] = [
            {"orderType": "STOP_MARKET", "algoId": "old_sl"}
        ]
        
        # Create position state
        sl_dist = 1.2  # 1.2%
        self.manager._position_states["BTCUSDT"] = PositionState(
            entry_price=100.0,
            entry_time=time.time() - 3600,
            sl_distance=sl_dist,
            is_long=True,
            original_size=0.1,
            current_sl_price=98.8,
        )
        
        # Run management
        results = self.manager.manage_open_positions()
        
        # Should have triggered breakeven
        assert "BTCUSDT" in results
        assert any("breakeven" in action for action in results["BTCUSDT"])
        assert self.manager._position_states["BTCUSDT"].breakeven_done is True
    
    def test_partial_close_at_1_5r(self):
        """Test partial close is triggered at +1.5R profit."""
        self.exchange.positions = [{
            "symbol": "ETHUSDT",
            "contracts": 1.0,
            "entryPrice": 2000.0,
        }]
        sl_dist = 24.0  # 1.2%
        self.exchange.current_prices["ETHUSDT"] = 2000 + sl_dist * 1.6  # +1.6R
        
        self.manager._position_states["ETHUSDT"] = PositionState(
            entry_price=2000.0,
            entry_time=time.time() - 3600,
            sl_distance=sl_dist,
            is_long=True,
            original_size=1.0,
            current_sl_price=2000 - sl_dist,
            breakeven_done=True,  # Already done
        )
        
        results = self.manager.manage_open_positions()
        
        assert "ETHUSDT" in results
        assert any("partial" in action for action in results["ETHUSDT"])
        assert self.manager._position_states["ETHUSDT"].partial_done is True
    
    def test_timeout_closes_position(self):
        """Test timeout closes position after ~3.3 hours."""
        self.exchange.positions = [{
            "symbol": "XRPUSDT",
            "contracts": 100.0,
            "entryPrice": 1.0,
        }]
        self.exchange.current_prices["XRPUSDT"] = 1.0  # No profit
        
        # Position opened 4 hours ago
        old_time = time.time() - (4 * 3600)
        self.manager._position_states["XRPUSDT"] = PositionState(
            entry_price=1.0,
            entry_time=old_time,
            sl_distance=0.012,
            is_long=True,
            original_size=100.0,
            current_sl_price=0.988,
        )
        
        results = self.manager.manage_open_positions()
        
        assert "XRPUSDT" in results
        assert any("timeout" in action for action in results["XRPUSDT"])
        # Position state should be removed after closure
        assert "XRPUSDT" not in self.manager._position_states


class TestInitialProtection:
    """Test initial protection placement."""
    
    def test_place_initial_protection_with_atr(self):
        """Test initial protection uses ATR when provided."""
        exchange = MockExchange()
        manager = SLTPManager(exchange)
        
        result = manager.place_initial_protection(
            symbol="BTCUSDT",
            size=0.01,
            entry_price=50000.0,
            is_long=True,
            atr=300.0,  # 0.6% of price
        )
        
        assert result is True
        assert len(exchange.orders) == 2  # SL and TP
        
        # Check SL order uses ATR-based distance
        sl_order = exchange.orders[0]
        assert sl_order["type"] == "stop_market"
        sl_price = sl_order["params"]["stopPrice"]
        expected_sl = 50000.0 - 300.0  # ATR-based
        assert abs(sl_price - expected_sl) < 1.0
    
    def test_position_state_created(self):
        """Test position state is created after initial protection."""
        exchange = MockExchange()
        manager = SLTPManager(exchange)
        
        manager.place_initial_protection(
            symbol="ETHUSDT",
            size=1.0,
            entry_price=2000.0,
            is_long=True,
            atr=20.0,
        )
        
        assert "ETHUSDT" in manager._position_states
        state = manager._position_states["ETHUSDT"]
        assert state.entry_price == 2000.0
        assert state.sl_distance == 20.0  # ATR
        assert state.breakeven_done is False
        assert state.partial_done is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
