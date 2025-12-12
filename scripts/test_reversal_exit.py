
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.experiment.strategies.alpha_strategy import AlphaStrategy, AlphaStrategyConfig
from src.experiment.simulator import Position, ExitReason
from src.experiment.models import SimulatorConfig

def test_signal_reversal_exit():
    """Test that strategy exits on signal reversal."""
    
    # Mock AlphaEngine
    mock_engine = MagicMock()
    
    # Config with reversal exit enabled
    config = AlphaStrategyConfig(
        exit_on_reversal=True,
        reversal_threshold=0.5
    )
    sim_config = SimulatorConfig()
    
    strategy = AlphaStrategy(mock_engine, config, sim_config)
    
    # Mock position (Long)
    position = Position(
        symbol="BTC/USDT",
        side="long",
        entry_price=50000,
        size=1.0,
        sl_price=49000,
        tp_price=53000,
        sl_distance=1000,
        entry_time=datetime.now(),
        entry_bar_idx=100
    )
    
    # Mock bar
    bar = pd.Series({
        "close": 50500,
        "high": 50900,  # Lower high to avoid SL (SL=51000)
        "low": 50000,
        "volume": 100
    })
    
    # Case 1: Strong Long Signal (Score 0.8) -> No Exit
    mock_engine.predict_single.return_value = {"alpha_score": 0.8}
    exit_signal = strategy.check_exit(position, bar, 105)
    assert exit_signal is None, "Should not exit on strong signal"
    
    # Case 2: Weak Long Signal (Score 0.55) -> No Exit
    mock_engine.predict_single.return_value = {"alpha_score": 0.55}
    exit_signal = strategy.check_exit(position, bar, 105)
    assert exit_signal is None, "Should not exit above threshold"
    
    # Case 3: Reversal (Score 0.4) -> Exit
    mock_engine.predict_single.return_value = {"alpha_score": 0.4}
    exit_signal = strategy.check_exit(position, bar, 105)
    assert exit_signal is not None, "Should exit on reversal"
    assert exit_signal.reason == ExitReason.SIGNAL_REVERSAL
    
    # Case 4: Short Position Reversal
    position_short = Position(
        symbol="BTC/USDT",
        side="short",
        entry_price=50000,
        size=1.0,
        sl_price=51000,
        tp_price=47000,
        sl_distance=1000,
        entry_time=datetime.now(),
        entry_bar_idx=100
    )
    
    # Strong Short (Score 0.2) -> No Exit
    mock_engine.predict_single.return_value = {"alpha_score": 0.2}
    exit_signal = strategy.check_exit(position_short, bar, 105)
    assert exit_signal is None
    
    # Reversal to Long (Score 0.6) -> Exit (Threshold 0.5 -> Exit if > 0.5)
    mock_engine.predict_single.return_value = {"alpha_score": 0.6}
    exit_signal = strategy.check_exit(position_short, bar, 105)
    assert exit_signal is not None
    assert exit_signal.reason == ExitReason.SIGNAL_REVERSAL

if __name__ == "__main__":
    test_signal_reversal_exit()
    print("Test passed!")
