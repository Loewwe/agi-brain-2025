
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict

@dataclass
class Signal:
    symbol: str
    side: str  # "LONG" or "SHORT"
    timestamp: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    strategy_id: str

class H074_PanicDumpCapitulation:
    """
    H074: Panic Dump Capitulation
    Logic: Mean reversion after extreme panic selling.
    Trigger: Price drop < -4.0% in 15m AND RSI(14) < 30.
    """
    
    def __init__(self):
        self.id = "H074"
        self.name = "Panic Dump Capitulation"
        self.drop_threshold_pct = -4.0
        self.rsi_threshold = 30
        self.timeframe_minutes = 15
        
        self.tp_pct = 0.015 # 1.5%
        self.sl_pct = 0.008 # 0.8%
        
    def check_signal(self, symbol: str, candle_15m: dict) -> Optional[Signal]:
        """
        Check for panic dump signal.
        candle_15m expected keys: 'open', 'close', 'high', 'low', 'rsi', 'timestamp'
        """
        open_price = candle_15m['open']
        close_price = candle_15m['close']
        rsi = candle_15m.get('rsi', 50)
        timestamp = candle_15m['timestamp']
        
        # Calculate drop
        drop_pct = (close_price - open_price) / open_price * 100
        
        if drop_pct < self.drop_threshold_pct and rsi < self.rsi_threshold:
            # Signal LONG
            entry_price = close_price
            tp = entry_price * (1 + self.tp_pct)
            sl = entry_price * (1 - self.sl_pct)
            
            return Signal(
                symbol=symbol,
                side="LONG",
                timestamp=timestamp,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                reason=f"Panic Dump {drop_pct:.2f}% RSI {rsi:.1f}",
                strategy_id=self.id
            )
            
        return None
