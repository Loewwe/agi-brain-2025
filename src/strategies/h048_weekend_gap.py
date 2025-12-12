
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
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

class H048_WeekendGapClose:
    """
    H048: Weekend Gap Close Strategy
    Logic: Fade the gap between Friday Close and Sunday/Monday Open.
    """
    
    def __init__(self):
        self.id = "H048"
        self.name = "Weekend Gap Close"
        self.gap_threshold_pct = 0.8  # 0.8% gap min
        self.tp_gap_closure_pct = 0.8 # Close at 80% gap fill
        self.sl_atr_mult = 1.5
        self.max_hold_hours = 48 # Close by Tuesday
        
        # State
        self.friday_closes: Dict[str, float] = {}
        
    def on_friday_close(self, symbol: str, close_price: float, timestamp: datetime):
        """Record Friday close price."""
        if timestamp.weekday() == 4: # Friday
            self.friday_closes[symbol] = close_price
            
    def check_signal(self, symbol: str, current_price: float, timestamp: datetime, atr: float) -> Optional[Signal]:
        """Check for gap signal on Sunday/Monday."""
        if symbol not in self.friday_closes:
            return None
            
        # Only check Sunday (6) or Monday (0)
        if timestamp.weekday() not in [6, 0]:
            return None
            
        friday_close = self.friday_closes[symbol]
        gap_pct = (current_price - friday_close) / friday_close * 100
        
        signal = None
        
        # Gap Down -> LONG
        if gap_pct <= -self.gap_threshold_pct:
            tp = friday_close # Target is gap fill (or close to it)
            sl = current_price * (1 - (self.sl_atr_mult * atr / current_price))
            
            signal = Signal(
                symbol=symbol,
                side="LONG",
                timestamp=timestamp,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                reason=f"Gap Down {gap_pct:.2f}%",
                strategy_id=self.id
            )
            
        # Gap Up -> SHORT
        elif gap_pct >= self.gap_threshold_pct:
            tp = friday_close
            sl = current_price * (1 + (self.sl_atr_mult * atr / current_price))
            
            signal = Signal(
                symbol=symbol,
                side="SHORT",
                timestamp=timestamp,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                reason=f"Gap Up {gap_pct:.2f}%",
                strategy_id=self.id
            )
            
        return signal
