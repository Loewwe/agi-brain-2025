
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import structlog

logger = structlog.get_logger()

class Regime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"

@dataclass
class RegimeConfig:
    adx_threshold: float = 25.0
    # Optional: slope threshold, etc.

class MarketRegimeDetector:
    """
    Detects market regime based on technical indicators.
    
    Logic v0:
    - SIDEWAYS: ADX < threshold
    - BULL_TREND: ADX >= threshold AND Close > EMA200
    - BEAR_TREND: ADX >= threshold AND Close < EMA200
    """
    
    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        
    def detect(self, bar: pd.Series | dict) -> Regime:
        """
        Detect regime for a single bar.
        Expects keys: 'adx', 'close', 'ema200'.
        """
        # Handle missing features gracefully
        if "adx" not in bar or "ema200" not in bar or "close" not in bar:
            # Default to sideways if data missing
            return Regime.SIDEWAYS
            
        adx = float(bar["adx"])
        close = float(bar["close"])
        ema200 = float(bar["ema200"])
        
        if adx < self.config.adx_threshold:
            return Regime.SIDEWAYS
        
        if close > ema200:
            return Regime.BULL_TREND
        else:
            return Regime.BEAR_TREND

    def detect_all(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime for entire dataframe.
        Returns Series of Regime enums.
        """
        return df.apply(self.detect, axis=1)
