
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..perception.stage6_adapter import Stage6TradingAdapter
from ..risk.advisor import get_risk_advisor, RiskState, RiskAction, RiskMode
from .h048_weekend_gap import H048_WeekendGapClose
from .h074_panic_dump import H074_PanicDumpCapitulation

logger = logging.getLogger(__name__)

class StrategyOrchestrator:
    """
    Mini-Brain for managing H048 and H074 strategies.
    """
    
    def __init__(self, config_path: str = "src/configs/live_strategies.json"):
        self.adapter = Stage6TradingAdapter()
        self.risk_advisor = get_risk_advisor()
        self.strategies = {}
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize strategies
        if self.config["strategies"]["H048"]["enabled"]:
            self.strategies["H048"] = H048_WeekendGapClose()
            mode = self.config["strategies"]["H048"].get("mode", "live")
            logger.info(f"H048 initialized in {mode} mode.")
            
        if self.config["strategies"]["H074"]["enabled"]:
            self.strategies["H074"] = H074_PanicDumpCapitulation()
            mode = self.config["strategies"]["H074"].get("mode", "live")
            logger.info(f"H074 initialized in {mode} mode.")
            
        logger.info(f"Orchestrator initialized with strategies: {list(self.strategies.keys())}")
        
    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)
            
    async def run(self, interval_seconds: int = 60):
        """Main execution loop."""
        self.running = True
        logger.info("Orchestrator started.")
        
        while self.running:
            try:
                await self._cycle()
            except Exception as e:
                logger.error(f"Error in orchestrator cycle: {e}", exc_info=True)
                
            await asyncio.sleep(interval_seconds)
            
    async def _cycle(self):
        """Single execution cycle."""
        # 0. Risk Check (Unsinkable)
        # We need to build RiskState. For now, we'll use a simplified version or fetch from adapter if possible.
        # The adapter has fetch_positions, but building full RiskState requires more.
        # Let's assume we can get a basic state.
        
        # TODO: Implement proper state building in adapter or here
        # For Shadow Mode, we proceed but log risk status
        
        symbols = ["BTC/USDT", "ETH/USDT"] # Target symbols
        
        for symbol in symbols:
            # 1. Fetch Data
            candles_15m = await self.adapter.fetch_ohlcv(symbol, '15m', limit=50)
            candles_1h = await self.adapter.fetch_ohlcv(symbol, '1h', limit=50) # For H048
            
            if not candles_15m or not candles_1h:
                continue
                
            current_price = candles_15m[-1]['close']
            timestamp = candles_15m[-1]['timestamp']
            
            # Calculate Indicators
            # ATR (14) on 1h for H048
            atr_1h = self._calculate_atr(candles_1h, 14)
            
            # RSI (14) on 15m for H074
            rsi_15m = self._calculate_rsi(candles_15m, 14)
            candles_15m[-1]['rsi'] = rsi_15m
            
            # 2. Check Signals
            active_signal = None
            
            # Priority 1: H074 (Panic)
            if "H074" in self.strategies:
                h074 = self.strategies["H074"]
                signal = h074.check_signal(symbol, candles_15m[-1])
                if signal:
                    active_signal = signal
                    logger.info(f"[H074_SIGNAL] {symbol} {signal.side} @ {signal.entry_price}")
            
            # Priority 2: H048 (Gap) - Only if no H074
            if not active_signal and "H048" in self.strategies:
                h048 = self.strategies["H048"]
                # Update Friday Close if needed
                h048.on_friday_close(symbol, current_price, timestamp)
                
                signal = h048.check_signal(symbol, current_price, timestamp, atr_1h)
                if signal:
                    active_signal = signal
                    logger.info(f"[H048_SIGNAL] {symbol} {signal.side} @ {signal.entry_price}")
            
            # 3. Execute (Shadow Mode Log)
            if active_signal:
                logger.info(f"[SHADOW_ORDER] {active_signal}")
                
    def _calculate_atr(self, candles: list, period: int) -> float:
        if len(candles) < period + 1: return 0.0
        tr_sum = 0.0
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            if i > len(candles) - period:
                tr_sum += tr
        return tr_sum / period

    def _calculate_rsi(self, candles: list, period: int) -> float:
        if len(candles) < period + 1: return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(len(candles) - period, len(candles)):
            change = candles[i]['close'] - candles[i-1]['close']
            if change > 0: gains += change
            else: losses -= change
        
        if losses == 0: return 100.0
        rs = gains / losses
        return 100 - (100 / (1 + rs))

    async def stop(self):
        self.running = False
        logger.info("Orchestrator stopped.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = StrategyOrchestrator()
    asyncio.run(orchestrator.run())
