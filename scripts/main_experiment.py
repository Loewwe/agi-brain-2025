"""
AGI Trader Stage 6 - EXPERIMENT VERSION
Filters OFF + Trading Window 16:00-00:00 KZT + Hard Stop at Midnight

CHANGES FROM REGULAR main.py:
1. filters_override_enabled = True (bypass all entry filters)
2. trading_window: 16:00-00:00 KZT (11:00-19:00 UTC)
3. hard_stop_after_midnight: Close all and exit at 00:00 KZT
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.decision.orchestrator import Orchestrator
from core.risk import UnsinkableGuardian, GrowthManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/stage6_experiment.log")],
)
logger = logging.getLogger("AGITrader.EXPERIMENT")


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

EXPERIMENT_CONFIG = {
    "filters_override_enabled": True,      # Bypass entry filters
    "unsinkable_pretrade_override": True,  # Bypass pre-trade check
    "trading_window_enabled": True,        # Enable time window
    "trading_window_tz": "Asia/Almaty",    # KZT timezone
    "trading_window_start_hour": 16,       # Start at 16:00 KZT
    "trading_window_end_hour": 24,         # End at 00:00 KZT
    "hard_stop_after_midnight": True,      # Hard stop at midnight
}


# =============================================================================
# TIME WINDOW HELPERS
# =============================================================================

def get_local_time(tz_name: str = "Asia/Almaty") -> datetime:
    """Get current time in local timezone."""
    tz = ZoneInfo(tz_name)
    return datetime.now(timezone.utc).astimezone(tz)


def is_in_trading_window(cfg: dict) -> bool:
    """Check if current time is within trading window."""
    if not cfg.get("trading_window_enabled", False):
        return True
    
    now_local = get_local_time(cfg.get("trading_window_tz", "Asia/Almaty"))
    hour = now_local.hour
    start = cfg.get("trading_window_start_hour", 16)
    end = cfg.get("trading_window_end_hour", 24)
    
    # Window: 16:00 - 24:00 (which is 00:00 next day)
    return start <= hour < end


def is_session_hard_stop(cfg: dict) -> bool:
    """Check if we should hard stop (after midnight)."""
    if not cfg.get("hard_stop_after_midnight", False):
        return False
    
    now_local = get_local_time(cfg.get("trading_window_tz", "Asia/Almaty"))
    hour = now_local.hour
    
    # After midnight but before 6am = hard stop zone
    return 0 <= hour < 6


# =============================================================================
# MAIN TRADER CLASS
# =============================================================================

class ExperimentTrader:
    def __init__(self):
        logger.warning("🧪 EXPERIMENT MODE ACTIVATED!")
        logger.warning("⚡ Filters: DISABLED")
        logger.warning("⚡ Unsinkable Pre-Trade: DISABLED")
        logger.warning("⏰ Trading Window: 16:00-00:00 KZT")
        logger.warning("🛑 Hard Stop: After 00:00 KZT")
        
        self.exp_config = EXPERIMENT_CONFIG.copy()
        
        self.config = {
            "unsinkable_enabled": True,
            "unsinkable_monitor_only": True,
            "growth_enabled": True,
        }
        
        from execution.exchange_connector import ExchangeConnector
        is_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        self.exchange = ExchangeConnector(testnet=is_testnet)
        
        # Pass experiment config to orchestrator
        self.orchestrator = Orchestrator(
            self.config, 
            dry_run=False, 
            exchange_client=self.exchange.exchange
        )
        # Inject experiment config into orchestrator
        self.orchestrator.experiment = self.exp_config
        
        # Risk management
        self.unsinkable = None
        self.growth = None
        self.current_day = None
        self._initialize_risk_management()
        
        # Web server
        from monitoring.web_server import WebServer
        self.web_server = WebServer(
            self.orchestrator.dashboard,
            close_callback=self.orchestrator.force_close_position,
        )
        self.web_server.start()
        
        self.running = False
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def _initialize_risk_management(self):
        """Initialize Unsinkable + Growth."""
        try:
            equity = self._get_equity()
            today_utc = datetime.now(timezone.utc).date()
            
            if self.config.get("unsinkable_enabled", False):
                self.unsinkable = UnsinkableGuardian(
                    policy_path=Path("config/unsinkable_policy.json"),
                    state_path=Path("data/unsinkable_state.json"),
                    notifier=None,
                    monitor_only=self.config.get("unsinkable_monitor_only", True)
                )
                self.unsinkable.resume_session(equity_current=equity, today_utc=today_utc)
                logger.info("✅ Unsinkable Guardian initialized")
            
            if self.config.get("growth_enabled", False):
                self.growth = GrowthManager(
                    policy_path=Path("config/growth_policy.json"),
                    state_path=Path("data/growth_state.json"),
                    max_cap_usd=None
                )
                self.growth.resume_session(equity_current=equity, today_utc=today_utc)
                logger.info(f"✅ Growth Manager (budget=${self.growth.trading_budget_usd:.2f})")
            
            self.current_day = today_utc
        except Exception as e:
            logger.error(f"⚠️ Risk management init failed: {e}")
    
    def _get_equity(self):
        try:
            return self.exchange.get_equity_usdt()
        except Exception as e:
            logger.warning(f"Could not fetch equity: {e}")
            return 0.0
    
    def _update_risk_managers(self):
        try:
            equity = self._get_equity()
            positions_dict = self.orchestrator.position_manager.get_all_positions()
            open_notional = sum(
                abs(pos.entry_price * pos.size * pos.leverage) 
                for pos in positions_dict.values()
            )
            
            if self.unsinkable:
                self.unsinkable.on_equity_update(
                    equity_current=equity,
                    open_positions_notional_usd=open_notional
                )
            
            if self.growth:
                self.growth.on_equity_update(equity_current=equity)
        except Exception as e:
            logger.error(f"Error updating risk managers: {e}")
    
    def _check_emergency_liquidation(self):
        """Check kill-switch (NOT bypassed!)"""
        if self.unsinkable and self.unsinkable.should_emergency_liquidate():
            logger.critical("🔴 UNSINKABLE: Kill-switch triggered!")
            try:
                self.orchestrator.emergency_stop_all_positions()
                logger.info("✅ Emergency liquidation complete")
            except Exception as e:
                logger.error(f"❌ Emergency liquidation failed: {e}")
            return True
        return False
    
    def _close_all_positions(self):
        """Close all positions for hard stop."""
        try:
            positions = self.orchestrator.position_manager.get_all_positions()
            if not positions:
                logger.info("No positions to close")
                return
            
            for symbol, pos in list(positions.items()):
                try:
                    self.orchestrator.force_close_position(symbol)
                    logger.info(f"✅ Closed {symbol}")
                except Exception as e:
                    logger.error(f"❌ Failed to close {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def shutdown(self, signum, frame):
        logger.info("🛑 Shutdown signal received")
        self.running = False
    
    def run(self):
        """Main loop with experiment logic."""
        self.running = True
        now_local = get_local_time(self.exp_config.get("trading_window_tz", "Asia/Almaty"))
        logger.info(f"✅ EXPERIMENT started at {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Initial sync
        self.orchestrator.sync_positions(self.exchange)
        
        while self.running:
            try:
                now_local = get_local_time(self.exp_config.get("trading_window_tz", "Asia/Almaty"))
                
                # === HARD STOP CHECK (priority!) ===
                if is_session_hard_stop(self.exp_config):
                    logger.critical(
                        f"⚡ SESSION_HARD_STOP: local time {now_local.strftime('%H:%M')} >= 00:00"
                    )
                    logger.warning("Closing all positions...")
                    self._close_all_positions()
                    logger.warning("👋 Stopping agent after midnight. Goodbye!")
                    break
                
                # Update risk managers
                self._update_risk_managers()
                
                # Check kill-switch (NOT bypassed!)
                if self._check_emergency_liquidation():
                    time.sleep(5)
                    continue
                
                # Sync positions
                self.orchestrator.sync_positions(self.exchange)
                self.orchestrator.ensure_sl_tp_protection(self.exchange)
                
                # === TRADING WINDOW CHECK ===
                if not is_in_trading_window(self.exp_config):
                    logger.info(
                        f"⏸ Outside trading window ({now_local.strftime('%H:%M')} KZT). "
                        f"Waiting for 16:00..."
                    )
                    time.sleep(60)
                    continue
                
                # === ACTIVE TRADING (with filters bypassed) ===
                logger.info(f"🟢 In trading window ({now_local.strftime('%H:%M')} KZT)")
                
                # Fetch market data
                universe = self.exchange.get_top_symbols(limit=20)
                market_data = self.exchange.fetch_market_data(universe)
                balance = self.exchange.fetch_balance()
                market_data["balance"] = balance
                
                if market_data:
                    # Run cycle (filters are bypassed via orchestrator.experiment)
                    self.orchestrator.run_cycle(market_data)
                else:
                    logger.warning("No market data fetched")
                
                time.sleep(15)
                
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                time.sleep(10)
        
        logger.info("👋 EXPERIMENT ended.")


if __name__ == "__main__":
    trader = ExperimentTrader()
    trader.run()
