"""
Shadow Runner for AGI-Brain Risk Advisor.

Runs in shadow mode over Stage6:
- Reads trading state periodically
- Generates risk decisions
- Logs to JSONL
- Sends updates to Telegram
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import structlog

from .advisor import RiskAdvisor, get_risk_advisor
from .models import RiskState, RiskDecision, RiskSnapshot, VolatilityRegime

logger = structlog.get_logger()


class TelegramNotifier:
    """Telegram bot for sending risk updates."""
    
    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self._client = httpx.AsyncClient(timeout=30.0)
        self._last_message_id: int | None = None
        
        if self.bot_token:
            logger.info("telegram.initialized", has_chat_id=bool(self.chat_id))
        else:
            logger.warning("telegram.no_token")
    
    @property
    def api_url(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"
    
    async def send_message(
        self, 
        text: str, 
        parse_mode: str = "Markdown",
        chat_id: str | None = None,
    ) -> dict | None:
        """Send message to Telegram."""
        if not self.bot_token:
            logger.warning("telegram.no_token_skip")
            return None
        
        target_chat = chat_id or self.chat_id
        if not target_chat:
            logger.warning("telegram.no_chat_id")
            return None
        
        try:
            response = await self._client.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": target_chat,
                    "text": text,
                    "parse_mode": parse_mode,
                }
            )
            result = response.json()
            if result.get("ok"):
                self._last_message_id = result["result"]["message_id"]
                logger.info("telegram.message_sent", message_id=self._last_message_id)
            else:
                logger.error("telegram.send_failed", error=result)
            return result
        except Exception as e:
            logger.error("telegram.send_error", error=str(e))
            return None
    
    async def get_updates(self, offset: int = 0) -> list[dict]:
        """Get updates (messages) from bot."""
        if not self.bot_token:
            return []
        
        try:
            response = await self._client.get(
                f"{self.api_url}/getUpdates",
                params={"offset": offset, "timeout": 5}
            )
            result = response.json()
            if result.get("ok"):
                return result.get("result", [])
            return []
        except Exception as e:
            logger.error("telegram.get_updates_error", error=str(e))
            return []
    
    async def handle_command(self, command: str, state: RiskState, decision: RiskDecision) -> str:
        """Handle bot command and return response."""
        cmd = command.lower().strip()
        
        if cmd in ["/brain", "/status", "/risk"]:
            return decision.to_telegram_message(state)
        
        elif cmd == "/why":
            reasons = decision.reasoning[:5] if decision.reasoning else ["Нет дополнительной информации"]
            return "🧠 *Почему такое решение:*\n" + "\n".join(f"• {r}" for r in reasons)
        
        elif cmd == "/dd":
            return (
                f"📊 *Drawdown Report*\n"
                f"DD сегодня: {state.drawdown_today_pct:.2f}%\n"
                f"DD от пика: {state.drawdown_from_peak_pct:.2f}%\n"
                f"Лимит дневной DD: -1.0%\n"
                f"Лимит пиковый DD: -5.0%"
            )
        
        elif cmd == "/config":
            return (
                f"⚙️ *Текущие лимиты (режим: {decision.mode})*\n"
                f"Max daily loss: {decision.max_daily_loss_pct}%\n"
                f"Max exposure: {decision.max_exposure_pct}%\n"
                f"Max trades: {decision.max_concurrent_trades}\n"
                f"Max single position: {decision.max_single_position_pct}%"
            )
        
        elif cmd == "/help":
            return (
                "🤖 *AGI-Brain Risk Advisor*\n\n"
                "/brain - текущий снапшот риска\n"
                "/why - почему такое решение\n"
                "/dd - информация о drawdown\n"
                "/config - текущие лимиты\n"
                "/help - эта справка"
            )
        
        return "❓ Неизвестная команда. /help для справки."


class ShadowRunner:
    """
    Shadow mode runner for AGI-Brain Risk Advisor.
    
    Connects to Stage6 (read-only), analyzes risk, logs decisions,
    and sends updates to Telegram.
    """
    
    def __init__(
        self,
        advisor: RiskAdvisor | None = None,
        log_path: Path | str = "shadow_decisions.jsonl",
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
    ):
        self.advisor = advisor or get_risk_advisor()
        self.log_path = Path(log_path)
        self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
        
        self._running = False
        self._last_decision: RiskDecision | None = None
        self._last_state: RiskState | None = None
        self._update_offset = 0
        
        # Stats
        self._cycles = 0
        self._decisions_logged = 0
        self._start_time: datetime | None = None
        
        logger.info(
            "shadow_runner.initialized",
            log_path=str(self.log_path),
        )
    
    async def build_state_from_stage6(self) -> RiskState:
        """
        Build RiskState from Stage6 adapter.
        
        This connects to the live Stage6 trading bot and reads current state.
        """
        # Import here to avoid circular imports
        from ..perception.stage6_adapter import Stage6TradingAdapter
        
        adapter = Stage6TradingAdapter()
        
        try:
            # Fetch positions
            positions = await adapter.fetch_positions()
            
            # Calculate metrics
            total_exposure = sum(float(p.size_usd) for p in positions)
            total_unrealized = sum(float(p.unrealized_pnl) for p in positions)
            avg_pnl = (
                sum(p.pnl_percent for p in positions) / len(positions)
                if positions else 0.0
            )
            max_exposure = max((float(p.size_usd) for p in positions), default=0.0)
            
            # Get risk config for limits
            risk_config = await adapter.fetch_risk_config()
            
            # Analyze risk
            risk_analysis = await adapter.analyze_risk()
            
            # Build stress flags
            stress_flags = []
            if risk_analysis.get("violations"):
                for v in risk_analysis["violations"]:
                    if "margin" in v.lower():
                        stress_flags.append("margin_warning")
                    if "sl" in v.lower() or "stop" in v.lower():
                        stress_flags.append("sl_missing")
            
            # Get equity (simplified - in real impl would fetch from exchange)
            equity = Decimal("7000")  # Placeholder
            
            # Calculate drawdowns (simplified)
            dd_today = -abs(total_unrealized / float(equity) * 100) if equity > 0 else 0.0
            
            return RiskState(
                equity_usd=equity,
                available_balance_usd=equity - Decimal(str(total_exposure)),
                unrealized_pnl_usd=Decimal(str(total_unrealized)),
                unrealized_pnl_pct=total_unrealized / float(equity) * 100 if equity > 0 else 0.0,
                drawdown_today_pct=min(0.0, dd_today),
                drawdown_from_peak_pct=min(0.0, dd_today * 1.5),  # Estimate
                open_positions_count=len(positions),
                total_exposure_pct=total_exposure / float(equity) * 100 if equity > 0 else 0.0,
                avg_position_pnl_pct=avg_pnl,
                max_position_exposure_pct=max_exposure / float(equity) * 100 if equity > 0 else 0.0,
                volatility_regime=VolatilityRegime.NORMAL,
                stress_flags=stress_flags,
            )
            
        except Exception as e:
            logger.error("shadow_runner.build_state_failed", error=str(e))
            # Return safe default state
            return RiskState(
                equity_usd=Decimal("7000"),
                available_balance_usd=Decimal("7000"),
                stress_flags=["api_issues"],
            )
    
    async def build_state_mock(self) -> RiskState:
        """Build mock state for testing without Stage6 connection."""
        import random
        
        # Simulate some variation
        pnl = random.uniform(-1.5, 2.0)
        exposure = random.uniform(20, 60)
        positions = random.randint(0, 5)
        
        stress_flags = []
        if pnl < -1.0:
            stress_flags.append("consecutive_losses")
        if exposure > 70:
            stress_flags.append("max_exposure_hit")
        
        return RiskState(
            equity_usd=Decimal("7000") + Decimal(str(pnl * 70)),
            available_balance_usd=Decimal(str(7000 - exposure * 70)),
            pnl_today_pct=pnl,
            drawdown_today_pct=min(0.0, pnl - 0.5),
            drawdown_from_peak_pct=min(0.0, pnl - 1.0),
            open_positions_count=positions,
            total_exposure_pct=exposure,
            volatility_regime=VolatilityRegime.NORMAL,
            stress_flags=stress_flags,
        )
    
    def log_decision(self, snapshot: RiskSnapshot) -> None:
        """Log decision to JSONL file."""
        entry = snapshot.to_log_entry()
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        
        self._decisions_logged += 1
        logger.debug("shadow_runner.logged", decisions=self._decisions_logged)
    
    async def run_cycle(self, use_mock: bool = False) -> RiskSnapshot:
        """Run single observation cycle."""
        self._cycles += 1
        
        # Build state
        if use_mock:
            state = await self.build_state_mock()
        else:
            state = await self.build_state_from_stage6()
        
        # Get decision
        decision = await self.advisor.analyze(state)
        
        # Create snapshot
        snapshot = RiskSnapshot(state=state, decision=decision)
        
        # Log
        self.log_decision(snapshot)
        
        # Store for command handling
        self._last_state = state
        self._last_decision = decision
        
        logger.info(
            "shadow_runner.cycle_complete",
            cycle=self._cycles,
            mode=decision.mode,
            action=decision.action,
        )
        
        return snapshot
    
    async def send_update(self, snapshot: RiskSnapshot) -> None:
        """Send risk update to Telegram."""
        message = snapshot.decision.to_telegram_message(snapshot.state)
        await self.telegram.send_message(message)
    
    async def check_commands(self) -> None:
        """Check for and handle Telegram commands."""
        if not self._last_state or not self._last_decision:
            return
        
        updates = await self.telegram.get_updates(self._update_offset)
        
        for update in updates:
            self._update_offset = update["update_id"] + 1
            
            message = update.get("message", {})
            text = message.get("text", "")
            chat_id = str(message.get("chat", {}).get("id", ""))
            
            if text.startswith("/"):
                response = await self.telegram.handle_command(
                    text, self._last_state, self._last_decision
                )
                await self.telegram.send_message(response, chat_id=chat_id)
                
                # If first message, save chat_id
                if not self.telegram.chat_id:
                    self.telegram.chat_id = chat_id
                    logger.info("shadow_runner.chat_id_set", chat_id=chat_id)
    
    async def run_continuous(
        self,
        interval_minutes: int = 5,
        duration_hours: float | None = None,
        use_mock: bool = False,
        send_updates: bool = True,
    ) -> None:
        """
        Run shadow mode continuously.
        
        Args:
            interval_minutes: Minutes between observations
            duration_hours: Optional duration limit (None = run forever)
            use_mock: Use mock data instead of Stage6
            send_updates: Send Telegram updates
        """
        self._running = True
        self._start_time = datetime.now()
        end_time = (
            self._start_time + timedelta(hours=duration_hours)
            if duration_hours else None
        )
        
        logger.info(
            "shadow_runner.starting",
            interval_minutes=interval_minutes,
            duration_hours=duration_hours,
            use_mock=use_mock,
        )
        
        # Send initial message
        if send_updates:
            await self.telegram.send_message(
                "🚀 *AGI-Brain Shadow Mode Started*\n"
                f"Interval: {interval_minutes} min\n"
                f"Duration: {duration_hours}h" if duration_hours else "∞"
            )
        
        try:
            while self._running:
                # Check time limit
                if end_time and datetime.now() >= end_time:
                    logger.info("shadow_runner.duration_reached")
                    break
                
                # Run cycle
                snapshot = await self.run_cycle(use_mock=use_mock)
                
                # Send update (every cycle or on mode change)
                if send_updates:
                    await self.send_update(snapshot)
                
                # Check commands
                await self.check_commands()
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
        except asyncio.CancelledError:
            logger.info("shadow_runner.cancelled")
        except KeyboardInterrupt:
            logger.info("shadow_runner.interrupted")
        finally:
            self._running = False
            if send_updates:
                await self.telegram.send_message(
                    f"⏹️ Shadow mode stopped.\n"
                    f"Cycles: {self._cycles}\n"
                    f"Logged: {self._decisions_logged}"
                )
    
    def stop(self) -> None:
        """Stop the runner."""
        self._running = False


async def main():
    """Main entry point for shadow runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI-Brain Shadow Runner")
    parser.add_argument("--interval", type=int, default=5, help="Interval in minutes")
    parser.add_argument("--duration", type=float, default=None, help="Duration in hours")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram")
    parser.add_argument("--log-path", type=str, default="shadow_decisions.jsonl")
    
    args = parser.parse_args()
    
    runner = ShadowRunner(log_path=args.log_path)
    
    await runner.run_continuous(
        interval_minutes=args.interval,
        duration_hours=args.duration,
        use_mock=args.mock,
        send_updates=not args.no_telegram,
    )


if __name__ == "__main__":
    asyncio.run(main())
