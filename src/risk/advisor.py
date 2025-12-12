"""
AGI-Brain Risk Advisor.

Core advisor that analyzes trading state and provides risk recommendations.
Uses hard rules (always enforced) + LLM for nuanced decisions.
"""

import json
from datetime import datetime
from typing import Any

import structlog
from openai import AsyncOpenAI

from .models import (
    RiskAction,
    RiskDecision,
    RiskMode,
    RiskState,
    RiskSnapshot,
    StressFlag,
    VolatilityRegime,
)

logger = structlog.get_logger()


class RiskAdvisor:
    """
    AGI-Brain Risk Advisor for Stage6.
    
    Three-layer decision process:
    1. Hard rules (always enforced, no exceptions)
    2. Soft rules (heuristics, can be overridden by LLM)
    3. LLM reasoning (nuanced context-aware decisions)
    """
    
    # === HARD LIMITS (NEVER VIOLATED) ===
    MAX_DD_TODAY_FOR_AGGRESSIVE = -1.0    # Never aggressive if DD today <= -1%
    MAX_DD_PEAK_FOR_ALLOW_MORE = -5.0     # Never allow_more if peak DD <= -5%
    MAX_DD_TODAY_FOR_ALLOW_MORE = -0.5    # Never allow_more if DD today <= -0.5%
    
    # Stress flags that ALWAYS trigger defensive mode (as string values)
    CRITICAL_STRESS_FLAGS = {
        "margin_warning",
        "liquidation_near",
        "sl_missing",
        "max_daily_loss_hit",
    }
    
    # Default limits per mode
    MODE_LIMITS = {
        RiskMode.DEFENSIVE: {
            "max_daily_loss_pct": 0.5,
            "max_exposure_pct": 30.0,
            "max_concurrent_trades": 2,
            "max_single_position_pct": 10.0,
        },
        RiskMode.NORMAL: {
            "max_daily_loss_pct": 1.0,
            "max_exposure_pct": 50.0,
            "max_concurrent_trades": 4,
            "max_single_position_pct": 20.0,
        },
        RiskMode.AGGRESSIVE: {
            "max_daily_loss_pct": 1.5,
            "max_exposure_pct": 70.0,
            "max_concurrent_trades": 5,
            "max_single_position_pct": 25.0,
        },
    }
    
    def __init__(
        self,
        openai_api_key: str | None = None,
        model: str = "gpt-4o-mini",
        use_llm: bool = True,
    ):
        """
        Initialize Risk Advisor.
        
        Args:
            openai_api_key: OpenAI API key (uses env if not provided)
            model: LLM model to use
            use_llm: Whether to use LLM for nuanced decisions (False = hard rules only)
        """
        self.model = model
        self.use_llm = use_llm
        self._client: AsyncOpenAI | None = None
        
        if openai_api_key:
            self._client = AsyncOpenAI(api_key=openai_api_key)
        elif use_llm:
            try:
                self._client = AsyncOpenAI()
            except Exception as e:
                logger.warning("risk_advisor.llm_init_failed", error=str(e))
                self.use_llm = False
        
        logger.info(
            "risk_advisor.initialized",
            model=model,
            use_llm=self.use_llm,
        )
    
    async def analyze(self, state: RiskState) -> RiskDecision:
        """
        Analyze trading state and return risk decision.
        
        Process:
        1. Apply hard rules (override everything)
        2. Calculate base recommendation from soft rules
        3. Optionally refine with LLM
        4. Validate final decision against hard rules
        """
        logger.info(
            "risk_advisor.analyzing",
            equity=float(state.equity_usd),
            dd_today=state.drawdown_today_pct,
            dd_peak=state.drawdown_from_peak_pct,
            stress_flags=state.stress_flags,
        )
        
        # Step 1: Check hard rule violations → immediate defensive mode
        hard_rule_decision = self._apply_hard_rules(state)
        if hard_rule_decision:
            logger.info("risk_advisor.hard_rule_triggered", decision=hard_rule_decision.action)
            return hard_rule_decision
        
        # Step 2: Calculate base decision from soft rules
        base_decision = self._apply_soft_rules(state)
        
        # Step 3: Optionally refine with LLM
        if self.use_llm and self._client:
            try:
                refined = await self._refine_with_llm(state, base_decision)
                # Step 4: Validate LLM decision against hard rules
                violations = self.validate_decision(state, refined)
                if violations:
                    logger.warning(
                        "risk_advisor.llm_decision_violated_rules",
                        violations=violations,
                    )
                    # Fall back to base decision
                    return base_decision
                return refined
            except Exception as e:
                logger.error("risk_advisor.llm_failed", error=str(e))
                return base_decision
        
        return base_decision
    
    def _apply_hard_rules(self, state: RiskState) -> RiskDecision | None:
        """
        Apply hard rules that ALWAYS override everything.
        
        Returns decision if hard rule triggered, None otherwise.
        """
        # Rule 1: Critical stress flags → PAUSE
        # Normalize flags to strings for comparison
        flags_as_strings = self._normalize_flags(state.stress_flags)
        critical_flags = flags_as_strings & self.CRITICAL_STRESS_FLAGS
        if critical_flags:
            return RiskDecision(
                mode=RiskMode.DEFENSIVE,
                action=RiskAction.PAUSE_TRADING,
                pause_duration_hours=4.0,
                comment=f"Критические флаги: {', '.join(critical_flags)}. Торговля приостановлена.",
                reasoning=[f"Hard rule: critical stress flag {f}" for f in critical_flags],
                confidence=1.0,
                **self.MODE_LIMITS[RiskMode.DEFENSIVE],
            )
        
        # Rule 2: DD today <= -1% with open positions → DEFENSIVE + REDUCE
        if state.drawdown_today_pct <= self.MAX_DD_TODAY_FOR_AGGRESSIVE and state.open_positions_count > 0:
            return RiskDecision(
                mode=RiskMode.DEFENSIVE,
                action=RiskAction.REDUCE_SIZE,
                pause_duration_hours=2.0,
                comment=f"Дневная просадка {state.drawdown_today_pct:.1f}% превысила лимит. Снижаем экспозицию.",
                reasoning=[
                    f"Hard rule: DD today {state.drawdown_today_pct}% <= {self.MAX_DD_TODAY_FOR_AGGRESSIVE}%",
                    f"Open positions: {state.open_positions_count}",
                ],
                confidence=1.0,
                **self.MODE_LIMITS[RiskMode.DEFENSIVE],
            )
        
        # Rule 3: DD from peak <= -5% → DEFENSIVE + PAUSE
        if state.drawdown_from_peak_pct <= self.MAX_DD_PEAK_FOR_ALLOW_MORE:
            return RiskDecision(
                mode=RiskMode.DEFENSIVE,
                action=RiskAction.PAUSE_TRADING,
                pause_duration_hours=8.0,
                comment=f"Просадка от пика {state.drawdown_from_peak_pct:.1f}% критична. Полная пауза.",
                reasoning=[
                    f"Hard rule: DD from peak {state.drawdown_from_peak_pct}% <= {self.MAX_DD_PEAK_FOR_ALLOW_MORE}%",
                ],
                confidence=1.0,
                **self.MODE_LIMITS[RiskMode.DEFENSIVE],
            )
        
        return None
    
    def _apply_soft_rules(self, state: RiskState) -> RiskDecision:
        """
        Apply soft rules (heuristics) to determine base recommendation.
        """
        reasoning = []
        
        # Determine mode based on state
        mode = self._determine_mode(state, reasoning)
        
        # Determine action based on state and mode
        action = self._determine_action(state, mode, reasoning)
        
        # Get limits for mode
        limits = self.MODE_LIMITS[mode].copy()
        
        # Adjust limits based on specific conditions
        if state.volatility_regime == VolatilityRegime.HIGH:
            limits["max_exposure_pct"] *= 0.7
            limits["max_single_position_pct"] *= 0.7
            reasoning.append("High volatility: reduced exposure limits")
        
        if state.consecutive_losses >= 3:
            limits["max_concurrent_trades"] = max(1, limits["max_concurrent_trades"] - 2)
            reasoning.append(f"Consecutive losses ({state.consecutive_losses}): reduced max trades")
        
        # Calculate pause duration
        pause_hours = 0.0
        if action == RiskAction.PAUSE_TRADING:
            if state.drawdown_today_pct <= -0.8:
                pause_hours = 4.0
            elif state.consecutive_losses >= 3:
                pause_hours = 2.0
            else:
                pause_hours = 1.0
        
        # Generate comment
        comment = self._generate_comment(state, mode, action)
        
        return RiskDecision(
            mode=mode,
            action=action,
            pause_duration_hours=pause_hours,
            comment=comment,
            reasoning=reasoning,
            confidence=0.8,
            **limits,
        )
    
    def _determine_mode(self, state: RiskState, reasoning: list[str]) -> RiskMode:
        """Determine risk mode based on state."""
        # Default to normal
        mode = RiskMode.NORMAL
        
        # Defensive triggers
        if state.drawdown_today_pct <= -0.5:
            mode = RiskMode.DEFENSIVE
            reasoning.append(f"DD today {state.drawdown_today_pct:.1f}% → defensive")
        elif state.consecutive_losses >= 2:
            mode = RiskMode.DEFENSIVE
            reasoning.append(f"Consecutive losses ({state.consecutive_losses}) → defensive")
        elif state.volatility_regime == VolatilityRegime.HIGH:
            mode = RiskMode.DEFENSIVE
            reasoning.append("High volatility → defensive")
        elif state.total_exposure_pct > 70:
            mode = RiskMode.DEFENSIVE
            reasoning.append(f"High exposure ({state.total_exposure_pct:.0f}%) → defensive")
        elif "consecutive_losses" in self._normalize_flags(state.stress_flags):
            mode = RiskMode.DEFENSIVE
            reasoning.append("Consecutive losses flag → defensive")
        
        # Aggressive triggers (only if nothing defensive)
        elif (
            state.pnl_today_pct >= 1.0
            and state.drawdown_today_pct >= -0.3
            and state.consecutive_wins >= 2
            and state.volatility_regime != VolatilityRegime.HIGH
            and not state.stress_flags
        ):
            mode = RiskMode.AGGRESSIVE
            reasoning.append(f"Winning ({state.pnl_today_pct:+.1f}%), low DD, streak → aggressive")
        else:
            reasoning.append("No triggers → normal mode")
        
        return mode
    
    def _determine_action(
        self, 
        state: RiskState, 
        mode: RiskMode,
        reasoning: list[str],
    ) -> RiskAction:
        """Determine recommended action."""
        # Pause triggers
        if state.consecutive_losses >= 4:
            reasoning.append("4+ consecutive losses → pause")
            return RiskAction.PAUSE_TRADING
        
        if state.drawdown_today_pct <= -0.8:
            reasoning.append(f"DD today {state.drawdown_today_pct:.1f}% → pause")
            return RiskAction.PAUSE_TRADING
        
        # Reduce triggers
        if mode == RiskMode.DEFENSIVE and state.total_exposure_pct > 40:
            reasoning.append(f"Defensive mode + high exposure → reduce")
            return RiskAction.REDUCE_SIZE
        
        if state.total_exposure_pct > 80:
            reasoning.append(f"Exposure {state.total_exposure_pct:.0f}% > 80% → reduce")
            return RiskAction.REDUCE_SIZE
        
        # Allow more triggers
        if (
            mode == RiskMode.AGGRESSIVE
            and state.total_exposure_pct < 40
            and state.drawdown_today_pct >= 0
        ):
            reasoning.append("Aggressive mode + low exposure + no DD → allow more")
            return RiskAction.ALLOW_MORE
        
        # Default: keep
        reasoning.append("No action triggers → keep")
        return RiskAction.KEEP
    
    def _generate_comment(
        self, 
        state: RiskState, 
        mode: RiskMode, 
        action: RiskAction,
    ) -> str:
        """Generate human-readable comment."""
        parts = []
        
        # PnL context
        if state.pnl_today_pct >= 1.0:
            parts.append(f"Хороший день (+{state.pnl_today_pct:.1f}%)")
        elif state.pnl_today_pct <= -0.5:
            parts.append(f"Сложный день ({state.pnl_today_pct:.1f}%)")
        
        # Action explanation
        if action == RiskAction.PAUSE_TRADING:
            parts.append("Рекомендую паузу для сброса тильта")
        elif action == RiskAction.REDUCE_SIZE:
            parts.append("Снижаем экспозицию для защиты")
        elif action == RiskAction.ALLOW_MORE:
            parts.append("Можно немного увеличить позиции")
        else:
            parts.append("Продолжаем в текущем режиме")
        
        # Volatility note
        if state.volatility_regime == VolatilityRegime.HIGH:
            parts.append("⚡ Высокая волатильность")
        
        return ". ".join(parts) + "."
    
    async def _refine_with_llm(
        self, 
        state: RiskState, 
        base_decision: RiskDecision,
    ) -> RiskDecision:
        """Use LLM to refine decision with additional context."""
        if not self._client:
            return base_decision
        
        system_prompt = """You are a risk management advisor for a crypto futures trading bot.

HARD RULES (NEVER VIOLATE):
1. If drawdown_today_pct <= -1.0: mode MUST be "defensive", NEVER "aggressive"
2. If drawdown_from_peak_pct <= -5.0: action MUST NOT be "allow_more"
3. If any critical flag (margin_warning, liquidation_near, sl_missing, max_daily_loss_hit): mode="defensive", action="pause_trading"

Your goal: Protect capital first, then allow upside when safe.

Respond ONLY with valid JSON matching the schema."""

        user_prompt = f"""Current state:
- Equity: ${state.equity_usd}
- PnL today: {state.pnl_today_pct:+.1f}%
- Drawdown today: {state.drawdown_today_pct:.1f}%
- Drawdown from peak: {state.drawdown_from_peak_pct:.1f}%
- Open positions: {state.open_positions_count}
- Total exposure: {state.total_exposure_pct:.0f}%
- Consecutive losses: {state.consecutive_losses}
- Consecutive wins: {state.consecutive_wins}
- Volatility: {state.volatility_regime}
- Stress flags: {state.stress_flags}

Base recommendation: mode={base_decision.mode}, action={base_decision.action}
Comment: {base_decision.comment}

Return JSON with: mode, action, max_daily_loss_pct, max_exposure_pct, max_concurrent_trades, pause_duration_hours, comment (1-2 sentences in Russian), confidence (0-1).
Values for mode: defensive, normal, aggressive
Values for action: pause_trading, reduce_size, keep, allow_more"""

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.3,
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return RiskDecision(
                mode=RiskMode(result.get("mode", base_decision.mode)),
                action=RiskAction(result.get("action", base_decision.action)),
                max_daily_loss_pct=result.get("max_daily_loss_pct", base_decision.max_daily_loss_pct),
                max_exposure_pct=result.get("max_exposure_pct", base_decision.max_exposure_pct),
                max_concurrent_trades=result.get("max_concurrent_trades", base_decision.max_concurrent_trades),
                max_single_position_pct=base_decision.max_single_position_pct,
                pause_duration_hours=result.get("pause_duration_hours", 0),
                comment=result.get("comment", base_decision.comment),
                reasoning=base_decision.reasoning + ["LLM refined"],
                confidence=result.get("confidence", 0.8),
            )
            
        except Exception as e:
            logger.error("risk_advisor.llm_refine_failed", error=str(e))
            return base_decision
    
    def validate_decision(
        self, 
        state: RiskState, 
        decision: RiskDecision,
    ) -> list[str]:
        """
        Validate decision against hard rules.
        
        Returns list of violations (empty = valid).
        """
        violations = []
        
        # Rule 1: No aggressive when DD today <= -1%
        if (
            decision.mode == RiskMode.AGGRESSIVE 
            and state.drawdown_today_pct <= self.MAX_DD_TODAY_FOR_AGGRESSIVE
        ):
            violations.append(
                f"FATAL: aggressive mode with DD today {state.drawdown_today_pct}% "
                f"<= {self.MAX_DD_TODAY_FOR_AGGRESSIVE}%"
            )
        
        # Rule 2: No allow_more when DD from peak <= -5%
        if (
            decision.action == RiskAction.ALLOW_MORE
            and state.drawdown_from_peak_pct <= self.MAX_DD_PEAK_FOR_ALLOW_MORE
        ):
            violations.append(
                f"FATAL: allow_more with DD from peak {state.drawdown_from_peak_pct}% "
                f"<= {self.MAX_DD_PEAK_FOR_ALLOW_MORE}%"
            )
        
        # Rule 3: No allow_more when DD today <= -0.5%
        if (
            decision.action == RiskAction.ALLOW_MORE
            and state.drawdown_today_pct <= self.MAX_DD_TODAY_FOR_ALLOW_MORE
        ):
            violations.append(
                f"FATAL: allow_more with DD today {state.drawdown_today_pct}% "
                f"<= {self.MAX_DD_TODAY_FOR_ALLOW_MORE}%"
            )
        
        # Rule 4: Critical stress must trigger defensive
        flags_as_strings = self._normalize_flags(state.stress_flags)
        critical_flags = flags_as_strings & self.CRITICAL_STRESS_FLAGS
        if critical_flags and decision.mode != RiskMode.DEFENSIVE:
            violations.append(
                f"FATAL: mode={decision.mode} with critical flags: {critical_flags}"
            )
        
        return violations
    
    def _normalize_flags(self, flags: list) -> set[str]:
        """Normalize stress flags to set of strings."""
        result = set()
        for f in flags:
            if isinstance(f, StressFlag):
                result.add(f.value)
            elif isinstance(f, str):
                result.add(f)
        return result
    
    async def get_snapshot(self, state: RiskState) -> RiskSnapshot:
        """Get complete snapshot with state and decision."""
        decision = await self.analyze(state)
        return RiskSnapshot(state=state, decision=decision)


# === Singleton ===

_risk_advisor: RiskAdvisor | None = None


def get_risk_advisor() -> RiskAdvisor:
    """Get singleton advisor."""
    global _risk_advisor
    if _risk_advisor is None:
        _risk_advisor = RiskAdvisor()
    return _risk_advisor


def configure_risk_advisor(
    openai_api_key: str | None = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
) -> RiskAdvisor:
    """Configure and get advisor."""
    global _risk_advisor
    _risk_advisor = RiskAdvisor(
        openai_api_key=openai_api_key,
        model=model,
        use_llm=use_llm,
    )
    return _risk_advisor
