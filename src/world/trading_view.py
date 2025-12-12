"""
Trading view builder for AGI-Brain.

Aggregates trading data into meaningful metrics and views.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from .model import (
    Position,
    PnLMetrics,
    RiskConfig,
    RiskLevel,
    RiskViolation,
    Trade,
    TradingAgent,
    TradingView,
)


@dataclass
class TradingAnalysis:
    """Результат анализа трейдинга."""
    risk_score: float  # 0-100
    violations: list[RiskViolation]
    recommendations: list[str]
    alerts: list[str]


class TradingViewBuilder:
    """Строитель TradingView из данных трейдинг-агента."""
    
    def build(self, agent: TradingAgent) -> TradingView:
        """Построить TradingView из данных агента."""
        total_positions_value = sum(
            p.size * p.current_price for p in agent.current_positions
        )
        total_unrealized_pnl = sum(
            p.unrealized_pnl for p in agent.current_positions
        )
        
        # Рассчёт использованной маржи
        total_margin_used = sum(
            p.size * p.current_price / p.leverage 
            for p in agent.current_positions
        )
        margin_used_percent = (
            float(total_margin_used / agent.pnl.total_balance_usd * 100)
            if agent.pnl.total_balance_usd > 0 else 0.0
        )
        
        # Анализ рисков
        analysis = self.analyze_risk(agent)
        
        return TradingView(
            agent=agent,
            total_positions_value=total_positions_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_margin_used_percent=margin_used_percent,
            current_risk_score=analysis.risk_score,
            risk_violations_active=analysis.violations,
            trades_today=self._count_trades_in_period(agent, days=1),
            trades_this_week=self._count_trades_in_period(agent, days=7),
        )
    
    def analyze_risk(self, agent: TradingAgent) -> TradingAnalysis:
        """Анализ рисков текущего состояния."""
        violations: list[RiskViolation] = []
        recommendations: list[str] = []
        alerts: list[str] = []
        risk_score = 0.0
        
        config = agent.risk_config
        pnl = agent.pnl
        now = datetime.now()
        
        # === Проверка лимитов ===
        
        # 1. Дневной убыток
        if pnl.pnl_today_percent < -config.max_daily_loss_percent:
            violations.append(RiskViolation(
                rule="max_daily_loss",
                severity=RiskLevel.HIGH,
                current_value=pnl.pnl_today_percent,
                limit_value=-config.max_daily_loss_percent,
                message=f"Daily loss {pnl.pnl_today_percent:.2f}% exceeds limit {-config.max_daily_loss_percent:.2f}%",
                detected_at=now,
            ))
            risk_score += 30
        
        # 2. Drawdown
        if pnl.max_drawdown_today > config.max_daily_loss_percent:
            violations.append(RiskViolation(
                rule="max_drawdown",
                severity=RiskLevel.MEDIUM,
                current_value=pnl.max_drawdown_today,
                limit_value=config.max_daily_loss_percent,
                message=f"Today's drawdown {pnl.max_drawdown_today:.2f}% is concerning",
                detected_at=now,
            ))
            risk_score += 15
        
        # 3. Количество позиций
        if len(agent.current_positions) > config.max_concurrent_positions:
            violations.append(RiskViolation(
                rule="max_positions",
                severity=RiskLevel.MEDIUM,
                current_value=len(agent.current_positions),
                limit_value=config.max_concurrent_positions,
                message=f"Too many positions: {len(agent.current_positions)} > {config.max_concurrent_positions}",
                detected_at=now,
            ))
            risk_score += 10
        
        # 4. Плечо
        for pos in agent.current_positions:
            if pos.leverage > config.max_leverage:
                violations.append(RiskViolation(
                    rule="max_leverage",
                    severity=RiskLevel.HIGH,
                    current_value=pos.leverage,
                    limit_value=config.max_leverage,
                    message=f"Position {pos.symbol} leverage {pos.leverage}x exceeds limit {config.max_leverage}x",
                    detected_at=now,
                ))
                risk_score += 20
        
        # 5. Близость к ликвидации
        for pos in agent.current_positions:
            distance = pos.distance_to_liquidation_percent
            if distance is not None and distance < 10:
                violations.append(RiskViolation(
                    rule="liquidation_proximity",
                    severity=RiskLevel.CRITICAL,
                    current_value=distance,
                    limit_value=10,
                    message=f"Position {pos.symbol} only {distance:.1f}% from liquidation!",
                    detected_at=now,
                ))
                risk_score += 40
                alerts.append(f"🚨 CRITICAL: {pos.symbol} близко к ликвидации!")
        
        # 6. UNSINKABLE баланс
        available = pnl.available_balance_usd
        if available < config.unsinkable_balance_usd:
            violations.append(RiskViolation(
                rule="unsinkable_balance",
                severity=RiskLevel.CRITICAL,
                current_value=float(available),
                limit_value=float(config.unsinkable_balance_usd),
                message=f"Available balance ${available} is below UNSINKABLE ${config.unsinkable_balance_usd}!",
                detected_at=now,
            ))
            risk_score += 50
            alerts.append(f"🚨 CRITICAL: Баланс ниже UNSINKABLE!")
        
        # === Рекомендации ===
        
        # Если много убыточных позиций
        losing_positions = [p for p in agent.current_positions if not p.is_profitable]
        if len(losing_positions) > len(agent.current_positions) / 2:
            recommendations.append("Consider reviewing losing positions strategy")
        
        # Если низкий win rate
        if pnl.win_rate < 0.4:
            recommendations.append(f"Win rate {pnl.win_rate:.0%} is low, review strategy parameters")
        
        # Если profit factor < 1
        if pnl.profit_factor < 1.0:
            recommendations.append(f"Profit factor {pnl.profit_factor:.2f} < 1.0, strategy is losing money")
        
        # Нормализация risk_score
        risk_score = min(100, risk_score)
        
        return TradingAnalysis(
            risk_score=risk_score,
            violations=violations,
            recommendations=recommendations,
            alerts=alerts,
        )
    
    def _count_trades_in_period(self, agent: TradingAgent, days: int) -> int:
        """Подсчёт сделок за период (заглушка, реальные данные из истории)."""
        # В реальности будет запрос к trading adapter
        return 0
    
    def build_daily_summary(self, agent: TradingAgent, trades: list[Trade]) -> dict[str, Any]:
        """Построить дневной отчёт."""
        today = datetime.now().date()
        today_trades = [t for t in trades if t.closed_at.date() == today]
        
        total_pnl = sum(t.realized_pnl for t in today_trades)
        total_fees = sum(t.fees for t in today_trades)
        
        winning = [t for t in today_trades if t.realized_pnl > 0]
        losing = [t for t in today_trades if t.realized_pnl < 0]
        
        return {
            "date": today.isoformat(),
            "trades_count": len(today_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(today_trades) if today_trades else 0,
            "total_pnl_usd": float(total_pnl),
            "total_fees_usd": float(total_fees),
            "net_pnl_usd": float(total_pnl - total_fees),
            "biggest_win": float(max((t.realized_pnl for t in winning), default=0)),
            "biggest_loss": float(min((t.realized_pnl for t in losing), default=0)),
            "current_positions": len(agent.current_positions),
            "unrealized_pnl": float(sum(p.unrealized_pnl for p in agent.current_positions)),
        }
