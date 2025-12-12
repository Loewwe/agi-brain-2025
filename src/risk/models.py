"""
Risk Models for AGI-Brain Risk Advisor.

Defines input state and output decision structures.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class StressFlag(str, Enum):
    """Stress indicators for risk assessment."""
    # Critical - require immediate attention
    MARGIN_WARNING = "margin_warning"
    LIQUIDATION_NEAR = "liquidation_near"
    SL_MISSING = "sl_missing"
    TP_MISSING = "tp_missing"
    
    # High priority
    MAX_DAILY_LOSS_HIT = "max_daily_loss_hit"
    MAX_EXPOSURE_HIT = "max_exposure_hit"
    MAX_POSITIONS_HIT = "max_positions_hit"
    CONSECUTIVE_LOSSES = "consecutive_losses"  # 3+ losses in a row
    
    # Medium priority
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    CORRELATION_RISK = "correlation_risk"  # All positions same direction
    OVEREXPOSED_SINGLE = "overexposed_single"  # >30% in one coin
    
    # Warnings
    NO_TRADES_24H = "no_trades_24h"
    WEEKEND_MARKET = "weekend_market"
    NEWS_EVENT = "news_event"
    API_ISSUES = "api_issues"


class VolatilityRegime(str, Enum):
    """Market volatility classification."""
    CALM = "calm"      # ATR < 1.5%
    NORMAL = "normal"  # ATR 1.5-3%
    HIGH = "high"      # ATR > 3%


class RiskMode(str, Enum):
    """Risk management mode."""
    DEFENSIVE = "defensive"  # Protect capital, minimal exposure
    NORMAL = "normal"        # Standard operations
    AGGRESSIVE = "aggressive"  # Allow higher exposure (only when winning)


class RiskAction(str, Enum):
    """Recommended action."""
    PAUSE_TRADING = "pause_trading"  # Stop all new trades
    REDUCE_SIZE = "reduce_size"      # Decrease exposure
    KEEP = "keep"                    # Maintain current approach
    ALLOW_MORE = "allow_more"        # Can increase exposure


class RiskState(BaseModel):
    """
    Input state for risk advisor.
    
    Snapshot of current trading situation for risk assessment.
    """
    # Account state
    equity_usd: Decimal = Field(..., description="Current account equity in USD")
    available_balance_usd: Decimal = Field(..., description="Available margin")
    
    # P&L
    pnl_today_usd: Decimal = Field(default=Decimal("0"), description="Realized P&L today")
    pnl_today_pct: float = Field(default=0.0, description="P&L today as % of equity")
    unrealized_pnl_usd: Decimal = Field(default=Decimal("0"), description="Unrealized P&L")
    unrealized_pnl_pct: float = Field(default=0.0, description="Unrealized P&L as %")
    
    # Drawdown
    drawdown_today_pct: float = Field(
        default=0.0, 
        le=0.0,
        description="Drawdown today from daily high (always <= 0)"
    )
    drawdown_from_peak_pct: float = Field(
        default=0.0,
        le=0.0,
        description="Drawdown from all-time peak (always <= 0)"
    )
    
    # Positions
    open_positions_count: int = Field(default=0, ge=0)
    total_exposure_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Total position size as % of equity"
    )
    avg_position_pnl_pct: float = Field(default=0.0, description="Average P&L across positions")
    max_position_exposure_pct: float = Field(default=0.0, description="Largest single position %")
    
    # Trading metrics
    trades_today: int = Field(default=0, ge=0)
    wins_today: int = Field(default=0, ge=0)
    losses_today: int = Field(default=0, ge=0)
    consecutive_losses: int = Field(default=0, ge=0)
    consecutive_wins: int = Field(default=0, ge=0)
    win_rate_7d: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Market conditions
    volatility_regime: VolatilityRegime = Field(default=VolatilityRegime.NORMAL)
    
    # Stress indicators
    stress_flags: list[StressFlag] = Field(default_factory=list)
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {"use_enum_values": True}

    @property
    def is_in_stress(self) -> bool:
        """Check if any critical stress flag is present."""
        critical = {
            StressFlag.MARGIN_WARNING,
            StressFlag.LIQUIDATION_NEAR,
            StressFlag.SL_MISSING,
            StressFlag.MAX_DAILY_LOSS_HIT,
        }
        return bool(set(self.stress_flags) & critical)
    
    @property
    def stress_level(self) -> int:
        """Calculate stress level 0-10 based on flags."""
        weights = {
            StressFlag.MARGIN_WARNING: 10,
            StressFlag.LIQUIDATION_NEAR: 10,
            StressFlag.SL_MISSING: 8,
            StressFlag.MAX_DAILY_LOSS_HIT: 9,
            StressFlag.MAX_EXPOSURE_HIT: 6,
            StressFlag.MAX_POSITIONS_HIT: 5,
            StressFlag.CONSECUTIVE_LOSSES: 7,
            StressFlag.HIGH_VOLATILITY: 4,
            StressFlag.CORRELATION_RISK: 5,
            StressFlag.OVEREXPOSED_SINGLE: 6,
        }
        total = sum(weights.get(f, 2) for f in self.stress_flags)
        return min(10, total)


class RiskDecision(BaseModel):
    """
    Output decision from risk advisor.
    
    Structured recommendation for risk management.
    """
    # Mode & Action
    mode: RiskMode = Field(..., description="Current risk mode")
    action: RiskAction = Field(..., description="Recommended action")
    
    # Recommended limits
    max_daily_loss_pct: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Recommended max daily loss limit"
    )
    max_exposure_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Recommended max total exposure"
    )
    max_concurrent_trades: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Recommended max open positions"
    )
    max_single_position_pct: float = Field(
        default=20.0,
        ge=0.0,
        le=50.0,
        description="Max size for single position"
    )
    
    # Timing
    pause_duration_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Recommended pause duration if action=pause_trading"
    )
    
    # Explanation
    comment: str = Field(..., max_length=500, description="1-2 sentence explanation")
    reasoning: list[str] = Field(
        default_factory=list,
        description="List of factors that led to this decision"
    )
    
    # Confidence & Metadata
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {"use_enum_values": True}

    def to_telegram_message(self, state: RiskState) -> str:
        """Format decision as Telegram message."""
        mode_emoji = {
            "defensive": "🛡️",
            "normal": "⚖️", 
            "aggressive": "🚀",
        }
        action_emoji = {
            "pause_trading": "⏸️",
            "reduce_size": "📉",
            "keep": "✅",
            "allow_more": "📈",
        }
        
        lines = [
            "🧠 **AGI-Brain Risk View**",
            f"Equity: ${state.equity_usd:,.0f} · PnL today: {state.pnl_today_pct:+.1f}% · DD from peak: {state.drawdown_from_peak_pct:.1f}%",
            f"Exposure: {state.total_exposure_pct:.0f}% · Open trades: {state.open_positions_count}",
            f"Mode: {mode_emoji.get(self.mode, '')} **{self.mode}**",
            f"Action: {action_emoji.get(self.action, '')} {self.action}",
        ]
        
        if self.pause_duration_hours > 0:
            lines.append(f"⏱️ Pause: {self.pause_duration_hours:.0f}h")
        
        lines.append(f"💬 {self.comment}")
        
        if state.stress_flags:
            flags = ", ".join(f.replace("_", " ") for f in state.stress_flags[:3])
            lines.append(f"⚠️ Flags: {flags}")
        
        return "\n".join(lines)


class RiskSnapshot(BaseModel):
    """Complete snapshot with state and decision."""
    state: RiskState
    decision: RiskDecision
    
    def to_log_entry(self) -> dict:
        """Convert to JSONL log entry."""
        return {
            "ts": self.state.timestamp.isoformat(),
            "state": self.state.model_dump(mode="json"),
            "decision": self.decision.model_dump(mode="json"),
        }
