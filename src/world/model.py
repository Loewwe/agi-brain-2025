"""
Core domain models for AGI-Brain world representation.

Описывает все сущности "мира" которые видит AGI-Brain:
- Trading Agent и связанные компоненты
- AGI система
- Владелец
- Внешние ресурсы
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field, ConfigDict


# === Enums ===

class AgentStatus(str, Enum):
    """Статус трейдинг-агента."""
    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class PositionSide(str, Enum):
    """Сторона позиции."""
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    """Тип ордера."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class ServiceStatus(str, Enum):
    """Статус сервиса."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Уровень риска."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AutonomyLevel(int, Enum):
    """Уровень автономии AGI-Brain."""
    L0_READ_ONLY = 0  # Только чтение и анализ
    L1_PROPOSE = 1    # Генерация патчей без применения
    L2_APPLY = 2      # Применение с подтверждением
    L3_AUTONOMOUS = 3 # Автономное выполнение (заблокировано в v1)


# === Trading Domain ===

class Position(BaseModel):
    """Открытая позиция."""
    symbol: str
    side: PositionSide
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    leverage: int
    unrealized_pnl: Decimal
    unrealized_pnl_percent: float
    liquidation_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    opened_at: datetime
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0
    
    @property
    def distance_to_liquidation_percent(self) -> float | None:
        if self.liquidation_price is None:
            return None
        if self.current_price == 0:
            return 0.0
        return abs(float(self.current_price - self.liquidation_price) / float(self.current_price) * 100)


class Trade(BaseModel):
    """Завершённая сделка."""
    id: str
    symbol: str
    side: PositionSide
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    realized_pnl: Decimal
    realized_pnl_percent: float
    fees: Decimal = Field(default=Decimal(0))
    opened_at: datetime
    closed_at: datetime
    reason: str  # strategy, stop_loss, take_profit, manual, liquidation
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Strategy(BaseModel):
    """Торговая стратегия."""
    id: str
    name: str
    enabled: bool
    symbols: list[str]
    parameters: dict[str, Any]
    performance_7d: float  # PnL за 7 дней в %
    win_rate: float
    trades_count: int


class RiskConfig(BaseModel):
    """Конфигурация риск-менеджмента."""
    max_position_size_usd: Decimal
    max_leverage: int
    max_daily_loss_usd: Decimal
    max_daily_loss_percent: float
    unsinkable_balance_usd: Decimal
    max_concurrent_positions: int
    stop_loss_percent: float
    take_profit_percent: float
    
    # UNSINKABLE / RiskGuardian параметры
    enable_risk_guardian: bool = True
    enable_liquidation_protection: bool = True
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PnLMetrics(BaseModel):
    """Метрики прибыли/убытка."""
    total_balance_usd: Decimal
    available_balance_usd: Decimal
    
    # Реализованный PnL
    realized_pnl_today: Decimal
    realized_pnl_week: Decimal
    realized_pnl_month: Decimal
    
    # Нереализованный PnL
    unrealized_pnl: Decimal
    
    # Проценты
    pnl_today_percent: float
    pnl_week_percent: float
    pnl_month_percent: float
    
    # Drawdown
    max_drawdown_today: float
    max_drawdown_week: float
    max_drawdown_month: float
    
    # Статистика
    win_rate: float
    profit_factor: float
    sharpe_ratio: float | None = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RiskViolation(BaseModel):
    """Нарушение риск-правила."""
    rule: str
    severity: RiskLevel
    current_value: Any
    limit_value: Any
    message: str
    detected_at: datetime


class TradingAgent(BaseModel):
    """Трейдинг-агент."""
    id: str
    name: str
    status: AgentStatus
    exchange: str
    strategies: list[Strategy]
    current_positions: list[Position]
    pnl: PnLMetrics
    risk_config: RiskConfig
    risk_violations: list[RiskViolation] = Field(default_factory=list)
    last_trade_at: datetime | None = None
    uptime_hours: float = 0.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# === AGI System Domain ===

class ServiceHealth(BaseModel):
    """Здоровье сервиса."""
    name: str
    status: ServiceStatus
    latency_ms: float | None = None
    error_rate: float = 0.0
    last_check: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class AGISystemState(BaseModel):
    """Состояние AGI системы."""
    version: str
    autonomy_level: AutonomyLevel
    
    # Сервисы
    services: list[ServiceHealth]
    
    # Метрики
    tasks_completed_today: int
    tasks_failed_today: int
    avg_task_duration_seconds: float
    
    # Память
    episodic_memory_size: int
    semantic_memory_size: int
    
    # Обучение
    hard_cases_pending: int
    last_learning_update: datetime | None = None
    
    @property
    def overall_health(self) -> ServiceStatus:
        if not self.services:
            return ServiceStatus.UNKNOWN
        unhealthy = sum(1 for s in self.services if s.status == ServiceStatus.UNHEALTHY)
        degraded = sum(1 for s in self.services if s.status == ServiceStatus.DEGRADED)
        if unhealthy > 0:
            return ServiceStatus.UNHEALTHY
        if degraded > len(self.services) / 2:
            return ServiceStatus.DEGRADED
        return ServiceStatus.HEALTHY


# === Owner Domain ===

class OwnerProfile(BaseModel):
    """Профиль Владельца."""
    id: str
    name: str
    
    # Риск
    max_daily_loss_percent: float
    max_weekly_loss_percent: float
    max_position_size_percent: float
    max_leverage: int
    unsinkable_balance_usd: Decimal
    
    # Цели
    target_daily_return_percent: float
    target_monthly_return_percent: float
    safety_priority: float  # 0-1
    
    # Предпочтения
    decision_style: str
    report_detail_level: str
    notification_hours_start: int
    notification_hours_end: int
    timezone: str
    
    # Ограничения
    forbidden_assets: list[str]
    max_concurrent_positions: int
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# === External Resources ===

class ExternalResource(BaseModel):
    """Внешний ресурс (курсы, документация)."""
    id: str
    name: str
    url: str
    category: str  # science, programming, trading
    last_accessed: datetime | None = None
    is_available: bool


# === World Snapshot ===

class TradingView(BaseModel):
    """Агрегированный вид трейдинга."""
    agent: TradingAgent
    
    # Агрегаты
    total_positions_value: Decimal
    total_unrealized_pnl: Decimal
    total_margin_used_percent: float
    
    # Риск
    current_risk_score: float  # 0-100
    risk_violations_active: list[RiskViolation]
    
    # За период
    trades_today: int
    trades_this_week: int
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def is_within_risk_limits(self) -> bool:
        return len(self.risk_violations_active) == 0


class WorldSnapshot(BaseModel):
    """
    Единый снимок состояния мира для мозга.
    
    Агрегирует данные из всех источников:
    - Трейдинг
    - AGI система
    - Профиль Владельца
    - Внешний контекст
    """
    timestamp: datetime
    
    # Компоненты
    trading: TradingView | None
    agi_system: AGISystemState
    owner_profile: OwnerProfile
    
    # Внешний контекст (новости, рыночные условия и т.д.)
    external_context: dict[str, Any] = Field(default_factory=dict)
    
    # Метаданные снимка
    snapshot_id: str = ""
    generation_time_ms: float = 0.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def model_post_init(self, __context: Any) -> None:
        if not self.snapshot_id:
            self.snapshot_id = f"ws_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    @property
    def overall_status_summary(self) -> str:
        """Краткий статус для мозга."""
        parts = []
        
        if self.trading:
            pnl = self.trading.agent.pnl
            parts.append(f"Trading: {self.trading.agent.status.value}, PnL today: {pnl.pnl_today_percent:+.2f}%")
            if self.trading.risk_violations_active:
                parts.append(f"⚠️ {len(self.trading.risk_violations_active)} risk violations")
        
        parts.append(f"AGI: {self.agi_system.overall_health.value}, L{self.agi_system.autonomy_level.value}")
        
        return " | ".join(parts)


class TradingAdapterProtocol(Protocol):
    """Protocol for trading adapter."""
    
    async def fetch_agent_state(self) -> TradingAgent | None:
        """Получить состояние трейдинг-агента."""
        ...
        
    async def fetch_history(self, days: int = 7, symbol: str | None = None) -> list[Trade]:
        """Получить историю сделок."""
        ...
        
    async def fetch_positions(self) -> list[Position]:
        """Получить открытые позиции."""
        ...
        
    async def fetch_config(self) -> RiskConfig:
        """Получить конфигурацию риск-менеджмента."""
        ...
        
    async def analyze_risk(self, include_recommendations: bool = True) -> Any:
        """Анализировать текущие риски."""
        ...
        
    async def health_check(self) -> bool:
        """Проверка доступности."""
        ...
        
    async def set_stop_loss(self, symbol: str, price: float) -> bool:
        """Установить Stop-Loss для позиции."""
        ...
