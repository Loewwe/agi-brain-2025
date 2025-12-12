"""
World Snapshot Builder for AGI-Brain.

Aggregates data from all perception adapters into a unified WorldSnapshot.
"""

import time
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from .model import (
    AGISystemState,
    AutonomyLevel,
    OwnerProfile,
    ServiceHealth,
    ServiceStatus,
    TradingAgent,
    TradingView,
    WorldSnapshot,
    TradingAdapterProtocol,
)
from .trading_view import TradingViewBuilder


class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collector."""
    
    async def get_service_health(self, service_name: str) -> ServiceHealth:
        """Получить здоровье сервиса."""
        ...
    
    async def get_task_stats(self) -> dict[str, int]:
        """Получить статистику задач."""
        ...


class MemoryProtocol(Protocol):
    """Protocol for memory access."""
    
    async def get_episodic_memory_size(self) -> int:
        """Получить размер эпизодической памяти."""
        ...
    
    async def get_semantic_memory_size(self) -> int:
        """Получить размер семантической памяти."""
        ...
    
    async def get_hard_cases_count(self) -> int:
        """Получить количество сложных кейсов."""
        ...


class SnapshotBuilder:
    """
    Строитель WorldSnapshot.
    
    Агрегирует данные из всех источников:
    - Trading Adapter
    - Metrics Collector
    - Memory
    - Owner Profile
    """
    
    def __init__(
        self,
        trading_adapter: TradingAdapterProtocol | None = None,
        metrics_collector: MetricsCollectorProtocol | None = None,
        episodic_memory: Any | None = None,
        semantic_memory: Any | None = None,
        owner_profile: OwnerProfile | None = None,
    ):
        self.trading_adapter = trading_adapter
        self.metrics_collector = metrics_collector
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.owner_profile = owner_profile
        self.trading_view_builder = TradingViewBuilder()
    
    async def build(
        self,
        include_trading: bool = True,
        include_metrics: bool = True,
        include_owner_profile: bool = True,
    ) -> WorldSnapshot:
        """
        Построить WorldSnapshot.
        
        Args:
            include_trading: Включить данные трейдинга
            include_metrics: Включить метрики
            include_owner_profile: Включить профиль владельца
            
        Returns:
            WorldSnapshot с агрегированными данными
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        # Trading View
        trading_view: TradingView | None = None
        if include_trading and self.trading_adapter:
            agent = await self.trading_adapter.fetch_agent_state()
            if agent:
                trading_view = self.trading_view_builder.build(agent)
        
        # AGI System State
        agi_state = await self._build_agi_state(include_metrics)
        
        # Owner Profile
        if include_owner_profile:
            owner = self.owner_profile if self.owner_profile else self._get_default_owner()
        else:
            owner = self._get_default_owner()
        
        generation_time_ms = (time.time() - start_time) * 1000
        
        return WorldSnapshot(
            timestamp=timestamp,
            trading=trading_view,
            agi_system=agi_state,
            owner_profile=owner,
            generation_time_ms=generation_time_ms,
        )
    
    async def _build_agi_state(self, include_metrics: bool) -> AGISystemState:
        """Построить состояние AGI системы."""
        services: list[ServiceHealth] = []
        task_stats = {"completed": 0, "failed": 0}
        avg_duration = 0.0
        episodic_size = 0
        semantic_size = 0
        hard_cases = 0
        last_learning: datetime | None = None
        
        if include_metrics and self.metrics_collector:
            # Собираем здоровье сервисов
            service_names = ["brain", "memory", "perception", "safety"]
            for name in service_names:
                try:
                    health = await self.metrics_collector.get_service_health(name)
                    services.append(health)
                except Exception:
                    services.append(ServiceHealth(
                        name=name,
                        status=ServiceStatus.UNKNOWN,
                        latency_ms=None,
                        error_rate=0.0,
                        last_check=datetime.now(),
                    ))
            
            # Статистика задач
            task_stats = await self.metrics_collector.get_task_stats()
        
        if self.episodic_memory:
            episodic_size = self.episodic_memory.get_size()
        
        if self.semantic_memory:
            semantic_size = self.semantic_memory.get_size()
            
        # Hard cases from episodic? Or separate?
        # Assuming episodic memory has hard cases buffer access or we ignore it for now
        # or we add hard_cases_buffer as dependency.
        # For now, let's skip hard cases count or get it from episodic if available.
        if self.episodic_memory and hasattr(self.episodic_memory, "get_hard_cases_count"):
             hard_cases = await self.episodic_memory.get_hard_cases_count()
        
        return AGISystemState(
            version="1.0.0",
            autonomy_level=AutonomyLevel.L0_READ_ONLY,  # v1 default
            services=services,
            tasks_completed_today=task_stats.get("completed", 0),
            tasks_failed_today=task_stats.get("failed", 0),
            avg_task_duration_seconds=avg_duration,
            episodic_memory_size=episodic_size,
            semantic_memory_size=semantic_size,
            hard_cases_pending=hard_cases,
            last_learning_update=last_learning,
        )
    
    def _get_default_owner(self) -> OwnerProfile:
        """Получить дефолтный профиль владельца."""
        return OwnerProfile(
            id="owner_default",
            name="Owner",
            max_daily_loss_percent=2.0,
            max_weekly_loss_percent=5.0,
            max_position_size_percent=10.0,
            max_leverage=5,
            unsinkable_balance_usd=Decimal("10000"),
            target_daily_return_percent=0.5,
            target_monthly_return_percent=15.0,
            safety_priority=0.7,
            decision_style="analytical",
            report_detail_level="detailed",
            notification_hours_start=9,
            notification_hours_end=22,
            timezone="Asia/Almaty",
            forbidden_assets=["SHIB", "DOGE", "PEPE"],
            max_concurrent_positions=5,
        )
    
    async def build_quick(self) -> WorldSnapshot:
        """Быстрый снимок без тяжёлых запросов."""
        return await self.build(
            include_trading=False,
            include_metrics=False,
            include_owner_profile=True,
        )


# === Singleton для глобального доступа ===

_snapshot_builder: SnapshotBuilder | None = None


def get_snapshot_builder() -> SnapshotBuilder:
    """Получить singleton builder."""
    global _snapshot_builder
    if _snapshot_builder is None:
        _snapshot_builder = SnapshotBuilder()
    return _snapshot_builder


def configure_snapshot_builder(
    trading_adapter: TradingAdapterProtocol | None = None,
    metrics_collector: MetricsCollectorProtocol | None = None,
    episodic_memory: Any | None = None,
    semantic_memory: Any | None = None,
    owner_profile: OwnerProfile | None = None,
) -> SnapshotBuilder:
    """Сконфигурировать и получить builder."""
    global _snapshot_builder
    _snapshot_builder = SnapshotBuilder(
        trading_adapter=trading_adapter,
        metrics_collector=metrics_collector,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        owner_profile=owner_profile,
    )
    return _snapshot_builder


async def world_snapshot(**kwargs) -> WorldSnapshot:
    """
    Удобная функция для получения снимка мира.
    
    Основная точка входа для мозга.
    """
    builder = get_snapshot_builder()
    return await builder.build(**kwargs)
