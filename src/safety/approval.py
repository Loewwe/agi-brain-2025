"""
Human Approval Workflow for AGI-Brain.

Workflow одобрения человеком:
- Очередь ожидающих одобрения
- Таймауты
- Обработка решений
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

import structlog

from .audit import get_audit_logger, AuditEventType

logger = structlog.get_logger()


class ApprovalStatus(str, Enum):
    """Статус запроса на одобрение."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """Запрос на одобрение."""
    request_id: str
    
    # Контекст
    plan_id: str
    step_id: str | None
    client_id: str
    
    # Описание
    action_type: str
    action_description: str
    risk_level: str
    
    # Данные
    details: dict[str, Any]
    requires_reason: bool = False
    
    # Статус
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_by: str | None = None
    decision_reason: str | None = None
    
    # Временные метки
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"approval_{uuid4().hex[:12]}"
        if self.expires_at is None:
            # Default: 24 hours
            self.expires_at = self.created_at + timedelta(hours=24)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at if self.expires_at else False
    
    @property
    def is_pending(self) -> bool:
        return self.status == ApprovalStatus.PENDING and not self.is_expired
    
    def approve(self, by: str, reason: str | None = None) -> None:
        """Одобрить запрос."""
        self.status = ApprovalStatus.APPROVED
        self.decision_by = by
        self.decision_reason = reason
        self.decided_at = datetime.now()
    
    def deny(self, by: str, reason: str | None = None) -> None:
        """Отклонить запрос."""
        self.status = ApprovalStatus.DENIED
        self.decision_by = by
        self.decision_reason = reason
        self.decided_at = datetime.now()
    
    def timeout(self) -> None:
        """Пометить как просроченный."""
        self.status = ApprovalStatus.TIMEOUT
        self.decided_at = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "risk_level": self.risk_level,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "decision_by": self.decision_by,
            "decision_reason": self.decision_reason,
        }


class ApprovalManager:
    """
    Менеджер одобрений AGI-Brain.
    
    Управляет очередью запросов на одобрение:
    - Создание запросов
    - Ожидание решения
    - Обработка таймаутов
    - Уведомления
    """
    
    def __init__(
        self,
        default_timeout_hours: float = 24,
        on_approval_needed: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
    ):
        """
        Args:
            default_timeout_hours: Таймаут по умолчанию
            on_approval_needed: Callback при создании запроса
        """
        self.default_timeout_hours = default_timeout_hours
        self.on_approval_needed = on_approval_needed
        
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        
        logger.info(
            "approval_manager.initialized",
            timeout_hours=default_timeout_hours,
        )
    
    async def request_approval(
        self,
        plan_id: str,
        action_type: str,
        action_description: str,
        risk_level: str = "high",
        step_id: str | None = None,
        client_id: str = "owner",
        details: dict[str, Any] | None = None,
        timeout_hours: float | None = None,
    ) -> ApprovalRequest:
        """
        Создать запрос на одобрение.
        
        Args:
            plan_id: ID плана
            action_type: Тип действия
            action_description: Описание
            risk_level: Уровень риска
            step_id: ID шага (опционально)
            client_id: ID клиента
            details: Дополнительные данные
            timeout_hours: Таймаут
            
        Returns:
            Созданный запрос
        """
        timeout = timeout_hours or self.default_timeout_hours
        
        request = ApprovalRequest(
            request_id="",
            plan_id=plan_id,
            step_id=step_id,
            client_id=client_id,
            action_type=action_type,
            action_description=action_description,
            risk_level=risk_level,
            details=details or {},
            expires_at=datetime.now() + timedelta(hours=timeout),
        )
        
        self._pending[request.request_id] = request
        
        # Аудит
        audit = get_audit_logger()
        audit.log_approval(plan_id, step_id, "requested", client_id)
        
        # Callback
        if self.on_approval_needed:
            await self.on_approval_needed(request)
        
        logger.info(
            "approval.requested",
            request_id=request.request_id,
            action=action_type,
            risk=risk_level,
        )
        
        return request
    
    async def wait_for_approval(
        self,
        request_id: str,
        poll_interval: float = 1.0,
    ) -> ApprovalRequest:
        """
        Дождаться решения по запросу.
        
        Args:
            request_id: ID запроса
            poll_interval: Интервал проверки (секунды)
            
        Returns:
            Запрос с решением
        """
        while True:
            request = self._pending.get(request_id)
            
            if not request:
                raise ValueError(f"Request not found: {request_id}")
            
            # Проверка таймаута
            if request.is_expired and request.status == ApprovalStatus.PENDING:
                request.timeout()
                self._finalize_request(request)
                audit = get_audit_logger()
                audit.log_approval(request.plan_id, request.step_id, "timeout", request.client_id)
            
            if not request.is_pending:
                return request
            
            await asyncio.sleep(poll_interval)
    
    def approve(
        self,
        request_id: str,
        by: str,
        reason: str | None = None,
    ) -> bool:
        """
        Одобрить запрос.
        
        Returns:
            True если успешно
        """
        request = self._pending.get(request_id)
        if not request or not request.is_pending:
            return False
        
        request.approve(by, reason)
        self._finalize_request(request)
        
        audit = get_audit_logger()
        audit.log_approval(request.plan_id, request.step_id, "granted", by, reason)
        
        logger.info(
            "approval.granted",
            request_id=request_id,
            by=by,
        )
        
        return True
    
    def deny(
        self,
        request_id: str,
        by: str,
        reason: str | None = None,
    ) -> bool:
        """
        Отклонить запрос.
        
        Returns:
            True если успешно
        """
        request = self._pending.get(request_id)
        if not request or not request.is_pending:
            return False
        
        request.deny(by, reason)
        self._finalize_request(request)
        
        audit = get_audit_logger()
        audit.log_approval(request.plan_id, request.step_id, "denied", by, reason)
        
        logger.info(
            "approval.denied",
            request_id=request_id,
            by=by,
            reason=reason,
        )
        
        return True
    
    def get_pending(self) -> list[ApprovalRequest]:
        """Получить все ожидающие запросы."""
        # Проверяем и обновляем таймауты
        for request in list(self._pending.values()):
            if request.is_expired and request.status == ApprovalStatus.PENDING:
                request.timeout()
                self._finalize_request(request)
        
        return [r for r in self._pending.values() if r.is_pending]
    
    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Получить запрос по ID."""
        return self._pending.get(request_id) or next(
            (r for r in self._history if r.request_id == request_id),
            None
        )
    
    def get_history(self, limit: int = 100) -> list[ApprovalRequest]:
        """Получить историю решений."""
        return self._history[-limit:]
    
    def get_stats(self) -> dict[str, Any]:
        """Статистика одобрений."""
        total = len(self._history)
        approved = sum(1 for r in self._history if r.status == ApprovalStatus.APPROVED)
        denied = sum(1 for r in self._history if r.status == ApprovalStatus.DENIED)
        timeouts = sum(1 for r in self._history if r.status == ApprovalStatus.TIMEOUT)
        
        return {
            "pending": len(self.get_pending()),
            "total_processed": total,
            "approved": approved,
            "denied": denied,
            "timeouts": timeouts,
            "approval_rate": approved / total if total > 0 else 0,
        }
    
    def _finalize_request(self, request: ApprovalRequest) -> None:
        """Переместить запрос в историю."""
        if request.request_id in self._pending:
            del self._pending[request.request_id]
        self._history.append(request)
        
        # Ограничение истории
        if len(self._history) > 1000:
            self._history = self._history[-500:]


# === Singleton ===

_approval_manager: ApprovalManager | None = None


def get_approval_manager() -> ApprovalManager:
    """Получить singleton approval manager."""
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalManager()
    return _approval_manager


def configure_approval_manager(
    default_timeout_hours: float = 24,
    on_approval_needed: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
) -> ApprovalManager:
    """Сконфигурировать approval manager."""
    global _approval_manager
    _approval_manager = ApprovalManager(
        default_timeout_hours=default_timeout_hours,
        on_approval_needed=on_approval_needed,
    )
    return _approval_manager
