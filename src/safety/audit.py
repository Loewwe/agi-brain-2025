"""
Audit Logger for AGI-Brain.

Полное логирование всех действий:
- Оркестрация
- Вызовы инструментов
- Решения об одобрении
- Ошибки и инциденты
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class AuditEventType(str, Enum):
    """Тип события аудита."""
    # Orchestration
    PLAN_CREATED = "plan_created"
    PLAN_EXECUTED = "plan_executed"
    PLAN_FAILED = "plan_failed"
    
    # Tools
    TOOL_CALLED = "tool_called"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    
    # Approval
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_TIMEOUT = "approval_timeout"
    
    # Security
    ACCESS_DENIED = "access_denied"
    VIOLATION_DETECTED = "violation_detected"
    RISK_ALERT = "risk_alert"
    
    # System
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGED = "config_changed"
    
    # Learning
    HARD_CASE_RECORDED = "hard_case_recorded"
    LEARNING_UPDATE = "learning_update"


@dataclass
class AuditEvent:
    """Событие аудита."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    
    # Контекст
    client_id: str
    session_id: str | None
    
    # Данные события
    resource: str  # tool, plan, etc.
    action: str
    details: dict[str, Any]
    
    # Результат
    success: bool
    error: str | None = None
    
    # Метаданные
    severity: str = "info"  # debug, info, warning, error, critical
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"audit_{uuid4().hex[:12]}"
    
    def to_dict(self) -> dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "client_id": self.client_id,
            "session_id": self.session_id,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "success": self.success,
            "error": self.error,
            "severity": self.severity,
            "tags": self.tags,
        }
    
    def to_json(self) -> str:
        """Сериализация в JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLogger:
    """
    Аудит-логгер AGI-Brain.
    
    Обеспечивает полную прозрачность действий системы:
    - Логирование всех событий
    - Хранение истории
    - Поиск и фильтрация
    - Экспорт для анализа
    """
    
    def __init__(
        self,
        storage_path: Path | None = None,
        max_memory_events: int = 10000,
    ):
        """
        Args:
            storage_path: Путь для хранения логов
            max_memory_events: Максимум событий в памяти
        """
        self.storage_path = storage_path
        self.max_memory_events = max_memory_events
        
        self._events: list[AuditEvent] = []
        self._events_by_type: dict[AuditEventType, list[str]] = {}
        self._events_by_session: dict[str, list[str]] = {}
        
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
        
        # Записываем событие старта
        self.log(
            event_type=AuditEventType.SYSTEM_START,
            client_id="system",
            resource="audit_logger",
            action="initialize",
            details={"storage_path": str(storage_path) if storage_path else None},
        )
    
    def log(
        self,
        event_type: AuditEventType,
        client_id: str,
        resource: str,
        action: str,
        details: dict[str, Any] | None = None,
        session_id: str | None = None,
        success: bool = True,
        error: str | None = None,
        severity: str = "info",
        tags: list[str] | None = None,
    ) -> AuditEvent:
        """
        Записать событие аудита.
        
        Returns:
            Созданное событие
        """
        event = AuditEvent(
            event_id="",
            event_type=event_type,
            timestamp=datetime.now(),
            client_id=client_id,
            session_id=session_id,
            resource=resource,
            action=action,
            details=details or {},
            success=success,
            error=error,
            severity=severity,
            tags=tags or [],
        )
        
        # Сохранение в память
        self._events.append(event)
        
        # Индексирование
        if event_type not in self._events_by_type:
            self._events_by_type[event_type] = []
        self._events_by_type[event_type].append(event.event_id)
        
        if session_id:
            if session_id not in self._events_by_session:
                self._events_by_session[session_id] = []
            self._events_by_session[session_id].append(event.event_id)
        
        # Ограничение размера
        if len(self._events) > self.max_memory_events:
            self._events = self._events[-self.max_memory_events // 2:]
        
        # Сохранение на диск
        if self.storage_path:
            self._persist_event(event)
        
        # Структурированный лог
        logger.info(
            f"audit.{event_type.value}",
            event_id=event.event_id,
            resource=resource,
            action=action,
            success=success,
            client_id=client_id,
        )
        
        return event
    
    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        client_id: str,
        session_id: str | None = None,
    ) -> str:
        """Логировать вызов инструмента."""
        event = self.log(
            event_type=AuditEventType.TOOL_CALLED,
            client_id=client_id,
            resource=tool_name,
            action="call",
            details={"args": args},
            session_id=session_id,
        )
        return event.event_id
    
    def log_tool_result(
        self,
        event_id: str,
        success: bool,
        output: Any = None,
        error: str | None = None,
        duration_ms: float = 0,
    ) -> None:
        """Логировать результат выполнения инструмента."""
        # Найти исходное событие
        original = next((e for e in self._events if e.event_id == event_id), None)
        
        self.log(
            event_type=AuditEventType.TOOL_SUCCEEDED if success else AuditEventType.TOOL_FAILED,
            client_id=original.client_id if original else "unknown",
            resource=original.resource if original else "unknown",
            action="result",
            details={
                "original_event_id": event_id,
                "duration_ms": duration_ms,
                "output_type": type(output).__name__ if output else None,
            },
            session_id=original.session_id if original else None,
            success=success,
            error=error,
            severity="info" if success else "error",
        )
    
    def log_approval(
        self,
        plan_id: str,
        step_id: str | None,
        decision: str,  # requested, granted, denied, timeout
        client_id: str,
        reason: str | None = None,
    ) -> None:
        """Логировать решение об одобрении."""
        event_type_map = {
            "requested": AuditEventType.APPROVAL_REQUESTED,
            "granted": AuditEventType.APPROVAL_GRANTED,
            "denied": AuditEventType.APPROVAL_DENIED,
            "timeout": AuditEventType.APPROVAL_TIMEOUT,
        }
        
        self.log(
            event_type=event_type_map.get(decision, AuditEventType.APPROVAL_REQUESTED),
            client_id=client_id,
            resource=f"plan:{plan_id}",
            action=decision,
            details={
                "plan_id": plan_id,
                "step_id": step_id,
                "reason": reason,
            },
            success=decision == "granted",
            severity="warning" if decision in ("denied", "timeout") else "info",
        )
    
    def log_security_event(
        self,
        event_type: AuditEventType,
        client_id: str,
        resource: str,
        details: dict[str, Any],
        severity: str = "warning",
    ) -> None:
        """Логировать событие безопасности."""
        self.log(
            event_type=event_type,
            client_id=client_id,
            resource=resource,
            action="security_event",
            details=details,
            success=False,
            severity=severity,
            tags=["security"],
        )
    
    def get_events(
        self,
        event_type: AuditEventType | None = None,
        session_id: str | None = None,
        client_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """
        Получить события с фильтрацией.
        
        Returns:
            Список событий
        """
        results = self._events
        
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        
        if session_id:
            results = [e for e in results if e.session_id == session_id]
        
        if client_id:
            results = [e for e in results if e.client_id == client_id]
        
        if since:
            results = [e for e in results if e.timestamp >= since]
        
        if until:
            results = [e for e in results if e.timestamp <= until]
        
        # Сортировка по времени (новые первые)
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)
        
        return results[:limit]
    
    def get_session_history(self, session_id: str) -> list[AuditEvent]:
        """Получить историю сессии."""
        event_ids = self._events_by_session.get(session_id, [])
        return [e for e in self._events if e.event_id in event_ids]
    
    def get_stats(self) -> dict[str, Any]:
        """Статистика аудита."""
        type_counts = {}
        for event_type, ids in self._events_by_type.items():
            type_counts[event_type.value] = len(ids)
        
        failed_events = sum(1 for e in self._events if not e.success)
        
        return {
            "total_events": len(self._events),
            "events_by_type": type_counts,
            "sessions_count": len(self._events_by_session),
            "failed_events": failed_events,
            "storage_enabled": self.storage_path is not None,
        }
    
    def export_session(self, session_id: str, output_path: Path) -> None:
        """Экспортировать сессию в файл."""
        events = self.get_session_history(session_id)
        
        with open(output_path, "w") as f:
            for event in events:
                f.write(event.to_json() + "\n")
    
    def _persist_event(self, event: AuditEvent) -> None:
        """Сохранить событие на диск."""
        if not self.storage_path:
            return
        
        # Файл по дате
        date_str = event.timestamp.strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit_{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(event.to_json() + "\n")


# === Singleton ===

_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Получить singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def configure_audit_logger(
    storage_path: Path | None = None,
) -> AuditLogger:
    """Сконфигурировать audit logger."""
    global _audit_logger
    _audit_logger = AuditLogger(storage_path=storage_path)
    return _audit_logger
