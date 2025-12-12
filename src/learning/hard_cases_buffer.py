"""
Hard Cases Buffer for AGI-Brain.

Буфер сложных кейсов для обучения:
- Низкая уверенность
- Неудовлетворённость Владельца
- Инциденты и ошибки
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class HardCaseType(str, Enum):
    """Тип сложного кейса."""
    LOW_CONFIDENCE = "low_confidence"
    OWNER_FEEDBACK = "owner_feedback"
    EXECUTION_ERROR = "execution_error"
    RISK_INCIDENT = "risk_incident"
    STRATEGY_FAILURE = "strategy_failure"
    UNKNOWN = "unknown"


class HardCasePriority(str, Enum):
    """Приоритет обработки."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class HardCase:
    """Сложный кейс для анализа."""
    case_id: str
    case_type: HardCaseType
    priority: HardCasePriority
    
    # Контекст
    goal: str
    plan_id: str | None
    episode_id: str | None
    
    # Данные
    context: dict[str, Any]
    error_message: str | None
    expected_result: str | None
    actual_result: str | None
    
    # Метаданные
    created_at: datetime = field(default_factory=datetime.now)
    processed: bool = False
    processed_at: datetime | None = None
    lessons_learned: str | None = None
    
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.case_id:
            self.case_id = f"hc_{uuid4().hex[:12]}"
    
    def mark_processed(self, lessons: str) -> None:
        """Отметить как обработанный."""
        self.processed = True
        self.processed_at = datetime.now()
        self.lessons_learned = lessons
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type.value,
            "priority": self.priority.value,
            "goal": self.goal,
            "plan_id": self.plan_id,
            "episode_id": self.episode_id,
            "error_message": self.error_message,
            "processed": self.processed,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


class HardCasesBuffer:
    """
    Буфер сложных кейсов AGI-Brain.
    
    Собирает кейсы для:
    - Анализа и обучения
    - Улучшения промптов
    - Выведения новых правил
    """
    
    def __init__(
        self,
        storage_path: Path | None = None,
        max_buffer_size: int = 1000,
    ):
        """
        Args:
            storage_path: Путь для хранения
            max_buffer_size: Максимальный размер буфера
        """
        self.storage_path = storage_path
        self.max_buffer_size = max_buffer_size
        
        self._cases: dict[str, HardCase] = {}
        self._by_type: dict[HardCaseType, list[str]] = {}
        self._by_priority: dict[HardCasePriority, list[str]] = {}
        
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
        
        logger.info(
            "hard_cases_buffer.initialized",
            storage_path=str(storage_path) if storage_path else None,
        )
    
    def add(
        self,
        case_type: HardCaseType,
        goal: str,
        context: dict[str, Any],
        priority: HardCasePriority = HardCasePriority.MEDIUM,
        plan_id: str | None = None,
        episode_id: str | None = None,
        error_message: str | None = None,
        expected_result: str | None = None,
        actual_result: str | None = None,
        tags: list[str] | None = None,
    ) -> HardCase:
        """
        Добавить сложный кейс.
        
        Returns:
            Созданный кейс
        """
        case = HardCase(
            case_id="",
            case_type=case_type,
            priority=priority,
            goal=goal,
            plan_id=plan_id,
            episode_id=episode_id,
            context=context,
            error_message=error_message,
            expected_result=expected_result,
            actual_result=actual_result,
            tags=tags or [],
        )
        
        self._cases[case.case_id] = case
        self._index_case(case)
        
        # Сохранение
        if self.storage_path:
            self._save_case(case)
        
        logger.info(
            "hard_cases.added",
            case_id=case.case_id,
            type=case_type.value,
            priority=priority.value,
        )
        
        # Ограничение размера
        self._enforce_size_limit()
        
        return case
    
    def add_low_confidence(
        self,
        goal: str,
        confidence: float,
        context: dict[str, Any],
        **kwargs,
    ) -> HardCase:
        """Добавить кейс с низкой уверенностью."""
        priority = (
            HardCasePriority.HIGH if confidence < 0.3
            else HardCasePriority.MEDIUM if confidence < 0.5
            else HardCasePriority.LOW
        )
        
        context["confidence"] = confidence
        
        return self.add(
            case_type=HardCaseType.LOW_CONFIDENCE,
            goal=goal,
            context=context,
            priority=priority,
            tags=["confidence"],
            **kwargs,
        )
    
    def add_owner_feedback(
        self,
        goal: str,
        feedback: str,
        rating: int,  # 1-5
        context: dict[str, Any],
        **kwargs,
    ) -> HardCase:
        """Добавить кейс с обратной связью Владельца."""
        priority = (
            HardCasePriority.CRITICAL if rating == 1
            else HardCasePriority.HIGH if rating == 2
            else HardCasePriority.MEDIUM
        )
        
        context["feedback"] = feedback
        context["rating"] = rating
        
        return self.add(
            case_type=HardCaseType.OWNER_FEEDBACK,
            goal=goal,
            context=context,
            priority=priority,
            tags=["owner_feedback"],
            **kwargs,
        )
    
    def add_execution_error(
        self,
        goal: str,
        error: str,
        context: dict[str, Any],
        **kwargs,
    ) -> HardCase:
        """Добавить кейс с ошибкой выполнения."""
        return self.add(
            case_type=HardCaseType.EXECUTION_ERROR,
            goal=goal,
            context=context,
            priority=HardCasePriority.HIGH,
            error_message=error,
            tags=["error"],
            **kwargs,
        )
    
    def add_risk_incident(
        self,
        description: str,
        loss_amount: float | None,
        context: dict[str, Any],
        **kwargs,
    ) -> HardCase:
        """Добавить кейс риск-инцидента."""
        context["loss_amount"] = loss_amount
        
        return self.add(
            case_type=HardCaseType.RISK_INCIDENT,
            goal=description,
            context=context,
            priority=HardCasePriority.CRITICAL,
            tags=["risk", "incident"],
            **kwargs,
        )
    
    def get(self, case_id: str) -> HardCase | None:
        """Получить кейс по ID."""
        return self._cases.get(case_id)
    
    def get_unprocessed(self, limit: int = 50) -> list[HardCase]:
        """Получить необработанные кейсы."""
        unprocessed = [c for c in self._cases.values() if not c.processed]
        # Сортировка по приоритету и дате
        unprocessed.sort(key=lambda c: (
            list(HardCasePriority).index(c.priority),
            c.created_at,
        ))
        return unprocessed[:limit]
    
    def get_by_type(self, case_type: HardCaseType, limit: int = 50) -> list[HardCase]:
        """Получить кейсы по типу."""
        case_ids = self._by_type.get(case_type, [])
        return [self._cases[id] for id in case_ids[:limit] if id in self._cases]
    
    def get_by_priority(self, priority: HardCasePriority, limit: int = 50) -> list[HardCase]:
        """Получить кейсы по приоритету."""
        case_ids = self._by_priority.get(priority, [])
        return [self._cases[id] for id in case_ids[:limit] if id in self._cases]

    def get_recent(self, limit: int = 50) -> list[HardCase]:
        """Получить последние кейсы."""
        cases = list(self._cases.values())
        cases.sort(key=lambda c: c.created_at, reverse=True)
        return cases[:limit]
    
    def mark_processed(self, case_id: str, lessons: str) -> bool:
        """Отметить кейс как обработанный."""
        case = self._cases.get(case_id)
        if not case:
            return False
        
        case.mark_processed(lessons)
        
        if self.storage_path:
            self._save_case(case)
        
        logger.info("hard_cases.processed", case_id=case_id)
        return True
    
    def get_count(self) -> int:
        """Количество кейсов."""
        return len(self._cases)
    
    def get_unprocessed_count(self) -> int:
        """Количество необработанных кейсов."""
        return sum(1 for c in self._cases.values() if not c.processed)
    
    def get_stats(self) -> dict[str, Any]:
        """Статистика буфера."""
        type_counts = {t.value: len(ids) for t, ids in self._by_type.items()}
        priority_counts = {p.value: len(ids) for p, ids in self._by_priority.items()}
        
        return {
            "total": len(self._cases),
            "unprocessed": self.get_unprocessed_count(),
            "by_type": type_counts,
            "by_priority": priority_counts,
        }
    
    def _index_case(self, case: HardCase) -> None:
        """Индексация кейса."""
        if case.case_type not in self._by_type:
            self._by_type[case.case_type] = []
        self._by_type[case.case_type].append(case.case_id)
        
        if case.priority not in self._by_priority:
            self._by_priority[case.priority] = []
        self._by_priority[case.priority].append(case.case_id)
    
    def _enforce_size_limit(self) -> None:
        """Ограничение размера буфера."""
        if len(self._cases) <= self.max_buffer_size:
            return
        
        # Удаляем обработанные старые кейсы
        processed = sorted(
            [c for c in self._cases.values() if c.processed],
            key=lambda c: c.created_at,
        )
        
        to_remove = len(self._cases) - self.max_buffer_size
        for case in processed[:to_remove]:
            del self._cases[case.case_id]
    
    def _save_case(self, case: HardCase) -> None:
        """Сохранить кейс на диск."""
        if not self.storage_path:
            return
        
        file_path = self.storage_path / f"{case.case_id}.json"
        with open(file_path, "w") as f:
            json.dump(case.to_dict(), f, indent=2)
    
    def _load_from_disk(self) -> None:
        """Загрузить кейсы с диска."""
        if not self.storage_path:
            return
        
        for file_path in self.storage_path.glob("hc_*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    logger.debug("hard_cases.loaded", case_id=data.get("case_id"))
            except Exception as e:
                logger.warning("hard_cases.load_failed", path=str(file_path), error=str(e))


# === Singleton ===

_hard_cases_buffer: HardCasesBuffer | None = None


def get_hard_cases_buffer() -> HardCasesBuffer:
    """Получить singleton буфера."""
    global _hard_cases_buffer
    if _hard_cases_buffer is None:
        _hard_cases_buffer = HardCasesBuffer()
    return _hard_cases_buffer


def configure_hard_cases_buffer(storage_path: Path | None = None) -> HardCasesBuffer:
    """Сконфигурировать буфер."""
    global _hard_cases_buffer
    _hard_cases_buffer = HardCasesBuffer(storage_path=storage_path)
    return _hard_cases_buffer
