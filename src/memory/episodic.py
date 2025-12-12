"""
Episodic Memory for AGI-Brain.

Хранит сессии работы мозга с полным контекстом:
- Цель
- Снимок мира
- План
- Шаги выполнения
- Результаты и ошибки
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class ExecutionStatus(str, Enum):
    """Статус выполнения эпизода."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


@dataclass
class ExecutionStep:
    """Шаг выполнения плана."""
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    
    def mark_started(self) -> None:
        self.started_at = datetime.now()
        self.status = ExecutionStatus.RUNNING
    
    def mark_completed(self, result: Any) -> None:
        self.completed_at = datetime.now()
        self.status = ExecutionStatus.COMPLETED
        self.result = result
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error: str) -> None:
        self.completed_at = datetime.now()
        self.status = ExecutionStatus.FAILED
        self.error = error
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


@dataclass
class Plan:
    """План выполнения задачи."""
    plan_id: str
    goal: str
    steps: list[ExecutionStep]
    created_at: datetime = field(default_factory=datetime.now)
    requires_approval: bool = False
    approval_reason: str | None = None
    
    @property
    def total_steps(self) -> int:
        return len(self.steps)
    
    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == ExecutionStatus.COMPLETED)
    
    @property
    def progress_percent(self) -> float:
        if not self.steps:
            return 100.0
        return (self.completed_steps / self.total_steps) * 100


@dataclass
class ExecutionResult:
    """Результат выполнения эпизода."""
    success: bool
    summary: str
    output: Any = None
    human_required: bool = False
    human_required_reason: str | None = None


@dataclass
class Error:
    """Ошибка во время выполнения."""
    code: str
    message: str
    step_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    recoverable: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """
    Эпизод работы мозга.
    
    Полная запись одной сессии выполнения задачи.
    """
    event_id: str
    goal: str
    context_hint: str | None
    client_id: str
    
    # Снимок мира на момент начала (сериализованный)
    world_snapshot_json: str
    
    # План и выполнение
    plan: Plan | None = None
    steps: list[ExecutionStep] = field(default_factory=list)
    result: ExecutionResult | None = None
    errors: list[Error] = field(default_factory=list)
    
    # Временные метки
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Статус
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Метаданные
    autonomy_level: int = 0
    tools_used: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"ep_{uuid.uuid4().hex[:12]}"
    
    def start(self) -> None:
        """Начать выполнение эпизода."""
        self.started_at = datetime.now()
        self.status = ExecutionStatus.RUNNING
    
    def complete(self, result: ExecutionResult) -> None:
        """Завершить эпизод."""
        self.completed_at = datetime.now()
        self.result = result
        self.status = ExecutionStatus.COMPLETED if result.success else ExecutionStatus.FAILED
    
    def add_error(self, error: Error) -> None:
        """Добавить ошибку."""
        self.errors.append(error)
    
    @property
    def duration_seconds(self) -> float | None:
        """Длительность эпизода."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_successful(self) -> bool:
        """Был ли эпизод успешным."""
        return self.result is not None and self.result.success
    
    def to_dict(self) -> dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "event_id": self.event_id,
            "goal": self.goal,
            "context_hint": self.context_hint,
            "client_id": self.client_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "is_successful": self.is_successful,
            "tools_used": self.tools_used,
            "tags": self.tags,
            "errors_count": len(self.errors),
        }


class EpisodicMemory:
    """
    Эпизодическая память AGI-Brain.
    
    Хранит историю всех сессий работы мозга.
    Позволяет:
    - Аудит и разбор полётов
    - Воспроизведение важных сессий
    - Обучение на успехах и ошибках
    """
    
    def __init__(self, storage_path: Path | None = None):
        """
        Инициализация эпизодической памяти.
        
        Args:
            storage_path: Путь для хранения эпизодов (опционально)
        """
        self.storage_path = storage_path
        self._episodes: dict[str, Episode] = {}
        self._index_by_date: dict[str, list[str]] = {}
        self._index_by_status: dict[ExecutionStatus, list[str]] = {}
        
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def create_episode(
        self,
        goal: str,
        world_snapshot_json: str,
        client_id: str = "owner",
        context_hint: str | None = None,
        tags: list[str] | None = None,
    ) -> Episode:
        """
        Создать новый эпизод.
        
        Args:
            goal: Цель эпизода
            world_snapshot_json: Сериализованный снимок мира
            client_id: ID клиента
            context_hint: Подсказка контекста
            tags: Теги для поиска
            
        Returns:
            Созданный эпизод
        """
        episode = Episode(
            event_id="",  # Будет сгенерирован в __post_init__
            goal=goal,
            context_hint=context_hint,
            client_id=client_id,
            world_snapshot_json=world_snapshot_json,
            tags=tags or [],
        )
        
        self._episodes[episode.event_id] = episode
        self._index_episode(episode)
        
        logger.info(
            "episodic_memory.episode_created",
            event_id=episode.event_id,
            goal=goal[:100],
            client_id=client_id,
        )
        
        return episode
    
    def get(self, event_id: str) -> Episode | None:
        """Получить эпизод по ID."""
        return self._episodes.get(event_id)
    
    def update(self, episode: Episode) -> None:
        """Обновить эпизод в памяти."""
        self._episodes[episode.event_id] = episode
        if self.storage_path:
            self._save_episode(episode)
    
    def search(
        self,
        query: str | None = None,
        status: ExecutionStatus | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        client_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[Episode]:
        """
        Поиск эпизодов.
        
        Args:
            query: Текстовый поиск по цели
            status: Фильтр по статусу
            date_from: Начало периода
            date_to: Конец периода
            client_id: Фильтр по клиенту
            tags: Фильтр по тегам
            limit: Максимальное количество
            
        Returns:
            Список найденных эпизодов
        """
        results = list(self._episodes.values())
        
        # Фильтрация
        if query:
            query_lower = query.lower()
            results = [e for e in results if query_lower in e.goal.lower()]
        
        if status:
            results = [e for e in results if e.status == status]
        
        if date_from:
            results = [e for e in results if e.created_at >= date_from]
        
        if date_to:
            results = [e for e in results if e.created_at <= date_to]
        
        if client_id:
            results = [e for e in results if e.client_id == client_id]
        
        if tags:
            results = [e for e in results if any(t in e.tags for t in tags)]
        
        # Сортировка по дате (новые первые)
        results.sort(key=lambda e: e.created_at, reverse=True)
        
        return results[:limit]
    
    def get_recent(self, limit: int = 10) -> list[Episode]:
        """Получить последние эпизоды."""
        return self.search(limit=limit)
    
    def get_failed(self, limit: int = 10) -> list[Episode]:
        """Получить неуспешные эпизоды для анализа."""
        return self.search(status=ExecutionStatus.FAILED, limit=limit)
    
    def get_size(self) -> int:
        """Размер памяти (количество эпизодов)."""
        return len(self._episodes)
    
    def get_stats(self) -> dict[str, Any]:
        """Статистика по памяти."""
        episodes = list(self._episodes.values())
        
        if not episodes:
            return {"total": 0}
        
        successful = sum(1 for e in episodes if e.is_successful)
        failed = sum(1 for e in episodes if e.status == ExecutionStatus.FAILED)
        
        durations = [e.duration_seconds for e in episodes if e.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total": len(episodes),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(episodes) if episodes else 0,
            "avg_duration_seconds": avg_duration,
            "oldest": min(e.created_at for e in episodes).isoformat(),
            "newest": max(e.created_at for e in episodes).isoformat(),
        }
    
    def _index_episode(self, episode: Episode) -> None:
        """Индексация эпизода."""
        date_key = episode.created_at.date().isoformat()
        if date_key not in self._index_by_date:
            self._index_by_date[date_key] = []
        self._index_by_date[date_key].append(episode.event_id)
        
        if episode.status not in self._index_by_status:
            self._index_by_status[episode.status] = []
        self._index_by_status[episode.status].append(episode.event_id)
    
    def _save_episode(self, episode: Episode) -> None:
        """Сохранить эпизод на диск."""
        if not self.storage_path:
            return
        
        file_path = self.storage_path / f"{episode.event_id}.json"
        with open(file_path, "w") as f:
            json.dump(episode.to_dict(), f, indent=2)
    
    def _load_from_disk(self) -> None:
        """Загрузить эпизоды с диска."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    # Упрощённое восстановление для индекса
                    # Полное восстановление будет при доступе
                    event_id = data.get("event_id")
                    if event_id:
                        logger.debug("episodic_memory.loaded", event_id=event_id)
            except Exception as e:
                logger.warning("episodic_memory.load_failed", path=str(file_path), error=str(e))


# === Singleton ===

_episodic_memory: EpisodicMemory | None = None


def get_episodic_memory() -> EpisodicMemory:
    """Получить singleton эпизодической памяти."""
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory


def configure_episodic_memory(storage_path: Path | None = None) -> EpisodicMemory:
    """Сконфигурировать эпизодическую память."""
    global _episodic_memory
    _episodic_memory = EpisodicMemory(storage_path=storage_path)
    return _episodic_memory
