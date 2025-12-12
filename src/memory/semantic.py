"""
Semantic Memory (RAG Layer) for AGI-Brain.

Индексирует и ищет:
- Внутренние документы
- Код и конфигурации
- Внешние учебные материалы

Использует:
- BM25 для keyword search
- pgvector для vector search
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class DocumentType(str, Enum):
    """Тип документа."""
    INTERNAL_DOC = "internal_doc"
    CODE = "code"
    CONFIG = "config"
    EXTERNAL_ARTICLE = "external_article"
    SCIENCE = "science"
    TRADING = "trading"
    PROGRAMMING = "programming"


class ContentLicense(str, Enum):
    """Лицензия контента."""
    PROPRIETARY = "proprietary"
    MIT = "mit"
    APACHE2 = "apache2"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    PUBLIC_DOMAIN = "public_domain"
    UNKNOWN = "unknown"


@dataclass
class Provenance:
    """Происхождение документа."""
    source_url: str | None = None
    source_type: str = "unknown"  # file, url, api
    retrieved_at: datetime = field(default_factory=datetime.now)
    license: ContentLicense = ContentLicense.UNKNOWN
    author: str | None = None
    version: str | None = None


@dataclass
class Document:
    """Документ в семантической памяти."""
    id: str
    title: str
    content: str
    doc_type: DocumentType
    provenance: Provenance
    
    # Метаданные
    tags: list[str] = field(default_factory=list)
    category: str = ""
    language: str = "en"
    
    # Индексация
    content_hash: str = ""
    embedding: list[float] | None = None
    indexed_at: datetime | None = None
    
    # Поиск
    chunk_index: int = 0  # Для разбитых документов
    parent_id: str | None = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"doc_{uuid4().hex[:12]}"
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """Результат поиска."""
    document: Document
    score: float
    match_type: str  # bm25, vector, hybrid
    highlights: list[str] = field(default_factory=list)


@dataclass
class SearchQuery:
    """Запрос поиска."""
    text: str
    doc_types: list[DocumentType] | None = None
    categories: list[str] | None = None
    tags: list[str] | None = None
    limit: int = 10
    use_vector: bool = True
    use_bm25: bool = True
    min_score: float = 0.0


class SemanticMemory:
    """
    Семантическая память AGI-Brain.
    
    Реализует RAG-слой с BM25 + vector search.
    
    В v1 использует in-memory хранилище.
    В будущем будет использовать PostgreSQL + pgvector.
    """
    
    def __init__(
        self,
        db_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Инициализация семантической памяти.
        
        Args:
            db_url: URL PostgreSQL (опционально, in-memory если None)
            embedding_model: Модель для эмбеддингов
        """
        self.db_url = db_url
        self.embedding_model = embedding_model
        
        # In-memory storage для v1
        self._documents: dict[str, Document] = {}
        self._index_by_type: dict[DocumentType, list[str]] = {}
        self._index_by_category: dict[str, list[str]] = {}
        
        # Заглушка для эмбеддингов
        self._embeddings_enabled = False
        
        logger.info(
            "semantic_memory.initialized",
            db_url=db_url,
            embedding_model=embedding_model,
            in_memory=db_url is None,
        )
    
    async def add_document(self, document: Document) -> str:
        """
        Добавить документ в память.
        
        Args:
            document: Документ для добавления
            
        Returns:
            ID документа
        """
        # Проверка дубликатов по хешу
        for existing in self._documents.values():
            if existing.content_hash == document.content_hash:
                logger.debug(
                    "semantic_memory.duplicate_skipped",
                    doc_id=document.id,
                    existing_id=existing.id,
                )
                return existing.id
        
        # Генерация эмбеддинга (заглушка)
        if self._embeddings_enabled:
            document.embedding = await self._generate_embedding(document.content)
        
        document.indexed_at = datetime.now()
        
        # Сохранение
        self._documents[document.id] = document
        self._index_document(document)
        
        logger.info(
            "semantic_memory.document_added",
            doc_id=document.id,
            title=document.title[:50],
            doc_type=document.doc_type.value,
        )
        
        return document.id
    
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Поиск документов.
        
        Args:
            query: Параметры поиска
            
        Returns:
            Список результатов с оценками релевантности
        """
        results: list[SearchResult] = []
        
        # Фильтрация по типам и категориям
        candidates = list(self._documents.values())
        
        if query.doc_types:
            candidates = [d for d in candidates if d.doc_type in query.doc_types]
        
        if query.categories:
            candidates = [d for d in candidates if d.category in query.categories]
        
        if query.tags:
            candidates = [d for d in candidates if any(t in d.tags for t in query.tags)]
        
        # Простой keyword search (BM25-like)
        if query.use_bm25:
            query_terms = query.text.lower().split()
            for doc in candidates:
                score = self._calculate_bm25_score(query_terms, doc)
                if score >= query.min_score:
                    results.append(SearchResult(
                        document=doc,
                        score=score,
                        match_type="bm25",
                        highlights=self._extract_highlights(query_terms, doc.content),
                    ))
        
        # Сортировка по релевантности
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:query.limit]
    
    async def get(self, doc_id: str) -> Document | None:
        """Получить документ по ID."""
        return self._documents.get(doc_id)
    
    async def delete(self, doc_id: str) -> bool:
        """Удалить документ."""
        if doc_id in self._documents:
            doc = self._documents.pop(doc_id)
            self._remove_from_indices(doc)
            logger.info("semantic_memory.document_deleted", doc_id=doc_id)
            return True
        return False
    
    async def get_by_type(self, doc_type: DocumentType, limit: int = 100) -> list[Document]:
        """Получить документы по типу."""
        doc_ids = self._index_by_type.get(doc_type, [])
        return [self._documents[id] for id in doc_ids[:limit] if id in self._documents]
    
    async def get_by_category(self, category: str, limit: int = 100) -> list[Document]:
        """Получить документы по категории."""
        doc_ids = self._index_by_category.get(category, [])
        return [self._documents[id] for id in doc_ids[:limit] if id in self._documents]
    
    def get_size(self) -> int:
        """Размер памяти (количество документов)."""
        return len(self._documents)
    
    def get_stats(self) -> dict[str, Any]:
        """Статистика по памяти."""
        type_counts = {
            doc_type.value: len(ids) 
            for doc_type, ids in self._index_by_type.items()
        }
        
        return {
            "total_documents": len(self._documents),
            "by_type": type_counts,
            "categories": list(self._index_by_category.keys()),
            "embeddings_enabled": self._embeddings_enabled,
        }
    
    def _index_document(self, doc: Document) -> None:
        """Индексация документа."""
        # По типу
        if doc.doc_type not in self._index_by_type:
            self._index_by_type[doc.doc_type] = []
        self._index_by_type[doc.doc_type].append(doc.id)
        
        # По категории
        if doc.category:
            if doc.category not in self._index_by_category:
                self._index_by_category[doc.category] = []
            self._index_by_category[doc.category].append(doc.id)
    
    def _remove_from_indices(self, doc: Document) -> None:
        """Удаление из индексов."""
        if doc.doc_type in self._index_by_type:
            self._index_by_type[doc.doc_type] = [
                id for id in self._index_by_type[doc.doc_type] if id != doc.id
            ]
        if doc.category in self._index_by_category:
            self._index_by_category[doc.category] = [
                id for id in self._index_by_category[doc.category] if id != doc.id
            ]
    
    def _calculate_bm25_score(self, query_terms: list[str], doc: Document) -> float:
        """Упрощённый BM25-like scoring."""
        content_lower = doc.content.lower()
        title_lower = doc.title.lower()
        
        score = 0.0
        for term in query_terms:
            # Title match (higher weight)
            if term in title_lower:
                score += 2.0
            # Content match
            if term in content_lower:
                score += 1.0
                # Bonus for multiple occurrences (with diminishing returns)
                count = content_lower.count(term)
                score += min(count * 0.1, 0.5)
        
        # Normalize by query length
        if query_terms:
            score = score / len(query_terms)
        
        return score
    
    def _extract_highlights(self, query_terms: list[str], content: str, context_size: int = 50) -> list[str]:
        """Извлечь фрагменты с совпадениями."""
        highlights = []
        content_lower = content.lower()
        
        for term in query_terms:
            idx = content_lower.find(term)
            if idx >= 0:
                start = max(0, idx - context_size)
                end = min(len(content), idx + len(term) + context_size)
                highlight = content[start:end]
                if start > 0:
                    highlight = "..." + highlight
                if end < len(content):
                    highlight = highlight + "..."
                highlights.append(highlight)
                
                if len(highlights) >= 3:
                    break
        
        return highlights
    
    async def _generate_embedding(self, text: str) -> list[float]:
        """Генерация эмбеддинга (заглушка для v1)."""
        # TODO: Интеграция с OpenAI Embeddings API
        return []


# === Singleton ===

_semantic_memory: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    """Получить singleton семантической памяти."""
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = SemanticMemory()
    return _semantic_memory


def configure_semantic_memory(
    db_url: str | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> SemanticMemory:
    """Сконфигурировать семантическую память."""
    global _semantic_memory
    _semantic_memory = SemanticMemory(db_url=db_url, embedding_model=embedding_model)
    return _semantic_memory
