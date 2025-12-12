"""
Web Fetcher for AGI-Brain.

Контролируемый доступ к интернету:
- Белый список доменов
- Rate limiting
- Логирование запросов
"""

import asyncio
import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml
import structlog

logger = structlog.get_logger()


@dataclass
class FetchResult:
    """Результат запроса."""
    url: str
    status_code: int
    content: str
    content_type: str
    fetch_time_ms: float
    cached: bool = False
    error: str | None = None
    
    @property
    def success(self) -> bool:
        return self.status_code == 200 and self.error is None


@dataclass
class RateLimitState:
    """Состояние rate limiter."""
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    minute_start: float = field(default_factory=time.time)
    hour_start: float = field(default_factory=time.time)


@dataclass
class DomainConfig:
    """Конфигурация домена."""
    domain: str
    description: str = ""
    category: str = ""
    subjects: list[str] = field(default_factory=list)
    enabled: bool = True


class WebFetcher:
    """
    Контролируемый web-fetcher для AGI-Brain.
    
    Особенности:
    - Белый список доменов
    - Rate limiting (per-minute, per-hour)
    - Кеширование ответов
    - Полное логирование
    """
    
    def __init__(
        self,
        config_path: Path | None = None,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Args:
            config_path: Путь к learning_sources.yaml
            cache_ttl_seconds: TTL кеша в секундах
        """
        self.config_path = config_path
        self.cache_ttl = cache_ttl_seconds
        
        # Настройки по умолчанию
        self.rate_limit_per_minute = 30
        self.rate_limit_per_hour = 500
        self.request_timeout = 30
        self.user_agent = "AGI-Brain/1.0 (Learning Agent)"
        
        # Белые/чёрные списки
        self._whitelist: dict[str, DomainConfig] = {}
        self._blacklist_patterns: list[re.Pattern] = []
        
        # Rate limiting
        self._rate_limits: dict[str, RateLimitState] = defaultdict(RateLimitState)
        
        # Кеш
        self._cache: dict[str, tuple[FetchResult, float]] = {}
        
        # HTTP клиент
        self._client: httpx.AsyncClient | None = None
        
        # Лог запросов
        self._request_log: list[dict[str, Any]] = []
        
        # Загрузка конфигурации
        if config_path and config_path.exists():
            self._load_config(config_path)
        
        logger.info(
            "web_fetcher.initialized",
            whitelist_domains=len(self._whitelist),
            rate_limit_minute=self.rate_limit_per_minute,
        )
    
    def _load_config(self, config_path: Path) -> None:
        """Загрузить конфигурацию из YAML."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Глобальные настройки
        global_config = config.get("global", {})
        self.rate_limit_per_minute = global_config.get("rate_limit_per_minute", 30)
        self.rate_limit_per_hour = global_config.get("rate_limit_per_hour", 500)
        self.request_timeout = global_config.get("request_timeout_seconds", 30)
        self.user_agent = global_config.get("user_agent", self.user_agent)
        
        # Категории доменов
        for category_name, category_config in config.get("categories", {}).items():
            if not category_config.get("enabled", True):
                continue
            
            for domain_config in category_config.get("domains", []):
                domain = domain_config.get("domain")
                if domain:
                    self._whitelist[domain] = DomainConfig(
                        domain=domain,
                        description=domain_config.get("description", ""),
                        category=category_name,
                        subjects=domain_config.get("subjects", []),
                    )
        
        # Blacklist
        for pattern in config.get("blacklist", []):
            try:
                regex = pattern.replace("*", ".*")
                self._blacklist_patterns.append(re.compile(regex))
            except re.error:
                logger.warning("web_fetcher.invalid_blacklist_pattern", pattern=pattern)
    
    async def fetch(self, url: str, use_cache: bool = True) -> FetchResult:
        """
        Получить контент по URL.
        
        Args:
            url: URL для запроса
            use_cache: Использовать кеш
            
        Returns:
            FetchResult с контентом или ошибкой
        """
        start_time = time.time()
        
        # Проверка URL
        validation_error = self._validate_url(url)
        if validation_error:
            return FetchResult(
                url=url,
                status_code=403,
                content="",
                content_type="",
                fetch_time_ms=0,
                error=validation_error,
            )
        
        # Проверка кеша
        if use_cache:
            cached = self._get_from_cache(url)
            if cached:
                logger.debug("web_fetcher.cache_hit", url=url)
                return cached
        
        # Проверка rate limit
        domain = urlparse(url).netloc
        if not self._check_rate_limit(domain):
            return FetchResult(
                url=url,
                status_code=429,
                content="",
                content_type="",
                fetch_time_ms=0,
                error="Rate limit exceeded",
            )
        
        # Выполнение запроса
        try:
            client = await self._get_client()
            response = await client.get(url)
            
            fetch_time_ms = (time.time() - start_time) * 1000
            
            result = FetchResult(
                url=url,
                status_code=response.status_code,
                content=response.text,
                content_type=response.headers.get("content-type", ""),
                fetch_time_ms=fetch_time_ms,
            )
            
            # Сохранение в кеш
            if response.status_code == 200 and use_cache:
                self._save_to_cache(url, result)
            
            # Логирование
            self._log_request(url, result)
            
            logger.info(
                "web_fetcher.request_completed",
                url=url,
                status=response.status_code,
                time_ms=round(fetch_time_ms, 2),
            )
            
            return result
            
        except httpx.TimeoutException:
            return FetchResult(
                url=url,
                status_code=504,
                content="",
                content_type="",
                fetch_time_ms=(time.time() - start_time) * 1000,
                error="Request timeout",
            )
        except Exception as e:
            logger.error("web_fetcher.request_failed", url=url, error=str(e))
            return FetchResult(
                url=url,
                status_code=500,
                content="",
                content_type="",
                fetch_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def is_allowed(self, url: str) -> tuple[bool, str]:
        """
        Проверить, разрешён ли URL.
        
        Returns:
            (allowed, reason)
        """
        validation = self._validate_url(url)
        return validation is None, validation or "Allowed"
    
    def get_whitelist(self) -> list[DomainConfig]:
        """Получить белый список доменов."""
        return list(self._whitelist.values())
    
    def get_request_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Получить лог запросов."""
        return self._request_log[-limit:]
    
    def get_stats(self) -> dict[str, Any]:
        """Статистика использования."""
        now = time.time()
        
        total_requests = len(self._request_log)
        successful = sum(1 for r in self._request_log if r.get("status") == 200)
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "cache_size": len(self._cache),
            "whitelist_domains": len(self._whitelist),
            "rate_limits": {
                domain: {
                    "minute": state.requests_this_minute,
                    "hour": state.requests_this_hour,
                }
                for domain, state in self._rate_limits.items()
            },
        }
    
    async def close(self) -> None:
        """Закрыть HTTP клиент."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Получить HTTP клиент."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.request_timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self._client
    
    def _validate_url(self, url: str) -> str | None:
        """
        Валидация URL.
        
        Returns:
            Сообщение об ошибке или None если OK
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return "Invalid URL format"
        
        # Только HTTP/HTTPS
        if parsed.scheme not in ("http", "https"):
            return f"Unsupported scheme: {parsed.scheme}"
        
        domain = parsed.netloc.lower()
        
        # Проверка blacklist
        for pattern in self._blacklist_patterns:
            if pattern.match(domain):
                return f"Domain is blacklisted: {domain}"
        
        # Проверка whitelist
        if not self._is_whitelisted(domain):
            return f"Domain not in whitelist: {domain}"
        
        return None
    
    def _is_whitelisted(self, domain: str) -> bool:
        """Проверить, есть ли домен в белом списке."""
        # Точное совпадение
        if domain in self._whitelist:
            return True
        
        # Проверка поддоменов
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self._whitelist:
                return True
        
        return False
    
    def _check_rate_limit(self, domain: str) -> bool:
        """Проверить и обновить rate limit."""
        now = time.time()
        state = self._rate_limits[domain]
        
        # Сброс минутного счётчика
        if now - state.minute_start >= 60:
            state.requests_this_minute = 0
            state.minute_start = now
        
        # Сброс часового счётчика
        if now - state.hour_start >= 3600:
            state.requests_this_hour = 0
            state.hour_start = now
        
        # Проверка лимитов
        if state.requests_this_minute >= self.rate_limit_per_minute:
            logger.warning("web_fetcher.rate_limit_minute", domain=domain)
            return False
        
        if state.requests_this_hour >= self.rate_limit_per_hour:
            logger.warning("web_fetcher.rate_limit_hour", domain=domain)
            return False
        
        # Обновление счётчиков
        state.requests_this_minute += 1
        state.requests_this_hour += 1
        
        return True
    
    def _get_from_cache(self, url: str) -> FetchResult | None:
        """Получить из кеша."""
        key = hashlib.md5(url.encode()).hexdigest()
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                result.cached = True
                return result
            else:
                del self._cache[key]
        return None
    
    def _save_to_cache(self, url: str, result: FetchResult) -> None:
        """Сохранить в кеш."""
        key = hashlib.md5(url.encode()).hexdigest()
        self._cache[key] = (result, time.time())
    
    def _log_request(self, url: str, result: FetchResult) -> None:
        """Логирование запроса."""
        self._request_log.append({
            "url": url,
            "domain": urlparse(url).netloc,
            "status": result.status_code,
            "time_ms": result.fetch_time_ms,
            "cached": result.cached,
            "error": result.error,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Ограничение размера лога
        if len(self._request_log) > 1000:
            self._request_log = self._request_log[-500:]


# === Singleton ===

_web_fetcher: WebFetcher | None = None


def get_web_fetcher() -> WebFetcher:
    """Получить singleton web fetcher."""
    global _web_fetcher
    if _web_fetcher is None:
        _web_fetcher = WebFetcher()
    return _web_fetcher


def configure_web_fetcher(config_path: Path | None = None) -> WebFetcher:
    """Сконфигурировать web fetcher."""
    global _web_fetcher
    _web_fetcher = WebFetcher(config_path=config_path)
    return _web_fetcher
