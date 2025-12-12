"""
Code Docs Fetcher for AGI-Brain.

Получение документации по программированию.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from .web_fetcher import WebFetcher, FetchResult, get_web_fetcher

logger = structlog.get_logger()


@dataclass
class DocPage:
    """Страница документации."""
    url: str
    title: str
    content: str
    language: str  # python, javascript, etc.
    framework: str | None  # fastapi, pydantic, etc.
    version: str | None
    cached: bool = False


class CodeDocsFetcher:
    """
    Fetcher для программной документации.
    
    Поддерживаемые источники:
    - Python docs
    - FastAPI docs
    - Pydantic docs
    - NumPy/Pandas docs
    - MDN (JavaScript)
    """
    
    KNOWN_DOCS = {
        "python": {
            "base_url": "https://docs.python.org/3/",
            "description": "Python Standard Library",
        },
        "fastapi": {
            "base_url": "https://fastapi.tiangolo.com/",
            "description": "FastAPI Framework",
        },
        "pydantic": {
            "base_url": "https://docs.pydantic.dev/latest/",
            "description": "Pydantic Data Validation",
        },
        "numpy": {
            "base_url": "https://numpy.org/doc/stable/",
            "description": "NumPy Scientific Computing",
        },
        "pandas": {
            "base_url": "https://pandas.pydata.org/docs/",
            "description": "Pandas Data Analysis",
        },
        "mdn": {
            "base_url": "https://developer.mozilla.org/en-US/docs/",
            "description": "MDN Web Docs",
        },
    }
    
    def __init__(self, web_fetcher: WebFetcher | None = None):
        """
        Args:
            web_fetcher: Web fetcher для запросов (или использовать singleton)
        """
        self.web_fetcher = web_fetcher or get_web_fetcher()
    
    async def fetch_doc(
        self,
        framework: str,
        path: str = "",
    ) -> DocPage | None:
        """
        Получить страницу документации.
        
        Args:
            framework: Название фреймворка (python, fastapi, etc.)
            path: Путь внутри документации
            
        Returns:
            DocPage или None при ошибке
        """
        framework_lower = framework.lower()
        
        if framework_lower not in self.KNOWN_DOCS:
            logger.warning("code_docs.unknown_framework", framework=framework)
            return None
        
        config = self.KNOWN_DOCS[framework_lower]
        url = config["base_url"] + path.lstrip("/")
        
        result = await self.web_fetcher.fetch(url)
        
        if not result.success:
            logger.warning(
                "code_docs.fetch_failed",
                framework=framework,
                url=url,
                error=result.error,
            )
            return None
        
        # Извлечение заголовка из HTML (упрощённо)
        title = self._extract_title(result.content) or path or framework
        
        return DocPage(
            url=url,
            title=title,
            content=self._clean_content(result.content),
            language=self._detect_language(framework_lower),
            framework=framework_lower,
            version=None,
            cached=result.cached,
        )
    
    async def search_docs(
        self,
        query: str,
        frameworks: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Поиск по документации.
        
        Примечание: В v1 возвращает только известные пути.
        Полнотекстовый поиск будет в будущих версиях.
        """
        results = []
        
        target_frameworks = frameworks or list(self.KNOWN_DOCS.keys())
        
        for fw in target_frameworks:
            if fw.lower() in self.KNOWN_DOCS:
                config = self.KNOWN_DOCS[fw.lower()]
                results.append({
                    "framework": fw,
                    "base_url": config["base_url"],
                    "description": config["description"],
                    "relevance": 1.0 if query.lower() in fw.lower() else 0.5,
                })
        
        return sorted(results, key=lambda x: x["relevance"], reverse=True)
    
    def get_available_frameworks(self) -> list[dict[str, str]]:
        """Получить список доступных источников."""
        return [
            {"name": name, "description": config["description"]}
            for name, config in self.KNOWN_DOCS.items()
        ]
    
    def _extract_title(self, html: str) -> str | None:
        """Извлечь заголовок из HTML."""
        # Простой regex для <title>
        import re
        match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _clean_content(self, html: str) -> str:
        """Очистить HTML контент."""
        # Простая очистка (в будущем использовать BeautifulSoup)
        import re
        # Удаление скриптов и стилей
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Удаление тегов
        text = re.sub(r"<[^>]+>", " ", text)
        # Нормализация пробелов
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _detect_language(self, framework: str) -> str:
        """Определить язык программирования."""
        if framework in ("python", "fastapi", "pydantic", "numpy", "pandas"):
            return "python"
        if framework == "mdn":
            return "javascript"
        return "unknown"


# === Singleton ===

_code_docs_fetcher: CodeDocsFetcher | None = None


def get_code_docs_fetcher() -> CodeDocsFetcher:
    """Получить singleton code docs fetcher."""
    global _code_docs_fetcher
    if _code_docs_fetcher is None:
        _code_docs_fetcher = CodeDocsFetcher()
    return _code_docs_fetcher
