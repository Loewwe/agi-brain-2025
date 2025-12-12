"""
Tools Registry for AGI-Brain.

Реестр всех доступных инструментов с:
- Метаданными (имя, описание, схемы)
- Уровнями риска
- ACL и permissions
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

import yaml
import structlog

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """Уровень риска инструмента."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolDefinition:
    """Определение инструмента."""
    name: str
    description: str
    category: str
    
    # Схемы
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    
    # Безопасность
    risk_level: RiskLevel
    autonomy_level: int  # 0, 1, 2, 3
    requires_approval: bool
    allowed_clients: set[str]
    
    # Метаданные
    version: str = "1.0.0"
    deprecated: bool = False
    tags: list[str] = field(default_factory=list)
    
    def can_execute(self, client_id: str, current_autonomy: int) -> tuple[bool, str]:
        """
        Проверить, можно ли выполнить инструмент.
        
        Args:
            client_id: ID клиента
            current_autonomy: Текущий уровень автономии
            
        Returns:
            (allowed, reason)
        """
        if client_id not in self.allowed_clients and "*" not in self.allowed_clients:
            return False, f"Client '{client_id}' not allowed"
        
        if self.autonomy_level > current_autonomy:
            return False, f"Requires autonomy level {self.autonomy_level}, current: {current_autonomy}"
        
        return True, "OK"


@dataclass
class RegisteredTool:
    """Зарегистрированный инструмент с реализацией."""
    definition: ToolDefinition
    handler: Callable[..., Awaitable[Any]]
    
    async def execute(self, **kwargs) -> Any:
        """Выполнить инструмент."""
        return await self.handler(**kwargs)


class ToolsRegistry:
    """
    Реестр инструментов AGI-Brain.
    
    Хранит определения и реализации всех доступных tools.
    """
    
    def __init__(self, config_path: Path | None = None):
        """
        Args:
            config_path: Путь к tools_registry.yaml
        """
        self.config_path = config_path
        self._tools: dict[str, RegisteredTool] = {}
        self._definitions: dict[str, ToolDefinition] = {}
        
        if config_path and config_path.exists():
            self._load_definitions(config_path)
        
        logger.info(
            "tools_registry.initialized",
            definitions_loaded=len(self._definitions),
        )
    
    def _load_definitions(self, config_path: Path) -> None:
        """Загрузить определения из YAML."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        for name, tool_config in config.get("tools", {}).items():
            try:
                definition = ToolDefinition(
                    name=name,
                    description=tool_config.get("description", ""),
                    category=tool_config.get("category", ""),
                    input_schema=tool_config.get("input_schema", {}),
                    output_schema=tool_config.get("output_schema", {}),
                    risk_level=RiskLevel(tool_config.get("risk_level", "low")),
                    autonomy_level=tool_config.get("autonomy_level", 0),
                    requires_approval=tool_config.get("requires_approval", False),
                    allowed_clients=set(tool_config.get("allowed_clients", ["owner"])),
                )
                self._definitions[name] = definition
            except Exception as e:
                logger.error("tools_registry.load_failed", tool=name, error=str(e))
    
    def register(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        definition: ToolDefinition | None = None,
    ) -> None:
        """
        Зарегистрировать инструмент.
        
        Args:
            name: Имя инструмента (e.g., "world.snapshot")
            handler: Async функция-обработчик
            definition: Определение (или из загруженных)
        """
        if definition is None:
            if name in self._definitions:
                definition = self._definitions[name]
            else:
                raise ValueError(f"No definition for tool: {name}")
        
        self._tools[name] = RegisteredTool(
            definition=definition,
            handler=handler,
        )
        
        logger.info(
            "tools_registry.tool_registered",
            name=name,
            risk_level=definition.risk_level.value,
        )
    
    def get(self, name: str) -> RegisteredTool | None:
        """Получить зарегистрированный инструмент."""
        return self._tools.get(name)
    
    def get_definition(self, name: str) -> ToolDefinition | None:
        """Получить определение инструмента."""
        if name in self._tools:
            return self._tools[name].definition
        return self._definitions.get(name)
    
    def list_all(self) -> list[str]:
        """Список всех зарегистрированных инструментов."""
        return list(self._tools.keys())
    
    def list_definitions(self) -> list[str]:
        """Список всех определений (включая незарегистрированные)."""
        return list(self._definitions.keys())
    
    def list_by_category(self, category: str) -> list[str]:
        """Список инструментов по категории."""
        return [
            name for name, tool in self._tools.items()
            if tool.definition.category == category
        ]
    
    def list_by_risk(self, max_risk: RiskLevel) -> list[str]:
        """Список инструментов до указанного уровня риска."""
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_idx = risk_order.index(max_risk)
        
        return [
            name for name, tool in self._tools.items()
            if risk_order.index(tool.definition.risk_level) <= max_idx
        ]
    
    def list_for_client(self, client_id: str) -> list[str]:
        """Список инструментов доступных клиенту."""
        return [
            name for name, tool in self._tools.items()
            if client_id in tool.definition.allowed_clients or "*" in tool.definition.allowed_clients
        ]
    
    def get_schema_for_llm(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Получить схемы инструментов для LLM (OpenAI function calling format).
        
        Args:
            names: Список имён (или все)
            
        Returns:
            Список function definitions для OpenAI API
        """
        target_names = names or list(self._tools.keys())
        
        functions = []
        for name in target_names:
            tool = self._tools.get(name)
            if not tool:
                continue
            
            func_def = {
                "type": "function",
                "function": {
                    "name": name.replace(".", "_"),  # OpenAI не любит точки
                    "description": tool.definition.description,
                    "parameters": tool.definition.input_schema,
                },
            }
            functions.append(func_def)
        
        return functions
    
    def validate_call(
        self,
        name: str,
        client_id: str,
        current_autonomy: int,
    ) -> tuple[bool, str]:
        """
        Валидировать вызов инструмента.
        
        Args:
            name: Имя инструмента
            client_id: ID клиента
            current_autonomy: Текущий уровень автономии
            
        Returns:
            (valid, reason)
        """
        tool = self._tools.get(name)
        if not tool:
            return False, f"Tool not found: {name}"
        
        return tool.definition.can_execute(client_id, current_autonomy)


# === Singleton ===

_registry: ToolsRegistry | None = None


def get_tools_registry() -> ToolsRegistry:
    """Получить singleton реестра."""
    global _registry
    if _registry is None:
        _registry = ToolsRegistry()
    return _registry


def configure_tools_registry(config_path: Path | None = None) -> ToolsRegistry:
    """Сконфигурировать реестр."""
    global _registry
    _registry = ToolsRegistry(config_path=config_path)
    return _registry
