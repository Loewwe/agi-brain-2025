"""
ACL (Access Control List) for AGI-Brain.

Контроль доступа:
- Кто может выполнять какие инструменты
- Проверка уровней автономии
- Валидация планов
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
import structlog

logger = structlog.get_logger()


class Permission(str, Enum):
    """Типы разрешений."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    APPROVE = "approve"
    ADMIN = "admin"


@dataclass
class Client:
    """Клиент системы."""
    id: str
    name: str
    permissions: set[Permission]
    max_autonomy_level: int
    allowed_tools: set[str]  # "*" = all
    denied_tools: set[str]
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AccessRule:
    """Правило доступа."""
    id: str
    description: str
    client_pattern: str  # regex or exact
    tool_pattern: str    # regex or "*"
    permission: Permission
    autonomy_level: int
    enabled: bool = True


@dataclass
class ValidationResult:
    """Результат валидации."""
    allowed: bool
    reason: str
    client_id: str
    resource: str
    checked_at: datetime = field(default_factory=datetime.now)


class ACL:
    """
    Access Control List для AGI-Brain.
    
    Проверяет:
    - Права клиента на выполнение инструментов
    - Соответствие уровня автономии
    - Валидность планов
    """
    
    # Предустановленные клиенты
    BUILTIN_CLIENTS = {
        "owner": Client(
            id="owner",
            name="Owner",
            permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.APPROVE, Permission.ADMIN},
            max_autonomy_level=3,
            allowed_tools={"*"},
            denied_tools=set(),
        ),
        "trader": Client(
            id="trader",
            name="Trading Agent",
            permissions={Permission.READ, Permission.EXECUTE},
            max_autonomy_level=0,
            allowed_tools={"world.snapshot", "memory.search", "trading.*"},
            denied_tools=set(),
        ),
        "internal": Client(
            id="internal",
            name="Internal Service",
            permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE},
            max_autonomy_level=1,
            allowed_tools={"*"},
            denied_tools={"trading.execute_*"},
        ),
    }
    
    def __init__(self, config_path: Path | None = None):
        """
        Args:
            config_path: Путь к конфигурации ACL (опционально)
        """
        self._clients: dict[str, Client] = dict(self.BUILTIN_CLIENTS)
        self._rules: list[AccessRule] = []
        self._audit_log: list[ValidationResult] = []
        
        if config_path and config_path.exists():
            self._load_config(config_path)
        
        logger.info(
            "acl.initialized",
            clients=len(self._clients),
            rules=len(self._rules),
        )
    
    def check_permission(
        self,
        client_id: str,
        tool: str,
        autonomy_level: int = 0,
    ) -> ValidationResult:
        """
        Проверить право на выполнение инструмента.
        
        Args:
            client_id: ID клиента
            tool: Имя инструмента
            autonomy_level: Требуемый уровень автономии
            
        Returns:
            ValidationResult
        """
        client = self._clients.get(client_id)
        
        if not client:
            result = ValidationResult(
                allowed=False,
                reason=f"Unknown client: {client_id}",
                client_id=client_id,
                resource=tool,
            )
            self._audit_log.append(result)
            return result
        
        if not client.active:
            result = ValidationResult(
                allowed=False,
                reason="Client is disabled",
                client_id=client_id,
                resource=tool,
            )
            self._audit_log.append(result)
            return result
        
        # Проверка denied_tools
        if self._matches_pattern(tool, client.denied_tools):
            result = ValidationResult(
                allowed=False,
                reason=f"Tool is denied for client: {tool}",
                client_id=client_id,
                resource=tool,
            )
            self._audit_log.append(result)
            return result
        
        # Проверка allowed_tools
        if not self._matches_pattern(tool, client.allowed_tools):
            result = ValidationResult(
                allowed=False,
                reason=f"Tool not in allowed list: {tool}",
                client_id=client_id,
                resource=tool,
            )
            self._audit_log.append(result)
            return result
        
        # Проверка уровня автономии
        if autonomy_level > client.max_autonomy_level:
            result = ValidationResult(
                allowed=False,
                reason=f"Autonomy level {autonomy_level} exceeds max {client.max_autonomy_level}",
                client_id=client_id,
                resource=tool,
            )
            self._audit_log.append(result)
            return result
        
        # Проверка permission EXECUTE
        if Permission.EXECUTE not in client.permissions:
            result = ValidationResult(
                allowed=False,
                reason="Client lacks EXECUTE permission",
                client_id=client_id,
                resource=tool,
            )
            self._audit_log.append(result)
            return result
        
        result = ValidationResult(
            allowed=True,
            reason="Access granted",
            client_id=client_id,
            resource=tool,
        )
        self._audit_log.append(result)
        return result
    
    def validate_plan(
        self,
        plan_steps: list[dict[str, Any]],
        client_id: str,
    ) -> tuple[bool, list[str]]:
        """
        Валидировать весь план.
        
        Args:
            plan_steps: Шаги плана [{tool, level, ...}]
            client_id: ID клиента
            
        Returns:
            (valid, list of violations)
        """
        violations = []
        
        for step in plan_steps:
            tool = step.get("tool", "")
            level = step.get("level", 0)
            step_id = step.get("id", "unknown")
            
            result = self.check_permission(client_id, tool, level)
            if not result.allowed:
                violations.append(f"Step {step_id}: {result.reason}")
        
        return len(violations) == 0, violations
    
    def can_approve(self, client_id: str) -> bool:
        """Может ли клиент давать одобрения."""
        client = self._clients.get(client_id)
        return client is not None and Permission.APPROVE in client.permissions
    
    def is_admin(self, client_id: str) -> bool:
        """Является ли клиент администратором."""
        client = self._clients.get(client_id)
        return client is not None and Permission.ADMIN in client.permissions
    
    def get_client(self, client_id: str) -> Client | None:
        """Получить клиента."""
        return self._clients.get(client_id)
    
    def add_client(self, client: Client) -> None:
        """Добавить клиента."""
        self._clients[client.id] = client
        logger.info("acl.client_added", client_id=client.id)
    
    def get_audit_log(self, limit: int = 100) -> list[ValidationResult]:
        """Получить лог проверок."""
        return self._audit_log[-limit:]
    
    def _matches_pattern(self, value: str, patterns: set[str]) -> bool:
        """Проверить соответствие паттернам."""
        if "*" in patterns:
            return True
        
        for pattern in patterns:
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if value.startswith(prefix):
                    return True
            elif pattern == value:
                return True
        
        return False
    
    def _load_config(self, config_path: Path) -> None:
        """Загрузить конфигурацию из YAML."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Загрузка дополнительных клиентов
        for client_config in config.get("clients", []):
            client = Client(
                id=client_config["id"],
                name=client_config.get("name", client_config["id"]),
                permissions={Permission(p) for p in client_config.get("permissions", [])},
                max_autonomy_level=client_config.get("max_autonomy_level", 0),
                allowed_tools=set(client_config.get("allowed_tools", [])),
                denied_tools=set(client_config.get("denied_tools", [])),
                active=client_config.get("active", True),
            )
            self._clients[client.id] = client


# === Singleton ===

_acl: ACL | None = None


def get_acl() -> ACL:
    """Получить singleton ACL."""
    global _acl
    if _acl is None:
        _acl = ACL()
    return _acl


def configure_acl(config_path: Path | None = None) -> ACL:
    """Сконфигурировать ACL."""
    global _acl
    _acl = ACL(config_path=config_path)
    return _acl
