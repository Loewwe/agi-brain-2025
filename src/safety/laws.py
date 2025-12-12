"""
Core Laws for AGI-Brain.

Высший закон и принципы работы системы:
1. Безопасность и интересы Владельца
2. Сохранность капитала и инфраструктуры
3. Эффективность и рост

Эти законы не могут быть изменены системой.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from ..memory.owner_profile import get_owner_profile_manager

logger = structlog.get_logger()


class LawPriority(int, Enum):
    """Приоритет закона (меньше = выше)."""
    SUPREME = 0      # Высший закон
    CRITICAL = 1     # Критические ограничения
    HIGH = 2         # Важные правила
    MEDIUM = 3       # Стандартные политики
    LOW = 4          # Рекомендации


@dataclass
class Law:
    """Закон системы."""
    id: str
    name: str
    description: str
    priority: LawPriority
    
    # Проверка
    check_function: str | None = None  # Имя функции проверки
    parameters: dict[str, Any] = field(default_factory=dict)
    
    # Метаданные
    immutable: bool = True  # Не может быть изменён
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LawViolation:
    """Нарушение закона."""
    law_id: str
    law_name: str
    priority: LawPriority
    description: str
    context: dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_critical(self) -> bool:
        return self.priority <= LawPriority.CRITICAL


class CoreLaws:
    """
    Ядро законов AGI-Brain.
    
    Определяет неизменяемые принципы работы системы.
    """
    
    # === ВЫСШИЙ ЗАКОН ===
    SUPREME_LAW = Law(
        id="supreme_law",
        name="Служение Владельцу",
        description="Максимизировать долгосрочную пользу и безопасность для Владельца в рамках законов, этики и заданных ограничений",
        priority=LawPriority.SUPREME,
        immutable=True,
    )
    
    # === КРИТИЧЕСКИЕ ЗАКОНЫ ===
    LAWS = [
        # Безопасность
        Law(
            id="owner_safety",
            name="Безопасность Владельца",
            description="Безопасность и интересы Владельца имеют высший приоритет",
            priority=LawPriority.CRITICAL,
            immutable=True,
        ),
        Law(
            id="capital_preservation",
            name="Сохранность капитала",
            description="Защита капитала и UNSINKABLE баланса",
            priority=LawPriority.CRITICAL,
            check_function="check_unsinkable_balance",
            immutable=True,
        ),
        Law(
            id="human_in_the_loop",
            name="Человек в контуре",
            description="Критические действия требуют одобрения Владельца",
            priority=LawPriority.CRITICAL,
            immutable=True,
        ),
        
        # Ограничения автономии
        Law(
            id="autonomy_limits",
            name="Ограничения автономии",
            description="Система не может расширять свои права без одобрения",
            priority=LawPriority.CRITICAL,
            immutable=True,
        ),
        Law(
            id="no_self_modification",
            name="Запрет самомодификации",
            description="Система не может изменять свои законы и ограничения",
            priority=LawPriority.CRITICAL,
            immutable=True,
        ),
        
        # Транспарентность
        Law(
            id="transparency",
            name="Прозрачность",
            description="Все действия логируются и объясняются",
            priority=LawPriority.HIGH,
            immutable=True,
        ),
        Law(
            id="audit_trail",
            name="Аудит",
            description="Полная история действий доступна для проверки",
            priority=LawPriority.HIGH,
            immutable=True,
        ),
        
        # Риск-менеджмент
        Law(
            id="risk_limits",
            name="Лимиты рисков",
            description="Соблюдение лимитов дневных убытков и плечей",
            priority=LawPriority.HIGH,
            check_function="check_risk_limits",
            immutable=True,
        ),
    ]
    
    def __init__(self):
        """Инициализация."""
        self._all_laws = {self.SUPREME_LAW.id: self.SUPREME_LAW}
        for law in self.LAWS:
            self._all_laws[law.id] = law
        
        logger.info("core_laws.initialized", laws_count=len(self._all_laws))
    
    def check_all(self, context: dict[str, Any]) -> list[LawViolation]:
        """
        Проверить соблюдение всех законов.
        
        Args:
            context: Контекст для проверки (план, действие и т.д.)
            
        Returns:
            Список нарушений (пустой если все OK)
        """
        violations = []
        
        for law in self._all_laws.values():
            if law.check_function:
                check_method = getattr(self, law.check_function, None)
                if check_method:
                    violation = check_method(law, context)
                    if violation:
                        violations.append(violation)
        
        return violations
    
    def check_action(
        self,
        action: str,
        autonomy_level: int,
        affects_capital: bool = False,
        affects_code: bool = False,
        affects_rights: bool = False,
    ) -> tuple[bool, str | None]:
        """
        Проверить допустимость действия.
        
        Returns:
            (allowed, reason_if_blocked)
        """
        # Human-in-the-loop для критических действий
        if affects_capital or affects_code or affects_rights:
            if autonomy_level < 2:
                return False, "This action requires human approval (affects capital/code/rights)"
        
        # Запрет самомодификации
        if affects_rights and action in ("modify_laws", "modify_autonomy", "modify_acl"):
            return False, "Self-modification of laws/autonomy/ACL is prohibited"
        
        return True, None
    
    def check_unsinkable_balance(
        self,
        law: Law,
        context: dict[str, Any],
    ) -> LawViolation | None:
        """Проверка UNSINKABLE баланса."""
        profile_manager = get_owner_profile_manager()
        profile = profile_manager.get()
        
        available_balance = context.get("available_balance")
        if available_balance is None:
            return None
        
        if available_balance < float(profile.unsinkable_balance_usd):
            return LawViolation(
                law_id=law.id,
                law_name=law.name,
                priority=law.priority,
                description=f"Balance ${available_balance} is below UNSINKABLE ${profile.unsinkable_balance_usd}",
                context=context,
            )
        
        return None
    
    def check_risk_limits(
        self,
        law: Law,
        context: dict[str, Any],
    ) -> LawViolation | None:
        """Проверка лимитов рисков."""
        profile_manager = get_owner_profile_manager()
        profile = profile_manager.get()
        
        daily_loss = context.get("daily_loss_percent")
        if daily_loss is not None:
            if daily_loss > profile.max_daily_loss_percent:
                return LawViolation(
                    law_id=law.id,
                    law_name=law.name,
                    priority=law.priority,
                    description=f"Daily loss {daily_loss}% exceeds limit {profile.max_daily_loss_percent}%",
                    context=context,
                )
        
        leverage = context.get("leverage")
        if leverage is not None:
            if leverage > profile.max_leverage:
                return LawViolation(
                    law_id=law.id,
                    law_name=law.name,
                    priority=law.priority,
                    description=f"Leverage {leverage}x exceeds limit {profile.max_leverage}x",
                    context=context,
                )
        
        return None
    
    def get_law(self, law_id: str) -> Law | None:
        """Получить закон по ID."""
        return self._all_laws.get(law_id)
    
    def get_all_laws(self) -> list[Law]:
        """Получить все законы."""
        return list(self._all_laws.values())
    
    def get_supreme_law(self) -> Law:
        """Получить высший закон."""
        return self.SUPREME_LAW
    
    def explain_laws(self) -> str:
        """Текстовое описание законов для LLM."""
        lines = ["# Законы AGI-Brain\n"]
        lines.append(f"## Высший закон\n{self.SUPREME_LAW.description}\n")
        lines.append("\n## Критические законы")
        
        for law in sorted(self._all_laws.values(), key=lambda l: l.priority):
            if law.id != "supreme_law":
                lines.append(f"\n### {law.name}")
                lines.append(f"{law.description}")
        
        return "\n".join(lines)


# === Singleton ===

_core_laws: CoreLaws | None = None


def get_core_laws() -> CoreLaws:
    """Получить singleton core laws."""
    global _core_laws
    if _core_laws is None:
        _core_laws = CoreLaws()
    return _core_laws
