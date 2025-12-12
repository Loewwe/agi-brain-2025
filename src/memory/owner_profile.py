"""
Owner Profile Manager for AGI-Brain.

Загружает и управляет профилем Владельца.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
import structlog

from ..world.model import OwnerProfile

logger = structlog.get_logger()


class OwnerProfileManager:
    """
    Менеджер профиля Владельца.
    
    Загружает конфигурацию из YAML и предоставляет
    доступ к параметрам для проверки соответствия планов.
    """
    
    def __init__(self, config_path: Path | None = None):
        """
        Инициализация менеджера.
        
        Args:
            config_path: Путь к owner_profile.yaml
        """
        self.config_path = config_path
        self._profile: OwnerProfile | None = None
        self._raw_config: dict[str, Any] = {}
        
        if config_path and config_path.exists():
            self.load(config_path)
    
    def load(self, config_path: Path) -> None:
        """Загрузить профиль из файла."""
        self.config_path = config_path
        
        try:
            with open(config_path) as f:
                self._raw_config = yaml.safe_load(f)
            
            self._profile = self._parse_config(self._raw_config)
            
            logger.info(
                "owner_profile.loaded",
                path=str(config_path),
                owner_id=self._profile.id,
            )
        except Exception as e:
            logger.error("owner_profile.load_failed", path=str(config_path), error=str(e))
            raise
    
    def get(self) -> OwnerProfile:
        """Получить профиль Владельца."""
        if self._profile is None:
            return self._get_default_profile()
        return self._profile
    
    def check_risk_compliance(
        self,
        daily_loss_percent: float | None = None,
        position_size_percent: float | None = None,
        leverage: int | None = None,
        asset: str | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Проверить соответствие параметров риск-профилю.
        
        Returns:
            (compliant, violations) - статус и список нарушений
        """
        profile = self.get()
        violations: list[str] = []
        
        if daily_loss_percent is not None:
            if daily_loss_percent > profile.max_daily_loss_percent:
                violations.append(
                    f"Daily loss {daily_loss_percent}% exceeds limit {profile.max_daily_loss_percent}%"
                )
        
        if position_size_percent is not None:
            if position_size_percent > profile.max_position_size_percent:
                violations.append(
                    f"Position size {position_size_percent}% exceeds limit {profile.max_position_size_percent}%"
                )
        
        if leverage is not None:
            if leverage > profile.max_leverage:
                violations.append(
                    f"Leverage {leverage}x exceeds limit {profile.max_leverage}x"
                )
        
        if asset is not None:
            if asset.upper() in [a.upper() for a in profile.forbidden_assets]:
                violations.append(f"Asset {asset} is forbidden")
        
        return len(violations) == 0, violations
    
    def is_within_notification_hours(self, hour: int, timezone: str | None = None) -> bool:
        """Проверить, находится ли время в разрешённых часах для уведомлений."""
        profile = self.get()
        return profile.notification_hours_start <= hour <= profile.notification_hours_end
    
    def get_safety_priority(self) -> float:
        """Получить приоритет безопасности (0-1)."""
        return self.get().safety_priority
    
    def get_decision_style(self) -> str:
        """Получить стиль принятия решений."""
        return self.get().decision_style
    
    def get_raw_config(self) -> dict[str, Any]:
        """Получить сырой конфиг для расширенного доступа."""
        return self._raw_config
    
    def _parse_config(self, config: dict[str, Any]) -> OwnerProfile:
        """Парсинг конфигурации в OwnerProfile."""
        owner = config.get("owner", {})
        risk = config.get("risk", {})
        goals = config.get("goals", {})
        preferences = config.get("preferences", {})
        constraints = config.get("constraints", {})
        notification_hours = preferences.get("notification_hours", {})
        
        return OwnerProfile(
            id=owner.get("id", "owner_default"),
            name=owner.get("name", "Owner"),
            
            # Risk
            max_daily_loss_percent=risk.get("max_daily_loss_percent", 2.0),
            max_weekly_loss_percent=risk.get("max_weekly_loss_percent", 5.0),
            max_position_size_percent=risk.get("max_position_size_percent", 10.0),
            max_leverage=risk.get("max_leverage", 5),
            unsinkable_balance_usd=Decimal(str(risk.get("unsinkable_balance_usd", 10000))),
            
            # Goals
            target_daily_return_percent=goals.get("target_daily_return_percent", 0.5),
            target_monthly_return_percent=goals.get("target_monthly_return_percent", 15.0),
            safety_priority=goals.get("safety_priority", 0.7),
            
            # Preferences
            decision_style=preferences.get("decision_style", "analytical"),
            report_detail_level=preferences.get("report_detail_level", "detailed"),
            notification_hours_start=notification_hours.get("start", 9),
            notification_hours_end=notification_hours.get("end", 22),
            timezone=notification_hours.get("timezone", "UTC"),
            
            # Constraints
            forbidden_assets=constraints.get("forbidden_assets", []),
            max_concurrent_positions=constraints.get("max_concurrent_positions", 5),
        )
    
    def _get_default_profile(self) -> OwnerProfile:
        """Дефолтный профиль."""
        return OwnerProfile(
            id="owner_default",
            name="Owner",
            max_daily_loss_percent=2.0,
            max_weekly_loss_percent=5.0,
            max_position_size_percent=10.0,
            max_leverage=5,
            unsinkable_balance_usd=Decimal("10000"),
            target_daily_return_percent=0.5,
            target_monthly_return_percent=15.0,
            safety_priority=0.7,
            decision_style="analytical",
            report_detail_level="detailed",
            notification_hours_start=9,
            notification_hours_end=22,
            timezone="UTC",
            forbidden_assets=[],
            max_concurrent_positions=5,
        )


# === Singleton ===

_owner_profile_manager: OwnerProfileManager | None = None


def get_owner_profile_manager() -> OwnerProfileManager:
    """Получить singleton менеджера профиля."""
    global _owner_profile_manager
    if _owner_profile_manager is None:
        _owner_profile_manager = OwnerProfileManager()
    return _owner_profile_manager


def configure_owner_profile(config_path: Path) -> OwnerProfileManager:
    """Сконфигурировать менеджер профиля."""
    global _owner_profile_manager
    _owner_profile_manager = OwnerProfileManager(config_path=config_path)
    return _owner_profile_manager
