"""
Trading Adapter for AGI-Brain.

Адаптер для чтения данных из трейдинг-агента.

v1:
- Stub/fixture реализация для разработки
- API-адаптер для подключения к живому agi_trader_stage6
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
import random

import httpx
import structlog

from ..world.model import (
    AgentStatus,
    OrderType,
    PnLMetrics,
    Position,
    PositionSide,
    RiskConfig,
    Strategy,
    Trade,
    TradingAgent,
    TradingAdapterProtocol,
)

logger = structlog.get_logger()


class AdapterMode(str, Enum):
    """Режим работы адаптера."""
    STUB = "stub"      # Заглушка с фиктивными данными
    API = "api"        # Подключение к реальному API
    STAGE6 = "stage6"  # Подключение к Stage 6 боту


@dataclass
class RiskAnalysis:
    """Результат анализа рисков."""
    risk_score: float  # 0-100
    violations: list[dict[str, Any]]
    recommendations: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


class TradingAdapterBase(ABC):
    """Базовый класс для trading adapter."""
    
    @abstractmethod
    async def fetch_agent_state(self) -> TradingAgent | None:
        """Получить текущее состояние агента."""
        ...
    
    @abstractmethod
    async def fetch_history(self, days: int = 7, symbol: str | None = None) -> list[Trade]:
        """Получить историю сделок."""
        ...
    
    @abstractmethod
    async def fetch_positions(self) -> list[Position]:
        """Получить открытые позиции."""
        ...
    
    @abstractmethod
    async def fetch_config(self) -> RiskConfig:
        """Получить конфигурацию риск-менеджмента."""
        ...
    
    @abstractmethod
    async def analyze_risk(self, include_recommendations: bool = True) -> RiskAnalysis:
        """Анализировать текущие риски (read-only)."""
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Проверка доступности трейдинг-агента."""
        ...


class StubTradingAdapter(TradingAdapterBase):
    """
    Stub-реализация для разработки и тестирования.
    
    Генерирует реалистичные фиктивные данные.
    """
    
    def __init__(self, scenario: str = "normal"):
        """
        Args:
            scenario: Сценарий симуляции
                - "normal": Стандартные данные
                - "profitable": Положительный PnL
                - "losing": Отрицательный PnL
                - "risky": Высокие риски
        """
        self.scenario = scenario
        logger.info("trading_adapter.stub_initialized", scenario=scenario)
    
    async def fetch_agent_state(self) -> TradingAgent | None:
        """Сгенерировать состояние агента."""
        positions = await self.fetch_positions()
        config = await self.fetch_config()
        pnl = self._generate_pnl()
        strategies = self._generate_strategies()
        
        return TradingAgent(
            id="trader_stage6",
            name="AGI Trader Stage 6",
            status=AgentStatus.RUNNING,
            exchange="bybit",
            strategies=strategies,
            current_positions=positions,
            pnl=pnl,
            risk_config=config,
            last_trade_at=datetime.now() - timedelta(minutes=random.randint(5, 120)),
            uptime_hours=random.uniform(24, 168),
        )
    
    async def fetch_history(self, days: int = 7, symbol: str | None = None) -> list[Trade]:
        """Сгенерировать историю сделок."""
        trades: list[Trade] = []
        now = datetime.now()
        
        for i in range(min(days * 5, 50)):  # ~5 сделок в день
            closed_at = now - timedelta(
                days=random.randint(0, days - 1),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            opened_at = closed_at - timedelta(minutes=random.randint(5, 480))
            
            side = random.choice([PositionSide.LONG, PositionSide.SHORT])
            entry_price = Decimal(str(random.uniform(30000, 45000)))
            
            # Generate exit based on scenario
            if self.scenario == "profitable":
                change = random.uniform(0.001, 0.02)
            elif self.scenario == "losing":
                change = random.uniform(-0.02, 0.001)
            else:
                change = random.uniform(-0.015, 0.015)
            
            if side == PositionSide.SHORT:
                change = -change
            
            exit_price = entry_price * Decimal(str(1 + change))
            size = Decimal(str(random.uniform(0.01, 0.5)))
            pnl = (exit_price - entry_price) * size if side == PositionSide.LONG else (entry_price - exit_price) * size
            
            trades.append(Trade(
                id=f"trade_{i}",
                symbol=symbol or random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
                side=side,
                entry_price=entry_price.quantize(Decimal("0.01")),
                exit_price=exit_price.quantize(Decimal("0.01")),
                size=size.quantize(Decimal("0.001")),
                realized_pnl=pnl.quantize(Decimal("0.01")),
                realized_pnl_percent=float(change) * 100,
                fees=Decimal(str(random.uniform(0.5, 5))).quantize(Decimal("0.01")),
                opened_at=opened_at,
                closed_at=closed_at,
                reason=random.choice(["strategy", "take_profit", "stop_loss"]),
            ))
        
        trades.sort(key=lambda t: t.closed_at, reverse=True)
        return trades
    
    async def fetch_positions(self) -> list[Position]:
        """Сгенерировать позиции."""
        if self.scenario == "risky":
            num_positions = random.randint(5, 8)
        else:
            num_positions = random.randint(1, 3)
        
        positions: list[Position] = []
        for i in range(num_positions):
            symbol = random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            side = random.choice([PositionSide.LONG, PositionSide.SHORT])
            entry_price = Decimal(str(random.uniform(30000, 45000)))
            leverage = random.randint(3, 10) if self.scenario == "risky" else random.randint(2, 5)
            
            # Current price relative to entry
            if self.scenario == "profitable":
                change = random.uniform(0.001, 0.03)
                if side == PositionSide.SHORT:
                    change = -change
            elif self.scenario == "losing":
                change = random.uniform(-0.03, -0.001)
                if side == PositionSide.SHORT:
                    change = -change
            else:
                change = random.uniform(-0.02, 0.02)
            
            current_price = entry_price * Decimal(str(1 + change))
            size = Decimal(str(random.uniform(0.05, 0.3)))
            
            if side == PositionSide.LONG:
                unrealized_pnl = (current_price - entry_price) * size
                liq_price = entry_price * Decimal(str(1 - 1 / leverage))
            else:
                unrealized_pnl = (entry_price - current_price) * size
                liq_price = entry_price * Decimal(str(1 + 1 / leverage))
            
            positions.append(Position(
                symbol=symbol,
                side=side,
                size=size.quantize(Decimal("0.001")),
                entry_price=entry_price.quantize(Decimal("0.01")),
                current_price=current_price.quantize(Decimal("0.01")),
                leverage=leverage,
                unrealized_pnl=unrealized_pnl.quantize(Decimal("0.01")),
                unrealized_pnl_percent=change * 100,
                liquidation_price=liq_price.quantize(Decimal("0.01")),
                stop_loss=entry_price * Decimal("0.97") if random.random() > 0.3 else None,
                take_profit=entry_price * Decimal("1.05") if random.random() > 0.3 else None,
                opened_at=datetime.now() - timedelta(hours=random.randint(1, 72)),
            ))
        
        return positions
    
    async def fetch_config(self) -> RiskConfig:
        """Вернуть конфигурацию."""
        return RiskConfig(
            max_position_size_usd=Decimal("5000"),
            max_leverage=5,
            max_daily_loss_usd=Decimal("500"),
            max_daily_loss_percent=2.0,
            unsinkable_balance_usd=Decimal("10000"),
            max_concurrent_positions=5,
            stop_loss_percent=3.0,
            take_profit_percent=5.0,
        )
    
    async def analyze_risk(self, include_recommendations: bool = True) -> RiskAnalysis:
        """Анализ рисков."""
        positions = await self.fetch_positions()
        config = await self.fetch_config()
        
        violations = []
        recommendations = []
        risk_score = 0.0
        
        # Check positions
        if len(positions) > config.max_concurrent_positions:
            violations.append({
                "rule": "max_positions",
                "current": len(positions),
                "limit": config.max_concurrent_positions,
            })
            risk_score += 20
        
        for pos in positions:
            if pos.leverage > config.max_leverage:
                violations.append({
                    "rule": "max_leverage",
                    "symbol": pos.symbol,
                    "current": pos.leverage,
                    "limit": config.max_leverage,
                })
                risk_score += 15
            
            dist = pos.distance_to_liquidation_percent
            if dist and dist < 10:
                violations.append({
                    "rule": "liquidation_warning",
                    "symbol": pos.symbol,
                    "distance": dist,
                })
                risk_score += 30
        
        if include_recommendations:
            if risk_score > 50:
                recommendations.append("Consider reducing position sizes")
            if any(p.stop_loss is None for p in positions):
                recommendations.append("Add stop losses to unprotected positions")
        
        return RiskAnalysis(
            risk_score=min(100, risk_score),
            violations=violations,
            recommendations=recommendations,
        )
    
    async def health_check(self) -> bool:
        """Всегда здоров в stub режиме."""
        return True
    
    def _generate_pnl(self) -> PnLMetrics:
        """Генерация PnL метрик."""
        base_balance = Decimal("25000")
        
        if self.scenario == "profitable":
            today_pct = random.uniform(0.5, 2.0)
            week_pct = random.uniform(3.0, 8.0)
            month_pct = random.uniform(10.0, 25.0)
        elif self.scenario == "losing":
            today_pct = random.uniform(-2.0, -0.5)
            week_pct = random.uniform(-5.0, -1.0)
            month_pct = random.uniform(-10.0, -3.0)
        else:
            today_pct = random.uniform(-1.0, 1.0)
            week_pct = random.uniform(-3.0, 5.0)
            month_pct = random.uniform(-5.0, 15.0)
        
        return PnLMetrics(
            total_balance_usd=base_balance,
            available_balance_usd=base_balance * Decimal("0.7"),
            realized_pnl_today=base_balance * Decimal(str(today_pct / 100)),
            realized_pnl_week=base_balance * Decimal(str(week_pct / 100)),
            realized_pnl_month=base_balance * Decimal(str(month_pct / 100)),
            unrealized_pnl=Decimal(str(random.uniform(-200, 300))),
            pnl_today_percent=today_pct,
            pnl_week_percent=week_pct,
            pnl_month_percent=month_pct,
            max_drawdown_today=abs(min(0, today_pct)) + random.uniform(0, 0.5),
            max_drawdown_week=abs(min(0, week_pct)) + random.uniform(0, 1),
            max_drawdown_month=abs(min(0, month_pct)) + random.uniform(0, 2),
            win_rate=random.uniform(0.45, 0.65),
            profit_factor=random.uniform(1.1, 1.8),
            sharpe_ratio=random.uniform(0.5, 2.0),
        )
    
    def _generate_strategies(self) -> list[Strategy]:
        """Генерация стратегий."""
        return [
            Strategy(
                id="trend_following",
                name="Trend Following",
                enabled=True,
                symbols=["BTCUSDT", "ETHUSDT"],
                parameters={"timeframe": "4h", "lookback": 20},
                performance_7d=random.uniform(-2, 5),
                win_rate=random.uniform(0.4, 0.6),
                trades_count=random.randint(10, 50),
            ),
            Strategy(
                id="mean_reversion",
                name="Mean Reversion",
                enabled=True,
                symbols=["BTCUSDT"],
                parameters={"timeframe": "1h", "std_dev": 2.0},
                performance_7d=random.uniform(-3, 4),
                win_rate=random.uniform(0.5, 0.65),
                trades_count=random.randint(20, 80),
            ),
        ]


class APITradingAdapter(TradingAdapterBase):
    """
    API-адаптер для подключения к реальному agi_trader_stage6.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Args:
            base_url: URL трейдинг-агента (e.g., http://localhost:8000)
            api_key: API ключ для авторизации
            timeout: Таймаут запросов
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        
        logger.info("trading_adapter.api_initialized", base_url=base_url)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Получить HTTP клиент."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Закрыть клиент."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def fetch_agent_state(self) -> TradingAgent | None:
        """Получить состояние агента через API."""
        try:
            client = await self._get_client()
            response = await client.get("/api/v1/agent/state")
            response.raise_for_status()
            data = response.json()
            return self._parse_agent_state(data)
        except Exception as e:
            logger.error("trading_adapter.fetch_state_failed", error=str(e))
            return None
    
    async def fetch_history(self, days: int = 7, symbol: str | None = None) -> list[Trade]:
        """Получить историю сделок."""
        try:
            client = await self._get_client()
            params = {"days": days}
            if symbol:
                params["symbol"] = symbol
            response = await client.get("/api/v1/trades/history", params=params)
            response.raise_for_status()
            data = response.json()
            return [self._parse_trade(t) for t in data.get("trades", [])]
        except Exception as e:
            logger.error("trading_adapter.fetch_history_failed", error=str(e))
            return []
    
    async def fetch_positions(self) -> list[Position]:
        """Получить позиции."""
        try:
            client = await self._get_client()
            response = await client.get("/api/v1/positions")
            response.raise_for_status()
            data = response.json()
            return [self._parse_position(p) for p in data.get("positions", [])]
        except Exception as e:
            logger.error("trading_adapter.fetch_positions_failed", error=str(e))
            return []
    
    async def fetch_config(self) -> RiskConfig:
        """Получить конфигурацию."""
        try:
            client = await self._get_client()
            response = await client.get("/api/v1/config/risk")
            response.raise_for_status()
            data = response.json()
            return self._parse_risk_config(data)
        except Exception as e:
            logger.error("trading_adapter.fetch_config_failed", error=str(e))
            # Return default config
            return RiskConfig(
                max_position_size_usd=Decimal("5000"),
                max_leverage=5,
                max_daily_loss_usd=Decimal("500"),
                max_daily_loss_percent=2.0,
                unsinkable_balance_usd=Decimal("10000"),
                max_concurrent_positions=5,
                stop_loss_percent=3.0,
                take_profit_percent=5.0,
            )
    
    async def analyze_risk(self, include_recommendations: bool = True) -> RiskAnalysis:
        """Анализ рисков."""
        try:
            client = await self._get_client()
            response = await client.get(
                "/api/v1/risk/analysis",
                params={"recommendations": include_recommendations},
            )
            response.raise_for_status()
            data = response.json()
            return RiskAnalysis(
                risk_score=data.get("risk_score", 0),
                violations=data.get("violations", []),
                recommendations=data.get("recommendations", []),
            )
        except Exception as e:
            logger.error("trading_adapter.analyze_risk_failed", error=str(e))
            return RiskAnalysis(risk_score=0, violations=[], recommendations=[])
    
    async def health_check(self) -> bool:
        """Проверка доступности."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def _parse_agent_state(self, data: dict) -> TradingAgent:
        """Парсинг состояния агента."""
        # TODO: Implement actual parsing based on API response format
        raise NotImplementedError("API response parsing not implemented")
    
    def _parse_trade(self, data: dict) -> Trade:
        """Парсинг сделки."""
        raise NotImplementedError("Trade parsing not implemented")
    
    def _parse_position(self, data: dict) -> Position:
        """Парсинг позиции."""
        raise NotImplementedError("Position parsing not implemented")
    
    def _parse_risk_config(self, data: dict) -> RiskConfig:
        """Парсинг конфигурации."""
        raise NotImplementedError("RiskConfig parsing not implemented")


class FixtureTradingAdapter(TradingAdapterBase):
    """
    Адаптер, загружающий данные из JSON-фикстур.
    """
    
    def __init__(self, trades_path: str | None = None, positions_path: str | None = None):
        self.trades_path = trades_path
        self.positions_path = positions_path
        self._trades: list[Trade] = []
        self._positions: list[Position] = []
        
        if trades_path:
            self._load_trades(trades_path)
        if positions_path:
            self._load_positions(positions_path)
            
    def _load_trades(self, path: str) -> None:
        import json
        from pathlib import Path
        with open(Path(path)) as f:
            data = json.load(f)
            self._trades = [Trade(**t) for t in data]
            
    def _load_positions(self, path: str) -> None:
        import json
        from pathlib import Path
        with open(Path(path)) as f:
            data = json.load(f)
            self._positions = [Position(**p) for p in data]

    async def fetch_agent_state(self) -> TradingAgent | None:
        # Minimal implementation for testing
        return TradingAgent(
            id="fixture_agent",
            name="Fixture Agent",
            status=AgentStatus.RUNNING,
            exchange="fixture",
            strategies=[],
            current_positions=self._positions,
            pnl=PnLMetrics(
                total_balance_usd=Decimal("10000"),
                available_balance_usd=Decimal("5000"),
                realized_pnl_today=Decimal("0"),
                realized_pnl_week=Decimal("0"),
                realized_pnl_month=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                pnl_today_percent=0.0,
                pnl_week_percent=0.0,
                pnl_month_percent=0.0,
                max_drawdown_today=0.0,
                max_drawdown_week=0.0,
                max_drawdown_month=0.0,
                win_rate=0.5,
                profit_factor=1.0,
            ),
            risk_config=await self.fetch_config(),
        )

    async def fetch_history(self, days: int = 7, symbol: str | None = None) -> list[Trade]:
        return self._trades

    async def fetch_positions(self) -> list[Position]:
        return self._positions

    async def fetch_config(self) -> RiskConfig:
        return RiskConfig(
            max_position_size_usd=Decimal("5000"),
            max_leverage=5,
            max_daily_loss_usd=Decimal("500"),
            max_daily_loss_percent=2.0,
            unsinkable_balance_usd=Decimal("10000"),
            max_concurrent_positions=5,
            stop_loss_percent=3.0,
            take_profit_percent=5.0,
        )

    async def analyze_risk(self, include_recommendations: bool = True) -> RiskAnalysis:
        positions = await self.fetch_positions()
        config = await self.fetch_config()
        
        violations = []
        recommendations = []
        risk_score = 0.0
        
        # Check positions
        if len(positions) > config.max_concurrent_positions:
            violations.append({
                "rule": "max_positions",
                "current": len(positions),
                "limit": config.max_concurrent_positions,
            })
            risk_score += 20
        
        for pos in positions:
            if pos.leverage > config.max_leverage:
                violations.append({
                    "rule": "max_leverage",
                    "symbol": pos.symbol,
                    "current": pos.leverage,
                    "limit": config.max_leverage,
                })
                risk_score += 15
            
            dist = pos.distance_to_liquidation_percent
            if dist and dist < 10:
                violations.append({
                    "rule": "liquidation_warning",
                    "symbol": pos.symbol,
                    "distance": dist,
                })
                risk_score += 30
        
        if include_recommendations:
            if risk_score > 50:
                recommendations.append("Consider reducing position sizes")
            if any(p.stop_loss is None for p in positions):
                recommendations.append("Add stop losses to unprotected positions")
        
        return RiskAnalysis(
            risk_score=min(100, risk_score),
            violations=violations,
            recommendations=recommendations,
        )

    async def health_check(self) -> bool:
        return True


# === Singleton ===

_trading_adapter: TradingAdapterProtocol | None = None


def get_trading_adapter() -> TradingAdapterProtocol:
    """Получить singleton адаптера."""
    global _trading_adapter
    if _trading_adapter is None:
        _trading_adapter = StubTradingAdapter()
    return _trading_adapter


def configure_trading_adapter(
    mode: AdapterMode = AdapterMode.STUB,
    scenario: str = "normal",
    api_url: str | None = None,
    api_key: str | None = None,
) -> TradingAdapterProtocol:
    """
    Фабрика адаптеров.
    
    Args:
        mode: Режим работы (stub, api, stage6)
        scenario: Сценарий для стаба
        api_url: URL для API/Stage6
        api_key: Ключ для API/Stage6
    """
    global _trading_adapter
    
    if mode == AdapterMode.STUB:
        _trading_adapter = StubTradingAdapter(scenario=scenario)
        logger.info("trading_adapter.stub_initialized", scenario=scenario)
        
    elif mode == AdapterMode.STAGE6:
        from .stage6_adapter import Stage6TradingAdapter
        # Allow initialization without explicit args, adapter will check env
        _trading_adapter = Stage6TradingAdapter(api_url=api_url, api_key=api_key)
        logger.info("trading_adapter.stage6_initialized", url=api_url)
            
    elif mode == AdapterMode.API:
        # Generic API adapter (legacy or external)
        if not api_url:
             logger.warning("trading_adapter.api_missing_url", msg="Using Stub instead")
             _trading_adapter = StubTradingAdapter(scenario=scenario)
        else:
             _trading_adapter = APITradingAdapter(base_url=api_url, api_key=api_key)
             logger.info("trading_adapter.api_initialized", url=api_url)
        
    return _trading_adapter
