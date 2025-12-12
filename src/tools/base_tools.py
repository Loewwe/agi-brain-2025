"""
Base Tools for AGI-Brain v1.

Реализации базовых инструментов:
- world.snapshot
- memory.search, memory.write
- trading.fetch_*, trading.analyze_risk
- reports.build_status_report
"""

import json
from datetime import datetime
from typing import Any

import structlog

from ..world.snapshot_builder import world_snapshot
from ..memory.episodic import get_episodic_memory
from ..memory.semantic import get_semantic_memory, SearchQuery, DocumentType
from ..perception.trading_adapter import get_trading_adapter

logger = structlog.get_logger()


# === World Tools ===

async def tool_world_snapshot(
    include_trading: bool = True,
    include_metrics: bool = True,
    include_owner_profile: bool = False,
) -> dict[str, Any]:
    """
    Получить текущий снимок состояния мира.
    
    Returns:
        WorldSnapshot как словарь
    """
    snapshot = await world_snapshot(
        include_trading=include_trading,
        include_metrics=include_metrics,
        include_owner_profile=include_owner_profile,
    )
    
    return {
        "snapshot_id": snapshot.snapshot_id,
        "timestamp": snapshot.timestamp.isoformat(),
        "trading": {
            "status": snapshot.trading.agent.status.value if snapshot.trading else None,
            "pnl_today": snapshot.trading.agent.pnl.pnl_today_percent if snapshot.trading else None,
            "positions_count": len(snapshot.trading.agent.current_positions) if snapshot.trading else 0,
            "risk_score": snapshot.trading.current_risk_score if snapshot.trading else 0,
            "violations": len(snapshot.trading.risk_violations_active) if snapshot.trading else 0,
        } if snapshot.trading else None,
        "agi_system": {
            "version": snapshot.agi_system.version,
            "autonomy_level": snapshot.agi_system.autonomy_level.value,
            "health": snapshot.agi_system.overall_health.value,
            "tasks_today": snapshot.agi_system.tasks_completed_today,
        },
        "summary": snapshot.overall_status_summary,
        "generation_time_ms": snapshot.generation_time_ms,
    }


# === Memory Tools ===

async def tool_memory_search(
    query: str,
    limit: int = 10,
    doc_types: list[str] | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Поиск в семантической памяти.
    
    Returns:
        Результаты поиска
    """
    memory = get_semantic_memory()
    
    # Преобразование типов
    types = None
    if doc_types:
        types = [DocumentType(t) for t in doc_types if t in DocumentType._value2member_map_]
    
    search_query = SearchQuery(
        text=query,
        doc_types=types,
        categories=categories,
        limit=limit,
    )
    
    results = await memory.search(search_query)
    
    return {
        "query": query,
        "total": len(results),
        "results": [
            {
                "id": r.document.id,
                "title": r.document.title,
                "score": r.score,
                "type": r.document.doc_type.value,
                "category": r.document.category,
                "highlights": r.highlights[:2],
            }
            for r in results
        ],
    }


async def tool_memory_write(
    content: dict[str, Any],
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Записать в эпизодическую память.
    
    Returns:
        event_id записи
    """
    memory = get_episodic_memory()
    
    episode = memory.create_episode(
        goal=content.get("goal", "Manual memory write"),
        world_snapshot_json=json.dumps(content.get("context", {})),
        client_id=content.get("client_id", "internal"),
        context_hint=content.get("hint"),
        tags=tags,
    )
    
    return {
        "event_id": episode.event_id,
        "created_at": episode.created_at.isoformat(),
    }


# === Trading Tools ===

async def tool_trading_fetch_history(
    days: int = 7,
    symbol: str | None = None,
) -> dict[str, Any]:
    """
    Получить историю сделок.
    
    Returns:
        Список сделок
    """
    adapter = get_trading_adapter()
    trades = await adapter.fetch_history(days=days, symbol=symbol)
    
    return {
        "period_days": days,
        "symbol": symbol,
        "total_trades": len(trades),
        "trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side.value,
                "pnl": float(t.realized_pnl),
                "pnl_percent": t.realized_pnl_percent,
                "closed_at": t.closed_at.isoformat(),
                "reason": t.reason,
            }
            for t in trades[:20]  # Лимит для ответа
        ],
    }


async def tool_trading_fetch_positions() -> dict[str, Any]:
    """
    Получить текущие позиции.
    
    Returns:
        Список позиций
    """
    adapter = get_trading_adapter()
    positions = await adapter.fetch_positions()
    
    return {
        "total_positions": len(positions),
        "positions": [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "size": float(p.size),
                "entry_price": float(p.entry_price),
                "current_price": float(p.current_price),
                "leverage": p.leverage,
                "unrealized_pnl": float(p.unrealized_pnl),
                "pnl_percent": p.unrealized_pnl_percent,
                "has_stop_loss": p.stop_loss is not None,
                "has_take_profit": p.take_profit is not None,
                "distance_to_liq": p.distance_to_liquidation_percent,
            }
            for p in positions
        ],
    }


async def tool_trading_fetch_config() -> dict[str, Any]:
    """
    Получить конфигурацию риск-менеджмента.
    
    Returns:
        Конфигурация
    """
    adapter = get_trading_adapter()
    config = await adapter.fetch_config()
    
    return {
        "max_position_size_usd": float(config.max_position_size_usd),
        "max_leverage": config.max_leverage,
        "max_daily_loss_percent": config.max_daily_loss_percent,
        "unsinkable_balance_usd": float(config.unsinkable_balance_usd),
        "max_concurrent_positions": config.max_concurrent_positions,
        "stop_loss_percent": config.stop_loss_percent,
        "take_profit_percent": config.take_profit_percent,
        "risk_guardian_enabled": config.enable_risk_guardian,
        "liquidation_protection": config.enable_liquidation_protection,
    }


async def tool_trading_analyze_risk(
    include_recommendations: bool = True,
) -> dict[str, Any]:
    """
    Анализ рисков текущих позиций.
    
    Returns:
        Анализ с рекомендациями
    """
    adapter = get_trading_adapter()
    analysis = await adapter.analyze_risk(include_recommendations=include_recommendations)
    
    return {
        "risk_score": analysis.risk_score,
        "risk_level": (
            "critical" if analysis.risk_score >= 70
            else "high" if analysis.risk_score >= 50
            else "medium" if analysis.risk_score >= 25
            else "low"
        ),
        "violations": analysis.violations,
        "recommendations": analysis.recommendations if include_recommendations else [],
        "timestamp": analysis.timestamp.isoformat(),
    }


async def tool_trading_set_stop_loss(
    symbol: str,
    price: float,
) -> dict[str, Any]:
    """
    Установить Stop-Loss для позиции.
    
    Args:
        symbol: Символ (e.g. BTCUSDT)
        price: Цена Stop-Loss
        
    Returns:
        Результат операции
    """
    adapter = get_trading_adapter()
    success = await adapter.set_stop_loss(symbol=symbol, price=price)
    
    return {
        "symbol": symbol,
        "price": price,
        "success": success,
        "timestamp": datetime.now().isoformat(),
    }
# === Reports Tools ===

async def tool_build_status_report(
    period: str = "daily",
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """
    Сформировать статусный отчёт.
    
    Args:
        period: daily, weekly, monthly
        sections: trading, agi, memory, risks
        
    Returns:
        Структурированный отчёт
    """
    all_sections = sections or ["trading", "agi", "risks"]
    report: dict[str, Any] = {
        "period": period,
        "generated_at": datetime.now().isoformat(),
        "sections": {},
    }
    
    if "trading" in all_sections:
        adapter = get_trading_adapter()
        agent = await adapter.fetch_agent_state()
        if agent:
            report["sections"]["trading"] = {
                "status": agent.status.value,
                "pnl_today": agent.pnl.pnl_today_percent,
                "pnl_week": agent.pnl.pnl_week_percent,
                "pnl_month": agent.pnl.pnl_month_percent,
                "positions": len(agent.current_positions),
                "win_rate": agent.pnl.win_rate,
                "strategies_active": len([s for s in agent.strategies if s.enabled]),
            }
    
    if "agi" in all_sections:
        episodic = get_episodic_memory()
        semantic = get_semantic_memory()
        report["sections"]["agi"] = {
            "episodic_memory_size": episodic.get_size(),
            "semantic_memory_size": semantic.get_size(),
            "episodic_stats": episodic.get_stats(),
        }
    
    if "risks" in all_sections:
        adapter = get_trading_adapter()
        analysis = await adapter.analyze_risk()
        report["sections"]["risks"] = {
            "risk_score": analysis.risk_score,
            "violations_count": len(analysis.violations),
            "violations": analysis.violations[:5],
            "top_recommendations": analysis.recommendations[:3],
        }
    
    return report


# === Tool Registration Helper ===

def register_base_tools(registry) -> None:
    """
    Зарегистрировать все базовые инструменты.
    
    Args:
        registry: ToolsRegistry instance
    """
    tools = {
        "world.snapshot": tool_world_snapshot,
        "memory.search": tool_memory_search,
        "memory.write": tool_memory_write,
        "trading.fetch_history": tool_trading_fetch_history,
        "trading.fetch_positions": tool_trading_fetch_positions,
        "trading.fetch_config": tool_trading_fetch_config,
        "trading.analyze_risk": tool_trading_analyze_risk,
        "trading.set_stop_loss": tool_trading_set_stop_loss,
        "reports.build_status_report": tool_build_status_report,
    }
    
    for name, handler in tools.items():
        try:
            registry.register(name, handler)
            logger.debug("base_tools.registered", tool=name)
        except Exception as e:
            logger.warning("base_tools.register_failed", tool=name, error=str(e))
