"""
Brain Core: LLM Orchestrator for AGI-Brain.

Главный мозг системы:
- Понимание целей
- Планирование
- Оркестрация инструментов
- Объяснение решений
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from openai import AsyncOpenAI

from ..tools.registry import ToolsRegistry, get_tools_registry
from ..memory.episodic import Episode, ExecutionStep, Plan, get_episodic_memory
from ..world.snapshot_builder import WorldSnapshot, world_snapshot
from ..memory.owner_profile import get_owner_profile_manager

logger = structlog.get_logger()


class BrainMode(str, Enum):
    """Режим работы мозга."""
    L0_READONLY = "L0"    # Только анализ и рекомендации
    L1_PROPOSE = "L1"     # Генерация планов/патчей
    L2_APPLY = "L2"       # Применение (заблокировано в v1)


@dataclass
class GoalAnalysis:
    """Результат анализа цели."""
    original_goal: str
    understood_intent: str
    required_tools: list[str]
    estimated_complexity: str  # low, medium, high
    requires_approval: bool
    approval_reason: str | None = None
    confidence: float = 0.0
    clarification_needed: str | None = None


@dataclass 
class PlanStep:
    """Шаг плана."""
    id: str
    tool: str
    args: dict[str, Any]
    level: int  # autonomy level required
    description: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    """Результат планирования."""
    plan_id: str
    goal: str
    steps: list[PlanStep]
    total_steps: int
    requires_approval: bool
    approval_reason: str | None
    estimated_duration_seconds: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_plan(self) -> Plan:
        """Преобразовать в Plan для Episode."""
        execution_steps = [
            ExecutionStep(
                step_id=step.id,
                tool_name=step.tool,
                arguments=step.args,
            )
            for step in self.steps
        ]
        return Plan(
            plan_id=self.plan_id,
            goal=self.goal,
            steps=execution_steps,
            requires_approval=self.requires_approval,
            approval_reason=self.approval_reason,
        )


class Brain:
    """
    LLM-Оркестратор AGI-Brain.
    
    Главный "мозг" системы, который:
    - Понимает цели в естественном языке
    - Строит планы достижения целей
    - Выбирает и вызывает инструменты
    - Объясняет свои решения
    
    Режимы работы:
    - L0: только анализ и рекомендации (read-only)
    - L1: генерация планов/патчей без применения
    - L2+: применение (заблокировано в v1)
    """
    
    # Системный промпт для мозга
    SYSTEM_PROMPT = """Ты - AGI-Brain, интеллектуальный ассистент, служащий интересам Владельца.

ВЫСШИЙ ЗАКОН: Максимизировать долгосрочную пользу и безопасность для Владельца в рамках законов, этики и заданных ограничений.

ПРИОРИТЕТЫ:
1. Безопасность и интересы Владельца
2. Сохранность капитала и инфраструктуры  
3. Эффективность и рост

ТЕКУЩИЙ РЕЖИМ: {mode}
- L0: Только анализ, отчёты, рекомендации (read-only)
- L1: Генерация планов и патчей без применения

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_description}

КОНТЕКСТ МИРА:
{world_context}

ПРОФИЛЬ ВЛАДЕЛЬЦА:
- Максимальный дневной убыток: {max_daily_loss}%
- Приоритет безопасности: {safety_priority}
- Стиль решений: {decision_style}

ПРАВИЛА:
1. Всегда объясняй свои решения
2. При любых сомнениях - спрашивай Владельца
3. Никогда не превышай разрешённый уровень автономии
4. Логируй все действия для аудита
"""

    def __init__(
        self,
        openai_api_key: str | None = None,
        model: str = "gpt-4o-mini",
        helper_model: str = "gpt-4o-mini",
        mode: BrainMode = BrainMode.L0_READONLY,
        registry: ToolsRegistry | None = None,
    ):
        """
        Args:
            openai_api_key: API ключ OpenAI
            model: Основная модель (gpt-5-mini/gpt-5.1)
            helper_model: Вспомогательная модель (gpt-4.1-mini)
            mode: Режим работы
            registry: Реестр инструментов
        """
        self.model = model
        self.helper_model = helper_model
        self.mode = mode
        # self.registry = registry or get_tools_registry() # Dynamic now
        
        # OpenAI клиент
        self._client: AsyncOpenAI | None = None
        if openai_api_key:
            self._client = AsyncOpenAI(api_key=openai_api_key)
        
        logger.info(
            "brain.initialized",
            model=model,
            mode=mode.value,
            tools_count=len(get_tools_registry().list_all()),
        )
    
    async def understand_goal(
        self,
        goal: str,
        context_hint: str | None = None,
        client_id: str = "owner",
        knowledge_section: str | None = None,
    ) -> GoalAnalysis:
        """
        Понять и проанализировать цель.
        
        Args:
            goal: Цель в естественном языке
            context_hint: Подсказка контекста
            client_id: ID клиента
            
        Returns:
            Анализ цели
        """
        # Получаем контекст
        snapshot = await world_snapshot(
            include_trading=True,
            include_metrics=True,
            include_owner_profile=True,
        )
        # Получаем доступные инструменты
        registry = get_tools_registry()
        available_tools = registry.list_for_client(client_id)
        
        # Анализ через LLM
        if self._client:
            analysis = await self._analyze_goal_with_llm(
                goal, context_hint, snapshot, available_tools, knowledge_section
            )
        else:
            # Fallback без LLM
            analysis = self._analyze_goal_locally(goal, available_tools, knowledge_section)
        
        logger.info(
            "brain.goal_understood",
            goal=goal[:100],
            tools_needed=analysis.required_tools,
            requires_approval=analysis.requires_approval,
        )
        
        return analysis
    
    async def create_plan(
        self,
        goal: str,
        goal_analysis: GoalAnalysis | None = None,
        snapshot: WorldSnapshot | None = None,
        client_id: str = "owner",
    ) -> PlanResult:
        """
        Создать план выполнения цели.
        
        Args:
            goal: Цель
            goal_analysis: Предварительный анализ (или создаётся)
            snapshot: Снимок мира (или создаётся)
            client_id: ID клиента
            
        Returns:
            План выполнения
        """
        if goal_analysis is None:
            goal_analysis = await self.understand_goal(goal, client_id=client_id)
        
        if snapshot is None:
            snapshot = await world_snapshot()
        
        # Генерация плана через LLM
        if self._client:
            plan = await self._create_plan_with_llm(goal, goal_analysis, snapshot, client_id)
        else:
            # Fallback
            plan = self._create_simple_plan(goal, goal_analysis)
        
        # Проверка на необходимость одобрения
        requires_approval = any(
            step.level >= 2 for step in plan.steps
        ) or goal_analysis.requires_approval
        
        plan.requires_approval = requires_approval
        if requires_approval:
            plan.approval_reason = "Plan contains actions requiring human approval"
        
        logger.info(
            "brain.plan_created",
            plan_id=plan.plan_id,
            steps=plan.total_steps,
            requires_approval=plan.requires_approval,
        )
        
        return plan
    
    async def explain_reasoning(
        self,
        plan: PlanResult,
        detail_level: str = "summary",
    ) -> str:
        """
        Объяснить логику плана.
        
        Args:
            plan: План для объяснения
            detail_level: summary, detailed, technical
            
        Returns:
            Текстовое объяснение
        """
        if self._client:
            return await self._explain_with_llm(plan, detail_level)
        
        # Fallback
        steps_desc = "\n".join(
            f"{i+1}. {step.tool}: {step.description}"
            for i, step in enumerate(plan.steps)
        )
        
        return f"""План для достижения цели: {plan.goal}

Шаги ({plan.total_steps}):
{steps_desc}

{"⚠️ Требуется одобрение: " + plan.approval_reason if plan.requires_approval else ""}
"""
    
    def get_mode(self) -> BrainMode:
        """Получить текущий режим."""
        return self.mode
    
    def set_mode(self, mode: BrainMode) -> None:
        """Установить режим (с проверками)."""
        if mode == BrainMode.L2_APPLY:
            logger.warning("brain.l2_blocked", reason="L2 mode is disabled in v1")
            return
        
        self.mode = mode
        logger.info("brain.mode_changed", mode=mode.value)
    
    async def _analyze_goal_with_llm(
        self,
        goal: str,
        context_hint: str | None,
        snapshot: WorldSnapshot,
        available_tools: list[str],
        knowledge_section: str | None = None,
    ) -> GoalAnalysis:
        """Анализ цели через LLM."""
        tools_desc = ", ".join(available_tools)
        
        prompt = f"""Проанализируй следующую цель и определи:
1. Что именно нужно сделать (understood_intent)
2. Какие инструменты понадобятся из: {tools_desc}
3. Сложность задачи (low/medium/high)
4. Нужно ли одобрение человека
5. Нужны ли уточнения

Цель: {goal}
{"Контекст: " + context_hint if context_hint else ""}
{"Раздел знаний: " + knowledge_section if knowledge_section else ""}

Текущее состояние:
{snapshot.overall_status_summary}

Ответь в формате JSON:
{{
  "understood_intent": "...",
  "required_tools": ["tool1", "tool2"],
  "complexity": "low|medium|high",
  "requires_approval": true|false,
  "approval_reason": "причина или null",
  "confidence": 0.0-1.0,
  "clarification_needed": "вопрос или null"
}}
"""
        
        response = await self._client.chat.completions.create(
            model=self.helper_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return GoalAnalysis(
            original_goal=goal,
            understood_intent=result.get("understood_intent", goal),
            required_tools=result.get("required_tools", []),
            estimated_complexity=result.get("complexity", "medium"),
            requires_approval=result.get("requires_approval", False),
            approval_reason=result.get("approval_reason"),
            confidence=result.get("confidence", 0.7),
            clarification_needed=result.get("clarification_needed"),
        )
    
    def _analyze_goal_locally(
        self,
        goal: str,
        available_tools: list[str],
        knowledge_section: str | None = None,
    ) -> GoalAnalysis:
        """Локальный анализ без LLM (fallback)."""
        goal_lower = goal.lower()
        
        # Простой keyword matching
        tools = []
        
        # Knowledge section hints
        if knowledge_section == "trading":
            tools.extend(["trading.fetch_positions", "trading.analyze_risk"])
        elif knowledge_section == "agi_core":
            tools.extend(["world.snapshot", "memory.search"])
            
        if "status" in goal_lower or "состояние" in goal_lower:
            tools.append("world.snapshot")
        if "trade" in goal_lower or "торг" in goal_lower or "позици" in goal_lower:
            tools.extend(["trading.fetch_positions", "trading.analyze_risk"])
        if "риск" in goal_lower or "risk" in goal_lower:
            tools.append("trading.analyze_risk")
        if "отчёт" in goal_lower or "report" in goal_lower:
            tools.append("reports.build_status_report")
        if "найти" in goal_lower or "search" in goal_lower or "поиск" in goal_lower:
            tools.append("memory.search")
        
        # Filter to available
        tools = [t for t in tools if t in available_tools]
        
        if not tools:
            tools = ["world.snapshot"]  # Default
        
        return GoalAnalysis(
            original_goal=goal,
            understood_intent=goal,
            required_tools=tools,
            estimated_complexity="medium",
            requires_approval=False,
            confidence=0.3,
        )
    
    async def _create_plan_with_llm(
        self,
        goal: str,
        analysis: GoalAnalysis,
        snapshot: WorldSnapshot,
        client_id: str,
    ) -> PlanResult:
        """Создание плана через LLM."""
        registry = get_tools_registry()
        tools_schema = registry.get_schema_for_llm(analysis.required_tools)
        
        prompt = f"""Создай пошаговый план для достижения цели.

Цель: {goal}
Понятый intent: {analysis.understood_intent}
Доступные инструменты: {json.dumps(analysis.required_tools)}

Текущее состояние:
{snapshot.overall_status_summary}

Создай план в формате JSON:
{{
  "steps": [
    {{
      "id": "step_1",
      "tool": "tool.name",
      "args": {{}},
      "level": 0,
      "description": "Что делает этот шаг",
      "depends_on": []
    }}
  ],
  "estimated_duration_seconds": 10
}}

Правила:
- level 0 = read-only операции
- level 1 = генерация/предложения
- level 2+ = изменения (заблокированы в v1)
- Минимизируй количество шагов
"""
        
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        steps = [
            PlanStep(
                id=s["id"],
                tool=s["tool"],
                args=s.get("args", {}),
                level=s.get("level", 0),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
            )
            for s in result.get("steps", [])
        ]
        
        return PlanResult(
            plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=goal,
            steps=steps,
            total_steps=len(steps),
            requires_approval=False,
            approval_reason=None,
            estimated_duration_seconds=result.get("estimated_duration_seconds", 10),
        )
    
    def _create_simple_plan(
        self,
        goal: str,
        analysis: GoalAnalysis,
    ) -> PlanResult:
        """Создание простого плана без LLM."""
        steps = []
        
        for i, tool in enumerate(analysis.required_tools):
            steps.append(PlanStep(
                id=f"step_{i+1}",
                tool=tool,
                args={},
                level=0,
                description=f"Execute {tool}",
                depends_on=[f"step_{i}"] if i > 0 else [],
            ))
        
        return PlanResult(
            plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=goal,
            steps=steps,
            total_steps=len(steps),
            requires_approval=False,
            approval_reason=None,
            estimated_duration_seconds=len(steps) * 5,
        )
    
    async def _explain_with_llm(
        self,
        plan: PlanResult,
        detail_level: str,
    ) -> str:
        """Объяснение через LLM."""
        steps_json = json.dumps([
            {"tool": s.tool, "description": s.description, "level": s.level}
            for s in plan.steps
        ], ensure_ascii=False)
        
        prompt = f"""Объясни следующий план на русском языке.
Уровень детализации: {detail_level}

Цель: {plan.goal}
Шаги: {steps_json}
Требует одобрения: {plan.requires_approval}
{"Причина: " + plan.approval_reason if plan.approval_reason else ""}

Объясни:
1. Что будет сделано
2. Почему выбраны эти шаги
3. Какие риски/ограничения
"""
        
        response = await self._client.chat.completions.create(
            model=self.helper_model,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content


# === Singleton ===

_brain: Brain | None = None


def get_brain() -> Brain:
    """Получить singleton мозга."""
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain


def configure_brain(
    openai_api_key: str | None = None,
    model: str = "gpt-4o-mini",
    mode: BrainMode = BrainMode.L0_READONLY,
) -> Brain:
    """Сконфигурировать мозг."""
    global _brain
    _brain = Brain(
        openai_api_key=openai_api_key,
        model=model,
        mode=mode,
    )
    return _brain
