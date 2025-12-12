"""
Plan Executor for AGI-Brain.

Безопасное выполнение планов:
- Step-by-step execution
- ACL enforcement
- Logging and audit
- Human approval workflow
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

import structlog

from ..tools.registry import ToolsRegistry, get_tools_registry
from ..memory.episodic import (
    Episode,
    ExecutionResult,
    ExecutionStatus,
    ExecutionStep,
    Error,
    Plan,
    get_episodic_memory,
)
from ..learning.hard_cases_buffer import get_hard_cases_buffer, HardCaseType, HardCasePriority
from .brain import PlanResult, PlanStep

logger = structlog.get_logger()


@dataclass
class StepResult:
    """Результат выполнения шага."""
    step_id: str
    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ExecutionContext:
    """Контекст выполнения плана."""
    plan: PlanResult
    episode: Episode
    client_id: str
    current_autonomy: int
    
    # Результаты шагов
    step_results: dict[str, StepResult] = field(default_factory=dict)
    
    # Флаги
    paused: bool = False
    cancelled: bool = False
    waiting_approval: bool = False
    approval_step_id: str | None = None


class PlanExecutor:
    """
    Исполнитель планов AGI-Brain.
    
    Особенности:
    - Пошаговое выполнение с логированием
    - Проверка ACL перед каждым шагом
    - Остановка при превышении уровня автономии
    - Human-in-the-loop для опасных операций
    """
    
    def __init__(self, max_autonomy_level: int = 0):
        """
        Args:
            max_autonomy_level: Максимальный разрешённый уровень
        """
        self.max_autonomy_level = max_autonomy_level
        # self._registry = get_tools_registry()  # Fetch dynamically
        
        # Callbacks
        self._on_step_start: Callable[[str, PlanStep], Awaitable[None]] | None = None
        self._on_step_complete: Callable[[str, StepResult], Awaitable[None]] | None = None
        self._on_approval_required: Callable[[str, PlanStep], Awaitable[bool]] | None = None
        
        logger.info(
            "executor.initialized",
            max_autonomy=max_autonomy_level,
            tools_available=len(get_tools_registry().list_all()),
        )
    
    async def execute(
        self,
        plan: PlanResult,
        client_id: str = "owner",
        autonomy_level: int = 0,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Выполнить план.
        
        Args:
            plan: План для выполнения
            client_id: ID клиента
            autonomy_level: Текущий уровень автономии
            dry_run: Только проверка без выполнения
            
        Returns:
            Результат выполнения
        """
        registry = get_tools_registry()
        logger.info(
            "executor.execute_start",
            plan_id=plan.plan_id,
            steps=plan.total_steps,
            dry_run=dry_run,
            tools_available=registry.list_all(),
            registry_id=id(registry),
        )
        
        # Создаём эпизод
        memory = get_episodic_memory()
        episode = memory.create_episode(
            goal=plan.goal,
            world_snapshot_json=json.dumps({"plan_id": plan.plan_id}),
            client_id=client_id,
            tags=["plan_execution"],
        )
        episode.plan = plan.to_plan()
        episode.start()
        
        # Контекст выполнения
        context = ExecutionContext(
            plan=plan,
            episode=episode,
            client_id=client_id,
            current_autonomy=autonomy_level,
        )
        
        # Pre-validation
        validation_error = self._validate_plan(plan, client_id, autonomy_level)
        if validation_error:
            return self._finish_with_error(context, validation_error)
        
        # Dry run - только проверка
        if dry_run:
            return ExecutionResult(
                success=True,
                summary=f"Dry run: {plan.total_steps} steps validated",
                output={"validated_steps": plan.total_steps},
            )
        
        # Выполнение шагов
        try:
            for step in plan.steps:
                if context.cancelled:
                    break
                
                # Проверка зависимостей
                if not self._dependencies_satisfied(step, context):
                    await self._wait_for_dependencies(step, context)
                
                # Выполнение шага
                result = await self._execute_step(step, context)
                context.step_results[step.id] = result
                
                if not result.success:
                    # Record hard case
                    buffer = get_hard_cases_buffer()
                    buffer.add_execution_error(
                        goal=context.plan.goal,
                        error=result.error or "Unknown error",
                        context={
                            "plan_id": context.plan.plan_id,
                            "step_id": step.id,
                            "tool": step.tool,
                            "args": step.args,
                        },
                        plan_id=context.plan.plan_id,
                        episode_id=context.episode.event_id,
                    )
                    
                    # Прерывание при ошибке
                    return self._finish_with_error(
                        context, 
                        f"Step {step.id} failed: {result.error}"
                    )
                
                # Обновление эпизода
                for ep_step in episode.steps:
                    if ep_step.step_id == step.id:
                        ep_step.mark_completed(result.output)
            
            # Успешное завершение
            return self._finish_success(context)
            
        except Exception as e:
            logger.exception("executor.unexpected_error", plan_id=plan.plan_id)
            
            # Record hard case
            buffer = get_hard_cases_buffer()
            buffer.add_execution_error(
                goal=plan.goal,
                error=str(e),
                context={"plan_id": plan.plan_id, "stage": "loop_exception"},
                plan_id=plan.plan_id,
                episode_id=context.episode.event_id,
                priority=HardCasePriority.CRITICAL,
            )
            
            return self._finish_with_error(context, str(e))
    
    async def execute_single_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        client_id: str = "owner",
    ) -> StepResult:
        """
        Выполнить единственный инструмент напрямую.
        
        Args:
            tool_name: Имя инструмента
            args: Аргументы
            client_id: ID клиента
            
        Returns:
            Результат выполнения
        """
        # Валидация
        registry = get_tools_registry()
        valid, reason = registry.validate_call(
            tool_name, client_id, self.max_autonomy_level
        )
        if not valid:
            return StepResult(
                step_id="direct",
                tool=tool_name,
                success=False,
                error=reason,
            )
        
        tool = registry.get(tool_name)
        if not tool:
            return StepResult(
                step_id="direct",
                tool=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
            )
        
        start_time = time.time()
        
        try:
            output = await tool.execute(**args)
            return StepResult(
                step_id="direct",
                tool=tool_name,
                success=True,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error("executor.tool_error", tool=tool_name, error=str(e))
            return StepResult(
                step_id="direct",
                tool=tool_name,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
    
    def set_callbacks(
        self,
        on_step_start: Callable[[str, PlanStep], Awaitable[None]] | None = None,
        on_step_complete: Callable[[str, StepResult], Awaitable[None]] | None = None,
        on_approval_required: Callable[[str, PlanStep], Awaitable[bool]] | None = None,
    ) -> None:
        """Установить callbacks для событий."""
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_approval_required = on_approval_required
    
    def _validate_plan(
        self,
        plan: PlanResult,
        client_id: str,
        autonomy_level: int,
    ) -> str | None:
        """Валидация плана перед выполнением."""
        for step in plan.steps:
            # 1. Получить инструмент
            registry = get_tools_registry()
            tool_def = registry.get_definition(step.tool)
            if not tool_def:
                return f"Tool not found: {step.tool}"
            
            # Проверка прав
            valid, reason = tool_def.can_execute(client_id, autonomy_level)
            if not valid:
                return f"Step {step.id}: {reason}"
            
            # Проверка уровня автономии
            if step.level > self.max_autonomy_level:
                return f"Step {step.id} requires autonomy L{step.level}, max allowed: L{self.max_autonomy_level}"
        
        return None
    
    async def _execute_step(
        self,
        step: PlanStep,
        context: ExecutionContext,
    ) -> StepResult:
        """Выполнить один шаг плана."""
        logger.info(
            "executor.step_start",
            plan_id=context.plan.plan_id,
            step_id=step.id,
            tool=step.tool,
        )
        
        # Callback
        if self._on_step_start:
            await self._on_step_start(context.plan.plan_id, step)
        
        # Обновление эпизода
        ep_step = ExecutionStep(
            step_id=step.id,
            tool_name=step.tool,
            arguments=step.args,
        )
        ep_step.mark_started()
        context.episode.steps.append(ep_step)
        context.episode.tools_used.append(step.tool)
        
        # Проверка на необходимость одобрения
        if step.level >= 2:
            if self._on_approval_required:
                approved = await self._on_approval_required(context.plan.plan_id, step)
                if not approved:
                    return StepResult(
                        step_id=step.id,
                        tool=step.tool,
                        success=False,
                        error="Approval denied",
                    )
            else:
                return StepResult(
                    step_id=step.id,
                    tool=step.tool,
                    success=False,
                    error="Human approval required but no callback set",
                )
        
        # Выполнение
        registry = get_tools_registry()
        tool = registry.get(step.tool)
        start_time = time.time()
        
        try:
            output = await tool.execute(**step.args)
            result = StepResult(
                step_id=step.id,
                tool=step.tool,
                success=True,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(
                "executor.step_failed",
                step_id=step.id,
                tool=step.tool,
                error=str(e),
            )
            result = StepResult(
                step_id=step.id,
                tool=step.tool,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
        
        logger.info(
            "executor.step_complete",
            step_id=step.id,
            success=result.success,
            duration=round(result.duration_seconds, 3),
        )
        
        # Callback
        if self._on_step_complete:
            await self._on_step_complete(context.plan.plan_id, result)
        
        return result
    
    def _dependencies_satisfied(
        self,
        step: PlanStep,
        context: ExecutionContext,
    ) -> bool:
        """Проверить, выполнены ли зависимости."""
        for dep_id in step.depends_on:
            if dep_id not in context.step_results:
                return False
            if not context.step_results[dep_id].success:
                return False
        return True
    
    async def _wait_for_dependencies(
        self,
        step: PlanStep,
        context: ExecutionContext,
        timeout: float = 60.0,
    ) -> None:
        """Ожидание выполнения зависимостей."""
        start = time.time()
        while not self._dependencies_satisfied(step, context):
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for dependencies of step {step.id}")
            await asyncio.sleep(0.1)
    
    def _finish_success(self, context: ExecutionContext) -> ExecutionResult:
        """Завершить выполнение успешно."""
        context.episode.complete(ExecutionResult(
            success=True,
            summary=f"Plan executed: {len(context.step_results)} steps",
        ))
        
        # Сохранение эпизода
        memory = get_episodic_memory()
        memory.update(context.episode)
        
        logger.info(
            "executor.plan_complete",
            plan_id=context.plan.plan_id,
            steps_executed=len(context.step_results),
        )
        
        return ExecutionResult(
            success=True,
            summary=f"Plan '{context.plan.plan_id}' executed successfully",
            output={
                "steps": [r.to_dict() for r in context.step_results.values()],
            },
        )
    
    def _finish_with_error(
        self,
        context: ExecutionContext,
        error_message: str,
    ) -> ExecutionResult:
        """Завершить выполнение с ошибкой."""
        context.episode.add_error(Error(
            code="EXECUTION_ERROR",
            message=error_message,
        ))
        context.episode.complete(ExecutionResult(
            success=False,
            summary=error_message,
        ))
        
        # Сохранение эпизода
        memory = get_episodic_memory()
        memory.update(context.episode)
        
        logger.error(
            "executor.plan_failed",
            plan_id=context.plan.plan_id,
            error=error_message,
        )
        
        return ExecutionResult(
            success=False,
            summary=f"Plan failed: {error_message}",
            human_required=True,
            human_required_reason=error_message,
        )


# === Singleton ===

_executor: PlanExecutor | None = None


def get_executor() -> PlanExecutor:
    """Получить singleton executor."""
    global _executor
    if _executor is None:
        _executor = PlanExecutor()
    return _executor


def configure_executor(
    max_autonomy_level: int = 1,
) -> PlanExecutor:
    """Сконфигурировать executor."""
    global _executor
    _executor = PlanExecutor(max_autonomy_level=max_autonomy_level)
    return _executor
