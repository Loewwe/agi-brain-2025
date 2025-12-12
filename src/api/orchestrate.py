"""
Orchestration API for AGI-Brain.

Внутренний API для оркестрации:
- /v1/orchestrate - главный endpoint
- Health checks
- Metrics
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..planning.brain import Brain, BrainMode, get_brain, PlanResult
from ..planning.executor import PlanExecutor, get_executor
from ..tools.registry import get_tools_registry
from ..memory.episodic import get_episodic_memory
from ..memory.semantic import get_semantic_memory
from ..safety.acl import get_acl
from ..safety.audit import get_audit_logger, AuditEventType
from ..safety.approval import get_approval_manager

router = APIRouter(prefix="/v1", tags=["orchestration"])


# === Request/Response Models ===

class OrchestrationRequest(BaseModel):
    """Запрос на оркестрацию."""
    goal: str = Field(..., description="Цель в естественном языке")
    context_hint: str | None = Field(None, description="Подсказка контекста")
    allowed_tools: list[str] | None = Field(None, description="Разрешённые инструменты")
    client_id: str = Field("owner", description="ID клиента")
    dry_run: bool = Field(False, description="Только планирование без выполнения")
    knowledge_section: str | None = Field(None, description="Раздел базы знаний для контекста")


class PlanStepResponse(BaseModel):
    """Шаг плана в ответе."""
    id: str
    tool: str
    description: str
    level: int
    status: str | None = None
    result: Any = None


class OrchestrationResponse(BaseModel):
    """Ответ на оркестрацию."""
    plan_id: str
    goal: str
    steps: list[PlanStepResponse]
    result_summary: str
    human_required: bool
    human_required_reason: str | None = None
    execution_time_ms: float | None = None


class ToolCallRequest(BaseModel):
    """Запрос на вызов инструмента."""
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    client_id: str = Field("owner")


class ToolCallResponse(BaseModel):
    """Ответ на вызов инструмента."""
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float


# === Endpoints ===

@router.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(
    request: OrchestrationRequest,
    background_tasks: BackgroundTasks,
) -> OrchestrationResponse:
    """
    Главный endpoint оркестрации.
    
    Принимает цель, создаёт план и выполняет его.
    """
    start_time = datetime.now()
    
    # Проверка ACL
    acl = get_acl()
    client = acl.get_client(request.client_id)
    if not client:
        raise HTTPException(status_code=403, detail=f"Unknown client: {request.client_id}")
    
    # Аудит
    audit = get_audit_logger()
    audit.log(
        event_type=AuditEventType.PLAN_CREATED,
        client_id=request.client_id,
        resource="orchestration",
        action="request",
        details={"goal": request.goal[:200]},
    )
    
    # Получаем мозг и создаём план
    brain = get_brain()
    
    try:
        # Понимание цели
        goal_analysis = await brain.understand_goal(
            goal=request.goal,
            context_hint=request.context_hint,
            client_id=request.client_id,
            knowledge_section=request.knowledge_section,
        )
        
        # Проверка уточнений
        if goal_analysis.clarification_needed:
            return OrchestrationResponse(
                plan_id="clarification_needed",
                goal=request.goal,
                steps=[],
                result_summary=f"Требуется уточнение: {goal_analysis.clarification_needed}",
                human_required=True,
                human_required_reason="clarification_needed",
            )
        
        # Создание плана
        plan = await brain.create_plan(
            goal=request.goal,
            goal_analysis=goal_analysis,
            client_id=request.client_id,
        )
        
        # Dry run
        if request.dry_run:
            return _build_response(plan, "Plan created (dry run)", start_time)
        
        # Выполнение
        executor = get_executor()
        result = await executor.execute(
            plan=plan,
            client_id=request.client_id,
            autonomy_level=client.max_autonomy_level,
        )
        
        return _build_response(
            plan,
            result.summary,
            start_time,
            human_required=result.human_required,
            human_required_reason=result.human_required_reason,
        )
        
    except Exception as e:
        audit.log(
            event_type=AuditEventType.PLAN_FAILED,
            client_id=request.client_id,
            resource="orchestration",
            action="execute",
            details={"error": str(e)},
            success=False,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest) -> ToolCallResponse:
    """
    Прямой вызов инструмента.
    """
    start_time = datetime.now()
    
    # Проверка ACL
    acl = get_acl()
    result = acl.check_permission(request.client_id, request.tool_name)
    if not result.allowed:
        raise HTTPException(status_code=403, detail=result.reason)
    
    # Выполнение
    executor = get_executor()
    step_result = await executor.execute_single_tool(
        tool_name=request.tool_name,
        args=request.args,
        client_id=request.client_id,
    )
    
    duration = (datetime.now() - start_time).total_seconds() * 1000
    
    return ToolCallResponse(
        tool_name=request.tool_name,
        success=step_result.success,
        output=step_result.output,
        error=step_result.error,
        duration_ms=duration,
    )


@router.get("/tools")
async def list_tools(client_id: str = "owner") -> dict[str, Any]:
    """Получить список доступных инструментов."""
    registry = get_tools_registry()
    acl = get_acl()
    
    tools = []
    for name in registry.list_all():
        definition = registry.get_definition(name)
        if not definition:
            continue
        
        check = acl.check_permission(client_id, name)
        tools.append({
            "name": name,
            "description": definition.description,
            "category": definition.category,
            "risk_level": definition.risk_level.value,
            "autonomy_level": definition.autonomy_level,
            "allowed": check.allowed,
        })
    
    return {"tools": tools, "total": len(tools)}


@router.get("/approvals/pending")
async def get_pending_approvals() -> dict[str, Any]:
    """Получить ожидающие одобрения."""
    manager = get_approval_manager()
    pending = manager.get_pending()
    
    return {
        "pending": [r.to_dict() for r in pending],
        "count": len(pending),
    }


@router.post("/approvals/{request_id}/approve")
async def approve_request(
    request_id: str,
    by: str = "owner",
    reason: str | None = None,
) -> dict[str, Any]:
    """Одобрить запрос."""
    manager = get_approval_manager()
    success = manager.approve(request_id, by, reason)
    
    if not success:
        raise HTTPException(status_code=404, detail="Request not found or already processed")
    
    return {"status": "approved", "request_id": request_id}


@router.post("/approvals/{request_id}/deny")
async def deny_request(
    request_id: str,
    by: str = "owner",
    reason: str | None = None,
) -> dict[str, Any]:
    """Отклонить запрос."""
    manager = get_approval_manager()
    success = manager.deny(request_id, by, reason)
    
    if not success:
        raise HTTPException(status_code=404, detail="Request not found or already processed")
    
    return {"status": "denied", "request_id": request_id}


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """Получить статус системы."""
    brain = get_brain()
    episodic = get_episodic_memory()
    semantic = get_semantic_memory()
    audit = get_audit_logger()
    approvals = get_approval_manager()
    
    return {
        "version": "1.0.0",
        "mode": brain.get_mode().value,
        "memory": {
            "episodic_size": episodic.get_size(),
            "semantic_size": semantic.get_size(),
        },
        "audit_events": audit.get_stats(),
        "approvals": approvals.get_stats(),
        "timestamp": datetime.now().isoformat(),
    }


# === Helper Functions ===

def _build_response(
    plan: PlanResult,
    summary: str,
    start_time: datetime,
    human_required: bool = False,
    human_required_reason: str | None = None,
) -> OrchestrationResponse:
    """Построить ответ оркестрации."""
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return OrchestrationResponse(
        plan_id=plan.plan_id,
        goal=plan.goal,
        steps=[
            PlanStepResponse(
                id=s.id,
                tool=s.tool,
                description=s.description,
                level=s.level,
            )
            for s in plan.steps
        ],
        result_summary=summary,
        human_required=human_required or plan.requires_approval,
        human_required_reason=human_required_reason or plan.approval_reason,
        execution_time_ms=execution_time,
    )
