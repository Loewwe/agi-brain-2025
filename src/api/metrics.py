"""
Prometheus Metrics for AGI-Brain.

Метрики для мониторинга:
- Задачи и выполнение
- Одобрения
- Ошибки
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# === Info ===

agi_brain_info = Info(
    "agi_brain",
    "AGI-Brain version and configuration",
)

# === Counters ===

tasks_total = Counter(
    "agi_brain_tasks_total",
    "Total number of tasks processed",
    ["client_id", "status"],
)

tool_calls_total = Counter(
    "agi_brain_tool_calls_total",
    "Total number of tool calls",
    ["tool_name", "status"],
)

approvals_total = Counter(
    "agi_brain_approvals_total",
    "Total number of approval requests",
    ["decision"],  # requested, approved, denied, timeout
)

errors_total = Counter(
    "agi_brain_errors_total",
    "Total number of errors",
    ["error_type", "component"],
)

# === Histograms ===

task_duration_seconds = Histogram(
    "agi_brain_task_duration_seconds",
    "Task execution duration in seconds",
    ["client_id"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
)

tool_duration_seconds = Histogram(
    "agi_brain_tool_duration_seconds",
    "Tool execution duration in seconds",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
)

llm_latency_seconds = Histogram(
    "agi_brain_llm_latency_seconds",
    "LLM API call latency in seconds",
    ["model", "operation"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

# === Gauges ===

pending_approvals = Gauge(
    "agi_brain_pending_approvals",
    "Number of pending approval requests",
)

episodic_memory_size = Gauge(
    "agi_brain_episodic_memory_size",
    "Size of episodic memory (episodes)",
)

semantic_memory_size = Gauge(
    "agi_brain_semantic_memory_size",
    "Size of semantic memory (documents)",
)

hard_cases_pending = Gauge(
    "agi_brain_hard_cases_pending",
    "Number of unprocessed hard cases",
)

current_autonomy_level = Gauge(
    "agi_brain_current_autonomy_level",
    "Current autonomy level",
)


# === Helper Functions ===

def record_task(client_id: str, success: bool, duration: float) -> None:
    """Записать метрики задачи."""
    status = "success" if success else "failure"
    tasks_total.labels(client_id=client_id, status=status).inc()
    task_duration_seconds.labels(client_id=client_id).observe(duration)


def record_tool_call(tool_name: str, success: bool, duration: float) -> None:
    """Записать метрики вызова инструмента."""
    status = "success" if success else "failure"
    tool_calls_total.labels(tool_name=tool_name, status=status).inc()
    tool_duration_seconds.labels(tool_name=tool_name).observe(duration)


def record_approval(decision: str) -> None:
    """Записать метрики одобрения."""
    approvals_total.labels(decision=decision).inc()


def record_error(error_type: str, component: str) -> None:
    """Записать ошибку."""
    errors_total.labels(error_type=error_type, component=component).inc()


def record_llm_call(model: str, operation: str, duration: float) -> None:
    """Записать вызов LLM."""
    llm_latency_seconds.labels(model=model, operation=operation).observe(duration)


def update_gauges(
    pending: int,
    episodic: int,
    semantic: int,
    hard_cases: int,
    autonomy: int,
) -> None:
    """Обновить все gauge метрики."""
    pending_approvals.set(pending)
    episodic_memory_size.set(episodic)
    semantic_memory_size.set(semantic)
    hard_cases_pending.set(hard_cases)
    current_autonomy_level.set(autonomy)


def init_info(version: str, mode: str) -> None:
    """Инициализировать info метрику."""
    agi_brain_info.info({
        "version": version,
        "mode": mode,
    })
