"""
FastAPI Main Application for AGI-Brain.

Главная точка входа для API.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import structlog

from .orchestrate import router as orchestrate_router
from .metrics import init_info, update_gauges
from ..planning.brain import configure_brain, BrainMode
from ..planning.executor import configure_executor
from ..tools.registry import configure_tools_registry
from ..tools.base_tools import register_base_tools
from ..memory.episodic import configure_episodic_memory, get_episodic_memory
from ..memory.semantic import configure_semantic_memory, get_semantic_memory
from ..memory.owner_profile import configure_owner_profile, get_owner_profile_manager
from ..perception.trading_adapter import configure_trading_adapter, AdapterMode
from ..perception.web_fetcher import configure_web_fetcher
from ..world.snapshot_builder import configure_snapshot_builder
from ..safety.acl import configure_acl
from ..safety.audit import configure_audit_logger
from ..safety.approval import get_approval_manager
from ..learning.hard_cases_buffer import get_hard_cases_buffer

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("agi_brain.starting")
    
    # Определяем пути к конфигам
    base_path = Path(__file__).parent.parent
    configs_path = base_path / "configs"
    
    # Инициализация компонентов
    
    # 1. Configs
    if (configs_path / "owner_profile.yaml").exists():
        configure_owner_profile(configs_path / "owner_profile.yaml")
    

    # 2. Tools Registry
    tools_config = configs_path / "tools_registry.yaml"
    
    registry = configure_tools_registry(
        tools_config if tools_config.exists() else None
    )
    register_base_tools(registry)
    
    # 3. Memory
    configure_episodic_memory()
    configure_semantic_memory()
    
    # 4. Perception
    import os
    exchange_key = os.environ.get("EXCHANGE_API_KEY")
    mode = AdapterMode.STAGE6 if exchange_key else AdapterMode.STUB
    trading_adapter = configure_trading_adapter(mode=mode, scenario="normal")
    if (configs_path / "learning_sources.yaml").exists():
        configure_web_fetcher(configs_path / "learning_sources.yaml")
        
    # 5. World Model
    configure_snapshot_builder(
        trading_adapter=trading_adapter,
        episodic_memory=get_episodic_memory(),
        semantic_memory=get_semantic_memory(),
        owner_profile=get_owner_profile_manager().get() if (configs_path / "owner_profile.yaml").exists() else None
    )
    
    # 5. Safety
    configure_acl()
    configure_audit_logger()
    
    # 6. Brain & Executor
    import os
    openai_key = os.environ.get("OPENAI_API_KEY")
    configure_brain(
        openai_api_key=openai_key,
        model="gpt-4o-mini",
        mode=BrainMode.L0_READONLY,
    )
    configure_executor(max_autonomy_level=1)
    
    # Info metric
    init_info(version="1.0.0", mode="L0")
    
    logger.info("agi_brain.started")
    
    yield
    
    # Shutdown
    logger.info("agi_brain.stopping")


# === Application ===

app = FastAPI(
    title="AGI-Brain",
    description="AGI-контейнер: наблюдает, анализирует, учится и советует Владельцу",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(orchestrate_router)
from .knowledge import router as knowledge_router
app.include_router(knowledge_router)
from .ui import router as ui_router
app.include_router(ui_router)
from .hard_cases import router as hard_cases_router
app.include_router(hard_cases_router)
from .trading import router as trading_router
app.include_router(trading_router)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# === Root Endpoints ===

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AGI-Brain",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check."""
    # Проверяем основные компоненты
    from ..planning.brain import get_brain
    from ..memory.episodic import get_episodic_memory
    
    brain = get_brain()
    memory = get_episodic_memory()
    
    return {
        "status": "ready",
        "brain_mode": brain.get_mode().value,
        "memory_size": memory.get_size(),
    }
