"""
Experiment - Experiment lifecycle management for AGI-Brain.

Provides:
- Experiment entity (goal, config, status, result)
- ExperimentRunner (pipeline: logs → dataset → backtest → result)
- Integration with EpisodicMemory
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .models import (
    DatasetConfig,
    ExperimentResult,
    SimulatorConfig,
)
from .event_log import EventLog
from .market_log import MarketLog
from .dataset_builder import DatasetBuilder
from .simulator import Simulator, Stage6Strategy, StrategyBase

logger = structlog.get_logger()


# =============================================================================
# EXPERIMENT STATUS
# =============================================================================

class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# EXPERIMENT ENTITY
# =============================================================================

@dataclass
class Experiment:
    """
    Experiment entity in AGI-Brain.
    
    Represents a single backtest/experiment run with:
    - Goal and configuration
    - Status tracking
    - Results storage
    - Link to EpisodicMemory
    """
    # Identity
    experiment_id: str
    name: str
    goal: str
    
    # Configuration
    strategy_config: dict[str, Any]
    symbols: list[str]
    date_from: date
    date_to: date
    
    # Status
    status: ExperimentStatus = ExperimentStatus.PLANNED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Results
    result: ExperimentResult | None = None
    error: str | None = None
    
    # Links
    episode_id: str | None = None
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "goal": self.goal,
            "strategy_config": self.strategy_config,
            "symbols": self.symbols,
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "episode_id": self.episode_id,
            "tags": self.tags,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        """Deserialize from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            goal=data["goal"],
            strategy_config=data["strategy_config"],
            symbols=data["symbols"],
            date_from=date.fromisoformat(data["date_from"]),
            date_to=date.fromisoformat(data["date_to"]),
            status=ExperimentStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=ExperimentResult.from_dict(data["result"]) if data.get("result") else None,
            error=data.get("error"),
            episode_id=data.get("episode_id"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Run experiments through the pipeline.
    
    Pipeline:
    1. Build dataset from market logs
    2. Run backtest with strategy
    3. Calculate metrics
    4. Save result to experiment
    5. Optionally save to EpisodicMemory
    """
    
    def __init__(
        self,
        market_log: MarketLog,
        event_log: EventLog | None = None,
        dataset_builder: DatasetBuilder | None = None,
        storage_path: Path | str | None = None,
    ):
        """
        Initialize ExperimentRunner.
        
        Args:
            market_log: MarketLog for OHLCV data
            event_log: Optional EventLog for trade history
            dataset_builder: Optional DatasetBuilder (created if not provided)
            storage_path: Path for experiment storage
        """
        self.market_log = market_log
        self.event_log = event_log
        
        self.dataset_builder = dataset_builder or DatasetBuilder(
            market_log=market_log,
            event_log=event_log,
        )
        
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory experiment cache
        self._experiments: dict[str, Experiment] = {}
        
        # Load existing experiments
        if self.storage_path:
            self._load_experiments()
        
        logger.info("experiment_runner.initialized")
    
    def create_experiment(
        self,
        name: str,
        goal: str,
        symbols: list[str],
        date_from: date,
        date_to: date,
        strategy_config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            goal: Experiment goal description
            symbols: List of symbols to test
            date_from: Start date
            date_to: End date
            strategy_config: Strategy configuration (uses defaults if None)
            tags: Optional tags
            
        Returns:
            Created Experiment
        """
        experiment = Experiment(
            experiment_id=str(uuid.uuid4())[:8],
            name=name,
            goal=goal,
            symbols=symbols,
            date_from=date_from,
            date_to=date_to,
            strategy_config=strategy_config or SimulatorConfig().to_dict(),
            tags=tags or [],
        )
        
        self._experiments[experiment.experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(
            "experiment_runner.created",
            experiment_id=experiment.experiment_id,
            name=name,
        )
        
        return experiment
    
    async def run(
        self,
        experiment: Experiment,
        strategy: StrategyBase | None = None,
        fetch_missing_data: bool = True,
    ) -> Experiment:
        """
        Execute experiment pipeline.
        
        Args:
            experiment: Experiment to run
            strategy: Strategy to use (defaults to Stage6Strategy)
            fetch_missing_data: Whether to fetch missing market data
            
        Returns:
            Updated experiment with results
        """
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()
        self._save_experiment(experiment)
        
        logger.info(
            "experiment_runner.starting",
            experiment_id=experiment.experiment_id,
            symbols=len(experiment.symbols),
        )
        
        try:
            # Step 1: Ensure market data
            if fetch_missing_data:
                await self.market_log.ensure_data(
                    symbols=experiment.symbols,
                    timeframe="5m",
                    date_from=experiment.date_from,
                    date_to=experiment.date_to,
                )
            
            # Step 2: Build dataset
            dataset_config = DatasetConfig(
                symbols=experiment.symbols,
                date_from=experiment.date_from,
                date_to=experiment.date_to,
                timeframe="5m",
                features_schema_version="v1",
                mode="backtest",
            )
            
            dataset, metadata = self.dataset_builder.build(dataset_config, save=False)
            
            logger.info(
                "experiment_runner.dataset_built",
                rows=len(dataset),
                symbols=len(metadata.symbols_included),
            )
            
            # Step 3: Create simulator and strategy
            sim_config = SimulatorConfig.from_dict(experiment.strategy_config)
            simulator = Simulator(sim_config)
            
            if strategy is None:
                strategy = Stage6Strategy(sim_config)
            
            # Step 4: Run backtest
            result = simulator.run(dataset, strategy)
            
            # Step 5: Update experiment
            experiment.result = result.metrics
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.now()
            
            self._save_experiment(experiment)
            
            logger.info(
                "experiment_runner.completed",
                experiment_id=experiment.experiment_id,
                trades=result.metrics.total_trades,
                win_rate=f"{result.metrics.win_rate * 100:.1f}%",
                total_return=f"{result.metrics.total_return_pct:.2f}%",
            )
            
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.error = str(e)
            experiment.completed_at = datetime.now()
            self._save_experiment(experiment)
            
            logger.error(
                "experiment_runner.failed",
                experiment_id=experiment.experiment_id,
                error=str(e),
            )
            raise
        
        return experiment
    
    def run_sync(
        self,
        experiment: Experiment,
        strategy: StrategyBase | None = None,
    ) -> Experiment:
        """
        Execute experiment synchronously (no data fetching).
        
        For use when data is already available.
        """
        import asyncio
        
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.run(experiment, strategy, fetch_missing_data=False)
        )
    
    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
        limit: int = 50,
    ) -> list[Experiment]:
        """
        List experiments.
        
        Args:
            status: Optional status filter
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiments, sorted by created_at descending
        """
        experiments = list(self._experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        experiments.sort(key=lambda e: e.created_at, reverse=True)
        
        return experiments[:limit]
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment."""
        if experiment_id not in self._experiments:
            return False
        
        del self._experiments[experiment_id]
        
        if self.storage_path:
            file_path = self.storage_path / f"exp_{experiment_id}.json"
            if file_path.exists():
                file_path.unlink()
        
        return True
    
    def get_stats(self) -> dict:
        """Get statistics about experiments."""
        experiments = list(self._experiments.values())
        
        by_status = {}
        for exp in experiments:
            by_status[exp.status.value] = by_status.get(exp.status.value, 0) + 1
        
        completed = [e for e in experiments if e.status == ExperimentStatus.COMPLETED and e.result]
        
        avg_win_rate = 0.0
        avg_return = 0.0
        if completed:
            avg_win_rate = sum(e.result.win_rate for e in completed) / len(completed)
            avg_return = sum(e.result.total_return_pct for e in completed) / len(completed)
        
        return {
            "total": len(experiments),
            "by_status": by_status,
            "completed_count": len(completed),
            "avg_win_rate": avg_win_rate,
            "avg_return_pct": avg_return,
        }
    
    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk."""
        self._experiments[experiment.experiment_id] = experiment
        
        if self.storage_path:
            file_path = self.storage_path / f"exp_{experiment.experiment_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(experiment.to_dict(), f, indent=2, default=str)
    
    def _load_experiments(self) -> None:
        """Load experiments from disk."""
        if not self.storage_path:
            return
        
        for file_path in self.storage_path.glob("exp_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                experiment = Experiment.from_dict(data)
                self._experiments[experiment.experiment_id] = experiment
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    "experiment_runner.load_error",
                    file=str(file_path),
                    error=str(e),
                )
        
        logger.info(
            "experiment_runner.loaded",
            count=len(self._experiments),
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_experiment_runner(
    data_path: Path | str = "data",
) -> ExperimentRunner:
    """
    Create ExperimentRunner with default configuration.
    
    Args:
        data_path: Base path for data storage
        
    Returns:
        Configured ExperimentRunner
    """
    data_path = Path(data_path)
    
    market_log = MarketLog(data_path / "market")
    event_log = EventLog(data_path / "events")
    
    dataset_builder = DatasetBuilder(
        market_log=market_log,
        event_log=event_log,
        output_path=data_path / "datasets",
    )
    
    return ExperimentRunner(
        market_log=market_log,
        event_log=event_log,
        dataset_builder=dataset_builder,
        storage_path=data_path / "experiments",
    )
