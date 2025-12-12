"""
Stage7 Experiment Module.

Provides the foundation for offline experimentation:
- Unified event logging (trades, positions)
- Market data storage
- Dataset building
- Deterministic backtesting
- Experiment lifecycle management
- ML labeling and alpha engine
"""

from .models import (
    TradeEvent,
    MarketBar,
    ExperimentResult,
    DatasetConfig,
    DatasetMetadata,
    SimulatorConfig,
    SimulatorResult,
)

from .event_log import EventLog
from .market_log import MarketLog
from .dataset_builder import DatasetBuilder
from .simulator import Simulator, StrategyBase
from .experiment import Experiment, ExperimentStatus, ExperimentRunner
from .labels import LabelConfig, LabelsBuilder, create_ml_dataset
from .alpha_engine import AlphaConfig, AlphaEngine, TrainResult
from .comparison import StrategyComparison, ComparisonResult, run_multi_period_comparison

__all__ = [
    # Models
    "TradeEvent",
    "MarketBar",
    "ExperimentResult",
    "DatasetConfig",
    "DatasetMetadata",
    "SimulatorConfig",
    "SimulatorResult",
    # Core components
    "EventLog",
    "MarketLog",
    "DatasetBuilder",
    "Simulator",
    "StrategyBase",
    # Experiment
    "Experiment",
    "ExperimentStatus",
    "ExperimentRunner",
    # Labels & ML
    "LabelConfig",
    "LabelsBuilder",
    "create_ml_dataset",
    # AlphaEngine
    "AlphaConfig",
    "AlphaEngine",
    "TrainResult",
    # Comparison
    "StrategyComparison",
    "ComparisonResult",
    "run_multi_period_comparison",
]

