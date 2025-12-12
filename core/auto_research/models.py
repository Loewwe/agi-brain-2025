"""Auto Research Models — Using dataclasses for compatibility."""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class StrategyConfig:
    strategy_id: str
    label: str
    engine: str
    symbols: List[str]
    timeframes: List[str]
    mode: str = "research"
    enabled: bool = True
    priority: int = 10
    params: Dict = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass
class BacktestResult:
    strategy_id: str
    engine: str
    symbol: str
    timeframe: str
    start: str
    end: str
    date_run: str = ""
    trades: int = 0
    pnl_abs: float = 0.0
    pnl_pct: float = 0.0
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_dd_pct: float = 0.0
    win_rate: Optional[float] = None
    avg_holding_time_min: Optional[float] = None
    rating: str = "watchlist"
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict())


@dataclass
class Policy:
    min_trades: int = 50
    min_sharpe: float = 1.5
    min_sortino: float = 1.2
    max_dd_pct: float = -5.0
    max_dd_hard: float = -15.0
    watchlist_min_sharpe: float = 1.0


@dataclass
class AutoResearchSummary:
    date: str
    total_strategies: int
    ratings_count: Dict
    top_candidates: List
