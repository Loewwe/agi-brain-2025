"""
Core data models for Stage7 Experiment system.

All models are frozen dataclasses for immutability and determinism.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
import hashlib
import json


# =============================================================================
# ENUMS
# =============================================================================

class TradeAction(str, Enum):
    """Trade action type."""
    OPEN = "open"
    CLOSE = "close"
    PARTIAL_CLOSE = "partial_close"


class ExitReason(str, Enum):
    """Exit reason for closed trades."""
    TP = "TP"
    SL = "SL"
    TRAILING = "trailing"
    TIMEOUT = "timeout"
    DAILY_STOP = "daily_stop"
    BREAKEVEN = "breakeven"
    SIGNAL_REVERSAL = "signal_reversal"
    MANUAL = "manual"


class DataSource(str, Enum):
    """Source of trade/market data."""
    STAGE6 = "stage6"
    STAGE7 = "stage7"
    BACKTEST = "backtest"


# =============================================================================
# TRADE EVENT
# =============================================================================

@dataclass(frozen=True)
class TradeEvent:
    """
    Unified trade event format for Stage6/Stage7/Backtest.
    
    Immutable, deterministic, and serializable.
    """
    # Event metadata
    event_id: str
    timestamp: datetime  # UTC
    source: DataSource
    
    # Trade info
    symbol: str
    side: Literal["long", "short"]
    action: TradeAction
    
    # Prices & sizes
    price: Decimal
    size: Decimal            # Base currency
    size_quote: Decimal      # Quote currency (USDT)
    
    # Position context (None for open, filled for close)
    entry_price: Decimal | None = None
    exit_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    leverage: int = 1
    
    # P&L (None for open, filled for close)
    realized_pnl: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    r_multiple: float | None = None
    
    # Exit reason (only for close/partial_close)
    exit_reason: ExitReason | None = None
    
    # IDs for tracking
    trade_id: str = ""
    position_id: str = ""
    strategy_id: str = ""
    agent_id: str | None = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action.value,
            "price": str(self.price),
            "size": str(self.size),
            "size_quote": str(self.size_quote),
            "entry_price": str(self.entry_price) if self.entry_price else None,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "leverage": self.leverage,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl else None,
            "r_multiple": self.r_multiple,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "trade_id": self.trade_id,
            "position_id": self.position_id,
            "strategy_id": self.strategy_id,
            "agent_id": self.agent_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TradeEvent":
        """Deserialize from dictionary."""
        return cls(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=DataSource(data["source"]),
            symbol=data["symbol"],
            side=data["side"],
            action=TradeAction(data["action"]),
            price=Decimal(data["price"]),
            size=Decimal(data["size"]),
            size_quote=Decimal(data["size_quote"]),
            entry_price=Decimal(data["entry_price"]) if data.get("entry_price") else None,
            exit_price=Decimal(data["exit_price"]) if data.get("exit_price") else None,
            stop_loss=Decimal(data["stop_loss"]) if data.get("stop_loss") else None,
            take_profit=Decimal(data["take_profit"]) if data.get("take_profit") else None,
            leverage=data.get("leverage", 1),
            realized_pnl=Decimal(data["realized_pnl"]) if data.get("realized_pnl") else None,
            unrealized_pnl=Decimal(data["unrealized_pnl"]) if data.get("unrealized_pnl") else None,
            r_multiple=data.get("r_multiple"),
            exit_reason=ExitReason(data["exit_reason"]) if data.get("exit_reason") else None,
            trade_id=data.get("trade_id", ""),
            position_id=data.get("position_id", ""),
            strategy_id=data.get("strategy_id", ""),
            agent_id=data.get("agent_id"),
        )


# =============================================================================
# MARKET DATA
# =============================================================================

@dataclass(frozen=True)
class MarketBar:
    """
    OHLCV bar for market data.
    
    Immutable for determinism.
    """
    timestamp: datetime
    symbol: str
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }


# =============================================================================
# DATASET
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    symbols: list[str]
    date_from: date
    date_to: date
    timeframe: str = "5m"
    features_schema_version: str = "v1"
    mode: Literal["training", "backtest"] = "backtest"
    
    def to_dict(self) -> dict:
        return {
            "symbols": self.symbols,
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
            "timeframe": self.timeframe,
            "features_schema_version": self.features_schema_version,
            "mode": self.mode,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        return cls(
            symbols=data["symbols"],
            date_from=date.fromisoformat(data["date_from"]),
            date_to=date.fromisoformat(data["date_to"]),
            timeframe=data.get("timeframe", "5m"),
            features_schema_version=data.get("features_schema_version", "v1"),
            mode=data.get("mode", "backtest"),
        )


@dataclass
class DatasetMetadata:
    """Metadata for dataset reproducibility."""
    config: DatasetConfig
    created_at: datetime
    row_count: int
    symbols_included: list[str]
    checksum: str  # MD5 of data for verification
    
    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "symbols_included": self.symbols_included,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        return cls(
            config=DatasetConfig.from_dict(data["config"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            row_count=data["row_count"],
            symbols_included=data["symbols_included"],
            checksum=data["checksum"],
        )


# =============================================================================
# SIMULATOR
# =============================================================================

@dataclass
class SimulatorConfig:
    """Simulator configuration."""
    # Account
    start_balance: float = 200.0
    leverage: int = 20
    
    # Risk
    risk_per_trade: float = 0.009  # 0.9%
    max_positions: int = 3
    max_trades_day: int = 12
    daily_stop_pct: float = -0.01  # -1%
    max_open_risk_pct: float = 0.027  # 3 × 0.9%
    
    # Trade management
    sl_max_pct: float = 0.012      # 1.2%
    atr_multiplier: float = 1.0
    tp_multiplier: float = 2.1
    be_trigger_r: float = 1.0
    be_offset_r: float = 0.15
    partial_trigger_r: float = 1.5
    partial_pct: float = 0.5
    trailing_distance_r: float = 1.0
    timeout_bars: int = 40
    
    # Simulation
    commission: float = 0.0004     # 0.04%
    slippage_pct: float = 0.0002   # 0.02%
    random_seed: int | None = None  # For determinism
    
    # Entry filters
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    volume_surge_mult: float = 1.5
    atr_min_pct: float = 0.5
    atr_max_pct: float = 3.0
    ema_long_threshold: float = 1.004
    ema_short_threshold: float = 0.996
    ema_dead_zone_pct: float = 0.004
    excluded_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    
    def to_dict(self) -> dict:
        return {
            "start_balance": self.start_balance,
            "leverage": self.leverage,
            "risk_per_trade": self.risk_per_trade,
            "max_positions": self.max_positions,
            "max_trades_day": self.max_trades_day,
            "daily_stop_pct": self.daily_stop_pct,
            "max_open_risk_pct": self.max_open_risk_pct,
            "sl_max_pct": self.sl_max_pct,
            "atr_multiplier": self.atr_multiplier,
            "tp_multiplier": self.tp_multiplier,
            "be_trigger_r": self.be_trigger_r,
            "be_offset_r": self.be_offset_r,
            "partial_trigger_r": self.partial_trigger_r,
            "partial_pct": self.partial_pct,
            "trailing_distance_r": self.trailing_distance_r,
            "timeout_bars": self.timeout_bars,
            "commission": self.commission,
            "slippage_pct": self.slippage_pct,
            "random_seed": self.random_seed,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "volume_surge_mult": self.volume_surge_mult,
            "atr_min_pct": self.atr_min_pct,
            "atr_max_pct": self.atr_max_pct,
            "ema_long_threshold": self.ema_long_threshold,
            "ema_short_threshold": self.ema_short_threshold,
            "ema_dead_zone_pct": self.ema_dead_zone_pct,
            "excluded_hours": list(self.excluded_hours),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SimulatorConfig":
        return cls(
            start_balance=data.get("start_balance", 200.0),
            leverage=data.get("leverage", 20),
            risk_per_trade=data.get("risk_per_trade", 0.009),
            max_positions=data.get("max_positions", 3),
            max_trades_day=data.get("max_trades_day", 12),
            daily_stop_pct=data.get("daily_stop_pct", -0.01),
            max_open_risk_pct=data.get("max_open_risk_pct", 0.027),
            sl_max_pct=data.get("sl_max_pct", 0.012),
            atr_multiplier=data.get("atr_multiplier", 1.0),
            tp_multiplier=data.get("tp_multiplier", 2.1),
            be_trigger_r=data.get("be_trigger_r", 1.0),
            be_offset_r=data.get("be_offset_r", 0.15),
            partial_trigger_r=data.get("partial_trigger_r", 1.5),
            partial_pct=data.get("partial_pct", 0.5),
            trailing_distance_r=data.get("trailing_distance_r", 1.0),
            timeout_bars=data.get("timeout_bars", 40),
            commission=data.get("commission", 0.0004),
            slippage_pct=data.get("slippage_pct", 0.0002),
            random_seed=data.get("random_seed"),
            rsi_oversold=data.get("rsi_oversold", 35.0),
            rsi_overbought=data.get("rsi_overbought", 65.0),
            volume_surge_mult=data.get("volume_surge_mult", 1.5),
            atr_min_pct=data.get("atr_min_pct", 0.5),
            atr_max_pct=data.get("atr_max_pct", 3.0),
            ema_long_threshold=data.get("ema_long_threshold", 1.004),
            ema_short_threshold=data.get("ema_short_threshold", 0.996),
            ema_dead_zone_pct=data.get("ema_dead_zone_pct", 0.004),
            excluded_hours=tuple(data.get("excluded_hours", (0, 1, 2, 3, 4, 5))),
        )


# =============================================================================
# EXPERIMENT RESULT
# =============================================================================

@dataclass
class ExperimentResult:
    """Results of a completed experiment."""
    # Core metrics
    total_trades: int
    win_rate: float           # 0.0 - 1.0
    profit_factor: float
    avg_r_multiple: float
    expectancy: float         # Expected R per trade
    
    # P&L
    total_return_pct: float
    avg_daily_pnl_pct: float
    max_drawdown_pct: float   # Negative value
    
    # Risk metrics
    sharpe_ratio: float
    risk_adjusted_return: float
    
    # Trade breakdown
    wins: int = 0
    losses: int = 0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    avg_hold_minutes: float = 0.0
    
    # Exit distribution
    exits_by_reason: dict[str, int] = field(default_factory=dict)
    
    # Validation
    checksum: str = ""        # For reproducibility verification
    
    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_r_multiple": self.avg_r_multiple,
            "expectancy": self.expectancy,
            "total_return_pct": self.total_return_pct,
            "avg_daily_pnl_pct": self.avg_daily_pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_adjusted_return": self.risk_adjusted_return,
            "wins": self.wins,
            "losses": self.losses,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
            "avg_hold_minutes": self.avg_hold_minutes,
            "exits_by_reason": self.exits_by_reason,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        return cls(
            total_trades=data["total_trades"],
            win_rate=data["win_rate"],
            profit_factor=data["profit_factor"],
            avg_r_multiple=data["avg_r_multiple"],
            expectancy=data["expectancy"],
            total_return_pct=data["total_return_pct"],
            avg_daily_pnl_pct=data["avg_daily_pnl_pct"],
            max_drawdown_pct=data["max_drawdown_pct"],
            sharpe_ratio=data["sharpe_ratio"],
            risk_adjusted_return=data["risk_adjusted_return"],
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            avg_win_r=data.get("avg_win_r", 0.0),
            avg_loss_r=data.get("avg_loss_r", 0.0),
            avg_hold_minutes=data.get("avg_hold_minutes", 0.0),
            exits_by_reason=data.get("exits_by_reason", {}),
            checksum=data.get("checksum", ""),
        )


@dataclass
class SimulatorResult:
    """Complete result from a simulation run."""
    # Metrics
    metrics: ExperimentResult
    
    # Configuration used
    config: SimulatorConfig
    
    # Detailed data (optional, for analysis)
    trades_count: int = 0
    days_count: int = 0
    
    # Reproducibility
    checksum: str = ""
    
    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "config": self.config.to_dict(),
            "trades_count": self.trades_count,
            "days_count": self.days_count,
            "checksum": self.checksum,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_checksum(data: Any) -> str:
    """Calculate MD5 checksum of data for verification."""
    if isinstance(data, dict):
        json_str = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, str):
        json_str = data
    else:
        json_str = str(data)
    
    return hashlib.md5(json_str.encode()).hexdigest()
