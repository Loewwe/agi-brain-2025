"""Live Metrics Reader — Ingest Stage 6 live performance data."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

# Stage 6 logs path
STAGE6_LOGS_PATH = Path("/root/Alien.agi/agi_brain/data/trading_logs")
TRADES_LOG = STAGE6_LOGS_PATH / "trades.jsonl"
PNL_LOG = STAGE6_LOGS_PATH / "daily_pnl.jsonl"


@dataclass
class LiveMetrics:
    strategy_id: str
    symbol: str
    timeframe: str
    pnl_7d: float = 0.0
    pnl_30d: float = 0.0
    trades_7d: int = 0
    trades_30d: int = 0
    max_dd_7d: float = 0.0
    win_rate_7d: Optional[float] = None
    last_trade_ts: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


def _parse_date(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s[:19])
    except:
        return datetime.min


def read_trades_log(days: int = 30) -> List[dict]:
    """Read recent trades from log."""
    if not TRADES_LOG.exists():
        return []
    
    cutoff = datetime.now() - timedelta(days=days)
    trades = []
    
    try:
        with open(TRADES_LOG) as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    ts = _parse_date(trade.get("ts", ""))
                    if ts >= cutoff:
                        trades.append(trade)
                except:
                    continue
    except:
        pass
    
    return trades


def get_live_metrics(strategy_id: str, symbol: str = "BTCUSDT", timeframe: str = "15m") -> LiveMetrics:
    """Get live metrics for a strategy from Stage 6 logs."""
    trades = read_trades_log(days=30)
    
    # Filter by strategy if available in logs
    relevant = [t for t in trades if t.get("symbol") == symbol]
    
    now = datetime.now()
    cutoff_7d = now - timedelta(days=7)
    
    trades_7d = [t for t in relevant if _parse_date(t.get("ts", "")) >= cutoff_7d]
    trades_30d = relevant
    
    pnl_7d = sum(t.get("pnl", 0) for t in trades_7d)
    pnl_30d = sum(t.get("pnl", 0) for t in trades_30d)
    
    wins_7d = sum(1 for t in trades_7d if t.get("pnl", 0) > 0)
    win_rate = wins_7d / len(trades_7d) if trades_7d else None
    
    last_trade_ts = max((t.get("ts") for t in relevant), default=None) if relevant else None
    
    return LiveMetrics(
        strategy_id=strategy_id,
        symbol=symbol,
        timeframe=timeframe,
        pnl_7d=round(pnl_7d, 2),
        pnl_30d=round(pnl_30d, 2),
        trades_7d=len(trades_7d),
        trades_30d=len(trades_30d),
        win_rate_7d=round(win_rate, 3) if win_rate else None,
        last_trade_ts=last_trade_ts
    )


def get_all_live_metrics() -> Dict[str, LiveMetrics]:
    """Get live metrics for all symbols."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    result = {}
    
    for sym in symbols:
        key = sym.lower()
        result[key] = get_live_metrics("stage6", sym)
    
    return result
