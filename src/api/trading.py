"""
Trading API.

Exposes trading data and previews.
"""

from fastapi import APIRouter

from ..world.snapshot_builder import get_snapshot_builder

router = APIRouter(prefix="/v1/trading", tags=["trading"])


@router.get("/preview")
async def get_trading_preview():
    """Get trading preview snapshot."""
    # Re-use the global snapshot builder
    # Note: In a real app we might want a dedicated service, 
    # but for now we use the builder configured in main.py
    builder = get_snapshot_builder()
    
    # Force a fresh build or just get the last state?
    # Let's build a fresh one to see live updates from the adapter
    snapshot = await builder.build()
    
    if not snapshot.trading:
        return {
            "pnl_metrics": None,
            "risk_status": None,
            "active_positions": [],
            "last_update": snapshot.timestamp,
        }

    agent = snapshot.trading.agent
    return {
        "pnl_metrics": agent.pnl,
        "risk_status": {
            "score": snapshot.trading.current_risk_score,
            "violations": snapshot.trading.risk_violations_active,
        },
        "active_positions": agent.current_positions,
        "last_update": snapshot.timestamp,
    }
