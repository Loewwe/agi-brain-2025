import pytest
from pathlib import Path
from src.perception.trading_adapter import FixtureTradingAdapter
from src.world.model import Trade

@pytest.mark.asyncio
async def test_trading_contract_ok_fixture():
    """Verify loading OK fixture matches Pydantic contract."""
    base_path = Path(__file__).parent.parent.parent
    fixture_path = base_path / "fixtures" / "trading" / "sample_week_ok.json"
    
    adapter = FixtureTradingAdapter(
        trades_path=str(fixture_path)
    )
    
    trades = await adapter.fetch_history()
    assert len(trades) == 2
    assert isinstance(trades[0], Trade)
    assert trades[0].symbol == "BTCUSDT"
    assert trades[0].realized_pnl > 0

@pytest.mark.asyncio
async def test_trading_contract_risky_fixture():
    """Verify loading Risky fixture matches Pydantic contract."""
    base_path = Path(__file__).parent.parent.parent
    fixture_path = base_path / "fixtures" / "trading" / "sample_week_risky.json"
    
    adapter = FixtureTradingAdapter(
        trades_path=str(fixture_path)
    )
    
    trades = await adapter.fetch_history()
    assert len(trades) == 2
    assert isinstance(trades[0], Trade)
    assert trades[0].realized_pnl < 0
    # Check strict types
    assert isinstance(trades[0].realized_pnl_percent, float)
