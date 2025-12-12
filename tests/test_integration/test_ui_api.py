"""
Integration tests for v1.2 UI and API endpoints.
"""

import pytest
from starlette.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_ui_endpoint(client):
    """Test that /ui returns HTML."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>AGI-Brain v1.2</title>" in response.text


def test_hard_cases_endpoint(client):
    """Test /v1/hard_cases endpoint."""
    response = client.get("/v1/hard_cases/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Even if empty, it should be a list


@pytest.mark.asyncio
async def test_trading_preview_endpoint(client):
    """Test /v1/trading/preview endpoint."""
    response = client.get("/v1/trading/preview")
    assert response.status_code == 200
    data = response.json()
    
    assert "pnl_metrics" in data
    assert "risk_status" in data
    assert "active_positions" in data
    
    pnl = data["pnl_metrics"]
    assert pnl is not None
    assert "pnl_today_percent" in pnl
    assert "total_balance_usd" in pnl
