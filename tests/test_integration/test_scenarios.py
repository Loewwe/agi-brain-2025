import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c

@pytest.mark.asyncio
async def test_scenario_trading_risk_review(client):
    """
    Scenario 1: Trading Risk Review
    """
    response = client.post("/v1/orchestrate", json={
        "goal": "Разбери последние 7 дней трейдинга, оцени риск и предложи изменения UNSINKABLE.",
        "client_id": "owner",
        "dry_run": False
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["plan_id"] is not None
    assert len(data["steps"]) > 0
    assert data["result_summary"] is not None
    
    tools_used = [s["tool"] for s in data["steps"]]
    assert "world.snapshot" in tools_used or "trading.analyze_risk" in tools_used

@pytest.mark.asyncio
async def test_scenario_agi_project_status(client):
    """
    Scenario 2: AGI-Project Status
    """
    response = client.post("/v1/orchestrate", json={
        "goal": "Собери сводный статус AGI-проекта по логам/докам и сделай краткий отчёт.",
        "client_id": "owner"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["steps"]) > 0
    tools_used = [s["tool"] for s in data["steps"]]
    
    assert any(t in tools_used for t in ["world.snapshot", "reports.build_status_report", "memory.search"])

@pytest.mark.asyncio
async def test_scenario_code_docs_insight(client):
    """
    Scenario 3: Code/Docs Insight
    """
    response = client.post("/v1/orchestrate", json={
        "goal": "Объясни, как устроен AGI-Brain v1 и какие у него ограничения по безопасности.",
        "client_id": "owner"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["steps"]) > 0
    tools_used = [s["tool"] for s in data["steps"]]
    
    assert "memory.search" in tools_used or "world.snapshot" in tools_used
