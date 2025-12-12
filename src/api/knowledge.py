"""
Knowledge Base API.

Endpoints for managing and retrieving knowledge sections.
"""

from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["knowledge"])

class KnowledgeSection(BaseModel):
    id: str
    title: str
    description: str
    icon: str | None = None

# Static sections for v1.1
SECTIONS = [
    {
        "id": "trading",
        "title": "Trading & Risk",
        "description": "Trading strategies, risk management, market analysis",
        "icon": "chart-line"
    },
    {
        "id": "agi_core",
        "title": "AGI Core",
        "description": "Brain architecture, memory systems, planning logic",
        "icon": "brain"
    },
    {
        "id": "math_physics",
        "title": "Math & Physics",
        "description": "Mathematical models, physical simulations",
        "icon": "calculator"
    },
    {
        "id": "coding",
        "title": "Coding & DevOps",
        "description": "Software engineering, deployment, infrastructure",
        "icon": "code"
    }
]

@router.get("/knowledge_sections", response_model=list[KnowledgeSection])
async def list_knowledge_sections() -> list[KnowledgeSection]:
    """Get available knowledge sections."""
    return [KnowledgeSection(**s) for s in SECTIONS]
