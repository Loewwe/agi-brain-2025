"""
Hard Cases API.

Exposes hard cases for review and analysis.
"""

from fastapi import APIRouter, Query

from ..learning.hard_cases_buffer import get_hard_cases_buffer, HardCase

router = APIRouter(prefix="/v1/hard_cases", tags=["hard_cases"])


@router.get("/", response_model=list[HardCase])
async def list_hard_cases(limit: int = Query(10, ge=1, le=100)):
    """Get recent hard cases."""
    buffer = get_hard_cases_buffer()
    return buffer.get_recent(limit)
