"""
UI Router for AGI-Brain.

Serves the minimal HTML interface.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Setup templates
BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/ui", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Render the main UI."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "AGI-Brain v1.2",
        }
    )
