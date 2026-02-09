from fastapi import APIRouter, HTTPException

from src.api.utils.history import history_manager

router = APIRouter()


@router.get("/recent")
async def get_recent_history(limit: int = 10, type: str | None = None):
    return history_manager.get_recent(limit, type)


@router.get("/{entry_id}")
async def get_history_entry(entry_id: str):
    entry = history_manager.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry
