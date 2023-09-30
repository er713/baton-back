import uuid
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response

from ..db import models, schemas
from ..db.database import get_db, Base, engine


# from app.oauth2 import require_user


router = APIRouter()


@router.get("/", response_model=schemas.CameraBaseSchema)
async def get_cameras(
    db: Session = Depends(get_db),
    limit: int = 100,
    skip: int = 0,
    search: str = "",
):
    cameras = db.execute(text("SELECT * FROM cameras")).mappings().all()

    return {"status": "success", "results": len(cameras), "cameras": cameras}
