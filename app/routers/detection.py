from datetime import datetime
import uuid

from ..db import models
from ..db import schemas
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response
from ..db.database import get_db

# from app.oauth2 import require_user


router = APIRouter()


@router.get("/")
async def get_detections(
    db: Session = Depends(get_db),
    limit: int = 100,
    skip: int = 0,
    search: str = "",
    # user_id: str = "",  # Depends(require_user),
):
    detections = (
        db.query(models.Detection)
        .filter(models.Detection.detected_animal.contains(search))
        .limit(limit)
        .offset(skip)
        .all()
    )

    return {"status": "success", "results": len(detections), "detections": detections}
