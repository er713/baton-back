from datetime import datetime
import uuid as uid
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response

from ..db import models
from ..db import schemas
from ..db.database import get_db

# from app.oauth2 import require_user


router = APIRouter()


@router.get("/")
async def get_users(
    db: Session = Depends(get_db),
    limit: int = 100,
    skip: int = 0,
    search: str = "",
):
    users = db.execute(text("SELECT * FROM users")).mappings().all()
    # print(users)
    return {"status": "success", "count": len(users), "results": users}
