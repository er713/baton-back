from datetime import datetime
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response, UploadFile
from fastapi.responses import FileResponse

from ..db import models
from ..db import schemas
from ..db.database import get_db
from ..utils import save_image

# from app.oauth2 import require_user


router = APIRouter()


@router.get("/")
async def get_submits(
    db: Session = Depends(get_db),
    limit: int = 100,
    skip: int = 0,
    search: str = "",
):
    submits = db.execute(text("SELECT * FROM submits")).mappings().all()
    """ TODO: read the image from the frame_path (base64 png) 
     (filename: UUID_data.png)
     """

    # for submit in submits:
    #     submit["frame"] = FileResponse(submit.frame_path)

    return {"status": "success", "results": len(submits), "submits": submits}
