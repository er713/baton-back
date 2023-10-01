import uuid
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response, UploadFile

from ..db import models, schemas
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

    return {"status": "success", "count": len(submits), "submits": submits}


@router.post("/", status_code=status.HTTP_201_CREATED)
async def post_submit(
    submit: schemas.SubmitBaseSchema,
    in_file: UploadFile = None,
    db: Session = Depends(get_db),
):
    """TODO: read the image from the frame_path (base64 png)
    (filename: UUID_data.png)
    """
    print("received post request")
    report_ts = datetime.now()
    submit_uuid = uuid.uuid4()
    if in_file:
        print("Saving image locally")
        save_image(
            filename=f"./examples/{submit_uuid}_data.png",
            file=in_file,
            size=(1280, 720),
        )
    print("Submitting to database")
    db_submit = models.Submit(
        uuid=submit_uuid,
        coordinates=submit.coordinates,
        reported_animal=submit.reported_animal,
        report_ts=report_ts,
    )

    db.add(db_submit)
    db.commit()
    db.refresh(db_submit)

    print("submit added to database")
    failed = False  # TODO: ai processing result (True or False) with detections results
    if failed:
        return status.HTTP_406_NOT_ACCEPTABLE
    return status.HTTP_202_ACCEPTED