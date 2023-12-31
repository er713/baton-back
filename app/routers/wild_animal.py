import uuid
from datetime import datetime
import cv2

from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import (
    Depends,
    HTTPException,
    status,
    APIRouter,
    Response,
    UploadFile,
    Request,
)

from ..db import models, schemas
from ..db.database import get_db
from ..utils import save_image
from ...connect import from_user

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


@router.post("/edge")
async def post_edge_submit(submit: Request, db: Session = Depends(get_db)):
    body = await submit.json()
    report_ts = datetime.now()
    submit_uuid = uuid.uuid4()
    frame = body["frame"].split(",", 1)[1]
    if frame:
        print("Saving image locally")
        save_image(
            filename=f"./examples/{submit_uuid}_data.png",
            in_file=frame,
            save_size=(1280, 720),
        )
    db.execute(
        text(
            f"INSERT INTO detections (uuid, detected_animal, confidence, detection_ts, camera_id) VALUES (uuid_generate_v4(), {body['class']}, {body['confidence']}, {datetime.isoformat(report_ts)}, {body['uuid']})"
        )
    )
    db.commit()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def post_submit(
    submit: Request,
    # in_file: UploadFile = None,
    db: Session = Depends(get_db),
):
    """TODO: read the image from the frame_path (base64 png)
    (filename: UUID_data.png)
    """
    body = await submit.json()
    print(body)
    # print(type(submit["frame"]))
    # print(submit["frame"])
    # breakpoint()
    frame = body["frame"].split(",", 1)[1]
    print("received post request")
    report_ts = datetime.now()
    submit_uuid = uuid.uuid4()
    if frame:
        print("Saving image locally")
        save_image(
            filename=f"./examples/{submit_uuid}_data.png",
            in_file=frame,
            save_size=(1280, 720),
        )
    print("Submitting to database")
    db_submit = models.Submit(
        uuid=submit_uuid,
        coordinates=body["localization"],
        report_ts=report_ts,
    )

    db.add(db_submit)
    db.commit()
    db.refresh(db_submit)

    print("submit added to database")
    results = from_user(cv2.imread(f"./examples/{submit_uuid}_data.png"))
    if not len(results):
        return status.HTTP_406_NOT_ACCEPTABLE
    return {"detectedAnimal": list(results.keys())[0]}
