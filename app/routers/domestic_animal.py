import uuid
import datetime
import json
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response, UploadFile

from ..db import models, schemas
from ..db.database import get_db
from ..utils import prepare_image, save_image

# from app.oauth2 import require_user

DOMESTIC_ANIMALS = "{dog, cat}"
router = APIRouter()


@router.get("/")
async def get_domestic_detections(
    db: Session = Depends(get_db),
    limit: int = 100,
    skip: int = 0,
    search: str = "",
):
    print()
    detections = (
        db.execute(
            text(
                f"SELECT detections.uuid as uuid, detected_animal, confidence, detection_ts, cameras.coordinates as coordinates, resolved FROM detections JOIN cameras ON detections.camera_id=cameras.uuid WHERE detected_animal = ANY('{DOMESTIC_ANIMALS}') AND resolved = false"
            )
        )
        .mappings()
        .all()
    )
    results = []
    for row in detections:
        dict_row = dict(row)
        print("Ssample row:", dict_row)
        new_row = {
            "uuid": str(dict_row["uuid"]),
            "detectedAnimal": dict_row["detected_animal"],
            "confidence": dict_row["confidence"],
            "timestamp": datetime.datetime.isoformat(dict_row["detection_ts"]),
            "localization": dict_row["coordinates"],
        }
        image_path = "./examples/" + new_row["uuid"] + "_data.png"
        new_row |= prepare_image(image_path)
        results.append(dict(new_row))
    return results


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
