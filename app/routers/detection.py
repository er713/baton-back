import datetime 
import uuid as uid

from app.utils import prepare_image

from ..db import models, schemas
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response
from ..db.database import get_db

# from app.oauth2 import require_user


DOMESTIC_ANIMALS = ["dog", "cat"]
router = APIRouter()


@router.get("/")
async def get_detections(
    db: Session = Depends(get_db),
    limit: int = 100,
    skip: int = 0,
    search: str = "",
):
    detections = db.execute(text(f"SELECT * FROM detections WHERE detected_animal IN {DOMESTIC_ANIMALS}")).mappings().all()
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
    return detections


@router.post("/")
async def post_detection(detection, db: Session = Depends(get_db)):
    db_detection = models.Detection(
        uuid=uid.uuid4(),
        detected_animal=detection["detectedAnimal"],
        detection_ts=datetime.now(),
        confidence=detection["confidence"],
        camera_id=detection["cameraId"] or None,
        submit_id=detection["submitId"] or None,
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return {"status": "success", "detection": db_detection.uuid}


