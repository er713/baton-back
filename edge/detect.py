from ..connect import CamInfo, from_camera, BACKGROUNDS
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import requests

BACKEND_URL = "http://localhost:8000/api/animal/edge"
THIS_CAMERA = CamInfo(
    "edge", None, "prefix_", "7f05cfff-532d-4dd3-af3b-cd0696ea4f28"
)
TMP_FILE = "tmp.png"


if __name__ == "__main__":
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    BACKGROUNDS[THIS_CAMERA] = camera.captre(rawCapture, format="rgb")
    # capture = cv2.VideoCapture(camera)
    while True:
        frame = camera.capture(rawCapture, format="rgb")
        results = from_camera(THIS_CAMERA, frame)
        if not results:
            time.sleep(0.1)
            continue
        cv2.imwrite(TMP_FILE, frame)
        for result in results.items():
            with open(TMP_FILE, "rb") as fd:
                requests.post(
                    BACKEND_URL,
                    json={
                        "class": result[0],
                        "confidence": result[1],
                        "uuid": THIS_CAMERA.id,
                        "frame": fd.write(),
                    },
                )
