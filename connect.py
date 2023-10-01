import asyncio
import httpx
import time
import os
import cv2
import imutils
import torch
import numpy as np
from pathlib import Path
from sqlalchemy.sql import text
from datetime import datetime

from typing import List, Dict, Tuple, Callable
from collections import namedtuple

from kenning.runtimes.onnx import ONNXRuntime
from kenning.modelwrappers.detectors.yolov4 import ONNXYOLOV4
from kenning.datasets.helpers.detection_and_segmentation import DetectObject
from mobilenet_v3 import PyTorchMobileNetV3

# CAMERAS
CamInfo = namedtuple("CamInfo", ["name", "url", "file_prefix", "id"])
PLAC = CamInfo(
    "poznan",
    "https://hoktastream1.webcamera.pl/czarnygron_cam_961902/czarnygron_cam_961902.stream/",  # noqa: E501
    "poz_",
)
POZN = CamInfo(
    "targ",
    "https://hoktastream2.webcamera.pl/poznan_cam_a2ec1b/poznan_cam_a2ec1b.stream/",  # noqa: E501
    "targ_",
)
TARG = CamInfo(
    "plac",
    "https://hstream1.webcamera.pl/nowytarg_cam_0f85a4/nowytarg_cam_0f85a4.stream/",  # noqa: E501
    "plac_",
)


START_SESSION = "playlist.m3u8"
COMMON_HAEDERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "hoktastream1.webcamera.pl",
    "Origin": "https://player.webcamera.pl",
    "Referer": "https://player.webcamera.pl/",
}
WAIT_DURATION = 1
IMG_DIR = Path("imgs")

QUEUE: List[cv2.UMat] = []
ALPHA = 0.01
BKG_FROM_N = 1
GB_KERNEL_SIZE = 21
BACKGROUNDS: Dict[CamInfo, cv2.typing.MatLike] = {}

MOBILENET_PATH = Path("build/mbnet.pth")
MOBILENET_MODEL = PyTorchMobileNetV3(MOBILENET_PATH, None)
CONFIDENCE_THRESHOLD = 0.6

YOLO_PATH = Path("build/yolo.onnx")
YOLO_MODEL = ONNXYOLOV4(YOLO_PATH, None)
YOLO_RUNTIME = ONNXRuntime(
    None, YOLO_PATH, disable_performance_measurements=True
)
YOLO_RUNTIME.prepare_model(None)

THRESHOLDS = [117, 120, 124, 129, 132, 133, 134]
CLASSES = ["dog", "wolf", "fox", "cat", "hare", "boar", "deer"]


async def process_queue(_func: Callable):
    while True:
        if not QUEUE:
            print("empty queue, waiting...")
            await asyncio.sleep(0.5)
            continue
        _func(QUEUE.pop(0))


def preprocess_image_cv(image: cv2.UMat):
    return cv2.GaussianBlur(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
        (GB_KERNEL_SIZE, GB_KERNEL_SIZE),
        0,
    )


def detect_motion(
    camera: CamInfo, frame: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    background = BACKGROUNDS[camera]
    print(frame)
    cv2.imwrite("tmp/test_frame.png", frame)
    frame_gray = preprocess_image_cv(frame)
    cv2.imwrite("tmp/test_frame_gray.png", frame_gray)
    cv2.imwrite("tmp/test_back_prev.png", background)
    diff = cv2.absdiff(background, frame_gray)
    motion_mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)[1]
    # motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, None)
    BACKGROUNDS[camera] = np.uint8(
        frame_gray * ALPHA + background * (1 - ALPHA)
    )
    cv2.imwrite(
        f"{IMG_DIR}/{camera.file_prefix}test_motion{time.time()}.png",
        motion_mask,
    )
    cv2.imwrite("tmp/test_background.png", background)
    return motion_mask


def get_moving_objects(
    camera: CamInfo, frame: cv2.UMat, mask: cv2.UMat
) -> List[cv2.UMat]:
    conturs = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    conturs = imutils.grab_contours(conturs)
    for contur in conturs:
        if cv2.contourArea(contur) < 0.2:
            continue
        (x, y, w, h) = cv2.boundingRect(contur)
        cv2.imwrite(str(IMG_DIR / Path()))
        yield (x, y, w, h)


def get_frame_part(frame: cv2.UMat, rect: Tuple[int]) -> cv2.UMat:
    x, y, w, h = rect
    middle = (x + w // 2, y + h // 2)
    a = max(w, h)
    if middle[0] - a // 2 < 0:
        x = 0
    elif middle[0] + a // 2 >= frame.shape[0]:
        x = frame.shape[0] - a - 1
    else:
        x = middle[0] - a // 2
    if middle[1] - a // 2 < 0:
        y = 0
    elif middle[1] + a // 2 >= frame.shape[1]:
        y = frame.shape[1] - a - 1
    else:
        y = middle[1] - a // 2
    return frame[x : x + a, y : y + a]


async def download_messages(camera: CamInfo, db):
    last_downloaded = [None] * 20
    async with httpx.AsyncClient() as client:
        start_response = await client.get(
            camera.url + START_SESSION, headers=COMMON_HAEDERS
        )
        while start_response.status_code != 200:
            await asyncio.sleep(0.5)
            print("Waiting for start of the session")
            print(start_response.content)
            start_response = await client.get(
                camera.url + START_SESSION, headers=COMMON_HAEDERS
            )
        print(start_response.content.splitlines())
        chunk_sufix = [
            chunk
            for chunk in start_response.content.splitlines()
            if not chunk.startswith(b"#")
        ][0]
        print("CHUNK:", chunk_sufix)
        while True:
            parts = await client.get(
                camera.url + chunk_sufix.decode("ascii"),
                headers=COMMON_HAEDERS,
            )
            parts_sufix = [
                p for p in parts.content.splitlines() if not p.startswith(b"#")
            ]
            for sufix in parts_sufix:
                if sufix in last_downloaded:
                    continue
                _sufix = sufix.decode("ascii")
                video = await client.get(
                    camera.url + _sufix, headers=COMMON_HAEDERS
                )
                if video.status_code != 200:
                    print("error when video")
                    print(video.content)
                    continue
                print(f"Chunk {_sufix.split('?', 1)[0]} downloaded")
                file = (
                    f"vids/{camera.file_prefix}vid_{_sufix.split('?', 1)[0]}"
                )
                image = IMG_DIR / Path(f'{file.split("/", 1)[1]}').with_suffix(
                    ".png"
                )
                with open(file, "wb") as fd:
                    fd.write(video.content)
                print(f"Saved to file {file}")
                status = os.system(
                    f"ffmpeg -n -i {file} -frames:v 1 {image}",
                )
                os.remove(file)
                if status != 0:
                    print("Cannot export to frame")
                    continue
                else:
                    print(f"Image saved to {image}")
                    last_downloaded.pop(0)
                    last_downloaded.append(sufix)
                img_cv = cv2.imread(str(image))
                if camera not in BACKGROUNDS:
                    BACKGROUNDS[camera] = preprocess_image_cv(img_cv)
                    continue
                else:
                    print("@@" * 10, image)
                    results = from_camera(camera, img_cv)
                    if len(results):
                        for result in results.items():
                            db.execute(
                                text(
                                    f"INSERT INTO detections (uuid, detected_animal, confidence, detection_ts, camera_id) VALUES (uuid_generate_v4(), {result[0]}, {result[1]}, {datetime.isoformat(datetime.now())}, {camera.id})"
                                )
                            )
                        db.commit()
                    # return
                os.remove(image)
                print("==" * 10)
                time.sleep(WAIT_DURATION)


def class_to_name(number: int) -> str:
    for thresh, name in zip(THRESHOLDS, CLASSES):
        if number <= thresh:
            return name


def add_boarder(frame: cv2.UMat, rect: Tuple[int]):
    (x, y, w, h) = rect
    return cv2.rectangle(frame, ((x, y), (x + w, y + h)), (0, 255, 0), 2)


def classify_moving_object(frame: cv2.UMat, model: PyTorchMobileNetV3):
    model.prepare_model()
    prepared_frame = cv2.resize(frame, (224, 224))
    prepared_frame = np.float32(np.asarray([prepared_frame])) / 255.0
    print("@@" * 10)
    print(prepared_frame.shape)
    if prepared_frame.shape[-1] == 3:
        prepared_frame = np.transpose(prepared_frame, (0, 3, 1, 2))
    prepared_frame = model.preprocess_input(prepared_frame)
    prediction = model.run_inference(prepared_frame)
    _class = torch.argmax(prediction)
    print(prediction.shape, _class)
    confidence = torch.softmax(prediction, -1)[0, _class]
    if confidence >= CONFIDENCE_THRESHOLD:
        return class_to_name(int(_class.cpu())), confidence
    else:
        return None, confidence


def detect_objects(frame: cv2.UMat) -> List[DetectObject]:
    # frame = np.transpose(np.asarray(frame), (2, 0, 1))
    frame = cv2.resize(frame, (416, 416))
    prepX = YOLO_MODEL.preprocess_input([frame])
    print(prepX.shape)
    from pprint import pprint

    pprint(YOLO_MODEL.perlayerparams)
    print("@@" * 5)
    print(prepX.shape)
    YOLO_RUNTIME.input = {}
    YOLO_RUNTIME.input["input_1:0"] = np.float32(prepX) / 255.0
    # YOLO_MODEL.perlayerparams["mask"]
    # print("@@@@@@@@@")x
    YOLO_RUNTIME._run()
    preds = YOLO_RUNTIME.extract_output()
    return YOLO_MODEL.postprocess_outputs(preds)


def postprocess_results(results: Dict, person_detected: bool) -> Dict:
    if person_detected:
        for _cls in ("cat", "dog"):
            if _cls in results:
                del results[_cls]
    return results


def from_user(frame: cv2.UMat):
    bboxes = detect_objects(frame)
    person_detected = False
    results = {}
    for bbox in bboxes[0]:
        if bbox.clsname == "person":
            person_detected = True
        elif bbox.clsname in ("cat", "dog", "horse", "sheep", "cow", "bear"):
            if (
                int(bbox.xmax) - int(bbox.xmin) < 20
                or int(bbox.ymax) - int(bbox.ymin) < 20
            ):
                continue
            frame_part = frame[
                int(bbox.xmin) : int(bbox.xmax),
                int(bbox.ymin) : int(bbox.ymax),
            ]
            _cls, conf = classify_moving_object(frame_part, MOBILENET_MODEL)
            if _cls:
                results[_cls] = conf
    _cls, conf = classify_moving_object(frame, MOBILENET_MODEL)
    print(conf)
    if _cls:
        results[_cls] = conf
    results = postprocess_results(results, person_detected)
    return results


def from_camera(camera: CamInfo, frame: cv2.UMat):
    motion_mask = detect_motion(camera, frame)
    frame_parts = get_moving_objects(camera, frame, motion_mask)
    person_detected = False
    results = {}
    for part in frame_parts:
        _cls, conf = classify_moving_object(part, MOBILENET_MODEL)
        if _cls:
            results[_cls] = conf
        else:
            bboxes = detect_objects(part)
            if [bbox for bbox in bboxes if bbox.clsname == "person"]:
                person_detected = True
    return postprocess_results(results, person_detected)


def main():
    loop = asyncio.get_event_loop()
    _all = asyncio.gather(
        download_messages(TARG),
        process_queue(lambda x: classify_moving_object(x, MOBILENET_MODEL)),
    )
    loop.run_until_complete(_all)
    loop.close()


if __name__ == "__main__":
    main()
