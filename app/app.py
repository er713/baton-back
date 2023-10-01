from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi_socketio import SocketManager
from socketio import AsyncServer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from app.config import settings
from sqlalchemy.sql import text
import datetime

from .utils import prepare_image
from .socket_handler import ConnectionManager
from app.routers import user, wild_animal, domestic_animal, detection, camera
from app.db.database import Base, engine, get_db, Session


app = FastAPI()
manager = ConnectionManager()


origins = [
    settings.CLIENT_ORIGIN,
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user.router, tags=["Users"], prefix="/api/users")
app.include_router(wild_animal.router, tags=["WildAnimals"], prefix="/api/wild-animal")
app.include_router(
    domestic_animal.router, tags=["DomesticAnimals"], prefix="/api/domestic-animal"
)
app.include_router(camera.router, tags=["Cameras"], prefix="/api/cameras")
app.include_router(detection.router, tags=["Detections"], prefix="/api/detections")


@app.websocket("/ws")
async def accept_client(websocket: WebSocket, db: Session = Depends(get_db)):
    await manager.connect(websocket)
    print("\twebsocket accepted")
    try:
        while True:
            data = await websocket.receive_text()
            if data != 'Connected':
                print("data is not None:", data)
                db.execute(
                    text("UPDATE detections SET resolved = true WHERE uuid = :uuid"),
                    {"uuid": data},
                )
            print("\n\n\n")
            print(data)
            print(type(data), data)
            print("\n\n\n")
            detections_to_send = (
                db.execute(
                    text(
                        "SELECT detections.uuid as uuid, detected_animal, confidence, detection_ts, cameras.coordinates as coordinates, resolved FROM detections JOIN cameras ON detections.camera_id=cameras.uuid WHERE resolved = false;"
                    )
                )
                .mappings()
                .all()
            )

            results = []
            for row in detections_to_send:
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
                new_row["frame"] = new_row["frame"].decode("utf-8")
                results.append(dict(new_row))
            print("sending message")
            await manager.send_personal_message(results, websocket)
            # await manager.broadcast(f"Client #{1} says: {data}")
            # await websocket.send_json({"detections": ["detection1", "detection2"]})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{1} left the chat")
        # await websocket.send_text(f"Accepted: {token}")


@app.websocket("/ws/resolve-detection")
async def resolve_detection(websocket: WebSocket):
    await websocket.accept()
    print("websocket accepted")
    while True:
        data = await websocket.receive_text()
        print("\n\n\n")
        print(type(data), data, data.split("token")[-1])
        print("\n\n\n")
        await websocket.send_json({"detections": ["detection1", "detection2"]})
        # await websocket.send_text(f"Accepted: {token}")


# TODO: update detections

# async def get_cookie_or_token(
#     websocket: WebSocket,
#     session: Annotated[str | None, Cookie()] = None,
#     token: Annotated[str | None, Query()] = None,
# ):
#     if session is None and token is None:
#         raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
#     return session or token


@app.get("/api/healthcheck")
def read_root():
    return {"Hello": "WildCity"}


@app.get("/api/create_tables")
def create_table(db: Session = Depends(get_db)):
    Base.metadata.create_all(bind=engine)
    db.commit()
    return {"status": "success"}
