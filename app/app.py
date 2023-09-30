from fastapi import FastAPI, Depends

from fastapi.middleware.cors import CORSMiddleware
from fastapi_sqlalchemy import DBSessionMiddleware, db
from app.config import settings

from app.routers import user, submit, detection, camera
from app.db.database import Base, engine, get_db, Session

# import uvicorn

app = FastAPI()

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
app.include_router(submit.router, tags=["Submits"], prefix="/api/submits")
app.include_router(camera.router, tags=["Cameras"], prefix="/api/cameras")
app.include_router(detection.router, tags=["Detections"], prefix="/api/detections")


@app.get("/api/healthcheck")
def read_root():
    return {"Hello": "WildCity"}
