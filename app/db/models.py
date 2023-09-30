import uuid as uid
from .database import Base
from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    Column,
    ForeignKey,
    Integer,
    String,
    Boolean,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime


class User(Base):
    __tablename__ = "users"
    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uid.uuid4)
    username = Column(String(64), nullable=False, unique=True)
    password = Column(String(64), nullable=False)
    email = Column(String(128), nullable=False, unique=True)
    # role = Column(String(64), nullable=False)
    # verified = Column(Boolean, nullable=False, default=False)


class Detection(Base):
    __tablename__ = "detections"
    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uid.uuid4)
    detected_animal = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    # frame_path = Column(String(255), nullable=False)
    detection_ts = Column(TIMESTAMP, nullable=False)
    camera_id = Column(UUID, ForeignKey("cameras.uuid"))
    submit_id = Column(Integer, ForeignKey("submits.id"))

    camera = relationship("Camera")
    submit = relationship("Submit")


class Submit(Base):
    __tablename__ = "submits"
    id = Column(Integer, primary_key=True, index=True)
    coordinates = Column(ARRAY(Float), nullable=False)
    # frame_path = Column(String(255), nullable=True)
    reported_animal = Column(String(255), nullable=True)
    report_ts = Column(TIMESTAMP, nullable=False)


class Camera(Base):
    __tablename__ = "cameras"
    uuid = Column(UUID, primary_key=True, default=uid.uuid4)
    coordinates = Column(ARRAY(Float), nullable=False)
    address = Column(String(255), nullable=True)
    active = Column(Boolean, nullable=False)
    url = Column(String(255), nullable=False)
