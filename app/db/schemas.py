from pydantic import BaseModel, constr, Base64UrlBytes, Field

# from pydantic.fields import LargeBinary
from datetime import datetime
import uuid
from typing import Tuple, Literal, Any


class UserBaseSchema(BaseModel):
    uuid: uuid.UUID
    username: str
    email: str
    password: str  # constr(min_length=8, max_length=64)

    class Config:
        from_attributes = True


class CreateUserSchema(UserBaseSchema):
    password: str  # constr(min_length=8, max_length=64)
    confirm_password: str
    # role: str = "user"
    # verified: bool = False


class LoginUserSchema(BaseModel):
    username: str
    password: str  # constr(min_length=8, max_length=64)


class UserResponse(UserBaseSchema):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class DetectionBaseSchema(BaseModel):
    uuid: uuid.UUID
    detected_animal: str
    # frame_path: str
    confidence: float
    detection_ts: datetime
    resolved: bool
    camera_id: uuid.UUID
    submit_id: int

    class Config:
        from_attributes = True


class CameraBaseSchema(BaseModel):
    uuid: uuid.UUID
    coordinates: list
    active: str
    address: str
    url: str

    class Config:
        from_attributes = True


class SubmitBaseSchema(BaseModel):
    id: int
    coordinates: list
    address: str
    reported_animal: str
    # frame: str
    report_ts: datetime

    class Config:
        from_attributes = True


class SubmitMinSchema(BaseModel):
    # ser_json_bytes: Literal['utf8', 'base64']
    frame: Any
    localization: Tuple[float, float]

    class Config:
        from_attributes = True


class SubmitResponse(SubmitBaseSchema):
    id: int
    report_ts: datetime
    updated_at: datetime
