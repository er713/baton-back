from datetime import datetime
import uuid as uid
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import (
    Depends,
    HTTPException,
    status,
    APIRouter,
    Response,
    Request,
    Header,
)
from fastapi.security import OAuth2PasswordRequestForm
from typing import Annotated

from ..db import models, schemas
from ..db.database import get_db

import jwt

# from app.oauth2 import require_user


router = APIRouter()


@router.get("/")
async def get_users(
    db: Session = Depends(get_db),
    limit: int = 100,
    skip: int = 0,
    search: str = "",
):
    users = db.execute(text("SELECT * FROM users")).mappings().all()
    # print(users)
    return {"status": "success", "count": len(users), "results": users}


@router.post("/login")
async def login_user(
    user: Request,
    db: Session = Depends(get_db),
    # response: Response,
):
    results = await user.json()
    print("\n\n", results, "\n\n")
    username = results["username"]
    password = results["password"]
    db_user = (
        db.execute(
            text(
                f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}';"
            )
        )
        .mappings()
        .first()
    )

    print("\n\n")
    print(db_user)
    print("\n\n")

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    user_uuid = str(db_user.uuid)
    print(user_uuid)

    jwt_token = jwt.encode({"uuid": user_uuid}, "secret", algorithm="HS256")

    return {"username": db_user.username, "token": f"Bearer {jwt_token}"}


def check_token(token, db) -> bool:
    try:
        decoded = jwt.decode(token, "secret", algorithms=["HS256"])
    except jwt.exceptions.DecodeError:
        return False
    user = (
        db.execute(
            text(f"SELECT uuid FROM users WHERE uuid='{decoded['uuid']}'")
        )
        .mappings()
        .all()
    )
    return bool(user)


@router.post("/auth")
def check_auth(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        bearer = request.headers.get("authorization").split(" ", 1)[1]
        if check_token(bearer, db):
            return Response(status_code=200)
        else:
            return Response(status_code=403)
    except Exception:
        return Response(status_code=403)


# @router.get


# @router.post(
#     "/", status_code=status.HTTP_201_CREATED, response_model=schemas.UserBaseSchema
# )
# def create_user(user: schemas.CreateUserSchema, db: Session = Depends(get_db)):
#     db_user = db.query(models.User).filter(models.User.email == user.email).first()
#     if db_user:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
#         )
#     db_user = models.User(
#         uuid=uid.uuid4(),
#         username=user.username,
#         email=user.email,
#         password=user.password,
#         role=user.role,
#         verified=user.verified,
#     )
#     db.add(db_user)
#     db.commit()
#     db.refresh(db_user)
#     return db_user
