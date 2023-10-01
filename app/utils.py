# import aiofiles
from base64 import b64decode, b64encode
from PIL import Image
from io import BytesIO
from passlib.context import CryptContext
from sqlalchemy import text


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str):
    return pwd_context.verify(password, hashed_password)


def get_format(content):
    extension = content.split(";")[0].split("/")[1]
    return extension


def prepare_image(image_path: str):
    with open(image_path, "rb") as image_file:
        print("loading file ", image_path)
        encoded_image_string = b64encode(image_file.read())
        print("loaded file ", image_path)
    return {
        "mime": "image/png",
        "frame": encoded_image_string,
    }


async def save_image(filename, in_file, save_size=None):
    """TODO: save in 1280x720"""
    try:
        decoded_image = b64decode(in_file)
        image = Image.open(BytesIO(decoded_image))
        if save_size:
            image = image.resize(save_size)
        image.save(filename)
    except Exception as e:
        return False, e
    return True
