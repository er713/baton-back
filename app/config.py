from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_HOSTNAME: str
    DATABASE_PORT: int
    POSTGRES_DB: str

    JWT_PUBLIC_KEY: str
    JWT_PRIVATE_KEY: str
    JWT_KEY: str
    JWT_ALGORITHM: str
    REFRESH_TOKEN_EXPIRE_MINUTES: int
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    CLIENT_ORIGIN: str

    class Config:
        env_file = ".env"


settings = ServerSettings()
