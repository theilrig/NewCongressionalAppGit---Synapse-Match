try:
    from pydantic_settings import BaseSettings 
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()
