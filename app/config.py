from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_host: str = "127.0.0.1"
    app_port: int = 8888
    allowed_origins: str = "http://127.0.0.1:8888,http://localhost:8888,null"

    text_api_url: AnyHttpUrl = "https://api.deepseek.com/v1/chat/completions"
    text_api_key: str = ""
    text_model: str = "deepseek-chat"

    dashscope_api_key: str = ""
    image_provider: str = "qwen"
    image_model: str = "qwen-image-2.0"
    image_size: str = "1024*1024"
    video_provider: str = "qwen"
    video_model: str = "happyhorse-1.0-i2v"
    video_duration: int = 30

    jimeng_api_key: str = ""
    jimeng_image_model: str = "jimeng_t2i_v40"
    jimeng_image_size: str = "1024"
    jimeng_video_model: str = "jimeng_t2v_v30"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
