from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OpenPlant ID API"
    model_path: str = "models/plantnet_efficientnetv2.onnx"
    disease_model_path: str = "models/disease_mobilenetv3.onnx"
    confidence_threshold: float = 0.25
    sqlite_url: str = "sqlite:///./openplant.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
