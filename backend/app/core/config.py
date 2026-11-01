from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "VisionMatch AI"
    ENVIRONMENT: str = "development"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "dev-secret-key-change-in-production-use-openssl-rand-hex-32"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str = (
        "postgresql+psycopg://visionmatch:visionmatch_pass@localhost:5432/visionmatch_db"
    )

    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_ALWAYS_EAGER: bool = False

    # MinIO / S3 Object Storage
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin_pass"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_PUBLIC_ENDPOINT: str = "localhost:9000"
    MINIO_BUCKET_NAME: str = "visionmatch-assets"
    MINIO_SECURE: bool = False

    # ML & MLOps
    DEFAULT_MODEL_KEY: str = "clip-vit-b32"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()