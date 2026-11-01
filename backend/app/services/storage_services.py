from pathlib import Path
import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from app.core.config import settings

class ObjectStorageService:
    """
    S3-compatible object storage abstraction supporting MinIO/AWS S3
    with automatic local-disk fallback for isolated testing environments.
    """
    def __init__(self) -> None:
        self.bucket = settings.MINIO_BUCKET_NAME
        self._fallback_dir = Path("./.local_storage")
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"{'https' if settings.MINIO_SECURE else 'http'}://{settings.MINIO_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ROOT_USER,
            aws_secret_access_key=settings.MINIO_ROOT_PASSWORD,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
        try:
            self._s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError):
            target = self._fallback_dir / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def download_bytes(self, key: str) -> bytes:
        try:
            response = self._s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except (BotoCoreError, ClientError):
            target = self._fallback_dir / key
            if target.exists():
                return target.read_bytes()
            raise FileNotFoundError(f"Storage object '{key}' not found.")

    def delete_object(self, key: str) -> None:
        try:
            self._s3.delete_object(Bucket=self.bucket, Key=key)
        except (BotoCoreError, ClientError):
            target = self._fallback_dir / key
            if target.exists():
                target.unlink()

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        try:
            url = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            if settings.MINIO_ENDPOINT != settings.MINIO_PUBLIC_ENDPOINT:
                url = url.replace(settings.MINIO_ENDPOINT, settings.MINIO_PUBLIC_ENDPOINT, 1)
            return url
        except (BotoCoreError, ClientError):
            return f"/api/v1/images/raw/{key}"


storage_service = ObjectStorageService()