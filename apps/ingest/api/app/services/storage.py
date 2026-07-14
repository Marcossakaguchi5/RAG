from io import BytesIO

from minio import Minio

from app.core.config import get_settings


def get_storage_client() -> Minio:
    settings = get_settings()
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket() -> None:
    settings = get_settings()
    client = get_storage_client()
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)


def put_pdf(object_name: str, content: bytes) -> None:
    settings = get_settings()
    get_storage_client().put_object(
        settings.minio_bucket,
        object_name,
        data=BytesIO(content),
        length=len(content),
        content_type="application/pdf",
    )


def delete_object(object_name: str) -> None:
    settings = get_settings()
    get_storage_client().remove_object(settings.minio_bucket, object_name)
