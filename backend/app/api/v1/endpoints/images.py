import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.models import (
    EmbeddingRecord,
    ImageRecord,
    MLModelRecord,
    ProcessingJob,
    User,
)
from app.db.session import get_db
from app.schemas.api_schemas import (
    ImageDetailResponse,
    ImageStatusResponse,
    ImageUploadResponse,
)
from app.services.storage_service import storage_service
from app.workers.tasks import process_image_embedding_task
from visionmatch_ml.preprocessing import (
    ImageValidationError,
    create_thumbnail_bytes,
    load_and_sanitize_image,
)

router = APIRouter(prefix="/images", tags=["Images"])


def _get_image_status_label(db: Session, img: ImageRecord) -> str:
    latest_job = db.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.image_id == img.id)
        .order_by(ProcessingJob.created_at.desc())
    )
    if latest_job:
        return latest_job.status.upper()
    has_emb = db.scalar(
        select(EmbeddingRecord.id).where(EmbeddingRecord.image_id == img.id)
    )
    return "COMPLETED" if has_emb else "PENDING"


def _to_image_detail(db: Session, img: ImageRecord) -> ImageDetailResponse:
    model_keys = list(
        db.scalars(
            select(MLModelRecord.model_key)
            .join(EmbeddingRecord, EmbeddingRecord.ml_model_id == MLModelRecord.id)
            .where(EmbeddingRecord.image_id == img.id)
        ).all()
    )
    status_label = _get_image_status_label(db, img)
    thumb_key = img.thumbnail_path or img.storage_path
    return ImageDetailResponse(
        id=img.id,
        original_filename=img.filename,
        mime_type=img.mime_type,
        file_size_bytes=img.file_size_bytes,
        width=img.width,
        height=img.height,
        processing_status=status_label,
        image_url=storage_service.get_image_url(img.storage_path),
        thumbnail_url=storage_service.get_image_url(thumb_key),
        indexed_models=model_keys,
        created_at=img.created_at,
    )


@router.post(
    "/upload",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_image(
    file: UploadFile = File(...),
    model_key: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ImageUploadResponse:
    raw_bytes = await file.read()
    try:
        sanitized = load_and_sanitize_image(raw_bytes)
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    target_model_key = model_key or settings.DEFAULT_MODEL_KEY
    model_rec = db.scalar(
        select(MLModelRecord).where(MLModelRecord.model_key == target_model_key)
    )
    if not model_rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{target_model_key}' not registered.",
        )

    existing_img = db.scalar(
        select(ImageRecord).where(
            ImageRecord.owner_id == current_user.id,
            ImageRecord.sha256_checksum == sanitized.sha256_checksum,
        )
    )
    if existing_img:
        latest_job = db.scalar(
            select(ProcessingJob)
            .where(ProcessingJob.image_id == existing_img.id)
            .order_by(ProcessingJob.created_at.desc())
        )
        return ImageUploadResponse(
            image_id=existing_img.id,
            job_id=latest_job.id if latest_job else uuid.uuid4(),
            processing_status=_get_image_status_label(db, existing_img),
            deduplicated=True,
            message="Identical image already exists in your library.",
        )

    image_id = uuid.uuid4()
    ext = sanitized.mime_type.split("/")[-1]
    storage_path = f"originals/{current_user.id}/{image_id}.{ext}"
    thumbnail_path = f"thumbnails/{current_user.id}/{image_id}.webp"

    thumb_bytes = create_thumbnail_bytes(sanitized.pil_image)
    storage_service.upload_bytes(storage_path, sanitized.clean_bytes, sanitized.mime_type)
    storage_service.upload_bytes(thumbnail_path, thumb_bytes, "image/webp")

    img_rec = ImageRecord(
        id=image_id,
        owner_id=current_user.id,
        filename=file.filename or f"upload.{ext}",
        storage_path=storage_path,
        thumbnail_path=thumbnail_path,
        mime_type=sanitized.mime_type,
        file_size_bytes=sanitized.file_size_bytes,
        width=sanitized.width,
        height=sanitized.height,
        sha256_checksum=sanitized.sha256_checksum,
    )
    db.add(img_rec)
    db.flush()

    job_rec = ProcessingJob(
        image_id=img_rec.id,
        job_type="embedding_extraction",
        status="pending",
    )
    db.add(job_rec)
    db.commit()

    # Execute embedding generation synchronously if Celery worker is not running separately
    process_image_embedding_task(str(img_rec.id), target_model_key, str(job_rec.id))

    return ImageUploadResponse(
        image_id=img_rec.id,
        job_id=job_rec.id,
        processing_status="COMPLETED",
        deduplicated=False,
        message="Image validated, EXIF stripped, and indexed into pgvector.",
    )


@router.get("", response_model=list[ImageDetailResponse])
def list_images(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ImageDetailResponse]:
    demo_users = select(User.id).where(User.is_superuser.is_(True))
    images = list(
        db.scalars(
            select(ImageRecord)
            .where(
                or_(
                    ImageRecord.owner_id == current_user.id,
                    ImageRecord.owner_id.in_(demo_users),
                )
            )
            .order_by(ImageRecord.created_at.desc())
            .offset(skip)
            .limit(min(limit, 100))
        ).all()
    )
    return [_to_image_detail(db, img) for img in images]


@router.get("/raw/{storage_path:path}")
def serve_raw_image(storage_path: str) -> Response:
    try:
        raw = storage_service.download_bytes(storage_path)
        media = "image/webp" if storage_path.endswith(".webp") else "image/png"
        return Response(content=raw, media_type=media)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Image file not found") from exc


@router.get("/{image_id}", response_model=ImageDetailResponse)
def get_image_details(
    image_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ImageDetailResponse:
    img = db.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found.")
    owner = db.get(User, img.owner_id)
    if img.owner_id != current_user.id and not (owner and owner.is_superuser):
        raise HTTPException(status_code=404, detail="Image not found.")
    return _to_image_detail(db, img)


@router.get("/{image_id}/status", response_model=ImageStatusResponse)
def get_image_status(
    image_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ImageStatusResponse:
    img = db.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found.")
    job = db.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.image_id == img.id)
        .order_by(ProcessingJob.created_at.desc())
    )
    status_label = _get_image_status_label(db, img)
    return ImageStatusResponse(
        image_id=img.id,
        processing_status=status_label,
        job_status=job.status if job else status_label.lower(),
        attempts=job.attempts if job else 1,
        error_message=job.error_message if job else None,
    )


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(
    image_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    img = db.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found.")
    if img.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not permitted.")
    storage_service.delete_object(img.storage_path)
    if img.thumbnail_path:
        storage_service.delete_object(img.thumbnail_path)
    db.delete(img)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)