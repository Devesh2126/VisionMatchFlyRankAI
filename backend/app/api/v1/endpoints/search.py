import io
import time
import uuid
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.models import (
    EmbeddingRecord,
    ImageRecord,
    MLModelRecord,
    SearchHistory,
    User,
)
from app.db.session import get_db
from app.repositories.vector_repository import VectorSearchRepository
from app.schemas.api_schemas import (
    MatchCandidateScore,
    MatchRequest,
    MatchResponse,
    SearchHistoryOut,
    SearchResponse,
    SearchResultItem,
    TextSearchRequest,
)
from app.services.storage_service import storage_service
from visionmatch_ml.models import get_model_registry
from visionmatch_ml.preprocessing import ImageValidationError, load_and_sanitize_image
from visionmatch_ml.similarity import compute_similarity_matrix

router = APIRouter(prefix="/search", tags=["Semantic Search"])


def _resolve_model(db: Session, model_key: str | None) -> MLModelRecord:
    key = model_key or settings.DEFAULT_MODEL_KEY
    model_rec = db.scalar(select(MLModelRecord).where(MLModelRecord.model_key == key))
    if not model_rec:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not registered.")
    return model_rec


def _format_results(
    ranked_rows: list[tuple[ImageRecord, float, int]], model_key: str
) -> list[SearchResultItem]:
    return [
        SearchResultItem(
            image_id=img.id,
            original_filename=img.filename,
            width=img.width,
            height=img.height,
            similarity_score=score,
            rank=rank,
            model_key=model_key,
            thumbnail_url=storage_service.get_image_url(
                img.thumbnail_path or img.storage_path
            ),
            image_url=storage_service.get_image_url(img.storage_path),
            created_at=img.created_at,
        )
        for img, score, rank in ranked_rows
    ]


@router.post("/text", response_model=SearchResponse)
def search_by_text(
    payload: TextSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SearchResponse:
    model_rec = _resolve_model(db, payload.model_key)
    vlm = get_model_registry().get_model(model_rec.model_key, auto_load=True)

    t0 = time.perf_counter()
    try:
        query_vec = vlm.encode_text(payload.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    emb_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    t1 = time.perf_counter()
    ranked = VectorSearchRepository.search_by_vector(
        db=db,
        query_vector=query_vec,
        model_record=model_rec,
        user_id=current_user.id,
        top_k=payload.top_k,
    )
    ret_ms = round((time.perf_counter() - t1) * 1000.0, 2)
    total_ms = round(emb_ms + ret_ms, 2)

    db.add(
        SearchHistory(
            user_id=current_user.id,
            query_text=payload.query,
            search_type="text_to_image",
            model_key=model_rec.model_key,
            result_count=len(ranked),
            latency_ms=total_ms,
        )
    )
    db.commit()

    return SearchResponse(
        query_type="TEXT_TO_IMAGE",
        query_text=payload.query,
        model_key=model_rec.model_key,
        embedding_dimension=model_rec.embedding_dimension,
        embedding_latency_ms=emb_ms,
        retrieval_latency_ms=ret_ms,
        total_latency_ms=total_ms,
        result_count=len(ranked),
        results=_format_results(ranked, model_rec.model_key),
    )


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile | None = File(default=None),
    image_id: uuid.UUID | None = Form(default=None),
    model_key: str | None = Form(default=None),
    top_k: int = Form(default=12),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SearchResponse:
    model_rec = _resolve_model(db, model_key)
    vlm = get_model_registry().get_model(model_rec.model_key, auto_load=True)

    t0 = time.perf_counter()
    query_label = "Uploaded Image"
    if file is not None:
        raw_bytes = await file.read()
        try:
            sanitized = load_and_sanitize_image(raw_bytes)
        except ImageValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        query_vec = vlm.encode_image(sanitized.pil_image)
        query_label = f"Visual Query: {file.filename}"
    elif image_id is not None:
        img_rec = db.get(ImageRecord, image_id)
        if not img_rec:
            raise HTTPException(status_code=404, detail="Reference image not found.")
        query_label = f"Similar to: {img_rec.filename}"
        existing_emb = db.scalar(
            select(EmbeddingRecord).where(
                EmbeddingRecord.image_id == img_rec.id,
                EmbeddingRecord.ml_model_id == model_rec.id,
            )
        )
        if existing_emb is not None:
            query_vec = np.asarray(existing_emb.vector, dtype=np.float32)
        else:
            raw_bytes = storage_service.download_bytes(img_rec.storage_path)
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                query_vec = vlm.encode_image(pil_img.convert("RGB"))
    else:
        raise HTTPException(status_code=400, detail="Provide file or image_id.")

    emb_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    t1 = time.perf_counter()
    ranked = VectorSearchRepository.search_by_vector(
        db=db,
        query_vector=query_vec,
        model_record=model_rec,
        user_id=current_user.id,
        top_k=min(max(top_k, 1), 50),
    )
    ret_ms = round((time.perf_counter() - t1) * 1000.0, 2)
    total_ms = round(emb_ms + ret_ms, 2)

    db.add(
        SearchHistory(
            user_id=current_user.id,
            query_text=query_label,
            query_image_id=image_id,
            search_type="image_to_image",
            model_key=model_rec.model_key,
            result_count=len(ranked),
            latency_ms=total_ms,
        )
    )
    db.commit()

    return SearchResponse(
        query_type="IMAGE_TO_IMAGE",
        query_text=query_label,
        model_key=model_rec.model_key,
        embedding_dimension=model_rec.embedding_dimension,
        embedding_latency_ms=emb_ms,
        retrieval_latency_ms=ret_ms,
        total_latency_ms=total_ms,
        result_count=len(ranked),
        results=_format_results(ranked, model_rec.model_key),
    )


@router.post("/match", response_model=MatchResponse)
def match_image_to_texts(
    payload: MatchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MatchResponse:
    model_rec = _resolve_model(db, payload.model_key)
    img_rec = db.get(ImageRecord, payload.image_id)
    if not img_rec:
        raise HTTPException(status_code=404, detail="Target image not found.")

    vlm = get_model_registry().get_model(model_rec.model_key, auto_load=True)
    t0 = time.perf_counter()

    existing_emb = db.scalar(
        select(EmbeddingRecord).where(
            EmbeddingRecord.image_id == img_rec.id,
            EmbeddingRecord.ml_model_id == model_rec.id,
        )
    )
    if existing_emb is not None:
        img_vec = np.asarray(existing_emb.vector, dtype=np.float32)
    else:
        raw_bytes = storage_service.download_bytes(img_rec.storage_path)
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            img_vec = vlm.encode_image(pil_img.convert("RGB"))

    text_vecs = vlm.encode_texts(payload.candidate_descriptions)
    sims = compute_similarity_matrix(img_vec, text_vecs)[0]

    scaled_logits = sims * 20.0
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    relative_pct = (exp_logits / np.sum(exp_logits)) * 100.0

    paired = list(
        zip(
            payload.candidate_descriptions,
            sims.tolist(),
            relative_pct.tolist(),
        )
    )
    paired.sort(key=lambda item: (-round(item[1], 7), item[0]))

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return MatchResponse(
        image_id=img_rec.id,
        model_key=model_rec.model_key,
        latency_ms=latency_ms,
        rankings=[
            MatchCandidateScore(
                description=desc,
                cosine_similarity=round(float(sim), 4),
                relative_alignment_pct=round(float(pct), 2),
                rank=idx + 1,
            )
            for idx, (desc, sim, pct) in enumerate(paired)
        ],
    )


@router.get("/history", response_model=list[SearchHistoryOut])
def get_search_history(
    limit: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SearchHistoryOut]:
    rows = list(
        db.scalars(
            select(SearchHistory)
            .where(SearchHistory.user_id == current_user.id)
            .order_by(SearchHistory.created_at.desc())
            .limit(min(limit, 100))
        ).all()
    )
    return [
        SearchHistoryOut(
            id=sh.id,
            query_type=sh.search_type.upper(),
            query_text=sh.query_text,
            model_key=sh.model_key,
            result_count=sh.result_count,
            latency_ms=float(sh.latency_ms or 0.0),
            created_at=sh.created_at,
        )
        for sh in rows
    ]