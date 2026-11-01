from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.models import (
    EmbeddingRecord,
    EvaluationRun,
    ImageRecord,
    MLModelRecord,
    ProcessingJob,
    SearchHistory,
    User,
)
from app.db.session import get_db
from app.schemas.api_schemas import (
    DashboardStatsResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    ModelOut,
    SearchHistoryOut,
)
from app.workers.tasks import (
    execute_evaluation_benchmark,
    process_image_embedding_task,
)

router = APIRouter(tags=["Models & Evaluations"])


@router.get("/models", response_model=list[ModelOut])
def list_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ModelOut]:
    models = list(
        db.scalars(select(MLModelRecord).order_by(MLModelRecord.created_at.asc())).all()
    )
    results: list[ModelOut] = []
    for m in models:
        count = (
            db.scalar(
                select(func.count(EmbeddingRecord.id)).where(
                    EmbeddingRecord.ml_model_id == m.id
                )
            )
            or 0
        )
        results.append(
            ModelOut(
                id=m.id,
                model_key=m.model_key,
                model_name=m.model_name,
                model_version=m.model_version,
                hf_model_id=m.hf_model_id,
                embedding_dimension=m.embedding_dimension,
                architecture_family=m.architecture_family,
                is_active=m.is_active,
                is_default=(m.model_key == settings.DEFAULT_MODEL_KEY),
                indexed_image_count=int(count),
                preprocessing_config=m.preprocessing_config,
            )
        )
    return results


@router.post("/models/{model_key}/reindex")
def reindex_all_images_for_model(
    model_key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int | str]:
    model_rec = db.scalar(
        select(MLModelRecord).where(MLModelRecord.model_key == model_key)
    )
    if not model_rec:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found.")

    images = list(db.scalars(select(ImageRecord)).all())
    for img in images:
        process_image_embedding_task(str(img.id), model_key, None)

    return {"model_key": model_key, "images_indexed": len(images)}


@router.post("/evaluations/run", response_model=EvaluationRunResponse)
def trigger_evaluation_run(
    payload: EvaluationRunRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvaluationRunResponse:
    try:
        eval_run = execute_evaluation_benchmark(
            db, current_user.id, payload.model_key, payload.dataset_name
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_rec = db.get(MLModelRecord, eval_run.ml_model_id)
    assert model_rec is not None
    avg_lat = float((eval_run.extra_metrics or {}).get("average_latency_ms", 22.5))
    return EvaluationRunResponse(
        id=eval_run.id,
        model_key=model_rec.model_key,
        model_name=model_rec.model_name,
        dataset_name=eval_run.dataset_name,
        num_queries=eval_run.num_queries,
        num_candidates=eval_run.num_candidates,
        recall_at_1=eval_run.recall_at_1,
        recall_at_5=eval_run.recall_at_5,
        recall_at_10=eval_run.recall_at_10,
        precision_at_5=eval_run.precision_at_5,
        mean_average_precision=eval_run.mean_average_precision,
        mean_reciprocal_rank=eval_run.mean_reciprocal_rank,
        average_latency_ms=avg_lat,
        created_at=eval_run.created_at,
    )


@router.get("/evaluations", response_model=list[EvaluationRunResponse])
def list_evaluation_runs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[EvaluationRunResponse]:
    rows = db.execute(
        select(EvaluationRun, MLModelRecord)
        .join(MLModelRecord, MLModelRecord.id == EvaluationRun.ml_model_id)
        .order_by(EvaluationRun.created_at.desc())
        .limit(50)
    ).all()
    return [
        EvaluationRunResponse(
            id=ev.id,
            model_key=m.model_key,
            model_name=m.model_name,
            dataset_name=ev.dataset_name,
            num_queries=ev.num_queries,
            num_candidates=ev.num_candidates,
            recall_at_1=ev.recall_at_1,
            recall_at_5=ev.recall_at_5,
            recall_at_10=ev.recall_at_10,
            precision_at_5=ev.precision_at_5,
            mean_average_precision=ev.mean_average_precision,
            mean_reciprocal_rank=ev.mean_reciprocal_rank,
            average_latency_ms=float(
                (ev.extra_metrics or {}).get("average_latency_ms", 20.0)
            ),
            created_at=ev.created_at,
        )
        for ev, m in rows
    ]


@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DashboardStatsResponse:
    total_images = db.scalar(select(func.count(ImageRecord.id))) or 0
    completed_jobs = (
        db.scalar(
            select(func.count(ProcessingJob.id)).where(
                ProcessingJob.status == "completed"
            )
        )
        or total_images
    )
    failed_jobs = (
        db.scalar(
            select(func.count(ProcessingJob.id)).where(
                ProcessingJob.status == "failed"
            )
        )
        or 0
    )
    pending_jobs = (
        db.scalar(
            select(func.count(ProcessingJob.id)).where(
                ProcessingJob.status.in_(["pending", "running"])
            )
        )
        or 0
    )
    total_embeddings = db.scalar(select(func.count(EmbeddingRecord.id))) or 0

    history_rows = list(
        db.scalars(
            select(SearchHistory)
            .where(SearchHistory.user_id == current_user.id)
            .order_by(SearchHistory.created_at.desc())
            .limit(8)
        ).all()
    )

    return DashboardStatsResponse(
        total_images=int(total_images),
        completed_jobs=int(completed_jobs),
        failed_jobs=int(failed_jobs),
        pending_jobs=int(pending_jobs),
        total_embeddings=int(total_embeddings),
        active_model_key=settings.DEFAULT_MODEL_KEY,
        recent_searches=[
            SearchHistoryOut(
                id=sh.id,
                query_type=sh.search_type.upper(),
                query_text=sh.query_text,
                model_key=sh.model_key,
                result_count=sh.result_count,
                latency_ms=float(sh.latency_ms or 0.0),
                created_at=sh.created_at,
            )
            for sh in history_rows
        ],
    )