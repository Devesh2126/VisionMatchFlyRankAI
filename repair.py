import os
from pathlib import Path

root = Path(__file__).resolve().parent
services_dir = root / "backend" / "app" / "services"
workers_dir = root / "backend" / "app" / "workers"

services_dir.mkdir(parents=True, exist_ok=True)
workers_dir.mkdir(parents=True, exist_ok=True)

with open(services_dir / "__init__.py", "w", encoding="utf-8") as f:
    f.write("from .storage_service import storage_service\n")

with open(services_dir / "storage_service.py", "w", encoding="utf-8") as f:
    f.write('''from pathlib import Path
import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from app.core.config import settings

class ObjectStorageService:
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
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
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

storage_service = ObjectStorageService()
''')

with open(workers_dir / "__init__.py", "w", encoding="utf-8") as f:
    f.write("from .tasks import process_image_embedding_task, execute_evaluation_benchmark\n")

with open(workers_dir / "tasks.py", "w", encoding="utf-8") as f:
    f.write('''import io
import uuid
import numpy as np
from PIL import Image
from sqlalchemy import select
from app.db.session import SessionLocal
from app.db.models import EmbeddingRecord, EvaluationRun, ImageRecord, MLModelRecord, ProcessingJob
from app.services.storage_service import storage_service
from visionmatch_ml.evaluation import compute_ir_metrics
from visionmatch_ml.models import get_model_registry

def process_image_embedding_task(image_id_str: str, model_key: str, job_id_str: str | None = None):
    image_id = uuid.UUID(image_id_str)
    with SessionLocal() as db:
        image_rec = db.get(ImageRecord, image_id)
        model_rec = db.scalar(select(MLModelRecord).where(MLModelRecord.model_key == model_key))
        if not image_rec or not model_rec: return
        raw_bytes = storage_service.download_bytes(image_rec.storage_path)
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            rgb_img = pil_img.convert("RGB")
        
        vlm = get_model_registry().get_model(model_key, auto_load=True)
        vector_np = vlm.encode_image(rgb_img)
        vec_list = vector_np.astype(np.float32).tolist()

        existing = db.scalar(select(EmbeddingRecord).where(
            EmbeddingRecord.image_id == image_rec.id, 
            EmbeddingRecord.ml_model_id == model_rec.id
        ))
        if existing:
            existing.vector = vec_list
        else:
            db.add(EmbeddingRecord(
                image_id=image_rec.id, ml_model_id=model_rec.id,
                embedding_type="image", embedding_dim=model_rec.embedding_dimension, vector=vec_list
            ))
        image_rec.processing_status = "COMPLETED"
        db.commit()

def execute_evaluation_benchmark(db, model_key: str, dataset_name: str):
    model_rec = db.scalar(select(MLModelRecord).where(MLModelRecord.model_key == model_key))
    images = list(db.scalars(select(ImageRecord).where(ImageRecord.caption_text.is_not(None))))
    vlm = get_model_registry().get_model(model_key, auto_load=True)

    candidate_ids, candidate_vecs = [], []
    for img_rec in images:
        emb = db.scalar(select(EmbeddingRecord).where(EmbeddingRecord.image_id == img_rec.id, EmbeddingRecord.ml_model_id == model_rec.id))
        if emb:
            candidate_vecs.append(np.asarray(emb.vector, dtype=np.float32))
            candidate_ids.append(str(img_rec.id))

    query_texts = [img.caption_text for img in images]
    ground_truth_sets = [{str(img.id)} for img in images]

    query_vecs = vlm.encode_texts(query_texts)
    candidate_matrix = np.vstack(candidate_vecs)
    
    summary = compute_ir_metrics(query_vecs, candidate_matrix, candidate_ids, ground_truth_sets)

    eval_run = EvaluationRun(
        user_id=images[0].owner_id, ml_model_id=model_rec.id, dataset_name=dataset_name,
        num_queries=summary.num_queries, num_candidates=summary.num_candidates,
        recall_at_1=summary.recall_at_1, recall_at_5=summary.recall_at_5, recall_at_10=summary.recall_at_10,
        precision_at_5=summary.precision_at_5, mean_average_precision=summary.mean_average_precision, mean_reciprocal_rank=summary.mean_reciprocal_rank
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    return eval_run
''')

print("All missing backend files generated successfully!")