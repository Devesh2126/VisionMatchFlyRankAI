import uuid
import numpy as np
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from app.db.models import EmbeddingRecord, ImageRecord, MLModelRecord, User
from visionmatch_ml.similarity import compute_cosine_similarity


class VectorSearchRepository:
    @staticmethod
    def search_by_vector(
        db: Session,
        query_vector: np.ndarray,
        model_record: MLModelRecord,
        user_id: uuid.UUID,
        top_k: int = 12,
    ) -> list[tuple[ImageRecord, float, int]]:
        vec_list = query_vector.astype(np.float32).tolist()

        # Users can search their own images plus public benchmark images owned by superuser demo@visionmatch.ai
        demo_user_ids = select(User.id).where(User.is_superuser.is_(True))
        conditions = [
            EmbeddingRecord.ml_model_id == model_record.id,
            EmbeddingRecord.embedding_dim == model_record.embedding_dimension,
            or_(
                ImageRecord.owner_id == user_id,
                ImageRecord.owner_id.in_(demo_user_ids),
            ),
        ]

        if db.bind and db.bind.dialect.name == "postgresql":
            distance_expr = EmbeddingRecord.vector.cosine_distance(vec_list)
            similarity_expr = (1.0 - distance_expr).label("similarity_score")

            stmt = (
                select(ImageRecord, similarity_expr)
                .join(EmbeddingRecord, EmbeddingRecord.image_id == ImageRecord.id)
                .where(and_(*conditions))
                .order_by(distance_expr.asc(), ImageRecord.id.asc())
                .limit(top_k)
            )
            rows = db.execute(stmt).all()
            return [
                (img, round(float(score), 4), idx + 1)
                for idx, (img, score) in enumerate(rows)
            ]

        stmt = (
            select(ImageRecord, EmbeddingRecord)
            .join(EmbeddingRecord, EmbeddingRecord.image_id == ImageRecord.id)
            .where(and_(*conditions))
        )
        rows = db.execute(stmt).all()
        scored: list[tuple[ImageRecord, float]] = []
        for img, emb in rows:
            cand_vec = np.asarray(emb.vector, dtype=np.float32)
            sim = compute_cosine_similarity(query_vector, cand_vec)
            scored.append((img, sim))

        scored.sort(key=lambda item: (-round(item[1], 7), str(item[0].id)))
        scored = scored[:top_k]
        return [
            (img, round(float(score), 4), idx + 1)
            for idx, (img, score) in enumerate(scored)
        ]