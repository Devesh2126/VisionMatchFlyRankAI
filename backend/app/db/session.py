import logging
from collections.abc import Generator
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from app.core.config import settings
from app.db.base import Base
from visionmatch_ml.models import MODEL_CATALOG

logger = logging.getLogger(__name__)


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app.db.models import MLModelRecord

    with engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "vector";'))

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        for key, meta in MODEL_CATALOG.items():
            existing = db.scalar(
                select(MLModelRecord).where(MLModelRecord.model_key == key)
            )
            if not existing:
                db.add(
                    MLModelRecord(
                        model_key=meta.model_key,
                        model_name=meta.model_name,
                        model_version=meta.model_version,
                        hf_model_id=meta.hf_model_id,
                        hf_revision=meta.hf_revision,
                        embedding_dimension=meta.embedding_dimension,
                        architecture_family=getattr(meta, "architecture_family", None) or "clip",
                        is_active=True,
                        preprocessing_config=meta.preprocessing_config,
                    )
                )
        db.commit()