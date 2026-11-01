"""EmbeddingRecord entity – vector storage with pgvector for VisionMatch AI."""

import uuid

from sqlalchemy import CheckConstraint, ForeignKey, Index, SmallInteger, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EmbeddingRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Persists a single L2-normalised embedding vector and its provenance.

    Each row ties a vector back to the exact image **and** model version
    that produced it, enabling multi-model A/B evaluation and safe
    re-indexing when a new model is deployed.

    Uses pgvector's ``Vector`` column type (unconstrained dimension) for storage
    and approximate nearest-neighbour search (HNSW / IVFFlat partial indexes
    are added at the migration level once access patterns stabilise).
    """

    __tablename__ = "embeddings"

    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    ml_model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    embedding_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Embedding modality: 'image' or 'text'.",
    )

    embedding_dim: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        comment="Actual dimensionality of the stored vector.",
    )

    vector: Mapped[list] = mapped_column(
        Vector(),
        nullable=False,
        comment="L2-normalised embedding vector (pgvector, variable dimension).",
    )

    # ── Relationships ────────────────────────────────────────────────────
    image: Mapped["ImageRecord"] = relationship(
        "ImageRecord",
        back_populates="embeddings",
    )

    ml_model: Mapped["MLModelRecord"] = relationship(
        "MLModelRecord",
        back_populates="embeddings",
    )

    __table_args__ = (
        # Prevent duplicate embeddings for the same image+model+type triple.
        Index(
            "ix_embeddings_image_model_type",
            "image_id",
            "ml_model_id",
            "embedding_type",
            unique=True,
        ),
        # Ensure embedding_dim reflects supported model dimensions (CLIP=512, SigLIP=768)
        CheckConstraint(
            "embedding_dim IN (512, 768)",
            name="ck_embeddings_supported_dim",
        ),
        # Ensure vector dimensions match declared embedding_dim
        CheckConstraint(
            "vector_dims(vector) = embedding_dim",
            name="ck_embeddings_vector_dim_matches",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EmbeddingRecord id={self.id!s} "
            f"type={self.embedding_type!r} dim={self.embedding_dim}>"
        )
