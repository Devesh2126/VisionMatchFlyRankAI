"""MLModelRecord entity – ML model registry metadata for VisionMatch AI."""

from sqlalchemy import Boolean, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class MLModelRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Stores metadata about each registered Vision-Language model.

    Maps 1-to-1 with entries in the ML package's MODEL_CATALOG,
    but persists the configuration in the database so that embeddings
    can reference the exact model version that produced them.
    """

    __tablename__ = "ml_models"

    model_key: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique short identifier (e.g. 'clip-vit-b32').",
    )

    model_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    hf_model_id: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
        comment="HuggingFace Hub model identifier.",
    )

    hf_revision: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        server_default=text("'main'"),
    )

    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    architecture_family: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Model family (e.g. 'clip', 'siglip').",
    )

    preprocessing_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="Model-specific preprocessing parameters.",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        comment="Soft-disable flag for model rotation.",
    )

    # ── Relationships ────────────────────────────────────────────────────
    embeddings: Mapped[list["EmbeddingRecord"]] = relationship(
        "EmbeddingRecord",
        back_populates="ml_model",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(
        "EvaluationRun",
        back_populates="ml_model",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<MLModelRecord id={self.id!s} "
            f"key={self.model_key!r} dim={self.embedding_dimension}>"
        )