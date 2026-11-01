"""EvaluationRun entity – IR metric snapshots for VisionMatch AI."""

import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EvaluationRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Stores a snapshot of Information Retrieval evaluation metrics
    (Recall@K, MRR, mAP, etc.) produced by the ML evaluation pipeline.

    Each run is tied to a specific user, model, and dataset so that
    historical quality trends can be tracked over time.
    """

    __tablename__ = "evaluation_runs"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    ml_model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    dataset_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Identifier for the evaluation dataset.",
    )

    num_queries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    num_candidates: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    recall_at_1: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recall_at_5: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recall_at_10: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    precision_at_5: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_average_precision: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_reciprocal_rank: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    extra_metrics: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional metrics not in the fixed schema.",
    )

    # ── Relationships ────────────────────────────────────────────────────
    user: Mapped["User"] = relationship(
        "User",
        back_populates="evaluation_runs",
    )

    ml_model: Mapped["MLModelRecord"] = relationship(
        "MLModelRecord",
        back_populates="evaluation_runs",
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationRun id={self.id!s} "
            f"dataset={self.dataset_name!r} "
            f"R@1={self.recall_at_1:.4f} MRR={self.mean_reciprocal_rank:.4f}>"
        )
