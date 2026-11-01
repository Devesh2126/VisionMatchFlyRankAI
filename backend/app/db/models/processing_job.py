"""ProcessingJob entity – async task tracking for VisionMatch AI."""

import uuid

from sqlalchemy import ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tracks the lifecycle of asynchronous image-processing tasks
    (e.g. embedding extraction, thumbnail generation).

    Status values: ``pending`` → ``running`` → ``completed`` | ``failed``.
    """

    __tablename__ = "processing_jobs"

    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    job_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Task category (e.g. 'embedding_extraction', 'thumbnail').",
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'pending'"),
        index=True,
        comment="One of: pending, running, completed, failed.",
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment="Higher value = higher priority.",
    )

    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("3"),
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    result_metadata: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Arbitrary output payload from the worker.",
    )

    # ── Relationships ────────────────────────────────────────────────────
    image: Mapped["ImageRecord"] = relationship(
        "ImageRecord",
        back_populates="processing_jobs",
    )

    __table_args__ = (
        Index("ix_processing_jobs_status_priority", "status", "priority"),
    )

    def __repr__(self) -> str:
        return (
            f"<ProcessingJob id={self.id!s} "
            f"type={self.job_type!r} status={self.status!r}>"
        )
