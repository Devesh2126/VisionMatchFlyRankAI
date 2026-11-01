"""SearchHistory entity – user search audit log for VisionMatch AI."""

import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class SearchHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Immutable audit record of every similarity search executed by a user.

    Captures the query, the model used, result count, and timing data
    so that search quality can be measured and reproduced offline.
    """

    __tablename__ = "search_history"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    query_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Sanitised text query (null for image-to-image search).",
    )

    query_image_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source image for image-to-image search.",
    )

    search_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        comment="Search modality: 'text_to_image', 'image_to_image'.",
    )

    model_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Model key used for embedding at search time.",
    )

    result_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    top_results: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Top-K result IDs and scores for offline analysis.",
    )

    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="End-to-end search latency in milliseconds.",
    )

    # ── Relationships ────────────────────────────────────────────────────
    user: Mapped["User"] = relationship(
        "User",
        back_populates="search_histories",
    )

    query_image: Mapped["ImageRecord | None"] = relationship(
        "ImageRecord",
    )

    def __repr__(self) -> str:
        return (
            f"<SearchHistory id={self.id!s} "
            f"type={self.search_type!r} results={self.result_count}>"
        )
