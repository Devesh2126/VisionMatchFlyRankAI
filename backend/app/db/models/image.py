"""ImageRecord entity – uploaded image metadata for VisionMatch AI."""

import uuid

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ImageRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Stores metadata for every uploaded image.

    Raw pixels live in object storage (MinIO); the database tracks
    the storage path, content hash for deduplication, and dimensional
    metadata needed by the preprocessing pipeline.
    """

    __tablename__ = "images"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    storage_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Object storage key (e.g. MinIO bucket path).",
    )

    thumbnail_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Object storage key for the WebP thumbnail.",
    )

    mime_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    width: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    height: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    sha256_checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hex digest of raw upload bytes.",
    )

    # ── Relationships ────────────────────────────────────────────────────
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="images",
    )

    embeddings: Mapped[list["EmbeddingRecord"]] = relationship(
        "EmbeddingRecord",
        back_populates="image",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    processing_jobs: Mapped[list["ProcessingJob"]] = relationship(
        "ProcessingJob",
        back_populates="image",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index(
            "ix_images_owner_checksum",
            "owner_id",
            "sha256_checksum",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ImageRecord id={self.id!s} "
            f"filename={self.filename!r} "
            f"{self.width}x{self.height}>"
        )
