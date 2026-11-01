"""User entity – authentication and ownership root for VisionMatch AI."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Represents a registered platform user.

    Owns uploaded images, search history, and evaluation runs.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320),
        unique=True,
        nullable=False,
        index=True,
    )

    hashed_password: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ── Relationships ────────────────────────────────────────────────────
    images: Mapped[list["ImageRecord"]] = relationship(
        "ImageRecord",
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    search_histories: Mapped[list["SearchHistory"]] = relationship(
        "SearchHistory",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(
        "EvaluationRun",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id!s} email={self.email!r}>"
