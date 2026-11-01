"""
VisionMatch AI – Database models package.

Importing this module ensures every ORM model is registered with
SQLAlchemy's metadata before Alembic or the application inspects it.
"""

from .embedding import EmbeddingRecord
from .evaluation_run import EvaluationRun
from .image import ImageRecord
from .ml_model import MLModelRecord
from .processing_job import ProcessingJob
from .search_history import SearchHistory
from .user import User

__all__ = [
    "User",
    "ImageRecord",
    "MLModelRecord",
    "EmbeddingRecord",
    "ProcessingJob",
    "SearchHistory",
    "EvaluationRun",
]
