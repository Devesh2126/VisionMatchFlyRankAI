import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    display_name: str
    is_superuser: bool = False
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut


class ImageUploadResponse(BaseModel):
    image_id: uuid.UUID
    job_id: uuid.UUID
    processing_status: str
    deduplicated: bool = False
    message: str


class ImageDetailResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    mime_type: str
    file_size_bytes: int
    width: int
    height: int
    processing_status: str
    category: str | None = None
    caption_text: str | None = None
    image_url: str
    thumbnail_url: str
    indexed_models: list[str] = []
    created_at: datetime


class ImageStatusResponse(BaseModel):
    image_id: uuid.UUID
    processing_status: str
    job_status: str | None = None
    attempts: int = 0
    error_message: str | None = None


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    model_key: str | None = None
    top_k: int = Field(default=12, ge=1, le=50)


class SearchResultItem(BaseModel):
    image_id: uuid.UUID
    original_filename: str
    width: int
    height: int
    similarity_score: float
    rank: int
    model_key: str
    thumbnail_url: str
    image_url: str
    created_at: datetime


class SearchResponse(BaseModel):
    query_type: str
    query_text: str | None = None
    model_key: str
    embedding_dimension: int
    embedding_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float
    result_count: int
    results: list[SearchResultItem]


class MatchRequest(BaseModel):
    image_id: uuid.UUID
    candidate_descriptions: list[str] = Field(min_length=2, max_length=20)
    model_key: str | None = None


class MatchCandidateScore(BaseModel):
    description: str
    cosine_similarity: float
    relative_alignment_pct: float
    rank: int


class MatchResponse(BaseModel):
    image_id: uuid.UUID
    model_key: str
    latency_ms: float
    rankings: list[MatchCandidateScore]


class ModelOut(BaseModel):
    id: uuid.UUID
    model_key: str
    model_name: str
    model_version: str
    hf_model_id: str
    embedding_dimension: int
    architecture_family: str
    is_active: bool
    is_default: bool
    indexed_image_count: int = 0
    preprocessing_config: dict[str, Any]


class EvaluationRunRequest(BaseModel):
    model_key: str
    dataset_name: str = "curated_multimodal_benchmark_v1"


class EvaluationRunResponse(BaseModel):
    id: uuid.UUID
    model_key: str
    model_name: str
    dataset_name: str
    num_queries: int
    num_candidates: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mean_average_precision: float
    mean_reciprocal_rank: float
    average_latency_ms: float
    created_at: datetime


class SearchHistoryOut(BaseModel):
    id: uuid.UUID
    query_type: str
    query_text: str | None
    model_key: str
    result_count: int
    latency_ms: float
    created_at: datetime


class DashboardStatsResponse(BaseModel):
    total_images: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    total_embeddings: int
    active_model_key: str
    recent_searches: list[SearchHistoryOut]