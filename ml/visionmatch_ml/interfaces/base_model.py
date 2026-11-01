from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModelMetadata:
    model_key: str
    model_name: str
    model_version: str
    hf_model_id: str
    hf_revision: str
    embedding_dimension: int
    architecture_family: str
    preprocessing_config: dict[str, Any]


class EmbeddingModel(ABC):
    """
    Abstract interface for Vision-Language Embedding Models.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Loads weights and processors into target device memory."""

    @abstractmethod
    def unload_model(self) -> None:
        """Releases model weights and clears device cache."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if weights are currently resident in memory."""

    @abstractmethod
    def encode_images(
        self, images: list[Image.Image], batch_size: int = 16
    ) -> np.ndarray:
        """Encodes images into L2-normalized embeddings."""

    @abstractmethod
    def encode_texts(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """Encodes texts into L2-normalized embeddings."""

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Returns a single image embedding."""
        return self.encode_images([image], batch_size=1)[0]

    def encode_text(self, text: str) -> np.ndarray:
        """Returns a single text embedding."""
        return self.encode_texts([text], batch_size=1)[0]

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Returns the fixed output vector dimension."""

    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata:
        """Returns model metadata."""