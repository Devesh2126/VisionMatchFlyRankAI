import gc

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor

from visionmatch_ml.interfaces.base_model import EmbeddingModel, ModelMetadata
from visionmatch_ml.preprocessing.text_ops import sanitize_query_text
from visionmatch_ml.similarity.metrics import l2_normalize


def detect_optimal_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


class HuggingFaceVLM(EmbeddingModel):
    """
    Production Vision-Language Model wrapper supporting CLIP and SigLIP.
    """

    def __init__(
        self,
        metadata: ModelMetadata,
        device: str | None = None,
    ) -> None:
        self._metadata = metadata
        self.device = detect_optimal_device(device)
        self._model: AutoModel | None = None
        self._processor: AutoProcessor | None = None

    def load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        self._processor = AutoProcessor.from_pretrained(
            self._metadata.hf_model_id,
            revision=self._metadata.hf_revision,
        )

        self._model = AutoModel.from_pretrained(
            self._metadata.hf_model_id,
            revision=self._metadata.hf_revision,
        )

        self._model.to(self.device)
        self._model.eval()

    def unload_model(self) -> None:
        self._model = None
        self._processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def encode_images(
        self,
        images: list[Image.Image],
        batch_size: int = 16,
    ) -> np.ndarray:

        if not images:
            return np.empty(
                (0, self._metadata.embedding_dimension),
                dtype=np.float32,
            )

        self.load_model()

        assert self._model is not None and self._processor is not None

        all_embeddings: list[np.ndarray] = []
        rgb_images = [img.convert("RGB") for img in images]

        with torch.inference_mode():
            for start_idx in range(0, len(rgb_images), batch_size):
                batch = rgb_images[start_idx:start_idx + batch_size]

                inputs = self._processor(
                    images=batch,
                    return_tensors="pt",
                )

                pixel_values = inputs["pixel_values"].to(self.device)

                image_features = self._model.get_image_features(
                    pixel_values=pixel_values
                )

                if hasattr(image_features, "pooler_output") and not isinstance(
                    image_features, torch.Tensor
                ):
                    image_features = image_features.pooler_output
                elif not isinstance(image_features, torch.Tensor):
                    image_features = image_features[0]

                batch_np = (
                    image_features.detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )

                normalized = l2_normalize(batch_np)
                all_embeddings.append(normalized)

        result = np.vstack(all_embeddings)

        if result.shape[1] != self._metadata.embedding_dimension:
            raise RuntimeError(
                f"Expected dimension {self._metadata.embedding_dimension}, "
                f"got {result.shape[1]}"
            )

        return result

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:

        if not texts:
            return np.empty(
                (0, self._metadata.embedding_dimension),
                dtype=np.float32,
            )

        self.load_model()

        assert self._model is not None and self._processor is not None

        cleaned_texts = [sanitize_query_text(t) for t in texts]
        all_embeddings: list[np.ndarray] = []

        padding_mode = (
            "max_length"
            if self._metadata.architecture_family == "siglip"
            else True
        )

        with torch.inference_mode():
            for start_idx in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[start_idx:start_idx + batch_size]

                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=padding_mode,
                    truncation=True,
                )

                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                    if k in {"input_ids", "attention_mask"}
                }

                text_features = self._model.get_text_features(**inputs)

                if hasattr(text_features, "pooler_output") and not isinstance(
                    text_features, torch.Tensor
                ):
                    text_features = text_features.pooler_output
                elif not isinstance(text_features, torch.Tensor):
                    text_features = text_features[0]

                batch_np = (
                    text_features.detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )

                normalized = l2_normalize(batch_np)
                all_embeddings.append(normalized)

        result = np.vstack(all_embeddings)

        if result.shape[1] != self._metadata.embedding_dimension:
            raise RuntimeError(
                f"Expected dimension {self._metadata.embedding_dimension}, "
                f"got {result.shape[1]}"
            )

        return result

    def get_embedding_dimension(self) -> int:
        return self._metadata.embedding_dimension

    def get_model_metadata(self) -> ModelMetadata:
        return self._metadata