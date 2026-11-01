from threading import Lock

from visionmatch_ml.interfaces.base_model import EmbeddingModel, ModelMetadata
from visionmatch_ml.models.hf_vlm import HuggingFaceVLM


# Pre-configured open-weight models
MODEL_CATALOG: dict[str, ModelMetadata] = {
    "clip-vit-b32": ModelMetadata(
        model_key="clip-vit-b32",
        model_name="CLIP ViT-B/32",
        model_version="1.0.0",
        hf_model_id="openai/clip-vit-base-patch32",
        hf_revision="main",
        embedding_dimension=512,
        architecture_family="clip",
        preprocessing_config={
            "image_size": 224,
            "patch_size": 32,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
    ),

    "clip-vit-b16": ModelMetadata(
        model_key="clip-vit-b16",
        model_name="CLIP ViT-B/16",
        model_version="1.0.0",
        hf_model_id="openai/clip-vit-base-patch16",
        hf_revision="main",
        embedding_dimension=512,
        architecture_family="clip",
        preprocessing_config={
            "image_size": 224,
            "patch_size": 16,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
    ),

    "siglip-base-224": ModelMetadata(
        model_key="siglip-base-224",
        model_name="SigLIP Base Patch16-224",
        model_version="1.0.0",
        hf_model_id="google/siglip-base-patch16-224",
        hf_revision="main",
        embedding_dimension=768,
        architecture_family="siglip",
        preprocessing_config={
            "image_size": 224,
            "patch_size": 16,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    ),
}


class ModelRegistry:
    """
    Thread-safe model registry with LRU-style memory management.
    """

    def __init__(self, max_loaded_models: int = 1) -> None:
        self._max_loaded_models = max_loaded_models
        self._instances: dict[str, EmbeddingModel] = {}
        self._load_order: list[str] = []
        self._lock = Lock()

    def list_available_models(self) -> list[ModelMetadata]:
        return list(MODEL_CATALOG.values())

    def get_metadata(self, model_key: str) -> ModelMetadata:
        if model_key not in MODEL_CATALOG:
            raise KeyError(
                f"Unknown model_key '{model_key}'. "
                f"Available: {list(MODEL_CATALOG.keys())}"
            )

        return MODEL_CATALOG[model_key]

    def get_model(
        self,
        model_key: str,
        auto_load: bool = True,
    ) -> EmbeddingModel:

        metadata = self.get_metadata(model_key)

        with self._lock:
            if model_key not in self._instances:
                self._instances[model_key] = HuggingFaceVLM(metadata=metadata)

            model = self._instances[model_key]

            if auto_load and not model.is_loaded():

                while len(self._load_order) >= self._max_loaded_models:
                    oldest_key = self._load_order.pop(0)

                    if oldest_key in self._instances:
                        self._instances[oldest_key].unload_model()

                model.load_model()
                self._load_order.append(model_key)

            return model


_GLOBAL_REGISTRY = ModelRegistry(max_loaded_models=2)


def get_model_registry() -> ModelRegistry:
    return _GLOBAL_REGISTRY