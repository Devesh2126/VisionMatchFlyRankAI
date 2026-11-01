
import io

import numpy as np
import pytest
from PIL import Image

from visionmatch_ml.preprocessing import (
    ImageValidationError,
    create_thumbnail_bytes,
    load_and_sanitize_image,
    sanitize_query_text,
)

from visionmatch_ml.similarity import (
    compute_cosine_similarity,
    l2_normalize,
    rank_candidates,
)

from visionmatch_ml.evaluation import compute_ir_metrics


def _make_test_image_bytes(
    width: int = 128,
    height: int = 128,
    fmt: str = "PNG",
) -> bytes:
    img = Image.new("RGBA", (width, height), color=(220, 40, 40, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_image_sanitization_converts_rgba_to_rgb_and_hashes():
    raw = _make_test_image_bytes(100, 80, "PNG")
    result = load_and_sanitize_image(raw)

    assert result.pil_image.mode == "RGB"
    assert result.width == 100
    assert result.height == 80
    assert result.mime_type == "image/png"
    assert len(result.sha256_checksum) == 64

    thumb_bytes = create_thumbnail_bytes(result.pil_image, max_dim=50)

    with Image.open(io.BytesIO(thumb_bytes)) as thumb:
        assert thumb.format == "WEBP"
        assert max(thumb.size) <= 50


def test_invalid_image_bytes_raises_validation_error():
    with pytest.raises(ImageValidationError):
        load_and_sanitize_image(b"not-an-image-file-payload")


def test_l2_normalization_and_cosine_similarity():
    vec_a = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    norm_a = l2_normalize(vec_a)

    assert np.isclose(np.linalg.norm(norm_a), 1.0, atol=1e-6)
    assert np.allclose(
        norm_a,
        np.array([0.6, 0.8, 0.0], dtype=np.float32),
    )

    assert np.isclose(
        compute_cosine_similarity(vec_a, vec_a * 5.0),
        1.0,
        atol=1e-6,
    )

    vec_ortho = np.array([0.0, 0.0, 10.0], dtype=np.float32)

    assert np.isclose(
        compute_cosine_similarity(vec_a, vec_ortho),
        0.0,
        atol=1e-6,
    )


def test_deterministic_candidate_ranking_and_ir_metrics():
    queries = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    candidates = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=np.float32,
    )

    c_ids = ["img_a", "img_b", "img_c"]
    ground_truth = [{"img_a"}, {"img_b"}]

    summary = compute_ir_metrics(
        queries,
        candidates,
        c_ids,
        ground_truth,
    )

    assert summary.recall_at_1 == 1.0
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.mean_average_precision == 1.0

    tied_candidates = np.array(
        [[1.0, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )

    ranked = rank_candidates(
        queries[0],
        tied_candidates,
        ["img_z", "img_a"],
    )

    assert ranked[0][0] == "img_a"
    assert ranked[1][0] == "img_z"


def test_sanitize_query_text():
    assert sanitize_query_text(
        "  A   red\n\tsports car  "
    ) == "A red sports car"

    with pytest.raises(ValueError):
        sanitize_query_text("   ")
