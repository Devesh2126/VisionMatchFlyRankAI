import numpy as np


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Projects a 1D vector (D,) or 2D batch of vectors (N, D) onto the unit hypersphere:
        v_hat = v / max(||v||_2, eps)

    Why normalization matters:
    1. Removes magnitude variance caused by image contrast or token length.
    2. Reduces Cosine Similarity to a fast inner dot product (u_hat · v_hat).
    """
    arr = np.asarray(vectors, dtype=np.float32)

    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr / max(float(norm), eps)

    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, eps)

    raise ValueError(
        f"Expected 1D or 2D array for L2 normalization, got {arr.ndim}D."
    )


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes exact cosine similarity between two 1D vectors.
    Returns a float in [-1.0, 1.0].
    This is NOT a calibrated classification probability.
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("compute_cosine_similarity expects two 1D vectors.")

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Incompatible embedding dimensions: {a.shape[0]} vs {b.shape[0]}."
        )

    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)

    score = float(np.dot(a_norm, b_norm))
    return float(np.clip(score, -1.0, 1.0))


def compute_similarity_matrix(
    queries: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Computes an (N_queries, N_candidates) cosine similarity matrix.
    """
    q = np.asarray(queries, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)

    if q.ndim == 1:
        q = q[np.newaxis, :]

    if c.ndim == 1:
        c = c[np.newaxis, :]

    if q.ndim != 2 or c.ndim != 2:
        raise ValueError("Queries and candidates must be 1D or 2D arrays.")

    if q.shape[1] != c.shape[1]:
        raise ValueError(
            f"Incompatible embedding dimensions: {q.shape[1]} vs {c.shape[1]}."
        )

    q_norm = l2_normalize(q)
    c_norm = l2_normalize(c)

    return np.clip(q_norm @ c_norm.T, -1.0, 1.0)


def rank_candidates(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: list[str],
    top_k: int | None = None,
) -> list[tuple[str, float, int]]:
    """
    Ranks candidates by cosine similarity in descending order with deterministic
    tie-breaking by candidate_id ascending.

    Returns list of (candidate_id, similarity_score, rank_1_indexed).
    """
    if len(candidate_ids) == 0:
        return []

    if top_k is not None and top_k < 0:
        raise ValueError("top_k cannot be negative.")

    if len(candidate_ids) != len(candidate_vecs):
        raise ValueError("Number of candidate IDs must match candidate vectors.")

    sim_matrix = compute_similarity_matrix(query_vec, candidate_vecs)[0]

    scored = list(zip(candidate_ids, sim_matrix.tolist()))

    scored.sort(key=lambda item: (-round(item[1], 7), item[0]))

    if top_k is not None:
        scored = scored[:top_k]

    return [
        (cid, float(score), idx + 1)
        for idx, (cid, score) in enumerate(scored)
    ]