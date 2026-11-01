from dataclasses import dataclass

import numpy as np

from visionmatch_ml.similarity.metrics import compute_similarity_matrix


@dataclass(frozen=True)
class EvaluationSummary:
    num_queries: int
    num_candidates: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mean_average_precision: float
    mean_reciprocal_rank: float


def compute_ir_metrics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: list[str],
    ground_truth_relevant_ids: list[set[str]],
) -> EvaluationSummary:
    """
    Computes standard Information Retrieval metrics across a set of queries:

    Recall@K: Fraction of relevant items retrieved in top-K,
    averaged over queries.

    Precision@5: Relevant items in top-5 divided by 5.

    MRR: Average reciprocal rank of the first relevant item.

    mAP: Mean Average Precision across all queries.
    """

    if len(ground_truth_relevant_ids) != query_embeddings.shape[0]:
        raise ValueError(
            "Mismatch between query count and ground truth list length."
        )

    sim_matrix = compute_similarity_matrix(
        query_embeddings,
        candidate_embeddings,
    )

    num_queries = sim_matrix.shape[0]
    num_candidates = len(candidate_ids)

    r1_list = []
    r5_list = []
    r10_list = []
    p5_list = []
    rr_list = []
    ap_list = []

    for q_idx in range(num_queries):
        relevant_set = ground_truth_relevant_ids[q_idx]

        if not relevant_set:
            continue

        scores = sim_matrix[q_idx]

        # Deterministic ranking
        ranked_indices = sorted(
            range(num_candidates),
            key=lambda i: (
                -round(float(scores[i]), 7),
                candidate_ids[i],
            ),
        )

        ranked_ids = [candidate_ids[i] for i in ranked_indices]

        relevance_binary = [
            1 if cid in relevant_set else 0
            for cid in ranked_ids
        ]

        total_rel = len(relevant_set)

        r1_list.append(sum(relevance_binary[:1]) / total_rel)
        r5_list.append(sum(relevance_binary[:5]) / total_rel)
        r10_list.append(sum(relevance_binary[:10]) / total_rel)

        p5_list.append(
            sum(relevance_binary[:5]) / min(5, num_candidates)
        )

        # Reciprocal Rank
        rr = 0.0

        for rank_idx, is_rel in enumerate(relevance_binary, start=1):
            if is_rel:
                rr = 1.0 / rank_idx
                break

        rr_list.append(rr)

        # Average Precision
        hits = 0
        sum_precisions = 0.0

        for rank_idx, is_rel in enumerate(relevance_binary, start=1):
            if is_rel:
                hits += 1
                sum_precisions += hits / rank_idx

        ap_list.append(
            sum_precisions / total_rel if total_rel > 0 else 0.0
        )

    return EvaluationSummary(
        num_queries=len(r1_list),
        num_candidates=num_candidates,
        recall_at_1=round(float(np.mean(r1_list or [0.0])), 4),
        recall_at_5=round(float(np.mean(r5_list or [0.0])), 4),
        recall_at_10=round(float(np.mean(r10_list or [0.0])), 4),
        precision_at_5=round(float(np.mean(p5_list or [0.0])), 4),
        mean_average_precision=round(float(np.mean(ap_list or [0.0])), 4),
        mean_reciprocal_rank=round(float(np.mean(rr_list or [0.0])), 4),
    )