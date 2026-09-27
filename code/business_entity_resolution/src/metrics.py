"""
Macro F_0.5 metric evaluation for Amazon ML Challenge 2026.
Precision is weighted 2x over Recall:
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
"""

from typing import Dict, Set, Iterable


def compute_entity_f05(pred_ids: Set[str], gt_ids: Set[str]) -> float:
    """
    Computes F_0.5 for a single Source 1 entity.
    Includes singleton evaluation logic:
    - If gt is empty and pred is empty -> 1.0
    - If gt is empty and pred is non-empty -> 0.0
    - If gt is non-empty and pred is empty -> 0.0
    """
    len_gt = len(gt_ids)
    len_pred = len(pred_ids)

    # Singleton ground truth
    if len_gt == 0:
        return 1.0 if len_pred == 0 else 0.0

    # Non-singleton ground truth, but predicted empty
    if len_pred == 0:
        return 0.0

    # Non-empty ground truth and non-empty prediction
    intersection_count = len(pred_ids & gt_ids)
    if intersection_count == 0:
        return 0.0

    precision = intersection_count / len_pred
    recall = intersection_count / len_gt

    denom = (0.25 * precision) + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Iterable[str],
) -> Dict[str, float]:
    """
    Computes Macro F_0.5 averaged over all provided S1 entities.
    Returns:
        dict with keys: 'macro_f05', 'mean_precision', 'mean_recall', 'num_entities', 'num_singletons'
    """
    total_f05 = 0.0
    total_p = 0.0
    total_r = 0.0
    evaluated_count = 0
    singleton_count = 0

    for s1_id in all_s1_ids:
        gt_ids = ground_truth.get(s1_id, set())
        pred_ids = predictions.get(s1_id, set())

        score = compute_entity_f05(pred_ids, gt_ids)
        total_f05 += score

        len_gt = len(gt_ids)
        len_pred = len(pred_ids)
        if len_gt == 0:
            singleton_count += 1
            p = 1.0 if len_pred == 0 else 0.0
            r = 1.0 if len_pred == 0 else 0.0
        else:
            tp = len(pred_ids & gt_ids)
            p = tp / len_pred if len_pred > 0 else 0.0
            r = tp / len_gt
        total_p += p
        total_r += r
        evaluated_count += 1

    if evaluated_count == 0:
        return {"macro_f05": 0.0, "mean_precision": 0.0, "mean_recall": 0.0}

    return {
        "macro_f05": total_f05 / evaluated_count,
        "mean_precision": total_p / evaluated_count,
        "mean_recall": total_r / evaluated_count,
        "num_entities": evaluated_count,
        "num_singletons": singleton_count,
        "singleton_ratio": singleton_count / evaluated_count,
    }