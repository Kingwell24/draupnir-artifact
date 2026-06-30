from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable

import numpy as np


def noise_as_singletons(labels: list[Any] | np.ndarray) -> list[Hashable]:
    out: list[Hashable] = []
    for idx, label in enumerate(labels):
        if str(label) == "-1":
            out.append(f"noise:{idx}")
        else:
            out.append(str(label))
    return out


def weighted_purity_inverse_f(
    true_labels: list[Any] | np.ndarray,
    pred_labels: list[Any] | np.ndarray,
    weights: list[float] | np.ndarray | None = None,
    split_noise: bool = True,
) -> dict[str, float | int]:
    true = [str(x) for x in true_labels]
    pred = noise_as_singletons(pred_labels) if split_noise else [str(x) for x in pred_labels]
    if len(true) != len(pred):
        raise ValueError("true_labels and pred_labels must have the same length")
    if weights is None:
        w = np.ones(len(true), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("sum of weights must be positive")

    label_weight: dict[str, float] = defaultdict(float)
    cluster_weight: dict[Hashable, float] = defaultdict(float)
    intersection: dict[tuple[str, Hashable], float] = defaultdict(float)
    for label, cluster, weight in zip(true, pred, w):
        label_weight[label] += float(weight)
        cluster_weight[cluster] += float(weight)
        intersection[(label, cluster)] += float(weight)

    labels = sorted(label_weight)
    clusters = sorted(cluster_weight, key=str)

    purity_num = 0.0
    for cluster in clusters:
        purity_num += max(intersection.get((label, cluster), 0.0) for label in labels)
    purity = purity_num / total

    inverse_num = 0.0
    f_num = 0.0
    for label in labels:
        best_recall = 0.0
        best_f = 0.0
        lw = label_weight[label]
        for cluster in clusters:
            iw = intersection.get((label, cluster), 0.0)
            if iw <= 0:
                continue
            precision = iw / cluster_weight[cluster]
            recall = iw / lw
            best_recall = max(best_recall, recall)
            denom = precision + recall
            f_val = 0.0 if denom == 0 else 2 * precision * recall / denom
            best_f = max(best_f, f_val)
        inverse_num += lw * best_recall
        f_num += lw * best_f
    inverse_purity = inverse_num / total
    f_measure = f_num / total

    raw_noise_count = sum(1 for label in pred_labels if str(label) == "-1")
    return {
        "purity": purity,
        "inverse_purity": inverse_purity,
        "f_measure": f_measure,
        "cluster_count": len(clusters),
        "label_count": len(labels),
        "item_count": len(true),
        "total_weight": total,
        "raw_noise_count": raw_noise_count,
        "noise_weight": float(sum(float(weight) for label, weight in zip(pred_labels, w) if str(label) == "-1")),
    }


def pairwise_weighted_counts(
    true_labels: list[Any] | np.ndarray,
    pred_labels: list[Any] | np.ndarray,
    weights: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    true = [str(x) for x in true_labels]
    pred = [str(x) for x in noise_as_singletons(pred_labels)]
    w = np.ones(len(true), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    tp = fp = fn = 0.0
    n = len(true)
    for i in range(n):
        for j in range(i + 1, n):
            pair_w = float(w[i] * w[j])
            same_true = true[i] == true[j]
            same_pred = pred[i] == pred[j]
            if same_true and same_pred:
                tp += pair_w
            elif same_pred and not same_true:
                fp += pair_w
            elif same_true and not same_pred:
                fn += pair_w
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "pair_tp_weight": tp,
        "pair_fp_weight": fp,
        "pair_fn_weight": fn,
    }
