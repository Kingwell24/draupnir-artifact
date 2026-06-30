from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score

from .metrics import noise_as_singletons, weighted_purity_inverse_f
from .vectorize import l2_normalize_dense


@dataclass
class ClusterResult:
    algorithm: str
    params: dict[str, Any]
    labels: list[Any]
    internal: dict[str, float | int]
    metrics: dict[str, float | int] | None = None

    def summary_row(self) -> dict[str, Any]:
        row = {"algorithm": self.algorithm, **self.params, **self.internal}
        if self.metrics:
            row.update(self.metrics)
        return row


def _internal_scores(x: np.ndarray, labels: list[Any]) -> dict[str, float | int]:
    split = noise_as_singletons(labels)
    cluster_count = len(set(split))
    raw_noise = sum(1 for label in labels if str(label) == "-1")
    counts = Counter(split)
    largest_cluster_fraction = max(counts.values()) / max(1, len(labels))
    score = -1.0
    if 1 < cluster_count < len(labels):
        try:
            score = float(silhouette_score(x, split, metric="euclidean"))
        except Exception:
            score = -1.0
    return {
        "cluster_count_internal": int(cluster_count),
        "raw_noise_count_internal": int(raw_noise),
        "noise_fraction": float(raw_noise / max(1, len(labels))),
        "largest_cluster_fraction": float(largest_cluster_fraction),
        "silhouette": score,
    }


def run_hdbscan_grid(
    x: np.ndarray,
    eps_values: list[float] | None = None,
    min_cluster_sizes: list[int] | None = None,
    min_samples_values: list[int | None] | None = None,
) -> list[ClusterResult]:
    x = l2_normalize_dense(x)
    eps_values = eps_values or [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.18, 0.22, 0.28, 0.35, 0.45, 0.6, 0.8]
    min_cluster_sizes = min_cluster_sizes or [2, 3, 5]
    min_samples_values = min_samples_values or [1]
    results: list[ClusterResult] = []
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            for eps in eps_values:
                model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=float(eps),
                    metric="euclidean",
                    allow_single_cluster=False,
                )
                labels = model.fit_predict(x).tolist()
                params = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_selection_epsilon": float(eps),
                }
                results.append(ClusterResult("hdbscan", params, labels, _internal_scores(x, labels)))
    return results


def _eps_grid_from_distances(x: np.ndarray, count: int = 28) -> list[float]:
    sample = x
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(sample @ sample.T, -1.0, 1.0)))
    distances[distances == 0] = np.nan
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return [0.5]
    qs = np.linspace(0.01, 0.25, count)
    values = sorted({float(np.quantile(finite, q)) for q in qs})
    return values


def run_dbscan_grid(
    x: np.ndarray,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
) -> list[ClusterResult]:
    x = l2_normalize_dense(x)
    eps_values = eps_values or _eps_grid_from_distances(x)
    min_samples_values = min_samples_values or [1, 2, 3]
    results: list[ClusterResult] = []
    for min_samples in min_samples_values:
        for eps in eps_values:
            model = DBSCAN(eps=float(eps), min_samples=min_samples, metric="euclidean", n_jobs=-1)
            labels = model.fit_predict(x).tolist()
            params = {"eps": float(eps), "min_samples": min_samples}
            results.append(ClusterResult("dbscan", params, labels, _internal_scores(x, labels)))
    return results


def run_agglomerative_grid(
    x: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[ClusterResult]:
    x = l2_normalize_dense(x)
    thresholds = thresholds or [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results: list[ClusterResult] = []
    for threshold in thresholds:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="euclidean",
            linkage="average",
            distance_threshold=float(threshold),
        )
        labels = model.fit_predict(x).tolist()
        params = {"distance_threshold": float(threshold)}
        results.append(ClusterResult("agglomerative", params, labels, _internal_scores(x, labels)))
    return results


def run_kmeans(x: np.ndarray, n_clusters: int, random_state: int = 13) -> ClusterResult:
    x = l2_normalize_dense(x)
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = model.fit_predict(x).tolist()
    params = {"n_clusters": n_clusters, "random_state": random_state}
    return ClusterResult("kmeans", params, labels, _internal_scores(x, labels))


def attach_metrics(
    results: list[ClusterResult],
    true_labels: list[Any],
    weights: list[float] | np.ndarray | None = None,
) -> list[ClusterResult]:
    for result in results:
        result.metrics = weighted_purity_inverse_f(true_labels, result.labels, weights, split_noise=True)
    return results


def select_unsupervised(
    results: list[ClusterResult],
    desired_cluster_count: int | None = None,
    silhouette_slack: float = 0.02,
    max_largest_cluster_fraction: float = 0.25,
) -> ClusterResult:
    valid = [r for r in results if 1 < r.internal["cluster_count_internal"] < len(r.labels)]
    bounded = [
        r
        for r in valid
        if float(r.internal.get("largest_cluster_fraction", 1.0)) <= max_largest_cluster_fraction
    ]
    if bounded:
        valid = bounded
    if not valid:
        return results[0]
    if desired_cluster_count is not None:
        return sorted(
            valid,
            key=lambda r: (
                abs(int(r.internal["cluster_count_internal"]) - desired_cluster_count),
                -float(r.internal["silhouette"]),
                float(r.internal["noise_fraction"]),
            ),
        )[0]
    max_sil = max(float(r.internal["silhouette"]) for r in valid)
    near = [r for r in valid if float(r.internal["silhouette"]) >= max_sil - silhouette_slack]
    return sorted(
        near,
        key=lambda r: (
            int(r.internal["cluster_count_internal"]),
            float(r.internal["noise_fraction"]),
            -float(r.internal["silhouette"]),
        ),
    )[0]


def select_oracle(results: list[ClusterResult]) -> ClusterResult:
    with_metrics = [r for r in results if r.metrics is not None]
    if not with_metrics:
        raise ValueError("select_oracle requires metrics")
    return sorted(
        with_metrics,
        key=lambda r: (
            -float(r.metrics["f_measure"]),
            -float(r.metrics["purity"]),
            abs(int(r.metrics["cluster_count"]) - int(r.metrics["label_count"])),
        ),
    )[0]
