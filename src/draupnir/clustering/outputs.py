from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from draupnir.clustering.metrics import pairwise_weighted_counts, weighted_purity_inverse_f
from draupnir.clustering.normalize import canonical_token, ordered_unique, token_to_words
from draupnir.clustering.vectorize import l2_normalize_dense


FINAL_FIELDS = ["object", "invariant", "propagation", "patch", "must", "should"]
TYPED_FIELDS = ["object", "invariant", "propagation", "patch"]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_field(name: str, values: list[str]) -> str:
    if not values:
        return ""
    lines = [f"[{name}]"]
    for value in values:
        token = canonical_token(value)
        words = token_to_words(token)
        if words and words != token:
            lines.append(f"token: {token} ; words: {words}")
        else:
            lines.append(f"token: {token}")
    return "\n".join(lines)


def build_final_fields(base_fields: dict[str, list[str]]) -> dict[str, list[str]]:
    typed = {field: ordered_unique(base_fields.get(field, [])) for field in TYPED_FIELDS}
    covered = {canonical_token(value) for field in TYPED_FIELDS for value in typed[field]}

    must = [
        token
        for token in ordered_unique(base_fields.get("must", []))
        if canonical_token(token) not in covered
    ]
    should = [
        token
        for token in ordered_unique(base_fields.get("should", []))
        if canonical_token(token) not in covered
    ]

    return {
        "object": typed["object"],
        "invariant": typed["invariant"],
        "propagation": typed["propagation"],
        "patch": typed["patch"],
        "must": must,
        "should": should,
    }


def final_row_from_base(base: dict[str, Any]) -> dict[str, Any]:
    fields = build_final_fields(base.get("fields", {}))
    field_texts = {field: format_field(field, fields[field]) for field in FINAL_FIELDS}
    all_text = "\n\n".join(field_texts[field] for field in FINAL_FIELDS if field_texts[field])
    return {
        "record_id": base["record_id"],
        "true_label": base["true_label"],
        "weight": float(base["weight"]),
        "unique_case_count": float(base.get("unique_case_count", base["weight"])),
        "cause_card_id": base.get("cause_card_id", ""),
        "cause_card_key": base.get("cause_card_key", ""),
        "source_crash_id": base.get("source_crash_id", ""),
        "copied_file_name": base.get("copied_file_name", ""),
        "fields": fields,
        "field_texts": field_texts,
        "all_text": all_text,
        "stats": {
            "field_token_counts": {field: len(fields[field]) for field in FINAL_FIELDS},
            "final_fields": FINAL_FIELDS,
            "deduplication_rule": "typed object/invariant/propagation/patch tokens are removed from must/should when identical",
        },
    }


def equal_field_text(row: dict[str, Any], fields: list[str] | None = None) -> str:
    fields = fields or FINAL_FIELDS
    texts = row.get("field_texts", {})
    return "\n\n".join(texts.get(field, "") for field in fields if texts.get(field, ""))


def tfidf_svd_vectors_equal_fields(
    rows: list[dict[str, Any]],
    analyzer: str = "word_char",
    svd_dims: int = 256,
    random_state: int = 13,
    fields: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    fields = list(fields or FINAL_FIELDS)
    texts = [equal_field_text(row, fields) for row in rows]
    matrices = []
    meta: dict[str, Any] = {
        "analyzer": analyzer,
        "svd_dims_requested": int(svd_dims),
        "field_weights": {field: 1.0 for field in fields},
        "fields": fields,
    }

    if analyzer in {"word", "word_char"}:
        word_vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)[A-Za-z0-9_.$:/()=+\-<>]+",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        word_x = word_vec.fit_transform(texts)
        matrices.append(word_x)
        meta["word_vocab_size"] = len(word_vec.vocabulary_)

    if analyzer in {"char", "word_char"}:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        char_x = char_vec.fit_transform(texts)
        matrices.append(char_x)
        meta["char_vocab_size"] = len(char_vec.vocabulary_)

    if not matrices:
        raise ValueError(f"Unsupported analyzer: {analyzer}")
    x = sparse.hstack(matrices, format="csr") if len(matrices) > 1 else matrices[0]
    max_dims = max(2, min(svd_dims, x.shape[0] - 1, x.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_dims, random_state=random_state)
    dense = svd.fit_transform(x)
    dense = normalize(dense, norm="l2").astype(np.float32)
    meta["input_dim"] = int(x.shape[1])
    meta["actual_svd_dims"] = int(max_dims)
    meta["explained_variance_ratio_sum"] = float(svd.explained_variance_ratio_.sum())
    return dense, meta


def default_thresholds() -> list[float]:
    return [round(x, 3) for x in np.arange(0.20, 1.201, 0.025)]


def _internal_scores(x: np.ndarray, labels: list[int]) -> dict[str, float | int]:
    cluster_count = len(set(labels))
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    score = -1.0
    if 1 < cluster_count < len(labels):
        try:
            score = float(silhouette_score(x, labels, metric="euclidean"))
        except Exception:
            score = -1.0
    return {
        "cluster_count_internal": int(cluster_count),
        "largest_cluster_fraction": float(max(counts.values()) / max(1, len(labels))),
        "silhouette": score,
    }


def agglomerative_silhouette_grid(
    x: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    x = l2_normalize_dense(x)
    results: list[dict[str, Any]] = []
    for threshold in thresholds or default_thresholds():
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="euclidean",
            linkage="average",
            distance_threshold=float(threshold),
        )
        labels = model.fit_predict(x).tolist()
        results.append(
            {
                "algorithm": "agglomerative",
                "distance_threshold": float(threshold),
                "labels": labels,
                **_internal_scores(x, labels),
            }
        )
    return results


def select_by_max_silhouette(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        result
        for result in results
        if 1 < int(result["cluster_count_internal"]) < len(result["labels"])
    ]
    if not valid:
        return results[0]
    return sorted(
        valid,
        key=lambda result: (
            -float(result["silhouette"]),
            int(result["cluster_count_internal"]),
            float(result["largest_cluster_fraction"]),
        ),
    )[0]


def attach_external_metrics(
    results: list[dict[str, Any]],
    true_labels: list[str],
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    enriched = []
    for result in results:
        metrics = weighted_purity_inverse_f(true_labels, result["labels"], weights, split_noise=True)
        pairwise = pairwise_weighted_counts(true_labels, result["labels"], weights)
        row = {key: value for key, value in result.items() if key != "labels"}
        row.update(metrics)
        row.update(pairwise)
        enriched.append({**row, "labels": result["labels"]})
    return enriched


def write_clustering_outputs(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    out_dir: str | Path,
    vector_source: str,
    vectorizer_meta: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["true_label"] for row in rows]
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=float)

    grid = attach_external_metrics(agglomerative_silhouette_grid(vectors), labels, weights)
    selected = select_by_max_silhouette(grid)
    oracle = sorted(
        grid,
        key=lambda result: (
            -float(result["f_measure"]),
            -float(result["purity"]),
            abs(int(result["cluster_count"]) - len(set(labels))),
        ),
    )[0]

    grid_rows = [{key: value for key, value in result.items() if key != "labels"} for result in grid]
    all_keys = sorted({key for row in grid_rows for key in row})
    write_csv(out_dir / "grid.csv", grid_rows, all_keys)

    for name, result in [("selected", selected), ("oracle_diagnostic", oracle)]:
        assignments = []
        for row, pred in zip(rows, result["labels"]):
            assignments.append(
                {
                    "record_id": row["record_id"],
                    "true_label": row["true_label"],
                    "pred_cluster": pred,
                    "weight": row["weight"],
                    "cause_card_id": row.get("cause_card_id", ""),
                    "source_crash_id": row.get("source_crash_id", ""),
                    "copied_file_name": row.get("copied_file_name", ""),
                }
            )
        write_csv(
            out_dir / f"assignments_{name}.csv",
            assignments,
            ["record_id", "true_label", "pred_cluster", "weight", "cause_card_id", "source_crash_id", "copied_file_name"],
        )

    def summarize(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if key != "labels"}

    summary = {
        "selection": "max_silhouette_over_average_linkage_agglomerative_threshold_grid",
        "vector_source": vector_source,
        "vectorizer": vectorizer_meta,
        "fields": vectorizer_meta.get("fields", FINAL_FIELDS),
        "field_weights": vectorizer_meta.get(
            "field_weights",
            {field: 1.0 for field in vectorizer_meta.get("fields", FINAL_FIELDS)},
        ),
        "record_count": len(rows),
        "ground_truth_count": len(set(labels)),
        "total_weight": float(weights.sum()),
        "selected": summarize(selected),
        "oracle_diagnostic": summarize(oracle),
        "candidate_count": len(grid),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return summary
