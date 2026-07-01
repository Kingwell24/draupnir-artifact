from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .features import FIELD_ORDER


def l2_normalize_dense(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def combine_field_vectors(
    vectors_by_field: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    fields: Iterable[str] | None = None,
    truncate_dims: int | None = None,
) -> np.ndarray:
    fields = list(fields or FIELD_ORDER)
    weights = weights or {field: 1.0 for field in fields if field != "negative"}
    combined: np.ndarray | None = None
    for field in fields:
        if field not in vectors_by_field:
            continue
        weight = float(weights.get(field, 0.0))
        if weight <= 0:
            continue
        vec = np.asarray(vectors_by_field[field], dtype=np.float32)
        if truncate_dims is not None:
            vec = vec[:, :truncate_dims]
        vec = l2_normalize_dense(vec)
        combined = vec * weight if combined is None else combined + vec * weight
    if combined is None:
        raise ValueError("No vectors available for requested fields")
    return l2_normalize_dense(combined)


def weighted_text(row: dict, field_weights: dict[str, float] | None = None) -> str:
    parts: list[str] = []
    texts = row.get("field_texts") or {}
    field_weights = field_weights or {field: 1.0 for field in FIELD_ORDER if field != "negative"}
    for field in FIELD_ORDER:
        text = texts.get(field, "")
        weight = field_weights.get(field, 0.0)
        if not text or weight <= 0:
            continue
        repeats = max(1, int(round(weight)))
        parts.extend([text] * repeats)
    return "\n\n".join(parts)


def tfidf_svd_vectors(
    rows: list[dict],
    analyzer: str = "word_char",
    svd_dims: int = 256,
    random_state: int = 13,
) -> tuple[np.ndarray, dict]:
    texts = [weighted_text(row) for row in rows]
    matrices = []
    meta = {"analyzer": analyzer, "svd_dims": svd_dims}

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
