from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI

from .config import ApiConfig


class SQLiteEmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def key(model: str, text: str, dimensions: int | None = None) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\0")
        h.update(str(dimensions or "").encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def get(self, model: str, text: str, dimensions: int | None = None) -> np.ndarray | None:
        key = self.key(model, text, dimensions)
        row = self.conn.execute("SELECT dim, vector FROM embeddings WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32).copy().reshape(int(dim))

    def put(self, model: str, text: str, vector: Iterable[float], dimensions: int | None = None) -> None:
        arr = np.asarray(list(vector), dtype=np.float32)
        key = self.key(model, text, dimensions)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings(cache_key, model, text_sha256, dim, vector, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (key, model, text_hash, int(arr.shape[0]), arr.tobytes(), time.time()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


class EmbeddingService:
    def __init__(self, api: ApiConfig, model: str, timeout: float = 120.0):
        self.model = model
        self.client = OpenAI(api_key=api.api_key, base_url=api.base_url, timeout=timeout)

    def embed_batch(self, texts: list[str], dimensions: int | None = None) -> list[list[float]]:
        kwargs = {"model": self.model, "input": texts}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = self.client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


def embed_texts(
    texts: list[str],
    service: EmbeddingService,
    cache: SQLiteEmbeddingCache,
    batch_size: int = 64,
    dimensions: int | None = None,
    max_retries: int = 5,
    sleep_base: float = 2.0,
) -> np.ndarray:
    vectors: list[np.ndarray | None] = [None] * len(texts)
    missing: list[tuple[int, str]] = []
    for idx, text in enumerate(texts):
        cached = cache.get(service.model, text, dimensions)
        if cached is None:
            missing.append((idx, text))
        else:
            vectors[idx] = cached

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        batch_texts = [text for _, text in batch]
        for attempt in range(max_retries):
            try:
                batch_vectors = service.embed_batch(batch_texts, dimensions=dimensions)
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(sleep_base * (2**attempt))
        for (idx, text), vec in zip(batch, batch_vectors):
            arr = np.asarray(vec, dtype=np.float32)
            cache.put(service.model, text, arr, dimensions)
            vectors[idx] = arr
        cache.commit()

    if any(vec is None for vec in vectors):
        raise RuntimeError("Embedding vector missing after API/cache pass")
    return np.vstack([vec for vec in vectors if vec is not None]).astype(np.float32)
