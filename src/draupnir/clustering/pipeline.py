#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from draupnir.clustering.config import load_openai_compatible_api
from draupnir.clustering.data import CauseCardRecord
from draupnir.clustering.embeddings import EmbeddingService, SQLiteEmbeddingCache, embed_texts
from draupnir.clustering.features import extract_features
from draupnir.clustering.outputs import FINAL_FIELDS, final_row_from_base, write_clustering_outputs, write_jsonl
from draupnir.clustering.vectorize import l2_normalize_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: extract six-field cause-card embeddings and cluster them with average-linkage agglomerative clustering."
    )
    parser.add_argument("--cause-cards", default="outputs/stage2_cause_cards")
    parser.add_argument("--representative-weights", default="outputs/stage1_crash_cards/representative_weights.csv")
    parser.add_argument("--out", default="outputs/stage3_clustering")
    parser.add_argument("--api-file", default="")
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--truncate-dims", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache", default="outputs/embedding_cache.sqlite")
    parser.add_argument("--cached-vectors", default="", help="Optional vectors.npz for fast verification without embedding API calls.")
    parser.add_argument("--max-chars-per-field", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_cause_card(root: Path, row: dict[str, str]) -> Path | None:
    bug_id = row.get("source_bug_id", "")
    cause_id = row.get("cause_card_id") or row.get("small_bucket_cluster_id") or row.get("cause_card_file_stem", "")
    candidates = [
        root / row.get("directory_name", "") / row.get("copied_file_name", ""),
        root / bug_id / f"{cause_id}.json",
        root / bug_id / f"{row.get('small_bucket_cluster_id', '')}.json",
        root / row.get("copied_file_name", ""),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if bug_id and cause_id and (root / bug_id).is_dir():
        matches = sorted((root / bug_id).glob(f"*{cause_id}*.json"))
        if matches:
            return matches[0]
    if cause_id:
        matches = sorted(root.glob(f"**/*{cause_id}*.json"))
        if matches:
            return matches[0]
    return None


def build_records(cause_root: Path, weights_path: Path) -> list[CauseCardRecord]:
    rows = read_csv(weights_path)
    records: list[CauseCardRecord] = []
    missing: list[str] = []
    for row in rows:
        card_path = find_cause_card(cause_root, row)
        if card_path is None:
            missing.append(f"{row.get('source_bug_id', '')}/{row.get('cause_card_id', '')}")
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        record_id = f"{row.get('directory_name', '')}/{row.get('copied_file_name', '')}"
        records.append(
            CauseCardRecord(
                record_id=record_id,
                final_ground_truth_id=row["final_ground_truth_id"],
                directory_name=row.get("directory_name", ""),
                copied_file_name=row.get("copied_file_name", ""),
                cause_card_key=row.get("cause_card_key", ""),
                cause_card_id=row.get("cause_card_id", card_path.stem),
                source_bug_id=row.get("source_bug_id", ""),
                source_small_bucket=row.get("source_small_bucket", ""),
                source_crash_id=row.get("source_crash_id", ""),
                copied_target_file=card_path,
                source_file=str(card_path),
                weight=float(row.get("represented_crash_row_count") or 1.0),
                unique_case_count=float(row.get("unique_case_count") or row.get("represented_crash_row_count") or 1.0),
                card=card,
            )
        )
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"Missing {len(missing)} cause cards under {cause_root}. First missing: {preview}")
    return records


def cap(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"


def embed_rows(rows: list[dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> tuple[np.ndarray, dict[str, Any], str]:
    if args.cached_vectors:
        vector_path = Path(args.cached_vectors)
        with np.load(vector_path, allow_pickle=False) as data:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            cached_record_ids = [str(x) for x in data["record_ids"]] if "record_ids" in data else []
        if vectors.shape[0] != len(rows):
            raise SystemExit(f"Cached vector row count {vectors.shape[0]} does not match feature rows {len(rows)}")
        row_ids = [row["record_id"] for row in rows]
        if cached_record_ids and cached_record_ids != row_ids:
            raise SystemExit("Cached vector record_ids do not match the cause-card feature order.")
        meta_path = vector_path.with_name("vectorizer_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta["cached_vectors"] = str(vector_path)
        return vectors, meta, str(vector_path)

    field_texts = {
        field: [cap(row.get("field_texts", {}).get(field, ""), args.max_chars_per_field) for row in rows]
        for field in FINAL_FIELDS
    }
    api = load_openai_compatible_api(args.api_file or None)
    service = EmbeddingService(api, model=args.model)
    cache = SQLiteEmbeddingCache(args.cache)
    combined: np.ndarray | None = None
    dim: int | None = None
    try:
        for field in FINAL_FIELDS:
            nonempty = [(idx, text) for idx, text in enumerate(field_texts[field]) if text.strip()]
            if not nonempty:
                continue
            indices = [idx for idx, _ in nonempty]
            embeddings = embed_texts(
                [text for _, text in nonempty],
                service=service,
                cache=cache,
                batch_size=args.batch_size,
                dimensions=args.dimensions,
            )
            if args.truncate_dims is not None:
                embeddings = embeddings[:, : args.truncate_dims]
            embeddings = l2_normalize_dense(embeddings)
            if combined is None:
                dim = int(embeddings.shape[1])
                combined = np.zeros((len(rows), dim), dtype=np.float32)
            if embeddings.shape[1] != dim:
                raise ValueError(f"Embedding dimension changed for field {field}")
            combined[np.asarray(indices, dtype=int)] += embeddings
    finally:
        cache.close()
    if combined is None:
        raise RuntimeError("No nonempty cause-card field texts were embedded")
    combined = l2_normalize_dense(combined)
    vector_path = out_dir / "vectors.npz"
    np.savez_compressed(
        vector_path,
        vectors=combined.astype(np.float32),
        true_labels=np.asarray([row["true_label"] for row in rows]),
        weights=np.asarray([float(row["weight"]) for row in rows], dtype=np.float32),
        record_ids=np.asarray([row["record_id"] for row in rows]),
        cause_card_ids=np.asarray([row["cause_card_id"] for row in rows]),
    )
    meta = {
        "model": args.model,
        "dimensions": args.dimensions,
        "truncate_dims": args.truncate_dims,
        "fields": FINAL_FIELDS,
        "field_weights": {field: 1.0 for field in FINAL_FIELDS},
        "field_combination": "L2-normalize each nonempty field embedding, sum fields with equal weight, then L2-normalize the card vector.",
        "max_chars_per_field": args.max_chars_per_field,
        "records": len(rows),
        "vector_shape": list(combined.shape),
        "cache": args.cache,
    }
    (out_dir / "vectorizer_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return combined, meta, str(vector_path)


def cached_record_order(path: str) -> list[str]:
    if not path:
        return []
    with np.load(Path(path), allow_pickle=False) as data:
        if "record_ids" not in data:
            return []
        return [str(x) for x in data["record_ids"]]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(Path(args.cause_cards), Path(args.representative_weights))
    rows = [final_row_from_base(extract_features(record).to_json()) for record in records]
    cached_order = cached_record_order(args.cached_vectors)
    if cached_order:
        rows_by_id = {row["record_id"]: row for row in rows}
        missing_from_features = [record_id for record_id in cached_order if record_id not in rows_by_id]
        extra_features = [row["record_id"] for row in rows if row["record_id"] not in set(cached_order)]
        if missing_from_features or extra_features:
            raise SystemExit(
                "Cached vector record_ids do not match the available cause-card records. "
                f"missing={len(missing_from_features)} extra={len(extra_features)}"
            )
        rows = [rows_by_id[record_id] for record_id in cached_order]
    write_jsonl(out_dir / "six_field_cause_cards.jsonl", rows)

    nonempty_counts = {
        field: sum(1 for row in rows if row["fields"].get(field))
        for field in FINAL_FIELDS
    }
    summary = {
        "record_count": len(rows),
        "ground_truth_count": len({row["true_label"] for row in rows}),
        "total_weight": sum(float(row["weight"]) for row in rows),
        "fields": FINAL_FIELDS,
        "nonempty_counts": nonempty_counts,
    }
    (out_dir / "feature_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({**summary, "embedding": "not run in dry-run"}, ensure_ascii=True, sort_keys=True))
        return

    vectors, meta, vector_source = embed_rows(rows, args, out_dir)
    cluster_summary = write_clustering_outputs(rows, vectors, out_dir, vector_source, meta)
    print(json.dumps(cluster_summary["selected"], ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
