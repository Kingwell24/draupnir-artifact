from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CauseCardRecord:
    record_id: str
    final_ground_truth_id: str
    directory_name: str
    copied_file_name: str
    cause_card_key: str
    cause_card_id: str
    source_bug_id: str
    source_small_bucket: str
    source_crash_id: str
    copied_target_file: Path
    source_file: str
    weight: float
    unique_case_count: float
    card: dict[str, Any]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _norm_path(value: str | Path) -> str:
    return str(value).replace("/", "\\").lower()


def _load_weight_rows(root: Path, weights_name: str) -> dict[tuple[str, str], dict[str, str]]:
    path = root / weights_name
    rows = _read_csv(path) if path.exists() else []
    by_dir_file: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("directory_name", ""), row.get("copied_file_name", ""))
        by_dir_file[key] = row
    return by_dir_file


def load_records(
    benchmark_root: str | Path,
    manifest_name: str = "_manifest.csv",
    weights_name: str = "_cause_card_crash_counts.csv",
    weight_column: str = "represented_crash_row_count",
) -> list[CauseCardRecord]:
    root = Path(benchmark_root)
    manifest_path = root / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    weight_rows = _load_weight_rows(root, weights_name)
    records: list[CauseCardRecord] = []
    for row in _read_csv(manifest_path):
        copied_file = Path(row["target_file"])
        if not copied_file.exists():
            copied_file = root / row["directory_name"] / row["copied_file_name"]
        with copied_file.open("r", encoding="utf-8") as f:
            card = json.load(f)

        weight_row = weight_rows.get((row["directory_name"], row["copied_file_name"]), {})
        weight = float(weight_row.get(weight_column) or 1.0)
        unique_case_count = float(weight_row.get("unique_case_count") or weight)
        record_id = f"{row['directory_name']}/{row['copied_file_name']}"
        records.append(
            CauseCardRecord(
                record_id=record_id,
                final_ground_truth_id=row["final_ground_truth_id"],
                directory_name=row["directory_name"],
                copied_file_name=row["copied_file_name"],
                cause_card_key=row.get("cause_card_key", ""),
                cause_card_id=row.get("cause_card_id", card.get("cause_id", "")),
                source_bug_id=row.get("source_bug_dir_id", card.get("source_bucket", "")),
                source_small_bucket=row.get("source_small_bucket", card.get("source_small_bucket", "")),
                source_crash_id=row.get("source_crash_id", card.get("source_crash_id", "")),
                copied_target_file=copied_file,
                source_file=row.get("source_file", ""),
                weight=weight,
                unique_case_count=unique_case_count,
                card=card,
            )
        )
    return records


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
