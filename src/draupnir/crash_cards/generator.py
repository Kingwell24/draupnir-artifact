#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from draupnir.crash_cards.parser import build_cluster_summary, cluster_exact, make_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: generate source-level crash cards from raw syzbot crashes, "
            "deduplicate identical normalized crash cards, and select one representative per small bucket."
        )
    )
    parser.add_argument("--input", default="data/crashes_by_bug")
    parser.add_argument("--ground-truth", default="data/ground_truth.csv")
    parser.add_argument("--out", default="outputs/stage1_crash_cards")
    parser.add_argument("--limit-bugs", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--limit-crashes-per-bug", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--resume", action="store_true", help="Reuse already generated per-bug crash-card summaries and representatives.")
    return parser.parse_args()


def read_ground_truth(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "row_count_by_case": defaultdict(int),
            "labels_by_case": defaultdict(Counter),
            "rows": [],
        }
    rows: list[dict[str, str]] = []
    row_count_by_case: defaultdict[tuple[str, str], int] = defaultdict(int)
    labels_by_case: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            bug_id = row["bug_id"]
            case_id = row["case_id"]
            label = row.get("final_ground_truth_id", "")
            rows.append(row)
            row_count_by_case[(bug_id, case_id)] += 1
            if label:
                labels_by_case[(bug_id, case_id)][label] += 1
    return {
        "row_count_by_case": row_count_by_case,
        "labels_by_case": labels_by_case,
        "rows": rows,
    }


def representative_score(card: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    repro = card.get("repro") or {}
    repro_sem = card.get("reproducer_semantics") or {}
    lockdep = card.get("lockdep_context") or {}
    sanitizer = card.get("sanitizer_context") or {}
    bpf_bridge = lockdep.get("bpf_tracepoint_bridge") or {}
    return (
        1 if repro.get("has_syz_repro") else 0,
        1 if repro.get("has_c_repro") else 0,
        len(repro_sem.get("semantic_tokens") or []),
        len(bpf_bridge.get("bpf_frames") or []) + len(bpf_bridge.get("map_operations") or []),
        len(lockdep.get("existing_dependency_chain") or []),
        len(sanitizer.get("alloc_trace") or [])
        + len(sanitizer.get("free_trace") or [])
        + len(sanitizer.get("origin_trace") or []),
        len(card.get("stack_semantic") or []),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_label_for_members(
    bug_id: str,
    members: list[str],
    row_count_by_case: dict[tuple[str, str], int],
    labels_by_case: dict[tuple[str, str], Counter[str]],
) -> str:
    counts: Counter[str] = Counter()
    for case_id in members:
        key = (bug_id, case_id)
        for label, count in labels_by_case.get(key, {}).items():
            counts[label] += count
    return counts.most_common(1)[0][0] if counts else ""


def make_representative_row(
    label: str,
    bug_id: str,
    idx: int,
    hash8: str,
    cluster_key: str,
    best_case_id: str,
    members: list[str],
    represented_rows: int,
    rep_path: Path,
) -> dict[str, Any]:
    directory_name = label.replace(":", "__") if label else ""
    copied_file_name = f"{bug_id}__{idx:03d}__{hash8}.json"
    return {
        "final_ground_truth_id": label,
        "directory_name": directory_name,
        "copied_file_name": copied_file_name,
        "cause_card_key": f"{bug_id}/{hash8}",
        "cause_card_id": hash8,
        "cause_card_file_stem": hash8,
        "source_bug_id": bug_id,
        "source_small_bucket": f"{idx:03d}",
        "source_crash_id": best_case_id,
        "small_bucket_cluster_id": hash8,
        "cluster_key": cluster_key,
        "cluster_representative_case_id": best_case_id,
        "unique_case_count": len(set(members)),
        "member_ref_count": len(members),
        "represented_crash_row_count": represented_rows,
        "duplicate_row_count": represented_rows - len(set(members)),
        "ndss_patch": label.split(":", 1)[1] if label.startswith("ndss:") else "",
        "cause_card_file": f"outputs/stage2_cause_cards/{bug_id}/{hash8}.json",
        "copied_target_file": str(rep_path).replace("\\", "/"),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    out_root = Path(args.out)
    crash_card_root = out_root / "crash_cards_by_bug"
    rep_root = out_root / "representative_crash_cards"
    ground_truth = read_ground_truth(Path(args.ground_truth))
    row_count_by_case = ground_truth["row_count_by_case"]
    labels_by_case = ground_truth["labels_by_case"]

    bug_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if args.limit_bugs:
        bug_dirs = bug_dirs[: args.limit_bugs]

    representative_rows: list[dict[str, Any]] = []
    cards_to_process: list[str] = []
    total_unique_crash_dirs = 0
    total_represented_rows = 0
    total_reps = 0
    processed_bugs = 0

    for bug_dir in bug_dirs:
        crashes_dir = bug_dir / "crashes"
        if not crashes_dir.is_dir():
            continue
        bug_id = bug_dir.name
        bug_out = crash_card_root / bug_id
        bug_rep_dir = rep_root / bug_id
        summary_path = bug_out / "_cluster_summary.json"
        if args.resume and summary_path.exists() and bug_rep_dir.is_dir():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if len(list(bug_rep_dir.glob("*.json"))) >= len(summary):
                bug_unique = 0
                bug_rows = 0
                for idx, cluster in enumerate(summary, 1):
                    members = sorted(cluster["members"])
                    cluster_key = cluster["cluster_key"]
                    hash8 = cluster_key.split("::")[-1][-8:]
                    rep_path = bug_rep_dir / f"{idx:03d}__{hash8}.json"
                    if not rep_path.exists():
                        continue
                    rep_card = json.loads(rep_path.read_text(encoding="utf-8"))
                    represented_rows = sum(row_count_by_case.get((bug_id, case_id), 1) for case_id in members)
                    label = best_label_for_members(bug_id, members, row_count_by_case, labels_by_case)
                    representative_rows.append(
                        make_representative_row(
                            label,
                            bug_id,
                            idx,
                            hash8,
                            cluster_key,
                            rep_card.get("case_id", ""),
                            members,
                            represented_rows,
                            rep_path,
                        )
                    )
                    cards_to_process.append(str(rep_path).replace("\\", "/"))
                    total_reps += 1
                    bug_unique += len(set(members))
                    bug_rows += represented_rows
                total_unique_crash_dirs += bug_unique
                total_represented_rows += bug_rows
                processed_bugs += 1
                print(f"[stage1] {bug_id}: reused representatives={len(summary)}")
                continue
        crash_dirs = sorted(path for path in crashes_dir.iterdir() if path.is_dir())
        if args.limit_crashes_per_bug:
            crash_dirs = crash_dirs[: args.limit_crashes_per_bug]
        if not crash_dirs:
            continue

        bug_out.mkdir(parents=True, exist_ok=True)
        cards: list[dict[str, Any]] = []
        for crash_dir in crash_dirs:
            if not (crash_dir / "crash_meta.json").exists():
                continue
            card = make_card(crash_dir, bug_id)
            cards.append(card)
            (bug_out / f"{card['case_id']}.json").write_text(
                json.dumps(card, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if not cards:
            continue

        clusters = cluster_exact(cards)
        summary = build_cluster_summary(cards, clusters)
        (bug_out / "_cluster_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cards_by_id = {card["case_id"]: card for card in cards}
        bug_rep_dir.mkdir(parents=True, exist_ok=True)
        for idx, cluster in enumerate(summary, 1):
            members = sorted(cluster["members"])
            member_cards = [cards_by_id[case_id] for case_id in members if case_id in cards_by_id]
            if not member_cards:
                continue
            best_card = sorted(member_cards, key=representative_score, reverse=True)[0]
            cluster_key = cluster["cluster_key"]
            hash8 = cluster_key.split("::")[-1][-8:]
            rep_name = f"{idx:03d}__{hash8}.json"
            rep_path = bug_rep_dir / rep_name
            rep_path.write_text(json.dumps(best_card, ensure_ascii=False, indent=2), encoding="utf-8")

            represented_rows = sum(row_count_by_case.get((bug_id, case_id), 1) for case_id in members)
            label = best_label_for_members(bug_id, members, row_count_by_case, labels_by_case)
            representative_rows.append(
                make_representative_row(
                    label,
                    bug_id,
                    idx,
                    hash8,
                    cluster_key,
                    best_card["case_id"],
                    members,
                    represented_rows,
                    rep_path,
                )
            )
            cards_to_process.append(str(rep_path).replace("\\", "/"))
            total_reps += 1
            total_represented_rows += represented_rows

        total_unique_crash_dirs += len(cards)
        processed_bugs += 1
        print(f"[stage1] {bug_id}: crashes={len(cards)} representatives={len(summary)}")

    fields = [
        "final_ground_truth_id",
        "directory_name",
        "copied_file_name",
        "cause_card_key",
        "cause_card_id",
        "cause_card_file_stem",
        "source_bug_id",
        "source_small_bucket",
        "source_crash_id",
        "small_bucket_cluster_id",
        "cluster_key",
        "cluster_representative_case_id",
        "unique_case_count",
        "member_ref_count",
        "represented_crash_row_count",
        "duplicate_row_count",
        "ndss_patch",
        "cause_card_file",
        "copied_target_file",
    ]
    write_csv(out_root / "representative_weights.csv", representative_rows, fields)
    write_csv(out_root / "representative_manifest.csv", representative_rows, fields)
    (out_root / "cards_to_process.txt").write_text("\n".join(cards_to_process) + "\n", encoding="utf-8")
    summary = {
        "input": str(input_root),
        "processed_bug_count": processed_bugs,
        "unique_crash_directory_count": total_unique_crash_dirs,
        "represented_crash_row_count": total_represented_rows,
        "representative_count": total_reps,
        "ground_truth_label_count": len({row["final_ground_truth_id"] for row in representative_rows if row["final_ground_truth_id"]}),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
