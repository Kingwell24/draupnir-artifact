#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from draupnir.cause_cards import harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: run the LLM + source-search harness to generate one cause card per representative crash card."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--representatives", default="outputs/stage1_crash_cards/representative_crash_cards")
    source.add_argument("--cards-file")
    parser.add_argument("--linux-src", default="external/linux", help="Local Linux source checkout used by the code-search tools.")
    parser.add_argument("--prompt-file", default="src/draupnir/cause_cards/cause_card_prompt.md")
    parser.add_argument("--out", default="outputs/stage2_cause_cards")
    parser.add_argument("--env-file", default=".env", help="Optional KEY=VALUE file for OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tool-rounds", type=int, default=12)
    parser.add_argument("--intervention-round", type=int, default=9)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_cards(args: argparse.Namespace) -> list[Path]:
    if args.cards_file:
        cards: list[Path] = []
        for raw in Path(args.cards_file).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                cards.append(Path(line))
        return cards[: args.limit] if args.limit else cards
    cards = sorted(Path(args.representatives).glob("*/*.json"))
    return cards[: args.limit] if args.limit else cards


def card_hash(card_path: Path) -> str:
    parts = card_path.stem.split("__")
    return parts[-1] if len(parts) > 1 else card_path.stem


def task_for_card(card_path: Path, out_root: Path) -> dict[str, Any]:
    bug_id = card_path.parent.name
    hash_part = card_hash(card_path)
    return {
        "card_path": card_path,
        "bug_id": bug_id,
        "hash_part": hash_part,
        "output_dir": out_root / bug_id,
        "output_file": out_root / bug_id / f"{hash_part}.json",
    }


def write_manifest(out_root: Path, tasks: list[dict[str, Any]]) -> None:
    rows = []
    for task in tasks:
        rows.append(
            {
                "bug_id": task["bug_id"],
                "cause_card_id": task["hash_part"],
                "representative_crash_card": str(task["card_path"]).replace("\\", "/"),
                "cause_card_file": str(task["output_file"]).replace("\\", "/"),
                "exists": str(task["output_file"].exists()).lower(),
            }
        )
    with (out_root / "cause_card_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bug_id", "cause_card_id", "representative_crash_card", "cause_card_file", "exists"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_task(task: dict[str, Any], system_prompt: str, args: argparse.Namespace) -> bool:
    client = None if args.dry_run else harness.create_openai_client()
    return harness.generate_one_cause_card(
        task["card_path"],
        task["output_dir"],
        client,
        harness.MODEL,
        system_prompt,
        temperature=args.temperature,
        max_tool_rounds=args.max_tool_rounds,
        intervention_round=args.intervention_round,
        save_debug=args.save_debug,
        force=args.force,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    harness.LINUX_SRC = Path(args.linux_src)
    harness.PROMPT_FILE = Path(args.prompt_file)
    harness.load_env_file(Path(args.env_file))
    harness.API_KEY = os.environ.get("OPENAI_API_KEY", harness.API_KEY)
    harness.BASE_URL = os.environ.get("OPENAI_BASE_URL", harness.BASE_URL)
    harness.MODEL = os.environ.get("OPENAI_MODEL", harness.MODEL)

    cards = read_cards(args)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = [task_for_card(card, out_root) for card in cards]
    system_prompt = harness.load_prompt()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "cards": len(tasks),
                    "linux_src": str(harness.LINUX_SRC),
                    "prompt_file": str(harness.PROMPT_FILE),
                    "out": str(out_root),
                    "model": harness.MODEL,
                    "api": "not used in dry-run",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        for task in tasks[:20]:
            print(f"[dry-run] {task['card_path']} -> {task['output_file']}")
        write_manifest(out_root, tasks)
        return

    if not harness.API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set. Put it in the environment or in .env.")
    if not harness.LINUX_SRC.is_dir():
        raise SystemExit(f"Linux source tree not found: {harness.LINUX_SRC}")

    ok = 0
    failed = 0
    parallelism = max(1, args.parallelism)
    if parallelism == 1:
        for task in tasks:
            print(f"[stage2] {task['bug_id']} {task['card_path'].name}")
            if run_task(task, system_prompt, args):
                ok += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            future_to_task = {executor.submit(run_task, task, system_prompt, args): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    if future.result():
                        ok += 1
                    else:
                        failed += 1
                except Exception as exc:
                    failed += 1
                    print(f"[ERROR] {task['card_path']}: {exc}")
                print(f"[progress] ok={ok} failed={failed} total={len(tasks)}")
    write_manifest(out_root, tasks)
    print(json.dumps({"processed": ok, "failed": failed, "total": len(tasks), "out": str(out_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
