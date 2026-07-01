from __future__ import annotations

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
