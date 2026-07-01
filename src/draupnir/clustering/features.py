from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data import CauseCardRecord
from .normalize import ascii_clean, canonical_token, compact_sentence, ordered_unique, token_to_words


FIELD_ORDER = [
    "primary_root",
    "must",
    "object",
    "invariant",
    "patch",
    "propagation",
    "should",
    "surface",
    "weak",
    "evidence",
    "negative",
]

@dataclass
class FeatureRow:
    record_id: str
    true_label: str
    weight: float
    unique_case_count: float
    cause_card_id: str
    cause_card_key: str
    source_crash_id: str
    copied_file_name: str
    fields: dict[str, list[str]]
    field_texts: dict[str, str]
    all_text: str
    stats: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _add(fields: dict[str, list[str]], field: str, values: Any) -> None:
    if not values:
        return
    if not isinstance(values, list):
        values = [values]
    fields.setdefault(field, []).extend(values)


def _is_usable_weight(value: str | None) -> bool:
    return value in {"high", "medium"}


def _is_usable_evidence_level(value: str | None) -> bool:
    return value in {"direct", "indirect"}


def _root_signal_field(signal: dict[str, Any]) -> str:
    stype = canonical_token(signal.get("signal_type", ""))
    role = canonical_token(signal.get("role", ""))
    if stype in {"object", "field", "struct", "member"}:
        return "object"
    if stype in {"fix_shape", "patch_shape"} or "patch" in stype:
        return "patch"
    if stype in {"state", "condition", "constraint", "invariant"}:
        return "invariant"
    if stype in {"lifetime", "propagation", "flow"}:
        return "propagation"
    if role in {"candidate_root_signal", "candidate_root_cause", "proximal_signal"}:
        return "must"
    return "should"


def _format_field(name: str, values: list[str]) -> str:
    if not values:
        return ""
    lines = [f"[{name}]"]
    for value in values:
        tok = canonical_token(value)
        words = token_to_words(tok)
        if words and words != tok:
            lines.append(f"token: {tok} ; words: {words}")
        else:
            lines.append(f"token: {tok}")
    return "\n".join(lines)


def extract_features(record: CauseCardRecord) -> FeatureRow:
    card = record.card
    fields: dict[str, list[str]] = {name: [] for name in FIELD_ORDER}

    rep = card.get("dedup_representation") or {}
    _add(fields, "must", rep.get("must_match_tokens"))
    _add(fields, "should", rep.get("should_match_tokens"))
    _add(fields, "weak", rep.get("weak_context_tokens"))
    _add(fields, "negative", rep.get("must_not_match_conditions"))
    _add(fields, "primary_root", rep.get("primary_root_tokens"))
    _add(fields, "propagation", rep.get("bridge_tokens"))
    _add(fields, "surface", rep.get("surface_tokens"))

    for token in rep.get("must_match_tokens", []) or []:
        tok = canonical_token(token)
        if tok.startswith(("object:", "field:", "struct:")):
            _add(fields, "object", tok)
        if tok.startswith(("invariant:", "condition:", "state:", "violation:")):
            _add(fields, "invariant", tok)
        if tok.startswith(("patch_shape:", "fix_shape:")):
            _add(fields, "patch", tok)

    for signal in card.get("root_cause_signals", []) or []:
        if not _is_usable_weight(signal.get("dedup_weight")):
            continue
        token = signal.get("normalized_token") or signal.get("name")
        if signal.get("stability") == "stable":
            _add(fields, _root_signal_field(signal), token)
            if signal.get("role") in {"candidate_root_signal", "candidate_root_cause", "proximal_signal"}:
                _add(fields, "primary_root", token)
        else:
            _add(fields, "should", token)
        text = compact_sentence("root_signal", signal.get("why_related"))
        if text and signal.get("stability") == "stable":
            _add(fields, "evidence", text)

    for inv in card.get("invariant_signals", []) or []:
        if _is_usable_weight(inv.get("dedup_weight")) and _is_usable_evidence_level(inv.get("evidence_level")):
            _add(fields, "invariant", compact_sentence("expected", inv.get("expected")))
            _add(fields, "invariant", compact_sentence("violation", inv.get("observed_or_suspected_violation")))

    for step in card.get("propagation_signals", []) or []:
        if _is_usable_weight(step.get("dedup_weight")):
            src = ascii_clean(step.get("from"))
            dst = ascii_clean(step.get("to"))
            mech = ascii_clean(step.get("mechanism"))
            if src or dst:
                _add(fields, "propagation", f"flow:{src}->{dst} via {mech}")

    for patch in card.get("patch_semantics_hypotheses", []) or []:
        if not _is_usable_weight(patch.get("dedup_weight")):
            continue
        if canonical_token(patch.get("confidence")) == "low":
            continue
        _add(fields, "patch", compact_sentence("patch_shape", patch.get("shape")))
        _add(fields, "patch", compact_sentence("patch_target", patch.get("target")))
        _add(fields, "patch", compact_sentence("patch_summary", patch.get("summary")))

    for hyp in card.get("hypotheses", []) or []:
        if hyp.get("dedup_usable") and canonical_token(hyp.get("confidence")) in {"high", "medium"}:
            _add(fields, "evidence", compact_sentence("hypothesis", hyp.get("summary")))

    for evidence in card.get("evidence_ledger", []) or []:
        if not _is_usable_weight(evidence.get("dedup_weight")):
            continue
        if not _is_usable_evidence_level(evidence.get("evidence_level")):
            continue
        summary = compact_sentence("evidence", evidence.get("content_summary"))
        location = compact_sentence("location", evidence.get("location"))
        _add(fields, "evidence", summary)
        if evidence.get("dedup_weight") == "high":
            _add(fields, "evidence", location)

    crash_surface = card.get("crash_surface") or {}
    _add(fields, "surface", compact_sentence("surface_sanitizer", crash_surface.get("sanitizer")))
    _add(fields, "surface", compact_sentence("surface_bug_type", crash_surface.get("bug_type")))
    _add(fields, "surface", compact_sentence("surface_operation", crash_surface.get("crash_operation")))
    crash_point = crash_surface.get("crash_point") or {}
    _add(fields, "surface", compact_sentence("surface_function", crash_point.get("function")))
    _add(fields, "surface", compact_sentence("surface_file", crash_point.get("file")))
    if crash_surface.get("surface_is_root_candidate") is True:
        _add(fields, "should", compact_sentence("surface_root_candidate", crash_surface.get("crash_operation")))

    repro = card.get("reproducer_semantics") or {}
    _add(fields, "weak", [f"syscall:{x}" for x in repro.get("syscalls", []) or []])
    _add(fields, "weak", repro.get("semantic_tokens"))
    _add(fields, "weak", [f"socket_op:{x}" for x in repro.get("socket_ops", []) or []])
    _add(fields, "weak", [f"bpf_helper:{x}" for x in repro.get("bpf_helpers", []) or []])
    _add(fields, "weak", [f"tracepoint:{x}" for x in repro.get("tracepoints", []) or []])

    for neg in card.get("negative_evidence", []) or []:
        strength = canonical_token(neg.get("conflict_strength"))
        if strength in {"high", "medium"}:
            _add(fields, "negative", compact_sentence("do_not_merge_with", neg.get("claim")))

    clean_fields = {name: ordered_unique(fields.get(name, [])) for name in FIELD_ORDER}
    field_texts = {name: _format_field(name, clean_fields[name]) for name in FIELD_ORDER}
    all_parts = [field_texts[name] for name in FIELD_ORDER if field_texts[name] and name != "negative"]
    all_text = "\n\n".join(all_parts)
    stats = {
        "field_token_counts": {name: len(clean_fields[name]) for name in FIELD_ORDER},
        "representation_confidence": (card.get("representation_confidence") or {}).get("level"),
        "input_quality": (card.get("input_quality") or {}).get("tier"),
    }
    return FeatureRow(
        record_id=record.record_id,
        true_label=record.final_ground_truth_id,
        weight=record.weight,
        unique_case_count=record.unique_case_count,
        cause_card_id=record.cause_card_id,
        cause_card_key=record.cause_card_key,
        source_crash_id=record.source_crash_id,
        copied_file_name=record.copied_file_name,
        fields=clean_fields,
        field_texts=field_texts,
        all_text=all_text,
        stats=stats,
    )


def extract_feature_rows(records: list[CauseCardRecord]) -> list[dict[str, Any]]:
    return [extract_features(record).to_json() for record in records]
