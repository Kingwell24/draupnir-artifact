#!/usr/bin/env python3
"""
Crash-card parsing utilities for syzbot crash artifacts.

The public entry point for the artifact pipeline is
`scripts/generate_crash_cards.py`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Regex / constants
# ---------------------------------------------------------------------------
HEX_RE = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{12,}\b")
OFFSET_RE = re.compile(r"\+0x[0-9a-fA-F]+/0x[0-9a-fA-F]+")
PID_TASK_RE = re.compile(r"\b(?:PID|CPU|UID|TID):\s*[^ ]+|\btask\s+[^/\s]+/\d+|\bComm:\s*\S+")
REGISTER_RE = re.compile(
    r"\b(?:RAX|RBX|RCX|RDX|RSI|RDI|RBP|RSP|RIP|EFLAGS|CR[0-4]|FS|GS|CS|DS|ES|R08|R09|R10|R11|R12|R13|R14|R15):.*"
)
PATH_RE = re.compile(r"\b([A-Za-z0-9_./+-]+\.(?:c|h|S|rs))(?::(-?\d+))?")
FUNC_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$]*$")
BPF_TRACE_RE = re.compile(r"\b(?:bpf_prog_[A-Za-z0-9_]+|__bpf_trace_run|bpf_trace_run\d*|__bpf_prog_run|bpf_prog_run|bpf_dispatcher_[A-Za-z0-9_]+|trace_[A-Za-z0-9_]+|__traceiter_[A-Za-z0-9_]+|__[A-Za-z0-9_]*trace_[A-Za-z0-9_]+)\b")
BPF_MAP_OP_RE = re.compile(r"\b(?:bpf_map_(?:delete|update|lookup)_elem|sock_hash_delete_elem|sock_map_delete_elem|__sock_map_delete|sock_map_unref|sock_hash_update_elem|sock_map_update_elem)\b")
TRACEPOINT_STRING_RE = re.compile(r"'([A-Za-z0-9_:.+-]+)\\x00'")

BPF_MAP_TYPE_NAMES = {
    0x0: "UNSPEC",
    0x1: "HASH",
    0x2: "ARRAY",
    0x3: "PROG_ARRAY",
    0x4: "PERF_EVENT_ARRAY",
    0x5: "PERCPU_HASH",
    0x6: "PERCPU_ARRAY",
    0x7: "STACK_TRACE",
    0x8: "CGROUP_ARRAY",
    0x9: "LRU_HASH",
    0xA: "LRU_PERCPU_HASH",
    0xB: "LPM_TRIE",
    0xC: "ARRAY_OF_MAPS",
    0xD: "HASH_OF_MAPS",
    0xE: "DEVMAP",
    0xF: "SOCKMAP",
    0x10: "CPUMAP",
    0x11: "XSKMAP",
    0x12: "SOCKHASH",
    0x13: "CGROUP_STORAGE",
    0x14: "REUSEPORT_SOCKARRAY",
    0x15: "PERCPU_CGROUP_STORAGE",
    0x16: "QUEUE",
    0x17: "STACK",
    0x18: "SK_STORAGE",
    0x19: "DEVMAP_HASH",
    0x1A: "STRUCT_OPS",
    0x1B: "RINGBUF",
}

BPF_PROG_TYPE_NAMES = {
    0x2: "KPROBE",
    0x3: "SCHED_CLS",
    0x4: "SCHED_ACT",
    0x5: "TRACEPOINT",
    0x11: "RAW_TRACEPOINT",
    0x1A: "TRACING",
}

# Functions that are crash-reporting, instrumentation, arch entry, or generic helpers.
GENERIC_FUNC_PREFIXES = (
    "show_stack", "dump_stack", "__dump_stack", "print_report",
    "print_address_description", "print_usage_bug",
    "kasan_report", "kmsan_report", "kmsan_internal", "__msan_warning",
    "__asan_report", "check_memory_region", "instrument_", "report_",
    "panic", "oops_", "die_", "exc_", "asm_exc_",
    "do_syscall", "__do_syscall", "entry_", "el0_", "el1_", "invoke_syscall",
    "ret_from_fork", "ret_from_fork_asm",
    "kthread", "worker_thread", "process_one_work", "process_scheduled_works",
    "__list_del_entry", "__list_add_valid", "__list_del_entry_valid_or_report",
    "__list_del_entry_valid",
    # lockdep report helpers
    "lock_acquire", "lock_release", "lockdep",
    "mark_lock", "mark_usage", "valid_state",
    "print_irqtrace_events",
    # KASAN/KMSAN save stack / track helpers
    "kasan_save_stack", "kasan_save_track", "kasan_save_free_info",
    "kasan_set_track", "__kasan_record_aux_stack",
    "poison_kmalloc_redzone", "poison_slab_object",
    "__kasan_kmalloc", "__kasan_slab_free",
    "kasan_kmalloc", "kasan_slab_free",
    "kmem_cache_free_bulk", "kvfree_rcu_bulk",
)
GENERIC_PATH_PREFIXES = (
    "arch/x86/entry/", "arch/x86/kernel/",
    "arch/arm64/kernel/entry", "arch/arm64/kernel/",
    "mm/kasan/", "mm/kmsan/",
    "lib/dump_stack.c", "lib/list_debug.c", "include/linux/list.h",
    "kernel/locking/lockdep.c", "kernel/softirq.c",
    "kernel/workqueue.c", "kernel/kthread.c",
    "kernel/rcu/tree.c", "mm/slub.c",
    "kernel/time/timer.c",
)
FRAME_STOP_MARKERS = (
    "Allocated by task", "Freed by task", "The buggy address",
    "Memory state around", "Modules linked", "---[ end trace",
    "Code disassembly", "Kernel panic",
    "Uninit was", "Local variable",
    "CPU:", "RIP:", "RSP:", "FS:", "CS:", "CR2:", "RBP:",
    "Last potentially related",
    "Secondary ac",  # "Secondary ac: ..."
    "other info that",
    "Possible unsafe",
    " *** DEADLOCK",
    "no locks held",
    "stack backtrace:",
    "irq event stamp:",
    "hardirqs last",
    "softirqs last",
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    func: str
    file: Optional[str] = None
    line: Optional[int] = None
    inline: bool = False
    raw: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def read_text(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def canon_func(func: str) -> str:
    """Normalise compiler-generated function suffixes."""
    func = func.strip()
    func = OFFSET_RE.sub("", func)
    func = func.split("+")[0]
    for suffix in (".cold", ".isra", ".constprop", ".part", ".llvm"):
        func = re.sub(re.escape(suffix) + r"(?:\.\d+)?$", "", func)
    return func.strip()


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------
def parse_frame_line(line: str) -> Optional[Frame]:
    raw = line.rstrip("\n")
    s = raw.strip()
    if not s or s in ("<TASK>", "</TASK>"):
        return None
    if s.startswith((
        "Code:", "RAX:", "RSP:", "RIP:", "FS:", "CS:", "CR2:", "RBP:",
        "RDX:", "RCX:", "RBX:", "RSI:", "RDI:", "R08:", "R09:", "R10:",
        "R11:", "R12:", "R13:", "R14:", "R15:", "EFLAGS:",
    )):
        return None

    # Remove known prefixes from fault-location / BUG lines.
    s = re.sub(r"^(?:RIP|PC|LR):\s*(?:[0-9a-fA-F]+:)?", "", s)
    s = re.sub(r"^BUG:\s+KASAN:.*?\s+in\s+", "", s)
    s = re.sub(r"^BUG:\s+KMSAN:.*?\s+in\s+", "", s)
    s = re.sub(r"^WARNING:.*?\s+in\s+", "", s)
    s = re.sub(r"^BUG:\s+unable.*?\s+in\s+", "", s)

    mpath = PATH_RE.search(s)
    fpath: Optional[str] = None
    lno: Optional[int] = None
    if mpath:
        fpath = mpath.group(1)
        if mpath.group(2) and mpath.group(2).lstrip("-").isdigit():
            lno = int(mpath.group(2))

    before = s[:mpath.start()].strip() if mpath else s
    before = before.replace("[inline]", "").strip()
    if not before:
        return None
    token = before.split()[-1]
    token = canon_func(token)
    token = token.strip("():")
    if not token or not FUNC_NAME_RE.match(token):
        return None

    is_inline = "[inline]" in raw
    return Frame(func=token, file=fpath, line=lno, inline=is_inline, raw=raw.strip())


def is_generic_frame(fr: Frame) -> bool:
    f = fr.func
    if any(f.startswith(p) for p in GENERIC_FUNC_PREFIXES):
        return True
    if fr.file and any(fr.file.startswith(p) or fr.file == p for p in GENERIC_PATH_PREFIXES):
        return True
    return False


# ---------------------------------------------------------------------------
# Trace extraction
# ---------------------------------------------------------------------------
def extract_call_trace(report: str) -> List[Frame]:
    """Extract the primary Call Trace section from a report."""
    lines = report.splitlines()
    start = None

    # Standard "Call Trace:" marker
    for i, line in enumerate(lines):
        if re.match(r"\s*Call [Tt]race:", line):
            start = i + 1
            break

    # KASAN/KMSAN BUG line may embed the top frame.
    if start is None:
        for i, line in enumerate(lines):
            if re.search(r"BUG:\s+(?:KASAN|KMSAN):", line):
                start = i + 1
                lines = lines[:i] + [line.split(" in ")[-1] if " in " in line else line] + lines[i+1:]
                start = i
                break

    # WARNING / BUG at / Oops fallback
    if start is None:
        for i, line in enumerate(lines):
            if re.search(r"(?:WARNING|BUG):.*\s+in\s+", line):
                start = i + 1
                lines = lines[:i] + [line.split(" in ")[-1] if " in " in line else line] + lines[i+1:]
                start = i
                break

    # "stack backtrace:" marker (lockdep reports)
    if start is None:
        for i, line in enumerate(lines):
            if "stack backtrace:" in line.lower():
                start = i + 1
                break

    if start is None:
        return []

    frames: List[Frame] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            if frames:
                break
            continue
        if s.startswith("</TASK>"):
            break
        if any(s.startswith(m) for m in FRAME_STOP_MARKERS):
            break
        fr = parse_frame_line(line)
        if fr:
            frames.append(fr)
    return frames


def extract_section_frames(report: str, start_pat: str, end_pats: Iterable[str]) -> List[Frame]:
    """Extract frames from a named section (Allocated by task, Freed by task, etc.)."""
    lines = report.splitlines()
    start = None
    rgx = re.compile(start_pat)
    for i, line in enumerate(lines):
        if rgx.search(line):
            start = i + 1
            break
    if start is None:
        return []
    frames: List[Frame] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            if frames:
                break
            continue
        if any(re.search(p, s) for p in end_pats):
            break
        fr = parse_frame_line(line)
        if fr:
            frames.append(fr)
    return frames


def extract_lockdep_registered_trace(report: str) -> List[Frame]:
    """Extract the '{STATE} state was registered at:' trace from lockdep reports."""
    lines = report.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.search(r"state was registered at:", line):
            start = i + 1
            break
    if start is None:
        return []
    frames: List[Frame] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            if frames:
                break
            continue
        if any(s.startswith(m) for m in FRAME_STOP_MARKERS):
            break
        if "irq event stamp" in s or "stack backtrace" in s:
            break
        fr = parse_frame_line(line)
        if fr:
            frames.append(fr)
    return frames


def _unique_keep_order(items: Iterable[str], limit: int = 32) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = str(item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _slice_between(lines: List[str], start_re: str, end_res: Iterable[str]) -> List[str]:
    start = None
    rgx = re.compile(start_re, re.I)
    end_rgxs = [re.compile(p, re.I) for p in end_res]
    for i, line in enumerate(lines):
        if rgx.search(line):
            start = i
            break
    if start is None:
        return []
    out: List[str] = []
    for line in lines[start:]:
        if out and any(r.search(line) for r in end_rgxs):
            break
        out.append(line)
    return out


def _extract_frames_from_lines(lines: Iterable[str], limit: int = 64) -> List[Frame]:
    frames: List[Frame] = []
    for line in lines:
        fr = parse_frame_line(line)
        if fr:
            frames.append(fr)
            if len(frames) >= limit:
                break
    return frames


def extract_lockdep_context(report: str) -> Dict[str, Any]:
    """Extract lockdep-specific causal sections that are not ordinary stacks."""
    low = report.lower()
    if not any(marker in low for marker in (
        "possible circular locking dependency",
        "possible deadlock",
        "inconsistent lock state",
        "lockdep",
    )):
        return {}

    lines = report.splitlines()
    acquire_sec = _slice_between(
        lines,
        r"is trying to acquire lock:",
        [r"but task is already holding lock:", r"which lock already depends", r"the existing dependency chain"],
    )
    holding_sec = _slice_between(
        lines,
        r"but task is already holding lock:",
        [r"which lock already depends", r"the existing dependency chain"],
    )
    dependency_sec = _slice_between(
        lines,
        r"the existing dependency chain",
        [r"other info that might help us debug this:", r"stack backtrace:"],
    )
    other_info_sec = _slice_between(
        lines,
        r"other info that might help us debug this:",
        [r"stack backtrace:", r"Call Trace:"],
    )
    held_locks_sec = _slice_between(
        lines,
        r"\d+\s+locks held by",
        [r"stack backtrace:", r"Call Trace:"],
    )

    dep_blocks: List[Dict[str, Any]] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []
    for line in dependency_sec:
        if re.match(r"\s*->\s*#\d+\s+", line):
            if current_header:
                dep_blocks.append({
                    "lock": current_header.strip(),
                    "frames": [asdict(f) for f in _extract_frames_from_lines(current_lines, limit=24)],
                    "raw_excerpt": "\n".join(current_lines[:28]).strip(),
                })
            current_header = line.strip()
            current_lines = []
        elif current_header:
            current_lines.append(line)
    if current_header:
        dep_blocks.append({
            "lock": current_header.strip(),
            "frames": [asdict(f) for f in _extract_frames_from_lines(current_lines, limit=24)],
            "raw_excerpt": "\n".join(current_lines[:28]).strip(),
        })

    held_locks: List[Dict[str, str]] = []
    for line in held_locks_sec:
        m = re.match(r"\s*#(\d+):\s+(.+?)(?:,\s+at:\s+(.+))?$", line)
        if m:
            held_locks.append({
                "index": m.group(1),
                "lock": m.group(2).strip(),
                "site": (m.group(3) or "").strip(),
            })

    unsafe = "\n".join(other_info_sec).strip()
    report_text_for_tokens = "\n".join(dependency_sec + other_info_sec + lines)
    bpf_frames = _unique_keep_order(BPF_TRACE_RE.findall(report_text_for_tokens), limit=24)
    map_ops = _unique_keep_order(BPF_MAP_OP_RE.findall(report_text_for_tokens), limit=16)
    tracepoints = _unique_keep_order(
        token for token in TRACEPOINT_STRING_RE.findall(report_text_for_tokens)
        if token not in {"GPL"}
    )
    lock_classes = _unique_keep_order(
        m.group(1).strip()
        for m in re.finditer(r"\(([^)\n]+)\)\{[^}\n]+\}-\{[^}\n]+\}", "\n".join(acquire_sec + holding_sec + dependency_sec + held_locks_sec))
    )

    raw_parts = [
        "\n".join(acquire_sec).strip(),
        "\n".join(holding_sec).strip(),
        "\n".join(dependency_sec).strip(),
        "\n".join(other_info_sec).strip(),
    ]
    raw_excerpt = "\n\n".join(part for part in raw_parts if part)
    if len(raw_excerpt) > 14000:
        raw_excerpt = raw_excerpt[:14000] + "\n...[truncated]..."

    return {
        "current_acquisition": {
            "raw_excerpt": "\n".join(acquire_sec).strip(),
            "frames": [asdict(f) for f in _extract_frames_from_lines(acquire_sec, limit=16)],
        },
        "already_holding": {
            "raw_excerpt": "\n".join(holding_sec).strip(),
            "frames": [asdict(f) for f in _extract_frames_from_lines(holding_sec, limit=16)],
        },
        "existing_dependency_chain": dep_blocks,
        "held_locks": held_locks,
        "unsafe_scenario": unsafe,
        "lock_classes": lock_classes,
        "bpf_tracepoint_bridge": {
            "present": bool(bpf_frames or map_ops or tracepoints),
            "bpf_frames": bpf_frames,
            "map_operations": map_ops,
            "tracepoints": tracepoints,
        },
        "raw_lockdep_excerpt": raw_excerpt,
    }


def extract_reproducer_semantics(syz_text: str, c_text: str = "") -> Dict[str, Any]:
    """Extract stable trigger-side semantics from syzkaller/C reproducers."""
    combined = "\n".join(part for part in (syz_text, c_text) if part)
    syscalls: List[str] = []
    for line in syz_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"(?:[A-Za-z0-9_]+\s*=\s*)?([A-Za-z_][A-Za-z0-9_$]*)\(", s)
        if m:
            syscalls.append(m.group(1))

    bpf_map_types: List[str] = []
    for line in syz_text.splitlines():
        if "bpf$MAP_CREATE" not in line:
            continue
        m = re.search(r"@base=\{(0x[0-9a-fA-F]+|\d+)", line)
        if not m:
            continue
        try:
            value = int(m.group(1), 0)
        except ValueError:
            continue
        bpf_map_types.append(BPF_MAP_TYPE_NAMES.get(value, f"UNKNOWN_{value}"))

    bpf_prog_types: List[str] = []
    for line in syz_text.splitlines():
        if "bpf$PROG_LOAD" not in line:
            continue
        m = re.search(r"bpf\$PROG_LOAD\([^)]*\)=\{(0x[0-9a-fA-F]+|\d+)", line)
        if not m:
            m = re.search(r"bpf\$PROG_LOAD\([^,]+,\s*&\([^)]*\)=\{(0x[0-9a-fA-F]+|\d+)", line)
        if not m:
            continue
        try:
            value = int(m.group(1), 0)
        except ValueError:
            continue
        bpf_prog_types.append(BPF_PROG_TYPE_NAMES.get(value, f"UNKNOWN_{value}"))

    tracepoints = _unique_keep_order(
        token for token in TRACEPOINT_STRING_RE.findall(combined)
        if token not in {"GPL"}
    )
    bpf_helpers = _unique_keep_order(
        m.group(1)
        for m in re.finditer(r"@([A-Za-z0-9_]*?(?:map|ringbuf|probe|trace|sock)[A-Za-z0-9_]*)", combined, re.I)
    )
    bpf_map_ops = _unique_keep_order(
        op for op in (
            "MAP_CREATE", "MAP_UPDATE_ELEM", "MAP_DELETE_ELEM", "MAP_LOOKUP_ELEM",
            "BPF_RAW_TRACEPOINT_OPEN", "PROG_LOAD"
        )
        if op in combined
    )
    socket_ops = _unique_keep_order(
        name for name in syscalls
        if name.startswith(("socket", "setsockopt", "bind", "listen", "connect", "send", "recv", "accept"))
    )

    semantic_tokens = []
    if any(t in {"SOCKMAP", "SOCKHASH"} for t in bpf_map_types):
        semantic_tokens.append("repro:sockmap_or_sockhash")
    if bpf_prog_types:
        semantic_tokens.append("repro:bpf_program")
    if tracepoints:
        semantic_tokens.append("repro:tracepoint_attach")
    if "MAP_DELETE_ELEM" in bpf_map_ops:
        semantic_tokens.append("repro:bpf_map_delete_elem")
    if "BPF_RAW_TRACEPOINT_OPEN" in bpf_map_ops:
        semantic_tokens.append("repro:bpf_raw_tracepoint_open")

    return {
        "syscalls": _unique_keep_order(syscalls, limit=40),
        "bpf_program_types": _unique_keep_order(bpf_prog_types, limit=16),
        "bpf_helpers": bpf_helpers,
        "bpf_map_types": _unique_keep_order(bpf_map_types, limit=16),
        "bpf_map_ops": bpf_map_ops,
        "tracepoints": tracepoints,
        "socket_ops": socket_ops,
        "semantic_tokens": _unique_keep_order(semantic_tokens, limit=16),
        "syz_repro_excerpt": syz_text[:6000] + ("\n...[truncated]..." if len(syz_text) > 6000 else ""),
    }


# ---------------------------------------------------------------------------
# Bug type classification
# ---------------------------------------------------------------------------
def normalize_bug_type(title: str, report: str) -> Dict[str, str]:
    """Classify crash into sanitizer + bug_type + access using title and report."""
    # Combine title and first lines of report for robust detection.
    report_head = "\n".join(report.splitlines()[:6]) if report else ""
    combined = (title or "") + " " + report_head
    low = combined.lower()

    out = {"sanitizer": "", "bug_type": "", "access": "", "title_func": ""}

    mfunc = re.search(r"\bin\s+([A-Za-z_][A-Za-z0-9_.$]*)", title or "")
    if not mfunc and report:
        mfunc = re.search(
            r"(?:BUG|WARNING|KASAN|KMSAN|UBSAN|general protection fault).*?\bin\s+([A-Za-z_][A-Za-z0-9_.$]*)",
            report_head,
        )
    if mfunc:
        out["title_func"] = canon_func(mfunc.group(1))

    # --- OOPS / hardware faults: check BEFORE KASAN since many OOPS reports
    #     also mention KASAN in the kernel configuration / sanitizer context. ---
    if "divide error" in low:
        out["sanitizer"] = "OOPS"
        out["bug_type"] = "divide-error"
        return out

    if "general protection fault" in low:
        out["sanitizer"] = "OOPS"
        out["bug_type"] = "general-protection-fault"
        return out

    if "unable to handle kernel paging request" in low or "unable to handle page fault" in low:
        out["sanitizer"] = "OOPS"
        out["bug_type"] = "page-fault"
        return out

    # --- Kernel BUG ---
    if "kernel bug" in low:
        out["sanitizer"] = "BUG"
        out["bug_type"] = "kernel-bug"
        return out

    # --- OOPS fallback ---
    if "kernel-oops" in low or re.search(r"\boops\b", low):
        out["sanitizer"] = "OOPS"
        out["bug_type"] = "oops"
        return out

    # --- Sanitizer-specific ---
    if "kasan:" in low:
        out["sanitizer"] = "KASAN"
        m = re.search(r"KASAN:\s*([A-Za-z0-9_-]+)(?:\s+(Read|Write))?", combined)
        if m:
            bt = m.group(1)
            if bt in ("slab-use-after-free", "use-after-free"):
                bt = "use-after-free"
            out["bug_type"] = bt
            if m.group(2):
                out["access"] = m.group(2).lower()
        return out

    if "kmsan:" in low:
        out["sanitizer"] = "KMSAN"
        m = re.search(r"KMSAN:\s*([A-Za-z0-9_-]+)", combined)
        if m:
            out["bug_type"] = m.group(1)
        return out

    if "kcserror:" in low or "kcsan:" in low:
        out["sanitizer"] = "KCSAN"
        m = re.search(r"KCSAN:\s*([A-Za-z0-9_-]+)", combined)
        if m:
            out["bug_type"] = m.group(1)
        return out

    if "ubsan" in low:
        out["sanitizer"] = "UBSAN"
        m = re.search(r"UBSAN:\s*([A-Za-z0-9_-]+)", combined)
        if m:
            out["bug_type"] = m.group(1)
        return out

    # --- Lockdep ---
    # Priority: check inconsistent > possible deadlock (in title) > possible circular
    # Syzbot titles say "possible deadlock in X" even for circular locking reports,
    # so "possible deadlock" in the title takes priority over "circular" in report body.
    if "inconsistent lock state" in low or "possible deadlock" in low or "possible circular locking" in low:
        out["sanitizer"] = "LOCKDEP"
        if "inconsistent lock" in low:
            out["bug_type"] = "inconsistent-lock-state"
        elif "possible deadlock" in low:
            out["bug_type"] = "possible-deadlock"
        else:
            out["bug_type"] = "possible-circular-lock"
        return out

    # --- WARNING ---
    if low.startswith("warning") or "warning in" in low or "warning:" in low or re.search(r"WARNING:\s+", combined):
        out["sanitizer"] = "WARN"
        out["bug_type"] = "warning"
        return out

    # --- List corruption ---
    if "corrupted list" in low or "list_del corruption" in report.lower():
        out["sanitizer"] = "BUG"
        out["bug_type"] = "list-corruption"
        return out

    # --- INFO patterns ---
    if "info: rcu detected stall" in low:
        out["sanitizer"] = "INFO"
        out["bug_type"] = "rcu-stall"
        return out

    if "info: task hung" in low:
        out["sanitizer"] = "INFO"
        out["bug_type"] = "task-hung"
        return out

    # --- Generic BUG ---
    if low.startswith("bug:"):
        out["sanitizer"] = "BUG"
        out["bug_type"] = "bug"
        return out

    # --- Fallback KASAN (even without explicit colon prefix) ---
    if "out-of-bounds" in low or "oob" in low:
        out["sanitizer"] = "KASAN"
        out["bug_type"] = "out-of-bounds"
        return out

    if "stack-out-of-bounds" in low:
        out["sanitizer"] = "KASAN"
        out["bug_type"] = "stack-out-of-bounds"
        return out

    out["bug_type"] = "unknown"
    return out


# ---------------------------------------------------------------------------
# Report normalisation
# ---------------------------------------------------------------------------
def normalize_report_text(report: str) -> str:
    """Remove high-entropy noise from report text for stable digest computation."""
    kept: List[str] = []
    in_disasm = False
    for line in report.splitlines():
        s = line.rstrip()
        if "Code disassembly" in s:
            in_disasm = True
            continue
        if in_disasm:
            continue
        if s.startswith((
            "Code:", "RAX:", "RBX:", "RSP:", "RIP:", "FS:", "CS:", "CR2:",
            "Modules linked", "---[ end trace",
        )):
            continue
        if REGISTER_RE.match(s):
            continue
        if re.match(r"\s*(CPU|Hardware name|Workqueue):", s):
            continue
        # Normalise noise
        s = HEX_RE.sub("<HEX>", s)
        s = OFFSET_RE.sub("+<OFF>", s)
        s = re.sub(r"\bsyz\.\d+\.\d+\b", "syz.N.N", s)
        s = re.sub(r"/\d+\b", "/N", s)
        s = PID_TASK_RE.sub("", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            kept.append(s)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Config feature extraction
# ---------------------------------------------------------------------------
def get_relevant_config(config_text: str) -> Dict[str, str]:
    keys = [
        "CONFIG_KASAN", "CONFIG_KASAN_GENERIC", "CONFIG_KASAN_SW_TAGS",
        "CONFIG_KMSAN", "CONFIG_KMSAN_CHECK_MEMINIT",
        "CONFIG_DEBUG_LIST", "CONFIG_DEBUG_NET", "CONFIG_REF_TRACKER",
        "CONFIG_LOCKDEP", "CONFIG_PROVE_LOCKING", "CONFIG_DEBUG_LOCK_ALLOC",
        "CONFIG_UBSAN", "CONFIG_UBSAN_BOUNDS", "CONFIG_UBSAN_SHIFT",
        "CONFIG_NET_SCHED", "CONFIG_INET", "CONFIG_IPV6", "CONFIG_XFRM",
        "CONFIG_COMEDI", "CONFIG_OCFS2_FS", "CONFIG_USB", "CONFIG_ATM",
        "CONFIG_NET_NSH", "CONFIG_OPENVSWITCH",
        "CONFIG_FAULT_INJECTION", "CONFIG_KASAN_HW_TAGS",
    ]
    values: Dict[str, str] = {}
    for line in config_text.splitlines():
        if not line.startswith("CONFIG_"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k in keys:
            values[k] = v
    return values


# ---------------------------------------------------------------------------
# Crash card construction
# ---------------------------------------------------------------------------
def make_card(crash_dir: pathlib.Path, big_bucket: str) -> Dict[str, Any]:
    meta = json.loads(read_text(crash_dir / "crash_meta.json") or "{}")
    report = read_text(crash_dir / "report.txt")
    config = read_text(crash_dir / "kernel.config")
    machine = read_text(crash_dir / "machine_info.txt")
    crash_log = read_text(crash_dir / "crash_log.txt")
    syz_repro_text = read_text(crash_dir / "repro.syz")
    c_repro_text = read_text(crash_dir / "repro.c")

    title = meta.get("title") or crash_dir.name
    crash_uid = meta.get("crash_uid", "")
    case_id = crash_dir.name

    bug = normalize_bug_type(title, report)

    # Primary call trace
    frames = extract_call_trace(report)
    semantic = [fr for fr in frames if not is_generic_frame(fr)]

    # KASAN / sanitizer extra traces
    alloc_frames = extract_section_frames(
        report, r"Allocated by task",
        [r"Freed by task", r"The buggy address", r"Memory state",
         r"Call [Tt]race:", r"Last potentially related"],
    )
    free_frames = extract_section_frames(
        report, r"Freed by task",
        [r"Allocated by task", r"The buggy address", r"Memory state",
         r"Call [Tt]race:", r"Last potentially related"],
    )
    origin_frames = extract_section_frames(
        report, r"Uninit.*(?:created|stored|was)|Local variable",
        [r"Bytes", r"CPU:", r"Call [Tt]race:", r"Memory access"],
    )

    # Lockdep registered-at trace
    lockdep_registered = extract_lockdep_registered_trace(report)
    lockdep_context = extract_lockdep_context(report)

    # Normalise report
    norm_text = normalize_report_text(report)

    # Primary function: prefer first semantic frame
    primary = (
        semantic[0].func if semantic
        else (bug.get("title_func") or (frames[0].func if frames else ""))
    )
    primary_file = semantic[0].file if semantic else (frames[0].file if frames else None)

    # Semantic frames (filter out generics)
    stack_funcs = [f.func for f in semantic[:18]]
    stack_paths = [f.file for f in semantic[:18] if f.file]

    # Anchor trace: non-generic frames that are domain-specific
    anchor = [f.func for f in semantic[:8]]

    # Sanitizer trace funcs
    alloc_funcs = [f.func for f in alloc_frames if not is_generic_frame(f)][:8]
    free_funcs = [f.func for f in free_frames if not is_generic_frame(f)][:8]
    origin_funcs = [f.func for f in origin_frames if not is_generic_frame(f)][:6]
    registered_funcs = [f.func for f in lockdep_registered if not is_generic_frame(f)][:8]

    # Repro availability
    has_syz_repro = bool(meta.get("syz_repro") or syz_repro_text)
    has_c_repro = bool(meta.get("c_repro") or c_repro_text)
    reproducer_semantics = extract_reproducer_semantics(syz_repro_text, c_repro_text)

    # Extract syscalls hint from crash_log
    syscall_hints: List[str] = []
    if crash_log:
        for line in crash_log.splitlines()[:200]:
            m = re.search(r"executing program \d+:", line)
            if m:
                continue
            # Extract first function-like token that looks like a syscall
            for prefix in ("syz_", "bpf$", "mmap$", "open$", "openat$",
                           "socket$", "sendmsg$", "recvmsg$", "ioctl$",
                           "setsockopt$", "fcntl$", "pipe", "splice",
                           "sendto$", "bind$", "r0 = "):
                if prefix in line and len(syscall_hints) < 12:
                    syscall_hints.append(line.strip()[:120])
                    break

    # Layer-2 signature tuple
    sig_tuple = (
        bug.get("sanitizer", ""),
        bug.get("bug_type", ""),
        bug.get("access", ""),
        primary,
        tuple(stack_funcs[:10]),
        tuple(alloc_funcs[:3]),
        tuple(free_funcs[:3]),
        tuple(origin_funcs[:3]),
        tuple(registered_funcs[:3]),
    )
    sig_json = json.dumps(sig_tuple, ensure_ascii=False, sort_keys=True)

    card = {
        "schema_version": "crash-card-v1.2",
        "case_id": case_id,
        "big_bucket": big_bucket,
        "crash_uid": crash_uid,
        "title": title,
        "time": meta.get("time", ""),
        "kernel_tree": meta.get("kernel", ""),
        "kernel_commit": meta.get("commit", ""),
        "manager": meta.get("manager", ""),

        # --- signature ---
        "bug": bug,

        # --- fault_evidence ---
        "fault": {
            "primary_function": primary,
            "primary_file": primary_file,
        },

        # --- reproducer ---
        "repro": {
            "has_syz_repro": has_syz_repro,
            "has_c_repro": has_c_repro,
            "syscall_hints": syscall_hints,
        },
        "reproducer_semantics": reproducer_semantics,

        # --- primary_trace ---
        "stack_raw": [asdict(f) for f in frames[:30]],
        "stack_semantic": [asdict(f) for f in semantic[:24]],

        # --- anchor_trace ---
        "anchor_trace": anchor,

        # --- sanitizer_context ---
        "sanitizer_context": {
            "alloc_trace": [asdict(f) for f in alloc_frames[:16]],
            "free_trace": [asdict(f) for f in free_frames[:16]],
            "origin_trace": [asdict(f) for f in origin_frames[:10]],
            "lockdep_registered_trace": [asdict(f) for f in lockdep_registered[:16]],
        },
        "lockdep_context": lockdep_context,

        # --- source_context ---
        "source_context": {
            "files": list(dict.fromkeys(
                f.file for f in semantic[:16] if f.file
            )),
            "subsystems": _guess_subsystems(semantic[:16]),
        },

        # --- config ---
        "config_features": get_relevant_config(config),
        "machine_info_digest": sha16(machine),

        # --- digests ---
        "raw_report_digest": sha16(report),
        "normalized_report_digest": sha16(norm_text),

        # --- signature fields for similarity ---
        "signature_fields": {
            "stack_funcs_top10": stack_funcs[:10],
            "stack_paths_top10": stack_paths[:10],
            "anchor_trace": anchor,
            "alloc_funcs_top3": alloc_funcs[:3],
            "free_funcs_top3": free_funcs[:3],
            "origin_funcs_top3": origin_funcs[:3],
            "registered_funcs_top3": registered_funcs[:3],
            "lockdep_bpf_frames": (lockdep_context.get("bpf_tracepoint_bridge") or {}).get("bpf_frames", [])[:8] if lockdep_context else [],
            "lockdep_map_ops": (lockdep_context.get("bpf_tracepoint_bridge") or {}).get("map_operations", [])[:8] if lockdep_context else [],
            "repro_semantic_tokens": reproducer_semantics.get("semantic_tokens", [])[:8],
        },

        # --- layer2 signature ---
        "layer2_signature_hash": sha16(sig_json),
        "layer2_signature_debug": sig_tuple,

        # --- metadata ---
        "kernel_commit_shas": meta.get("kernel_commit_shas", []),
        "syzkaller_commit_shas": meta.get("syzkaller_commit_shas", []),
    }
    return card


def _guess_subsystems(frames: List[Frame]) -> List[str]:
    """Heuristic subsystem extraction from source file paths."""
    hits: List[str] = []
    for fr in frames:
        if not fr.file:
            continue
        parts = fr.file.split("/")
        if len(parts) >= 2:
            subsys = parts[0]
            if subsys not in hits:
                hits.append(subsys)
    return hits


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------
def cluster_exact(cards: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for c in cards:
        key = c["big_bucket"] + "::" + c["layer2_signature_hash"]
        buckets[key].append(c["case_id"])
    return dict(buckets)


def build_cluster_summary(cards: List[Dict[str, Any]], clusters: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Summary for each cluster: representative info + membership."""
    by_id = {c["case_id"]: c for c in cards}
    summaries = []
    for key, ids in sorted(clusters.items()):
        rep = by_id.get(ids[0], {})
        summaries.append({
            "cluster_key": key,
            "size": len(ids),
            "members": ids,
            "representative": {
                "case_id": rep.get("case_id"),
                "title": rep.get("title"),
                "bug": rep.get("bug"),
                "primary_func": (rep.get("fault") or {}).get("primary_function"),
                "anchor": rep.get("anchor_trace"),
            },
        })
    return summaries
