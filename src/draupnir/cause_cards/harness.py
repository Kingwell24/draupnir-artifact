#!/usr/bin/env python3
"""Generate cause cards with an LLM and local Linux source-search tools."""

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[2]
PROJECT_DIR = RELEASE_ROOT
LAYER3_DIR = HERE
BUGS_DIR = RELEASE_ROOT / "data" / "crashes_by_bug"
LINUX_SRC = RELEASE_ROOT / "external" / "linux"
PROMPT_FILE = HERE / "cause_card_prompt.md"

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("OPENAI_MODEL", "deepseek-v4-pro")
DEFAULT_ENV_FILE = RELEASE_ROOT / ".env"

MAX_TOOL_ROUNDS = 12          # hard cap; prompt uses a softer target below this
TOOL_CALL_TIMEOUT = 30        # seconds per code_searcher call
LLM_TIMEOUT = 300             # seconds per LLM API call
RETRY_MAX = 2                 # retries on LLM error
INTERVENTION_ROUND = 9        # nudge to wrap up after sufficient exploration
DEFAULT_TEMPERATURE = 0.1     # low temperature helps cause-card stability
TOOL_RESULT_CHAR_LIMIT = 24000
PREFETCH_CHAR_BUDGET = 28000
PREFETCH_DOMAIN_FRAMES = 3
PREFETCH_SANITIZER_FRAMES = 2


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "code_searcher_batch",
            "description": "Execute multiple code_searcher operations in one call. Prefer this in the first round and whenever a wider evidence pack can replace several tiny lookups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "description": "One batch request. Supported ops: line, function, callers, callees, grep, struct, field, macro, containing, search_symbol, file_outline"
                        },
                        "description": "List of search requests to execute"
                    }
                },
                "required": ["requests"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_line",
            "description": "Get a wider source window around a specific file:line in the Linux kernel source tree. Prefer using a sufficiently large window instead of repeated one-line lookups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Source file path relative to kernel root, e.g. sound/core/timer.c"},
                    "line": {"type": "integer", "description": "Line number in the file"},
                    "before": {"type": "integer", "description": "Lines to show before target line (default: 80; can be larger if needed)"},
                    "after": {"type": "integer", "description": "Lines to show after target line (default: 120; can be larger if needed)"}
                },
                "required": ["file", "line"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_function",
            "description": "Get the full source code of a function definition. Prefer this over fragmented line lookups when you need variable, macro, and control-flow context together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Function name to search for"},
                    "file": {"type": "string", "description": "Optional: restrict search to this file"},
                    "around_line": {"type": "integer", "description": "Optional: prefer definition near this line number"},
                    "before": {"type": "integer", "description": "Extra context lines before function start (default: 8)"},
                    "after": {"type": "integer", "description": "Extra context lines after function end (default: 8)"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_callers",
            "description": "Find all call sites (callers) of a given function. Returns file:line:code for each match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Function name whose callers to find"},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional: glob patterns to restrict search, e.g. ['sound/core/*.c', 'include/sound/*.h']"},
                    "limit": {"type": "integer", "description": "Optional character budget for the result (default: 20000; increase if a broader caller set is needed)"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_callees",
            "description": "Extract all function calls made within a function's body. Useful for tracing what a function does internally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Function name to analyze"},
                    "file": {"type": "string", "description": "Optional: restrict to this file"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_grep",
            "description": "Search for a regex pattern across kernel source files. Returns file:line:code matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for (POSIX or Python syntax)"},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional: glob patterns to restrict search scope"},
                    "limit": {"type": "integer", "description": "Optional character budget for the result (default: 20000)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_struct",
            "description": "Get the full definition of a struct, including all fields with their types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Struct name, e.g. snd_timer, task_struct"},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional: restrict search scope"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_field",
            "description": "Find all places where a struct field is accessed via ->field or .field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "struct": {"type": "string", "description": "Struct type name (not required for matching, just for context)"},
                    "field": {"type": "string", "description": "Field name to search for"},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional: restrict search scope"},
                    "limit": {"type": "integer", "description": "Optional character budget for the result (default: 20000)"}
                },
                "required": ["struct", "field"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_macro",
            "description": "Find the #define definition of a macro.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Macro name, e.g. GFP_KERNEL, WARN_ON"},
                    "limit": {"type": "integer", "description": "Optional character budget for the result (default: 20000)"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_file_outline",
            "description": "Get a compact overview of a source file near a target line, including nearby macros and function layout. Use this when you need more than a single line but less than whole-file grep noise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Source file path relative to kernel root"},
                    "around_line": {"type": "integer", "description": "Optional target line used to center the overview window"},
                    "radius": {"type": "integer", "description": "Optional window radius around around_line (default: 220)"},
                    "max_macros": {"type": "integer", "description": "Optional maximum number of macros to return (default: 60)"},
                    "max_functions": {"type": "integer", "description": "Optional maximum number of functions to return (default: 40)"}
                },
                "required": ["file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_containing",
            "description": "Find which function contains a given file:line. Useful when you have a line number but need to know the enclosing function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Source file path"},
                    "line": {"type": "integer", "description": "Line number"}
                },
                "required": ["file", "line"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_searcher_search_symbol",
            "description": "Search for a symbol (function, variable, struct name) as a whole word across the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol to search for"},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional: restrict search scope"},
                    "limit": {"type": "integer", "description": "Optional character budget for the result (default: 20000)"}
                },
                "required": ["symbol"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

# Import the searcher lazily to keep the script importable even without it
_searcher_tls = threading.local()
_client_tls = threading.local()

def _get_searcher():
    searcher = getattr(_searcher_tls, "instance", None)
    searcher_repo = getattr(_searcher_tls, "repo", None)
    repo_str = str(LINUX_SRC.resolve())
    if searcher is None or searcher_repo != repo_str:
        from draupnir.cause_cards.code_search import KernelCodeSearcher
        searcher = KernelCodeSearcher(repo_str)
        _searcher_tls.instance = searcher
        _searcher_tls.repo = repo_str
    return searcher


def create_openai_client() -> Any:
    import openai
    return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _get_thread_client() -> Any:
    client = getattr(_client_tls, "instance", None)
    client_key = getattr(_client_tls, "key", None)
    desired_key = (API_KEY, BASE_URL)
    if client is None or client_key != desired_key:
        client = create_openai_client()
        _client_tls.instance = client
        _client_tls.key = desired_key
    return client


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a code_searcher tool call and return the result as a string."""
    ks = _get_searcher()
    try:
        if name == "code_searcher_batch":
            r = ks.batch(args["requests"])
            return json.dumps(r, ensure_ascii=False, indent=2)

        elif name == "code_searcher_line":
            r = ks.get_lines(args["file"], args["line"],
                             args.get("before", 80), args.get("after", 120))
            return r["source"]

        elif name == "code_searcher_function":
            r = ks.get_function(
                args["symbol"],
                file=args.get("file"),
                around_line=args.get("around_line"),
                before=args.get("before", 8),
                after=args.get("after", 8))
            return json.dumps(r, ensure_ascii=False, indent=2)

        elif name == "code_searcher_callers":
            return ks.find_callers(args["symbol"], args.get("paths"),
                                   limit=args.get("limit", 20000))

        elif name == "code_searcher_callees":
            r = ks.find_callees(args["symbol"], args.get("file"))
            return json.dumps(r, ensure_ascii=False, indent=2)

        elif name == "code_searcher_grep":
            return ks.grep(args["pattern"], args.get("paths"),
                           limit=args.get("limit", 20000))

        elif name == "code_searcher_struct":
            r = ks.get_struct_def(args["name"], args.get("paths"))
            return json.dumps(r, ensure_ascii=False, indent=2)

        elif name == "code_searcher_field":
            return ks.find_field_usage(args["struct"], args["field"], args.get("paths"),
                                       limit=args.get("limit", 20000))

        elif name == "code_searcher_macro":
            r = ks.get_macro_def(args["name"], limit=args.get("limit", 20000))
            return json.dumps(r, ensure_ascii=False, indent=2)

        elif name == "code_searcher_file_outline":
            r = ks.get_file_outline(
                args["file"],
                around_line=args.get("around_line"),
                radius=args.get("radius", 220),
                max_macros=args.get("max_macros", 60),
                max_functions=args.get("max_functions", 40),
            )
            return json.dumps(r, ensure_ascii=False, indent=2)

        elif name == "code_searcher_containing":
            r = ks.find_containing_function(args["file"], args["line"])
            return json.dumps(r, ensure_ascii=False, indent=2)

        elif name == "code_searcher_search_symbol":
            return ks.search_symbol(args["symbol"], args.get("paths"),
                                    limit=args.get("limit", 20000))

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e), "tool": name})


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def load_prompt() -> str:
    """Load the full prompt document so schema/rules are preserved."""
    if not PROMPT_FILE.exists():
        raise SystemExit(f"Prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def load_env_file(env_file: pathlib.Path) -> None:
    """Load env vars from a simple KEY=VALUE file or PowerShell $env:KEY='VALUE' lines."""
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("$env:"):
            key, sep, value = line[5:].partition("=")
        else:
            key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


GENERIC_FILE_PREFIXES = (
    "arch/",
    "kernel/",
    "lib/",
    "mm/",
    "scripts/",
    "tools/",
)

GENERIC_FILES = {
    "fs/readdir.c",
    "fs/read_write.c",
    "mm/filemap.c",
}

GENERIC_FUNCTIONS = {
    "__dump_stack",
    "dump_stack",
    "dump_stack_lvl",
    "print_report",
    "kasan_report",
    "kasan_save_stack",
    "kasan_save_track",
    "__kasan_slab_alloc",
    "__kasan_slab_free",
    "lock_acquire",
    "__lock_acquire",
    "__mutex_lock_common",
    "mutex_lock_nested",
    "validate_chain",
    "check_noncircular",
    "check_prevs_add",
    "check_prev_add",
    "wrap_directory_iterator",
    "iterate_dir",
    "generic_perform_write",
    "__generic_file_write_iter",
    "generic_file_write_iter",
    "new_sync_write",
    "vfs_write",
    "ksys_write",
    "__do_sys_write",
    "__se_sys_write",
}

LOW_VALUE_SINK_FUNCTIONS = {
    "__warn",
    "warn_slowpath_fmt",
    "warn_slowpath_null",
    "panic",
    "BUG",
    "BUG_ON",
    "__dump_stack",
    "dump_stack",
    "dump_stack_lvl",
    "strlen",
    "strnlen",
    "strcmp",
    "strncmp",
    "memcpy",
    "memmove",
    "memset",
    "copy_to_user",
    "copy_from_user",
}


def _frame_func(frame: Dict[str, Any]) -> str:
    return str(frame.get("func") or frame.get("function") or "").strip()


def _frame_file(frame: Dict[str, Any]) -> str:
    return str(frame.get("file") or "").strip()


def _frame_line(frame: Dict[str, Any]) -> Optional[int]:
    line = frame.get("line")
    if isinstance(line, int):
        return line
    try:
        return int(line)
    except Exception:
        return None


def _is_generic_frame(frame: Dict[str, Any]) -> bool:
    func = _frame_func(frame)
    file = _frame_file(frame)
    func_lower = func.lower()
    file_lower = file.lower()
    if func in GENERIC_FUNCTIONS:
        return True
    if func_lower.startswith(("do_syscall", "entry_", "__se_sys_", "__do_sys_", "sys_", "ksys_")):
        return True
    if file in GENERIC_FILES:
        return True
    return any(file_lower.startswith(prefix) for prefix in GENERIC_FILE_PREFIXES)


def _domain_scope(file_path: str) -> str:
    parts = [part for part in file_path.split("/") if part]
    if not parts:
        return ""
    if parts[0] == "drivers" and len(parts) >= 3:
        return "/".join(parts[:3]) + "/"
    if len(parts) >= 2:
        return "/".join(parts[:2]) + "/"
    return parts[0] + "/"


def _choose_domain_prefix(crash_card: Dict[str, Any]) -> str:
    candidate_lists = [
        crash_card.get("stack_semantic", []),
        crash_card.get("stack_raw", []),
    ]
    for frames in candidate_lists:
        for frame in frames:
            if not isinstance(frame, dict) or _is_generic_frame(frame):
                continue
            file_path = _frame_file(frame)
            if file_path:
                return _domain_scope(file_path)
    primary_file = str(crash_card.get("fault", {}).get("primary_file") or "").strip()
    if primary_file and primary_file not in GENERIC_FILES and not any(primary_file.startswith(p) for p in GENERIC_FILE_PREFIXES):
        return _domain_scope(primary_file)
    return ""


def _pick_primary_source_frame(crash_card: Dict[str, Any]) -> Dict[str, Any]:
    for frames in (crash_card.get("stack_semantic", []), crash_card.get("stack_raw", [])):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            if _is_generic_frame(frame):
                continue
            if _frame_file(frame) and _frame_line(frame):
                return frame
    primary_func = str(crash_card.get("fault", {}).get("primary_function") or "").strip()
    primary_file = str(crash_card.get("fault", {}).get("primary_file") or "").strip()
    for frame in crash_card.get("stack_raw", []):
        if not isinstance(frame, dict):
            continue
        if _frame_func(frame) == primary_func and _frame_file(frame) == primary_file:
            return frame
    return {
        "func": primary_func,
        "file": primary_file,
        "line": None,
    }


def _collect_domain_frames(crash_card: Dict[str, Any], domain_prefix: str,
                           max_items: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for frame in crash_card.get("stack_semantic", []) + crash_card.get("stack_raw", []):
        if not isinstance(frame, dict):
            continue
        file_path = _frame_file(frame)
        func = _frame_func(frame)
        if not file_path or not func or _is_generic_frame(frame):
            continue
        if domain_prefix and not file_path.startswith(domain_prefix):
            continue
        key = (file_path, func, _frame_line(frame))
        if key in seen:
            continue
        seen.add(key)
        out.append(frame)
        if len(out) >= max_items:
            break
    return out


def _collect_sanitizer_frames(crash_card: Dict[str, Any], domain_prefix: str,
                              max_items: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    sanitizer_ctx = crash_card.get("sanitizer_context", {})
    for key in ("access_trace", "free_trace", "alloc_trace", "origin_trace"):
        for frame in sanitizer_ctx.get(key, []) or []:
            if not isinstance(frame, dict):
                continue
            file_path = _frame_file(frame)
            func = _frame_func(frame)
            if not file_path or not func or _is_generic_frame(frame):
                continue
            if domain_prefix and not file_path.startswith(domain_prefix):
                continue
            marker = (key, file_path, func, _frame_line(frame))
            if marker in seen:
                continue
            seen.add(marker)
            out.append({"trace_kind": key, **frame})
            if len(out) >= max_items:
                return out
    return out


def _has_reproducer_semantics(crash_card: Dict[str, Any]) -> bool:
    repro = crash_card.get("repro") if isinstance(crash_card.get("repro"), dict) else {}
    if repro.get("syscall_hints") or repro.get("has_syz_repro") or repro.get("has_c_repro"):
        return True
    sem = crash_card.get("reproducer_semantics") if isinstance(crash_card.get("reproducer_semantics"), dict) else {}
    for key in (
        "syscalls",
        "bpf_program_types",
        "bpf_helpers",
        "bpf_map_types",
        "bpf_map_ops",
        "tracepoints",
        "socket_ops",
        "semantic_tokens",
        "syz_repro_excerpt",
    ):
        if sem.get(key):
            return True
    return False


def _compute_input_quality(crash_card: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize crash-card evidence density and conservative confidence caps."""
    bug = crash_card.get("bug") if isinstance(crash_card.get("bug"), dict) else {}
    fault = crash_card.get("fault") if isinstance(crash_card.get("fault"), dict) else {}
    sanitizer_ctx = crash_card.get("sanitizer_context") if isinstance(crash_card.get("sanitizer_context"), dict) else {}

    stack = crash_card.get("stack_semantic") or crash_card.get("stack_raw") or []
    if not isinstance(stack, list):
        stack = []
    stack_len = len(stack)
    anchor = crash_card.get("anchor_trace") if isinstance(crash_card.get("anchor_trace"), list) else []
    primary_func = str(fault.get("primary_function") or "").strip()
    primary_file = str(fault.get("primary_file") or "").strip()
    primary_location_present = bool(primary_func and primary_file)
    domain_prefix = _choose_domain_prefix(crash_card)
    domain_frames = _collect_domain_frames(crash_card, domain_prefix, 4)
    domain_frame_count = len(domain_frames)
    has_repro = _has_reproducer_semantics(crash_card)
    top_func = _frame_func(stack[0]) if stack and isinstance(stack[0], dict) else primary_func
    top_frame = stack[0] if stack and isinstance(stack[0], dict) else {"func": primary_func, "file": primary_file}
    top_is_generic_or_sink = bool(
        top_func in LOW_VALUE_SINK_FUNCTIONS
        or top_func.lower().startswith(("__warn", "warn_", "dump_stack"))
        or _is_generic_frame(top_frame)
    )

    missing_reasons: List[str] = []
    caps: List[str] = []
    recommended_tool_rounds = "normal"
    tier = "sufficient"
    confidence_cap = "none"

    if not stack_len:
        missing_reasons.append("no_stack")
    if not anchor:
        missing_reasons.append("no_anchor_trace")
    if not primary_func:
        missing_reasons.append("missing_primary_function")
    if not primary_file:
        missing_reasons.append("missing_primary_file")
    if top_is_generic_or_sink and domain_frame_count == 0:
        missing_reasons.append("generic_or_low_value_sink_only")

    sanitizer = str(bug.get("sanitizer") or "")
    bug_type = str(bug.get("bug_type") or "")
    if "use-after-free" in bug_type.lower() and sanitizer.upper() in {"KASAN", "KMSAN"}:
        if not sanitizer_ctx.get("alloc_trace") or not sanitizer_ctx.get("free_trace"):
            caps.append("uaf_missing_alloc_or_free_trace")
    if sanitizer.upper() == "KMSAN" and "uninit" in bug_type.lower():
        if not sanitizer_ctx.get("origin_trace"):
            caps.append("kmsan_missing_origin_trace")

    if (
        (not stack_len and not primary_location_present)
        or (stack_len <= 1 and not has_repro and not sanitizer_ctx.get("alloc_trace")
            and not sanitizer_ctx.get("free_trace") and not sanitizer_ctx.get("origin_trace"))
        or (not primary_location_present and domain_frame_count == 0 and not has_repro)
    ):
        tier = "minimal"
        confidence_cap = "low"
        recommended_tool_rounds = "minimal"
    elif top_is_generic_or_sink and domain_frame_count == 0:
        tier = "limited"
        confidence_cap = "medium" if stack_len >= 3 or has_repro else "low"
        recommended_tool_rounds = "short"
    elif not primary_location_present or stack_len <= 2 or not anchor:
        tier = "limited"
        confidence_cap = "medium"
        recommended_tool_rounds = "short"

    if confidence_cap == "none" and caps:
        confidence_cap = "medium"

    if tier == "minimal":
        model_instruction = (
            "Crash evidence is minimal. Do not invent candidate root-cause tokens. "
            "Use at most one broad source retrieval if it can validate a surface location; "
            "otherwise emit low-confidence uncertainty with only weak context tokens."
        )
    elif confidence_cap == "medium":
        model_instruction = (
            "Crash evidence is limited or a sanitizer-specific trace is incomplete. "
            "Prefer direct source-backed representation, but do not promote lifecycle/origin "
            "hypotheses to high confidence without matching trace evidence."
        )
    else:
        model_instruction = (
            "Crash evidence is sufficient for normal source-guided representation. "
            "Use the standard evidence-first workflow."
        )

    return {
        "tier": tier,
        "confidence_cap": confidence_cap,
        "recommended_tool_rounds": recommended_tool_rounds,
        "missing_or_weak_evidence": sorted(set(missing_reasons + caps)),
        "metrics": {
            "stack_len": stack_len,
            "anchor_len": len(anchor),
            "domain_frame_count": domain_frame_count,
            "primary_location_present": primary_location_present,
            "has_reproducer_semantics": has_repro,
            "has_alloc_trace": bool(sanitizer_ctx.get("alloc_trace")),
            "has_free_trace": bool(sanitizer_ctx.get("free_trace")),
            "has_origin_trace": bool(sanitizer_ctx.get("origin_trace")),
        },
        "model_instruction": model_instruction,
    }


def _format_input_quality_for_prompt(input_quality: Dict[str, Any]) -> str:
    return json.dumps(input_quality, ensure_ascii=False, indent=2)


def _format_prefetch_function(ks: Any, frame: Dict[str, Any]) -> str:
    symbol = _frame_func(frame)
    file_path = _frame_file(frame)
    line = _frame_line(frame)
    if not symbol:
        return ""
    matches = ks.get_function(
        symbol,
        file=file_path or None,
        around_line=line,
        before=8,
        after=8,
    )
    if matches:
        best = matches[0]
        return f"{best.get('file', file_path)}:{best.get('start_line')}..{best.get('end_line')}\n{best.get('source', '')}"
    if file_path and line:
        window = ks.get_lines(file_path, line, 40, 80)
        return f"{file_path}:{line}\n{window.get('source', '')}"
    return ""


def _extract_macro_candidates(source_text: str, limit: int = 8) -> List[str]:
    seen = set()
    out: List[str] = []
    for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", source_text or ""):
        if token in {"NULL", "WARN", "BUG", "KERN"}:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _extract_ranked_macro_candidates(source_text: str, target_line: Optional[int],
                                     limit: int = 8) -> List[str]:
    if not source_text or not target_line:
        return _extract_macro_candidates(source_text, limit=limit)
    scored: List[tuple[int, str]] = []
    seen = set()
    for raw_line in source_text.splitlines():
        match = re.match(r"^[ >=>]*\s*(\d+):\s*(.*)$", raw_line)
        if not match:
            continue
        line_no = int(match.group(1))
        code = match.group(2)
        for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", code):
            if token in {"NULL", "WARN", "BUG", "KERN"}:
                continue
            if token in seen:
                continue
            seen.add(token)
            scored.append((abs(line_no - target_line), token))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [token for _, token in scored[:limit]]


def build_prefetched_source_packet(crash_card: Dict[str, Any]) -> str:
    """Deterministically prefetch core source context before the LLM starts."""
    try:
        ks = _get_searcher()
        sections: List[str] = []
        total_chars = 0

        def add_section(title: str, body: str) -> None:
            nonlocal total_chars
            body = (body or "").strip()
            if not body:
                return
            section = f"## {title}\n{body}\n"
            if total_chars + len(section) > PREFETCH_CHAR_BUDGET and sections:
                return
            sections.append(section)
            total_chars += len(section)

        domain_prefix = _choose_domain_prefix(crash_card)
        primary = _pick_primary_source_frame(crash_card)
        add_section(
            "Prefetch Notes",
            "Deterministic source packet prepared by the script. Treat it as initial evidence, not ground truth.\n"
            f"Domain scope hint: {domain_prefix or '<unknown>'}",
        )
        if crash_card.get("lockdep_context"):
            add_section(
                "Crash Card LOCKDEP Context",
                json.dumps(crash_card.get("lockdep_context"), ensure_ascii=False, indent=2),
            )
        if crash_card.get("reproducer_semantics"):
            add_section(
                "Crash Card Reproducer Semantics",
                json.dumps(crash_card.get("reproducer_semantics"), ensure_ascii=False, indent=2),
            )

        primary_file = _frame_file(primary)
        primary_line = _frame_line(primary)
        primary_func = _frame_func(primary)
        primary_func_src = ""

        if primary_file:
            outline = ks.get_file_outline(primary_file, around_line=primary_line, radius=220)
            add_section("Crash File Outline", json.dumps(outline, ensure_ascii=False, indent=2))
        if primary_func:
            primary_func_src = _format_prefetch_function(ks, primary)
        if primary_file and primary_line:
            line_ctx = ks.get_lines(primary_file, primary_line, 80, 120)
            add_section("Crash Line Context", line_ctx.get("source", ""))
            macro_defs: List[Dict[str, Any]] = []
            macro_candidates = _extract_ranked_macro_candidates(primary_func_src, primary_line, limit=8)
            if not macro_candidates:
                macro_candidates = _extract_macro_candidates(line_ctx.get("source", ""), limit=8)
            for macro_name in macro_candidates:
                macro_def = ks.get_macro_def(macro_name, limit=4000)
                if macro_def.get("matches"):
                    macro_defs.append(macro_def)
            if macro_defs:
                add_section("Nearby Macro Definitions", json.dumps(macro_defs, ensure_ascii=False, indent=2))
        if primary_func:
            add_section(f"Primary Function: {primary_func}", primary_func_src)

        seen_functions = {(primary_file, primary_func)}
        for frame in _collect_domain_frames(crash_card, domain_prefix, PREFETCH_DOMAIN_FRAMES):
            key = (_frame_file(frame), _frame_func(frame))
            if key in seen_functions:
                continue
            seen_functions.add(key)
            add_section(f"Related Domain Function: {_frame_func(frame)}", _format_prefetch_function(ks, frame))

        for frame in _collect_sanitizer_frames(crash_card, domain_prefix, PREFETCH_SANITIZER_FRAMES):
            key = (_frame_file(frame), _frame_func(frame))
            if key in seen_functions:
                continue
            seen_functions.add(key)
            trace_kind = str(frame.get("trace_kind") or "sanitizer_trace")
            add_section(f"{trace_kind}: {_frame_func(frame)}", _format_prefetch_function(ks, frame))

        return "\n".join(sections).strip()
    except Exception:
        return ""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from LLM output (may be wrapped in ```json ... ```)."""
    # Try parsing directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json ... ``` block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding the outermost { ... }
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_root_locations(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    root_loc = _first_non_empty(
        raw.get("root_cause_location"),
        raw.get("root_cause", {}).get("root_cause_location"),
        raw.get("root_cause", {}).get("location"),
    )
    if root_loc is None:
        return []
    if isinstance(root_loc, str):
        parts = root_loc.split(":")
        if len(parts) >= 3:
            return [{
                "file": parts[0],
                "function": parts[1],
                "line": parts[2],
                "role": "root",
                "description": "",
            }]
        return [{
            "file": "",
            "function": "",
            "line": root_loc,
            "role": "root",
            "description": "",
        }]
    if isinstance(root_loc, dict):
        root_loc = [root_loc]

    out: List[Dict[str, Any]] = []
    for item in root_loc:
        if not isinstance(item, dict):
            continue
        out.append({
            "file": item.get("file", ""),
            "function": item.get("function", ""),
            "line": _first_non_empty(item.get("line"), item.get("line_range"), item.get("lines"), ""),
            "role": item.get("role", "root"),
            "description": _first_non_empty(
                item.get("description"),
                item.get("why_root"),
                item.get("code"),
                item.get("code_snippet"),
                "",
            ),
        })
    return out


def _normalize_source_evidence(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    source_evidence = raw.get("source_evidence")
    if isinstance(source_evidence, list):
        out: List[Dict[str, str]] = []
        for item in source_evidence:
            if isinstance(item, dict):
                out.append({
                    "ref": _first_non_empty(
                        item.get("code_snippet_ref"),
                        item.get("location"),
                        item.get("file"),
                        "",
                    ),
                    "relevance": _first_non_empty(
                        item.get("relevance"),
                        item.get("detail"),
                        item.get("description"),
                        "",
                    ),
                })
        if out:
            return out

    evidence = raw.get("evidence")
    if isinstance(evidence, dict):
        return [{"ref": key, "relevance": str(value)} for key, value in evidence.items()]
    return []


def _normalize_fix_semantics(raw: Dict[str, Any]) -> Dict[str, str]:
    analysis = raw.get("analysis", {}) if isinstance(raw.get("analysis"), dict) else {}
    analysis_root = analysis.get("root_cause", {}) if isinstance(analysis.get("root_cause"), dict) else {}
    fix = _first_non_empty(
        raw.get("fix_semantics"),
        raw.get("fix_guidance"),
        raw.get("fix_suggestion"),
        analysis_root.get("missing_validation"),
        {},
    )
    if isinstance(fix, str):
        return {"summary": fix, "shape": "", "location": ""}
    if not isinstance(fix, dict):
        return {"summary": "", "shape": "", "location": ""}
    return {
        "summary": _first_non_empty(fix.get("summary"), fix.get("suggested_fix"), fix.get("change"), ""),
        "shape": _first_non_empty(fix.get("expected_shape"), fix.get("expected_patch_shape"), "",),
        "location": _first_non_empty(
            fix.get("affected_code"),
            fix.get("affected_code_location"),
            fix.get("file"),
            "",
        ),
    }


def _normalize_dedup_fingerprint(raw: Dict[str, Any]) -> Dict[str, Any]:
    fp = raw.get("dedup_fingerprint")
    if isinstance(fp, str):
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9_]+", fp) if tok]
        return {"semantic_id": fp, "tokens": tokens}
    if isinstance(fp, dict):
        semantic_id = _first_non_empty(
            fp.get("semantic_id"),
            fp.get("root_object"),
            raw.get("cause_id"),
            "",
        )
        token_values: List[str] = []
        for key in ("tokens", "root_object_tokens", "invariant_tokens", "function_tokens", "subsystem_tokens"):
            value = fp.get(key)
            if isinstance(value, list):
                token_values.extend(str(v) for v in value if v)
            elif isinstance(value, str) and value:
                token_values.append(value)
        for key in ("root_object", "invariant", "subsystem"):
            value = fp.get(key)
            if isinstance(value, str) and value:
                token_values.append(value)
        return {"semantic_id": semantic_id, "tokens": token_values}
    return {"semantic_id": "", "tokens": []}


def _collect_tool_usage(tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    functions: List[str] = []
    files: List[str] = []
    rounds = 0
    for item in tool_trace:
        rounds = max(rounds, int(item.get("round", 0) or 0))
        args = item.get("args", {})
        if not isinstance(args, dict):
            continue
        for key in ("symbol", "name"):
            value = args.get(key)
            if isinstance(value, str) and value:
                functions.append(value)
        value = args.get("file")
        if isinstance(value, str) and value:
            files.append(value)
        for req in args.get("requests", []) if isinstance(args.get("requests"), list) else []:
            if not isinstance(req, dict):
                continue
            for key in ("symbol", "name"):
                value = req.get(key)
                if isinstance(value, str) and value:
                    functions.append(value)
            value = req.get("file")
            if isinstance(value, str) and value:
                files.append(value)
    return {
        "rounds_used": rounds,
        "functions_examined": sorted(set(functions)),
        "files_examined": sorted(set(files)),
    }


def _refine_root_object_name(name: Any,
                             object_type: Any,
                             description: Any,
                             source_evidence: List[Dict[str, str]],
                             dedup_fingerprint: Dict[str, Any]) -> Any:
    if not isinstance(name, str) or not name:
        name = ""
    text_parts: List[str] = []
    if isinstance(description, str):
        text_parts.append(description)
    for item in source_evidence:
        if isinstance(item, dict):
            for key in ("ref", "relevance"):
                value = item.get(key)
                if isinstance(value, str):
                    text_parts.append(value)
    text = "\n".join(text_parts)
    if name in {"lock", "timer", "object", "entry", "name"}:
        for pattern in (
            r"([A-Za-z_][A-Za-z0-9_]*->[A-Za-z_][A-Za-z0-9_]*)",
            r"([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]{2,})",
        ):
            m = re.search(pattern, text)
            if m:
                candidate = m.group(1)
                if candidate.rsplit(".", 1)[-1] not in {"c", "h"}:
                    return candidate
        for tok in dedup_fingerprint.get("tokens", []) if isinstance(dedup_fingerprint, dict) else []:
            if isinstance(tok, str) and "." in tok and tok.rsplit(".", 1)[-1] not in {"c", "h"}:
                return tok
        if isinstance(object_type, str) and object_type.startswith("struct "):
            struct_name = object_type.split(None, 1)[1]
            if name:
                return f"{struct_name}.{name}"
            return struct_name
    if not name and isinstance(object_type, str) and object_type.startswith("struct "):
        return object_type.split(None, 1)[1]
    return name or None


def _write_debug_record(debug_file: pathlib.Path,
                        *,
                        card_path: pathlib.Path,
                        output_file: pathlib.Path,
                        model: str,
                        prompt_file: pathlib.Path,
                        status: str,
                        tool_trace: List[Dict[str, Any]],
                        last_content: str = "",
                        error: str = "") -> None:
    debug_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "card_path": str(card_path),
        "output_file": str(output_file),
        "model": model,
        "prompt_file": str(prompt_file),
        "status": status,
        "tool_trace": tool_trace,
        "last_content_preview": last_content[:4000],
        "error": error,
    }
    debug_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_text_list(value: Any) -> List[str]:
    out: List[str] = []
    for item in _as_list(value):
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _clean_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip().lower()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^a-z0-9_:\-./>\[\]]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _infer_fix_shape(summary: str) -> str:
    text = summary.lower()
    if any(kw in text for kw in ("bounds check", "range check", "validate index", "oob", "out-of-bounds")):
        return "add_bounds_check"
    if any(kw in text for kw in ("initialize", "clear", "zero", "uninitialized")):
        return "initialize_field"
    if any(kw in text for kw in ("reference", "refcount", "use-after-free", "hold ref")):
        return "fix_lifetime"
    if any(kw in text for kw in ("cancel", "timer", "workqueue", "callback", "defer")):
        return "cancel_async"
    if any(kw in text for kw in ("lock", "unlock", "irq-disable", "deadlock")):
        return "add_locking"
    if any(kw in text for kw in ("validate input", "sanity check", "check flags")):
        return "validate_input"
    return "other"


def _guess_signal_type(name: str, summary: str) -> str:
    text = f"{name} {summary}".lower()
    if "->" in name or "." in name:
        return "field"
    if any(kw in text for kw in ("index", "slot", "stbl[", "array", "bound")):
        return "index"
    if any(kw in text for kw in ("len", "length", "size")):
        return "length"
    if any(kw in text for kw in ("lock", "mutex", "spin", "irq")):
        return "lock"
    if any(kw in text for kw in ("state", "flag", "mode", "context")):
        return "state"
    if any(kw in text for kw in ("lifetime", "free", "refcount", "use-after-free")):
        return "lifetime"
    if any(kw in text for kw in ("pointer", "ptr", "address")):
        return "pointer"
    return "object"


def _build_evidence_ledger(raw: Dict[str, Any],
                           crash_card: Dict[str, Any],
                           source_evidence: List[Dict[str, str]],
                           root_locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = raw.get("evidence_ledger")
    if isinstance(existing, list) and existing:
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(existing, 1):
            if not isinstance(item, dict):
                continue
            out.append({
                "evidence_id": _first_non_empty(item.get("evidence_id"), f"E{idx}"),
                "source": _first_non_empty(item.get("source"), "inference"),
                "location": _first_non_empty(item.get("location"), item.get("ref"), ""),
                "content_summary": _first_non_empty(item.get("content_summary"), item.get("summary"), item.get("content"), ""),
                "evidence_level": _first_non_empty(item.get("evidence_level"), "indirect"),
                "dedup_weight": _first_non_empty(item.get("dedup_weight"), "medium"),
            })
        if out:
            return out

    evidence: List[Dict[str, Any]] = []
    evidence.append({
        "evidence_id": "E1",
        "source": "crash_card",
        "location": f"crash_card:{crash_card.get('case_id', '')}",
        "content_summary": f"{crash_card.get('title', '')} | anchor_trace={', '.join(_ensure_text_list(crash_card.get('anchor_trace'))[:3])}",
        "evidence_level": "direct",
        "dedup_weight": "medium",
    })
    next_id = 2
    for loc in root_locations:
        location = ":".join(str(loc.get(key, "")) for key in ("file", "function", "line") if loc.get(key, "") not in (None, ""))
        summary = _first_non_empty(loc.get("description"), "candidate source location")
        evidence.append({
            "evidence_id": f"E{next_id}",
            "source": "source_code",
            "location": location,
            "content_summary": str(summary),
            "evidence_level": "direct",
            "dedup_weight": "high",
        })
        next_id += 1
    for item in source_evidence:
        evidence.append({
            "evidence_id": f"E{next_id}",
            "source": "source_code",
            "location": _first_non_empty(item.get("ref"), ""),
            "content_summary": _first_non_empty(item.get("relevance"), ""),
            "evidence_level": "indirect",
            "dedup_weight": "medium",
        })
        next_id += 1

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in evidence:
        key = (item["source"], item["location"], item["content_summary"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _normalize_root_cause_signals(raw: Dict[str, Any],
                                  crash_card: Dict[str, Any],
                                  dedup_fingerprint: Dict[str, Any],
                                  root_locations: List[Dict[str, Any]],
                                  source_evidence: List[Dict[str, str]],
                                  evidence_ledger: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = raw.get("root_cause_signals")
    if isinstance(existing, list) and existing:
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(existing, 1):
            if not isinstance(item, dict):
                continue
            name = _first_non_empty(item.get("name"), item.get("signal"), item.get("object"), "")
            out.append({
                "signal_id": _first_non_empty(item.get("signal_id"), f"S{idx}"),
                "signal_type": _first_non_empty(item.get("signal_type"), _guess_signal_type(str(name), "")),
                "name": name,
                "normalized_token": _first_non_empty(item.get("normalized_token"), _clean_token(name)),
                "role": _first_non_empty(item.get("role"), "candidate_root_signal"),
                "why_related": _first_non_empty(item.get("why_related"), item.get("summary"), ""),
                "supporting_evidence": _ensure_text_list(item.get("supporting_evidence")),
                "contradicting_evidence": _ensure_text_list(item.get("contradicting_evidence")),
                "stability": _first_non_empty(item.get("stability"), "stable"),
                "dedup_weight": _first_non_empty(item.get("dedup_weight"), "medium"),
            })
        return out

    raw_analysis = raw.get("analysis", {}) if isinstance(raw.get("analysis"), dict) else {}
    raw_root = raw.get("root_cause", {}) if isinstance(raw.get("root_cause"), dict) else {}
    raw_root_object = raw.get("root_object", {}) if isinstance(raw.get("root_object"), dict) else {}
    if not raw_root_object and isinstance(raw_root.get("root_object"), dict):
        raw_root_object = raw_root.get("root_object", {})

    evidence_ids = [item.get("evidence_id") for item in evidence_ledger if item.get("evidence_id")]
    high_evidence = evidence_ids[1:3] or evidence_ids[:1]
    medium_evidence = evidence_ids[2:5] or evidence_ids[:2]

    root_object_name = _first_non_empty(
        raw_root.get("object"),
        raw_root.get("root_object"),
        raw_root_object.get("name"),
        raw_root_object.get("field"),
        "",
    )
    root_object_type = _first_non_empty(
        raw_root_object.get("type"),
        raw_root_object.get("field_type"),
        "",
    )
    root_object_desc = _first_non_empty(
        raw_root_object.get("description"),
        raw_root.get("violated_invariant"),
        raw_root.get("missing_validation"),
        "",
    )
    root_object_name = _refine_root_object_name(
        root_object_name,
        root_object_type,
        root_object_desc,
        source_evidence,
        dedup_fingerprint,
    ) or ""

    signals: List[Dict[str, Any]] = []
    if root_object_name:
        signals.append({
            "signal_id": "S1",
            "signal_type": _guess_signal_type(root_object_name, root_object_desc),
            "name": root_object_name,
            "normalized_token": _clean_token(root_object_name),
            "role": "candidate_root_signal",
            "why_related": root_object_desc or "primary source-level object tied to the crash representation",
            "supporting_evidence": high_evidence,
            "contradicting_evidence": [],
            "stability": "stable",
            "dedup_weight": "high",
        })

    operation_text = ""
    if root_locations:
        operation_text = _first_non_empty(root_locations[0].get("description"), "")
    operation_text = _first_non_empty(
        operation_text,
        raw.get("crash_surface", {}).get("crash_operation") if isinstance(raw.get("crash_surface"), dict) else "",
        "",
    )
    if operation_text:
        signals.append({
            "signal_id": f"S{len(signals)+1}",
            "signal_type": "source_operation",
            "name": operation_text,
            "normalized_token": _clean_token(operation_text)[:96],
            "role": "proximal_signal",
            "why_related": "exact or near-exact source operation around the crash or candidate cause site",
            "supporting_evidence": high_evidence,
            "contradicting_evidence": [],
            "stability": "version_sensitive",
            "dedup_weight": "high",
        })

    for tok in dedup_fingerprint.get("tokens", []):
        clean = _clean_token(tok)
        if not clean:
            continue
        if any(sig.get("normalized_token") == clean for sig in signals):
            continue
        signals.append({
            "signal_id": f"S{len(signals)+1}",
            "signal_type": _guess_signal_type(clean, clean),
            "name": tok,
            "normalized_token": clean,
            "role": "context_signal",
            "why_related": "legacy dedup token preserved during normalization",
            "supporting_evidence": medium_evidence,
            "contradicting_evidence": [],
            "stability": "stable",
            "dedup_weight": "medium",
        })

    function_name = _first_non_empty(
        crash_card.get("fault", {}).get("primary_function"),
        raw_analysis.get("crash_type"),
        "",
    )
    if function_name:
        signals.append({
            "signal_id": f"S{len(signals)+1}",
            "signal_type": "state",
            "name": function_name,
            "normalized_token": f"function:{_clean_token(function_name)}",
            "role": "context_signal",
            "why_related": "subsystem-local function context from crash surface",
            "supporting_evidence": evidence_ids[:1],
            "contradicting_evidence": [],
            "stability": "stable",
            "dedup_weight": "low",
        })
    return signals


def _normalize_invariant_signals(raw: Dict[str, Any],
                                 evidence_ledger: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = raw.get("invariant_signals")
    if isinstance(existing, list) and existing:
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(existing, 1):
            if not isinstance(item, dict):
                continue
            out.append({
                "invariant_id": _first_non_empty(item.get("invariant_id"), f"I{idx}"),
                "expected": _first_non_empty(item.get("expected"), ""),
                "observed_or_suspected_violation": _first_non_empty(item.get("observed_or_suspected_violation"), item.get("actual"), item.get("summary"), ""),
                "source_basis": _ensure_text_list(item.get("source_basis")),
                "evidence_level": _first_non_empty(item.get("evidence_level"), "indirect"),
                "dedup_weight": _first_non_empty(item.get("dedup_weight"), "medium"),
            })
        return out

    raw_root = raw.get("root_cause", {}) if isinstance(raw.get("root_cause"), dict) else {}
    raw_violation_text = raw.get("violated_invariant") if isinstance(raw.get("violated_invariant"), str) else ""
    raw_violation = raw.get("violated_invariant", {}) if isinstance(raw.get("violated_invariant"), dict) else {}
    statement = _first_non_empty(
        raw_violation.get("invariant"),
        raw_violation.get("violated_invariant"),
        raw_root.get("violated_invariant"),
        raw_root.get("missing_validation"),
        (
            f"expected: {raw_violation.get('expected')}; actual: {raw_violation.get('actual')}"
            if raw_violation.get("expected") or raw_violation.get("actual")
            else None
        ),
        raw_violation_text,
        "",
    )
    if not statement:
        return []
    expected = raw_violation.get("expected") if isinstance(raw_violation, dict) else ""
    observed = _first_non_empty(raw_violation.get("actual") if isinstance(raw_violation, dict) else "", statement)
    return [{
        "invariant_id": "I1",
        "expected": expected or statement,
        "observed_or_suspected_violation": observed,
        "source_basis": [item.get("evidence_id") for item in evidence_ledger[:3] if item.get("evidence_id")],
        "evidence_level": "indirect",
        "dedup_weight": "high" if expected else "medium",
    }]


def _normalize_propagation_signals(raw: Dict[str, Any],
                                   evidence_ledger: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = raw.get("propagation_signals")
    if isinstance(existing, list) and existing:
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(existing, 1):
            if not isinstance(item, dict):
                continue
            out.append({
                "step": _first_non_empty(item.get("step"), idx),
                "from": _first_non_empty(item.get("from"), ""),
                "to": _first_non_empty(item.get("to"), ""),
                "mechanism": _first_non_empty(item.get("mechanism"), "other"),
                "source_evidence": _ensure_text_list(item.get("source_evidence")),
                "dedup_weight": _first_non_empty(item.get("dedup_weight"), "medium"),
            })
        return out

    raw_analysis = raw.get("analysis", {}) if isinstance(raw.get("analysis"), dict) else {}
    path = _as_list(_first_non_empty(raw.get("propagation_path"), raw_analysis.get("propagation_path"), []))
    if not path:
        return []
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(path, 1):
        if isinstance(item, dict):
            src = _first_non_empty(item.get("from"), item.get("location"), item.get("step"), "")
            dst = _first_non_empty(item.get("to"), item.get("detail"), item.get("action"), "")
            mechanism_text = " ".join(_ensure_text_list([
                item.get("detail"),
                item.get("action"),
                item.get("location"),
            ])).lower()
            mechanism = "other"
            for key, label in (
                ("callback", "callback"),
                ("workqueue", "workqueue"),
                ("timer", "timer"),
                ("rcu", "rcu"),
                ("softirq", "softirq"),
                ("syscall", "syscall"),
                ("metadata", "metadata_parse"),
                ("call", "direct_call"),
            ):
                if key in mechanism_text:
                    mechanism = label
                    break
            out.append({
                "step": _first_non_empty(item.get("step"), idx),
                "from": str(src),
                "to": str(dst),
                "mechanism": mechanism,
                "source_evidence": [item2.get("evidence_id") for item2 in evidence_ledger[:2] if item2.get("evidence_id")],
                "dedup_weight": "medium",
            })
        elif isinstance(item, str) and item.strip():
            out.append({
                "step": idx,
                "from": item.strip(),
                "to": "",
                "mechanism": "other",
                "source_evidence": [item2.get("evidence_id") for item2 in evidence_ledger[:1] if item2.get("evidence_id")],
                "dedup_weight": "low",
            })
    return out


def _normalize_hypotheses(raw: Dict[str, Any],
                          evidence_ledger: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = raw.get("hypotheses")
    if isinstance(existing, list) and existing:
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(existing, 1):
            if not isinstance(item, dict):
                continue
            out.append({
                "hypothesis_id": _first_non_empty(item.get("hypothesis_id"), f"H{idx}"),
                "summary": _first_non_empty(item.get("summary"), ""),
                "role": _first_non_empty(item.get("role"), "candidate_root_cause"),
                "supporting_evidence": _ensure_text_list(item.get("supporting_evidence")),
                "contradicting_evidence": _ensure_text_list(item.get("contradicting_evidence")),
                "confidence": _first_non_empty(item.get("confidence"), "medium"),
                "dedup_usable": bool(item.get("dedup_usable", True)),
            })
        return out

    raw_analysis = raw.get("analysis", {}) if isinstance(raw.get("analysis"), dict) else {}
    raw_root = raw.get("root_cause", {}) if isinstance(raw.get("root_cause"), dict) else {}
    raw_violation = raw.get("violated_invariant", {}) if isinstance(raw.get("violated_invariant"), dict) else {}
    summary = _first_non_empty(
        raw_root.get("root_cause_summary"),
        raw_root.get("violated_invariant"),
        raw_root.get("missing_validation"),
        raw_violation.get("actual"),
        raw.get("summary"),
        raw.get("analysis_summary"),
        raw.get("root_cause_hypothesis", {}).get("summary") if isinstance(raw.get("root_cause_hypothesis"), dict) else "",
        raw_analysis.get("summary"),
        "",
    )
    if not summary:
        return []
    return [{
        "hypothesis_id": "H1",
        "summary": summary,
        "role": "candidate_root_cause",
        "supporting_evidence": [item.get("evidence_id") for item in evidence_ledger[:3] if item.get("evidence_id")],
        "contradicting_evidence": [],
        "confidence": str(_first_non_empty(raw.get("confidence"), raw_analysis.get("confidence"), "medium")).lower() or "medium",
        "dedup_usable": True,
    }]


def _normalize_patch_semantics_hypotheses(raw: Dict[str, Any],
                                          evidence_ledger: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = raw.get("patch_semantics_hypotheses")
    if isinstance(existing, list) and existing:
        out: List[Dict[str, Any]] = []
        for item in existing:
            if not isinstance(item, dict):
                continue
            summary = _first_non_empty(item.get("summary"), "")
            out.append({
                "shape": _first_non_empty(item.get("shape"), _infer_fix_shape(summary)),
                "target": _first_non_empty(item.get("target"), ""),
                "summary": summary,
                "supporting_evidence": _ensure_text_list(item.get("supporting_evidence")),
                "confidence": _first_non_empty(item.get("confidence"), "medium"),
                "dedup_weight": _first_non_empty(item.get("dedup_weight"), "medium"),
            })
        return out

    fix = _normalize_fix_semantics(raw)
    if not any(fix.values()):
        return []
    return [{
        "shape": fix.get("shape") or _infer_fix_shape(fix.get("summary", "")),
        "target": fix.get("location", ""),
        "summary": fix.get("summary", ""),
        "supporting_evidence": [item.get("evidence_id") for item in evidence_ledger[:3] if item.get("evidence_id")],
        "confidence": "medium",
        "dedup_weight": "medium",
    }]


def _normalize_negative_evidence(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(_as_list(raw.get("negative_evidence")), 1):
        if isinstance(item, dict):
            out.append({
                "claim": _first_non_empty(item.get("claim"), item.get("summary"), ""),
                "why_it_matters_for_dedup": _first_non_empty(item.get("why_it_matters_for_dedup"), item.get("reason"), ""),
                "evidence": _ensure_text_list(item.get("evidence")),
                "conflict_strength": _first_non_empty(item.get("conflict_strength"), "medium"),
            })
        elif isinstance(item, str) and item.strip():
            out.append({
                "claim": item.strip(),
                "why_it_matters_for_dedup": "",
                "evidence": [],
                "conflict_strength": "low",
            })
    return out


def _normalize_uncertainty(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(_as_list(raw.get("uncertainty")), 1):
        if isinstance(item, dict):
            out.append({
                "aspect": _first_non_empty(item.get("aspect"), f"uncertainty_{idx}"),
                "reason": _first_non_empty(item.get("reason"), item.get("summary"), ""),
                "needed_evidence": _first_non_empty(item.get("needed_evidence"), ""),
                "dedup_impact": _first_non_empty(item.get("dedup_impact"), "medium"),
            })
        elif isinstance(item, str) and item.strip():
            out.append({
                "aspect": f"uncertainty_{idx}",
                "reason": item.strip(),
                "needed_evidence": "",
                "dedup_impact": "medium",
            })
    return out


def _normalize_lockdep_context(raw: Dict[str, Any], crash_card: Dict[str, Any]) -> Dict[str, Any]:
    existing = raw.get("lockdep_context")
    base = crash_card.get("lockdep_context") if isinstance(crash_card.get("lockdep_context"), dict) else {}
    if not isinstance(existing, dict):
        existing = {}
    bridge = existing.get("bpf_tracepoint_bridge") if isinstance(existing.get("bpf_tracepoint_bridge"), dict) else {}
    base_bridge = base.get("bpf_tracepoint_bridge") if isinstance(base.get("bpf_tracepoint_bridge"), dict) else {}
    return {
        "current_chain": _first_non_empty(existing.get("current_chain"), base.get("current_acquisition"), {}),
        "existing_dependency_chain": _first_non_empty(existing.get("existing_dependency_chain"), base.get("existing_dependency_chain"), []),
        "held_locks": _first_non_empty(existing.get("held_locks"), base.get("held_locks"), []),
        "unsafe_scenario": _first_non_empty(existing.get("unsafe_scenario"), base.get("unsafe_scenario"), ""),
        "lock_classes": _ensure_text_list(_first_non_empty(existing.get("lock_classes"), base.get("lock_classes"), [])),
        "irq_context": existing.get("irq_context") if isinstance(existing.get("irq_context"), dict) else {
            "irqs_disabled": "unknown",
            "hardirq_context": "unknown",
            "softirq_context": "unknown",
        },
        "bpf_tracepoint_bridge": {
            "present": bool(_first_non_empty(bridge.get("present"), base_bridge.get("present"), False)),
            "tracepoints": _ensure_text_list(_first_non_empty(bridge.get("tracepoints"), base_bridge.get("tracepoints"), [])),
            "bpf_frames": _ensure_text_list(_first_non_empty(bridge.get("bpf_frames"), base_bridge.get("bpf_frames"), [])),
            "map_operations": _ensure_text_list(_first_non_empty(bridge.get("map_operations"), base_bridge.get("map_operations"), [])),
            "source_evidence": _ensure_text_list(bridge.get("source_evidence")),
        },
    }


def _normalize_reproducer_semantics(raw: Dict[str, Any], crash_card: Dict[str, Any]) -> Dict[str, Any]:
    existing = raw.get("reproducer_semantics")
    if isinstance(existing, dict):
        base = existing
    else:
        base = crash_card.get("reproducer_semantics") if isinstance(crash_card.get("reproducer_semantics"), dict) else {}
    return {
        "syscalls": _ensure_text_list(base.get("syscalls")),
        "bpf_program_types": _ensure_text_list(base.get("bpf_program_types")),
        "bpf_helpers": _ensure_text_list(base.get("bpf_helpers")),
        "bpf_map_types": _ensure_text_list(base.get("bpf_map_types")),
        "bpf_map_ops": _ensure_text_list(base.get("bpf_map_ops")),
        "tracepoints": _ensure_text_list(base.get("tracepoints")),
        "socket_ops": _ensure_text_list(base.get("socket_ops")),
        "semantic_tokens": _ensure_text_list(base.get("semantic_tokens")),
    }


def _build_dedup_representation(raw: Dict[str, Any],
                                crash_card: Dict[str, Any],
                                root_cause_signals: List[Dict[str, Any]],
                                invariant_signals: List[Dict[str, Any]],
                                patch_hypotheses: List[Dict[str, Any]],
                                negative_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing = raw.get("dedup_representation")
    if isinstance(existing, dict):
        return {
            "must_match_tokens": _ensure_text_list(existing.get("must_match_tokens")),
            "should_match_tokens": _ensure_text_list(existing.get("should_match_tokens")),
            "weak_context_tokens": _ensure_text_list(existing.get("weak_context_tokens")),
            "must_not_match_conditions": _ensure_text_list(existing.get("must_not_match_conditions")),
            "primary_root_tokens": _ensure_text_list(existing.get("primary_root_tokens")),
            "bridge_tokens": _ensure_text_list(existing.get("bridge_tokens")),
            "surface_tokens": _ensure_text_list(existing.get("surface_tokens")),
        }

    must_match: List[str] = []
    should_match: List[str] = []
    weak_context: List[str] = []
    must_not: List[str] = []

    subsystem = crash_card.get("fault", {}).get("primary_file", "")
    if subsystem:
        parts = subsystem.split("/")
        if len(parts) >= 2:
            must_match.append(f"subsystem:{_clean_token(parts[1] if parts[0] == 'fs' else parts[0])}")

    for signal in root_cause_signals:
        token = _clean_token(signal.get("normalized_token") or signal.get("name"))
        if not token:
            continue
        weight = signal.get("dedup_weight", "low")
        prefixed = token
        if signal.get("signal_type") == "source_operation":
            prefixed = f"operation:{token}"
        elif signal.get("signal_type") in {"object", "field", "index", "length", "pointer", "lock", "state", "lifetime"}:
            prefixed = f"{signal.get('signal_type')}:{token}"
        if weight == "high":
            must_match.append(prefixed)
        elif weight == "medium":
            should_match.append(prefixed)
        else:
            weak_context.append(prefixed)

    for inv in invariant_signals:
        expected = _clean_token(inv.get("expected"))
        if expected:
            token = f"invariant:{expected[:96]}"
            if inv.get("dedup_weight") == "high":
                must_match.append(token)
            else:
                should_match.append(token)

    for patch in patch_hypotheses:
        shape = _clean_token(patch.get("shape"))
        if shape:
            should_match.append(f"patch_shape:{shape}")

    sanitizer = crash_card.get("bug", {}).get("sanitizer", "")
    if sanitizer:
        weak_context.append(f"sanitizer:{_clean_token(sanitizer)}")
    title_func = crash_card.get("bug", {}).get("title_func", "")
    if title_func:
        weak_context.append(f"function:{_clean_token(title_func)}")

    for item in negative_evidence:
        claim = _clean_token(item.get("claim"))
        if claim and item.get("conflict_strength") in {"high", "medium"}:
            must_not.append(f"conflict:{claim[:96]}")

    def _dedup_tokens(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    lockdep = crash_card.get("lockdep_context") if isinstance(crash_card.get("lockdep_context"), dict) else {}
    bridge = lockdep.get("bpf_tracepoint_bridge") if isinstance(lockdep.get("bpf_tracepoint_bridge"), dict) else {}
    repro_sem = crash_card.get("reproducer_semantics") if isinstance(crash_card.get("reproducer_semantics"), dict) else {}
    bridge_tokens = []
    for value in bridge.get("bpf_frames") or []:
        bridge_tokens.append(f"bpf_frame:{_clean_token(value)}")
    for value in bridge.get("map_operations") or []:
        bridge_tokens.append(f"map_op:{_clean_token(value)}")
    for value in bridge.get("tracepoints") or []:
        bridge_tokens.append(f"tracepoint:{_clean_token(value)}")
    for value in repro_sem.get("semantic_tokens") or []:
        bridge_tokens.append(_clean_token(value))
    for value in repro_sem.get("bpf_map_types") or []:
        bridge_tokens.append(f"repro_map_type:{_clean_token(value)}")
    for value in repro_sem.get("tracepoints") or []:
        bridge_tokens.append(f"repro_tracepoint:{_clean_token(value)}")

    surface_tokens = list(weak_context)

    return {
        "must_match_tokens": _dedup_tokens(must_match),
        "should_match_tokens": _dedup_tokens(should_match),
        "weak_context_tokens": _dedup_tokens(weak_context),
        "must_not_match_conditions": _dedup_tokens(must_not),
        "primary_root_tokens": _dedup_tokens(must_match),
        "bridge_tokens": _dedup_tokens(bridge_tokens),
        "surface_tokens": _dedup_tokens(surface_tokens),
    }


def _normalize_representation_confidence(raw: Dict[str, Any],
                                         evidence_ledger: List[Dict[str, Any]],
                                         dedup_representation: Dict[str, Any],
                                         uncertainty: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing = raw.get("representation_confidence")
    if isinstance(existing, dict):
        return {
            "level": _first_non_empty(existing.get("level"), "medium"),
            "reason": _first_non_empty(existing.get("reason"), ""),
            "direct_source_evidence_ratio": float(existing.get("direct_source_evidence_ratio", 0.0) or 0.0),
            "stable_token_count": int(existing.get("stable_token_count", 0) or 0),
            "speculative_token_count": int(existing.get("speculative_token_count", 0) or 0),
        }

    direct = sum(1 for item in evidence_ledger if item.get("evidence_level") == "direct")
    total = len(evidence_ledger)
    stable_tokens = len(dedup_representation.get("must_match_tokens", [])) + len(dedup_representation.get("should_match_tokens", []))
    speculative = sum(1 for item in evidence_ledger if item.get("evidence_level") == "speculative")
    level = str(_first_non_empty(raw.get("confidence"), "medium")).lower()
    reason = "normalized from available evidence density and dedup-ready tokens"

    if not total or stable_tokens == 0:
        level = "low"
        reason = "missing evidence ledger or dedup representation tokens"
    elif direct == 0:
        level = "low"
        reason = "no direct source-backed evidence after normalization"
    elif level not in {"high", "medium", "low"}:
        level = "medium"
    if uncertainty and level == "high" and any(item.get("dedup_impact") == "high" for item in uncertainty):
        level = "medium"
        reason = "high-impact uncertainty remains"

    return {
        "level": level,
        "reason": reason,
        "direct_source_evidence_ratio": round(direct / total, 3) if total else 0.0,
        "stable_token_count": stable_tokens,
        "speculative_token_count": speculative,
    }


def _confidence_rank(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(level or "").lower(), 1)


def _apply_input_quality_caps(cause_card: Dict[str, Any],
                              input_quality: Dict[str, Any]) -> Dict[str, Any]:
    cap = str(input_quality.get("confidence_cap") or "none").lower()
    if cap not in {"low", "medium"}:
        return cause_card

    confidence = cause_card.get("representation_confidence")
    if not isinstance(confidence, dict):
        confidence = {}
        cause_card["representation_confidence"] = confidence

    current = str(confidence.get("level") or "medium").lower()
    if _confidence_rank(current) > _confidence_rank(cap):
        confidence["level"] = cap
        reason = confidence.get("reason") or ""
        cap_reason = (
            f"input_quality cap applied: {cap}; "
            f"tier={input_quality.get('tier')}; "
            f"weak_evidence={', '.join(_ensure_text_list(input_quality.get('missing_or_weak_evidence')))}"
        )
        confidence["reason"] = f"{reason}; {cap_reason}" if reason else cap_reason

    if cap == "low":
        dedup = cause_card.get("dedup_representation")
        if isinstance(dedup, dict):
            demoted = []
            demoted.extend(_ensure_text_list(dedup.get("must_match_tokens")))
            demoted.extend(_ensure_text_list(dedup.get("should_match_tokens")))
            demoted.extend(_ensure_text_list(dedup.get("primary_root_tokens")))
            weak = _ensure_text_list(dedup.get("weak_context_tokens")) + demoted
            seen = set()
            dedup["weak_context_tokens"] = [tok for tok in weak if tok and not (tok in seen or seen.add(tok))]
            dedup["must_match_tokens"] = []
            dedup["should_match_tokens"] = []
            dedup["primary_root_tokens"] = []

    uncertainty = cause_card.get("uncertainty")
    if not isinstance(uncertainty, list):
        uncertainty = []
        cause_card["uncertainty"] = uncertainty
    uncertainty.append({
        "aspect": "input_quality_cap",
        "reason": str(input_quality.get("model_instruction") or ""),
        "needed_evidence": "more complete crash stack, source location, sanitizer lifecycle/origin trace, or reproducer semantics",
        "dedup_impact": "high" if cap == "low" else "medium",
    })
    return cause_card


def normalize_cause_card(raw: Dict[str, Any],
                         crash_card: Dict[str, Any],
                         small_bucket_id: str,
                         hash_part: str,
                         tool_trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Normalize variant LLM outputs into the v3 representation-card schema."""
    input_quality = _compute_input_quality(crash_card)
    raw_analysis = raw.get("analysis", {}) if isinstance(raw.get("analysis"), dict) else {}
    crash_surface = raw.get("crash_surface", {}) if isinstance(raw.get("crash_surface"), dict) else {}
    crash_sink = raw_analysis.get("crash_sink", {}) if isinstance(raw_analysis.get("crash_sink"), dict) else {}
    raw_root = raw.get("root_cause", {}) if isinstance(raw.get("root_cause"), dict) else {}
    if not raw_root and isinstance(raw_analysis.get("root_cause"), dict):
        raw_root = raw_analysis.get("root_cause", {})

    root_loc_source = dict(raw)
    if raw_root:
        root_loc_source.setdefault("root_cause", raw_root)
    if raw_root.get("root_cause_location") and not root_loc_source.get("root_cause_location"):
        root_loc_source["root_cause_location"] = raw_root.get("root_cause_location")
    root_locations = _normalize_root_locations(root_loc_source)

    source_evidence = _normalize_source_evidence(raw)
    if not source_evidence and isinstance(raw_analysis.get("source_evidence"), list):
        for item in raw_analysis.get("source_evidence", []):
            if isinstance(item, dict):
                source_evidence.append({
                    "ref": _first_non_empty(item.get("file"), item.get("location"), ""),
                    "relevance": _first_non_empty(item.get("evidence"), item.get("relevance"), ""),
                })

    dedup_fingerprint = _normalize_dedup_fingerprint(raw)
    evidence_ledger = _build_evidence_ledger(raw, crash_card, source_evidence, root_locations)
    root_cause_signals = _normalize_root_cause_signals(
        raw, crash_card, dedup_fingerprint, root_locations, source_evidence, evidence_ledger
    )
    invariant_signals = _normalize_invariant_signals(raw, evidence_ledger)
    propagation_signals = _normalize_propagation_signals(raw, evidence_ledger)
    hypotheses = _normalize_hypotheses(raw, evidence_ledger)
    patch_semantics_hypotheses = _normalize_patch_semantics_hypotheses(raw, evidence_ledger)
    negative_evidence = _normalize_negative_evidence(raw)
    uncertainty = _normalize_uncertainty(raw)
    dedup_representation = _build_dedup_representation(
        raw, crash_card, root_cause_signals, invariant_signals, patch_semantics_hypotheses, negative_evidence
    )
    representation_confidence = _normalize_representation_confidence(
        raw, evidence_ledger, dedup_representation, uncertainty
    )
    lockdep_context = _normalize_lockdep_context(raw, crash_card)
    reproducer_semantics = _normalize_reproducer_semantics(raw, crash_card)

    if crash_card.get("bug", {}).get("sanitizer") == "LOCKDEP":
        has_chain = bool(lockdep_context.get("existing_dependency_chain"))
        has_bridge = bool((lockdep_context.get("bpf_tracepoint_bridge") or {}).get("present"))
        surface_only = not has_chain or not has_bridge
        if surface_only and representation_confidence.get("level") == "high":
            representation_confidence["level"] = "medium"
            representation_confidence["reason"] = (
                "LOCKDEP representation capped because dependency chain or BPF/map bridge evidence is incomplete"
            )

    available_evidence = ["crash_report", "stack", "source_code"]
    if crash_card.get("sanitizer_context", {}).get("alloc_trace"):
        available_evidence.append("alloc_trace")
    if crash_card.get("sanitizer_context", {}).get("free_trace"):
        available_evidence.append("free_trace")
    if crash_card.get("sanitizer_context", {}).get("origin_trace"):
        available_evidence.append("origin_trace")
    if crash_card.get("lockdep_context"):
        available_evidence.append("lockdep_context")
    if crash_card.get("reproducer_semantics"):
        available_evidence.append("reproducer_semantics")

    cause_card = {
        "cause_card_version": "3.0-representation-normalized",
        "cause_id": _first_non_empty(raw.get("cause_id"), dedup_fingerprint.get("semantic_id"), hash_part),
        "source_bucket": crash_card.get("big_bucket", ""),
        "source_small_bucket": small_bucket_id,
        "source_crash_id": crash_card.get("case_id", ""),
        "analysis_contract": {
            "task": "collect_root_cause_representations_not_final_rca",
            "claim_policy": "all causal claims must be evidence-typed",
            "dedup_policy": "only stable high/medium evidence fields should drive dedup",
        },
        "input_scope": {
            "kernel_tree": crash_card.get("kernel_tree", ""),
            "kernel_commit": crash_card.get("kernel_commit", ""),
            "crash_card_schema": crash_card.get("schema_version", ""),
            "available_evidence": available_evidence,
        },
        "crash_surface": {
            "sanitizer": _first_non_empty(crash_surface.get("sanitizer"), crash_card.get("bug", {}).get("sanitizer"), "OTHER"),
            "bug_type": _first_non_empty(
                crash_surface.get("bug_type"),
                raw.get("crash_type"),
                raw_analysis.get("crash_type"),
                crash_card.get("bug", {}).get("bug_type"),
                "",
            ),
            "crash_point": {
                "function": _first_non_empty(
                    crash_surface.get("crash_point", {}).get("function") if isinstance(crash_surface.get("crash_point"), dict) else None,
                    crash_surface.get("function"),
                    crash_sink.get("function"),
                    crash_card.get("fault", {}).get("primary_function"),
                    "",
                ),
                "file": _first_non_empty(
                    crash_surface.get("crash_point", {}).get("file") if isinstance(crash_surface.get("crash_point"), dict) else None,
                    crash_surface.get("file"),
                    crash_sink.get("file"),
                    crash_card.get("fault", {}).get("primary_file"),
                    "",
                ),
                "line": _first_non_empty(
                    crash_surface.get("crash_point", {}).get("line") if isinstance(crash_surface.get("crash_point"), dict) else None,
                    crash_surface.get("line"),
                    crash_sink.get("line"),
                    0,
                ),
            },
            "crash_operation": _first_non_empty(
                crash_surface.get("crash_operation"),
                root_locations[0].get("description") if root_locations else "",
                "",
            ),
            "surface_helpers": _ensure_text_list(crash_surface.get("surface_helpers")),
            "surface_is_root_candidate": bool(crash_surface.get("surface_is_root_candidate", False)),
            "why_surface_may_be_misleading": _first_non_empty(
                crash_surface.get("why_surface_may_be_misleading"),
                crash_surface.get("why_not_root"),
                raw.get("crash_surface", {}).get("why_not_root") if isinstance(raw.get("crash_surface"), dict) else "",
                "",
            ),
        },
        "evidence_ledger": evidence_ledger,
        "root_cause_signals": root_cause_signals,
        "invariant_signals": invariant_signals,
        "propagation_signals": propagation_signals,
        "hypotheses": hypotheses,
        "patch_semantics_hypotheses": patch_semantics_hypotheses,
        "lockdep_context": lockdep_context,
        "reproducer_semantics": reproducer_semantics,
        "dedup_representation": dedup_representation,
        "negative_evidence": negative_evidence,
        "uncertainty": uncertainty,
        "input_quality": input_quality,
        "representation_confidence": representation_confidence,
        "tool_usage": _collect_tool_usage(tool_trace or []),
    }
    return _apply_input_quality_caps(cause_card, input_quality)


def generate_one_cause_card(card_path: pathlib.Path,
                            output_dir: pathlib.Path,
                            client: Any,
                            model: str,
                            system_prompt: str,
                            temperature: float = DEFAULT_TEMPERATURE,
                            max_tool_rounds: int = MAX_TOOL_ROUNDS,
                            intervention_round: int = INTERVENTION_ROUND,
                            save_debug: bool = False,
                            force: bool = False,
                            dry_run: bool = False) -> bool:
    """
    Process a single representative crash card → cause card.

    Returns True on success (or if skipped in dry-run).
    """
    # Derive output filename from the card filename stem
    stem = card_path.stem  # e.g. "001__3cf6302f"
    # Keep just the hash part for the cause card filename
    parts = stem.split("__")
    hash_part = parts[-1] if len(parts) > 1 else stem
    output_file = output_dir / f"{hash_part}.json"
    debug_file = output_dir / "_debug" / f"{hash_part}.debug.json"

    if output_file.exists() and not force:
        print(f"  [SKIP] cause card already exists: {output_file.name}")
        return True

    if dry_run:
        print(f"  [DRY-RUN] would process: {card_path.name} → {output_file.name}")
        return True

    # Load crash card
    try:
        with open(card_path, "r", encoding="utf-8") as f:
            crash_card = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Failed to load crash card: {e}")
        return False

    # Truncate very large crash cards (remove raw stack with 30+ frames to save tokens)
    # Keep semantic stack and sanitizer context intact
    crash_card_str = json.dumps(crash_card, ensure_ascii=False)

    # Build messages
    input_quality = _compute_input_quality(crash_card)
    prefetched_source_packet = build_prefetched_source_packet(crash_card)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "Use the prompt contract above and produce a Cause Representation Card JSON.\n\n"
            "IMPORTANT: actively use the code_searcher tools to retrieve kernel source code. "
            "Do NOT guess. Focus on stable, source-grounded representations for deduplication rather than proving one final root cause. "
            "A deterministic prefetched source packet is provided below to reduce wasted micro-lookups. "
            "Use it first, then use the tools to validate or expand only where evidence is still missing. "
            "Prefer wider line windows, full function bodies, file outlines, and batch requests over single-line or single-macro probes. "
            "Do not spend extra rounds expanding accessor-style macros unless they materially change lock, lifetime, bounds, refcount, or validation semantics. "
            "Retrieve the crash sink context, the first 2-4 non-generic domain-specific frames, relevant sanitizer traces, "
            "and enough source evidence to fill evidence_ledger and dedup_representation.\n\n"
            "Input quality assessment prepared by the script:\n"
            + _format_input_quality_for_prompt(input_quality)
            + "\n\n"
            "Follow the input_quality confidence_cap. If it is low, converge quickly and output a conservative low-confidence card. "
            "If it is medium, do not force high-confidence lifecycle/origin/root-cause claims without direct matching evidence. "
            "If it is none, proceed normally.\n\n"
            "Prefetched source packet:\n"
            + (prefetched_source_packet or "<none>")
            + "\n\n"
            "Output ONLY valid JSON — no explanation outside the JSON.\n\n"
            "Crash card:\n" + crash_card_str
        )}
    ]

    # Tool calling loop
    tool_rounds = 0
    last_error = None
    tool_trace: List[Dict[str, Any]] = []
    last_content = ""

    for attempt in range(RETRY_MAX + 1):
        try:
            messages_current = [m.copy() for m in messages]  # fresh copy for retry
            tool_rounds = 0
            tool_trace = []
            last_content = ""
            intervention_sent = False

            while tool_rounds < max_tool_rounds:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages_current,
                    tools=TOOLS,
                    tool_choice="auto",
                    timeout=LLM_TIMEOUT,
                    temperature=temperature,
                )
                msg = response.choices[0].message

                if msg.tool_calls:
                    # LLM wants to use tools
                    messages_current.append(msg)
                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}
                        print(f"    [tool R{tool_rounds+1}] {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:120]})")
                        tool_trace.append({
                            "round": tool_rounds + 1,
                            "tool": fn_name,
                            "args": fn_args,
                        })
                        result = execute_tool(fn_name, fn_args)
                        # Truncate very long results
                        if len(result) > TOOL_RESULT_CHAR_LIMIT:
                            result = result[:TOOL_RESULT_CHAR_LIMIT] + "\n...[truncated]..."
                        messages_current.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        })
                    tool_rounds += 1
                    print(f"    [round {tool_rounds}/{max_tool_rounds} complete, {len(msg.tool_calls)} tool call(s)]")

                    # Intervention: nudge LLM to wrap up (only once at or beyond the threshold)
                    if intervention_round > 0 and tool_rounds >= intervention_round and not intervention_sent:
                        messages_current.append({
                            "role": "user",
                            "content": (
                                f"You have used {tool_rounds} tool-calling rounds (limit: {max_tool_rounds}). "
                                "You should only continue if one missing subsystem-specific source retrieval would materially change dedup_representation. "
                                "Otherwise STOP calling tools and produce the final Cause Representation Card JSON IMMEDIATELY. "
                                "Output ONLY valid JSON — no explanation outside the JSON."
                            )
                        })
                        intervention_sent = True

                    if tool_rounds >= max_tool_rounds:
                        final_response = client.chat.completions.create(
                            model=model,
                            messages=messages_current + [{
                                "role": "user",
                                "content": (
                                    f"Tool budget exhausted at {tool_rounds} rounds. "
                                    "Do not call any more tools. Produce the final Cause Representation Card JSON now. "
                                    "Output ONLY valid JSON."
                                )
                            }],
                            timeout=LLM_TIMEOUT,
                            temperature=temperature,
                        )
                        final_msg = final_response.choices[0].message
                        content = final_msg.content or ""
                        last_content = content
                        cause_card_raw = extract_json(content)
                        if cause_card_raw is None:
                            print(f"  [WARN] Could not parse JSON after forced finalization. Raw preview:")
                            print(f"    {content[:300]}...")
                            raw_file = output_dir / f"{hash_part}_raw.txt"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            raw_file.write_text(content, encoding="utf-8")
                            _write_debug_record(
                                debug_file,
                                card_path=card_path,
                                output_file=output_file,
                                model=model,
                                prompt_file=PROMPT_FILE,
                                status="forced_finalize_json_parse_failed",
                                tool_trace=tool_trace,
                                last_content=content,
                            )
                            return False
                        small_bucket_id = parts[0] if len(parts) > 1 else ""
                        cause_card = normalize_cause_card(
                            cause_card_raw,
                            crash_card,
                            small_bucket_id,
                            hash_part,
                            tool_trace=tool_trace,
                        )
                        output_dir.mkdir(parents=True, exist_ok=True)
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(cause_card, f, ensure_ascii=False, indent=2)
                        if save_debug:
                            _write_debug_record(
                                debug_file,
                                card_path=card_path,
                                output_file=output_file,
                                model=model,
                                prompt_file=PROMPT_FILE,
                                status="ok_forced_finalize",
                                tool_trace=tool_trace,
                                last_content=content,
                            )
                        print(f"  [OK] → {output_file.name} (forced finalize)")
                        return True

                else:
                    # Final response — should contain cause card JSON
                    content = msg.content or ""
                    last_content = content
                    cause_card_raw = extract_json(content)
                    if cause_card_raw is None:
                        print(f"  [WARN] Could not parse JSON from response. Raw preview:")
                        print(f"    {content[:300]}...")
                        # Save raw response for debugging
                        raw_file = output_dir / f"{hash_part}_raw.txt"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        raw_file.write_text(content, encoding="utf-8")
                        _write_debug_record(
                            debug_file,
                            card_path=card_path,
                            output_file=output_file,
                            model=model,
                            prompt_file=PROMPT_FILE,
                            status="json_parse_failed",
                            tool_trace=tool_trace,
                            last_content=content,
                        )
                        return False
                    small_bucket_id = parts[0] if len(parts) > 1 else ""
                    cause_card = normalize_cause_card(
                        cause_card_raw,
                        crash_card,
                        small_bucket_id,
                        hash_part,
                        tool_trace=tool_trace,
                    )

                    # Save cause card
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(cause_card, f, ensure_ascii=False, indent=2)
                    if save_debug:
                        _write_debug_record(
                            debug_file,
                            card_path=card_path,
                            output_file=output_file,
                            model=model,
                            prompt_file=PROMPT_FILE,
                            status="ok",
                            tool_trace=tool_trace,
                            last_content=content,
                        )
                    print(f"  [OK] → {output_file.name}")
                    return True

            # Exceeded max tool rounds — save what we have
            print(f"  [WARN] Exceeded {max_tool_rounds} tool rounds, no final cause card")
            _write_debug_record(
                debug_file,
                card_path=card_path,
                output_file=output_file,
                model=model,
                prompt_file=PROMPT_FILE,
                status="tool_rounds_exceeded",
                tool_trace=tool_trace,
                last_content=last_content,
            )
            return False

        except Exception as e:
            last_error = e
            if attempt < RETRY_MAX:
                wait = 2 ** attempt
                print(f"  [RETRY] attempt {attempt+1}/{RETRY_MAX} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  [ERROR] All retries exhausted: {e}")
                traceback.print_exc()
                _write_debug_record(
                    debug_file,
                    card_path=card_path,
                    output_file=output_file,
                    model=model,
                    prompt_file=PROMPT_FILE,
                    status="exception",
                    tool_trace=tool_trace,
                    last_content=last_content,
                    error=str(e),
                )
                return False

    return False


def iter_all_cards(bugs_dir: pathlib.Path, output_dir_name: str):
    """Yield (card_path, output_dir) for all representative crash cards."""
    for bug_dir in sorted(bugs_dir.iterdir()):
        if not bug_dir.is_dir():
            continue
        rep_dir = bug_dir / "crash_cards" / "small_bucket_rep"
        if not rep_dir.is_dir():
            continue
        output_dir = bug_dir / output_dir_name
        for card_file in sorted(rep_dir.glob("*.json")):
            yield card_file, output_dir


def iter_selected_cards(cards_file: pathlib.Path, output_dir_name: str):
    """Yield (card_path, output_dir) for selected representative crash cards."""
    for raw_line in cards_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        card_path = pathlib.Path(line)
        if not card_path.is_absolute():
            card_path = (PROJECT_DIR / line).resolve()
        if not card_path.exists():
            print(f"  [WARN] card not found, skip: {card_path}")
            continue
        yield card_path, card_path.parent.parent.parent / output_dir_name


def _card_hash_part(card_path: pathlib.Path) -> str:
    stem = card_path.stem
    parts = stem.split("__")
    return parts[-1] if len(parts) > 1 else stem


def collect_batch_tasks(iterator,
                        limit: int,
                        force: bool) -> Dict[str, Any]:
    """Collect pending batch tasks and precompute skip counts."""
    tasks: List[Dict[str, Any]] = []
    skipped = 0
    seen = 0

    for card_path, output_dir in iterator:
        if limit and seen >= limit:
            break
        seen += 1
        hash_part = _card_hash_part(card_path)
        output_file = output_dir / f"{hash_part}.json"
        if not force and output_file.exists():
            print(f"  [SKIP] cause card already exists: {hash_part}.json")
            skipped += 1
            continue
        bug_name = card_path.parent.parent.parent.name
        short_bug = bug_name[:60] if len(bug_name) > 60 else bug_name
        tasks.append({
            "card_path": card_path,
            "output_dir": output_dir,
            "bug_name": bug_name,
            "short_bug": short_bug,
            "hash_part": hash_part,
        })

    return {
        "tasks": tasks,
        "skipped": skipped,
        "seen": seen,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global LINUX_SRC, PROMPT_FILE, API_KEY, BASE_URL, MODEL

    parser = argparse.ArgumentParser(
        description="Generate cause cards using LLM + kernel_code_searcher"
    )
    default_bugs = str(BUGS_DIR)
    default_linux = str(LINUX_SRC)
    default_prompt = str(PROMPT_FILE)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--card", type=str, help="Process a single representative crash card file")
    group.add_argument("--all", action="store_true", help="Process all representative crash cards")
    group.add_argument("--cards-file", type=str,
                       help="Process a newline-separated list of representative crash card paths")
    parser.add_argument("--bugs-dir", type=str, default=default_bugs,
                        help=f"Path to bugs directory (default: {default_bugs})")
    parser.add_argument("--linux-src", type=str, default=default_linux,
                        help=f"Path to Linux source tree (default: {default_linux})")
    parser.add_argument("--prompt-file", type=str, default=default_prompt,
                        help=f"Path to prompt file (default: {default_prompt})")
    parser.add_argument("--env-file", type=str, default=str(DEFAULT_ENV_FILE),
                        help=f"Path to env file with OPENAI_* settings (default: {DEFAULT_ENV_FILE})")
    parser.add_argument("--output-dir-name", type=str, default="cause_cards",
                        help="Per-bug output directory name (default: cause_cards)")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save per-card debug sidecars with tool traces and raw final response previews")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing cause card outputs instead of skipping")
    parser.add_argument("--resume", action="store_true", help="(deprecated: skip is now default) Skip already-existing cause cards")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without calling LLM")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N cards (for testing)")
    parser.add_argument("--parallelism", type=int, default=1,
                        help="Number of representative crash cards to process concurrently in batch mode (default: 1)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"LLM temperature for stability (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tool-rounds", type=int, default=MAX_TOOL_ROUNDS,
                        help=f"Maximum tool-calling rounds per card (default: {MAX_TOOL_ROUNDS})")
    parser.add_argument("--intervention-round", type=int, default=INTERVENTION_ROUND,
                        help=f"Round at which to inject a wrap-up nudge (default: {INTERVENTION_ROUND})")
    args = parser.parse_args()

    # Resolve paths from args
    bugs_dir = pathlib.Path(args.bugs_dir)
    linux_src = pathlib.Path(args.linux_src)
    prompt_file = pathlib.Path(args.prompt_file)
    env_file = pathlib.Path(args.env_file)
    parallelism = max(1, int(args.parallelism))

    # Override module-level globals so _get_searcher / load_prompt use the right paths
    LINUX_SRC = linux_src
    PROMPT_FILE = prompt_file

    load_env_file(env_file)
    API_KEY = os.environ.get("OPENAI_API_KEY", API_KEY)
    BASE_URL = os.environ.get("OPENAI_BASE_URL", BASE_URL)
    MODEL = os.environ.get("OPENAI_MODEL", MODEL)

    # Load prompt
    system_prompt = load_prompt()
    print(f"Loaded system prompt ({len(system_prompt)} chars)")

    if args.dry_run:
        print("DRY-RUN mode — no LLM calls will be made")

    if not args.dry_run:
        if not API_KEY:
            raise SystemExit("OPENAI_API_KEY environment variable not set. "
                             "Run: $env:OPENAI_API_KEY='sk-...' (PowerShell)")
        if not linux_src.is_dir():
            raise SystemExit(f"Linux source not found: {linux_src}")

        client = create_openai_client()
        print(f"LLM client: {BASE_URL} / {MODEL}")
        print(f"Config: temperature={args.temperature}, max_tool_rounds={args.max_tool_rounds}, intervention_round={args.intervention_round}, parallelism={parallelism}")
    else:
        client = None

    # Process
    if args.card:
        card_path = pathlib.Path(args.card)
        if not card_path.exists():
            raise SystemExit(f"Card file not found: {card_path}")
        # Infer output dir from card path
        output_dir = card_path.parent.parent.parent / args.output_dir_name
        print(f"Processing single card: {card_path}")
        generate_one_cause_card(card_path, output_dir, client, MODEL,
                                system_prompt,
                                temperature=args.temperature,
                                max_tool_rounds=args.max_tool_rounds,
                                intervention_round=args.intervention_round,
                                save_debug=args.save_debug,
                                force=args.force,
                                dry_run=args.dry_run)

    elif args.cards_file or args.all:
        iterator = (
            iter_selected_cards(pathlib.Path(args.cards_file), args.output_dir_name)
            if args.cards_file else
            iter_all_cards(bugs_dir, args.output_dir_name)
        )
        batch = collect_batch_tasks(iterator, args.limit, args.force)
        tasks = batch["tasks"]
        skipped = batch["skipped"]
        seen = batch["seen"]
        processed = 0
        failed = 0

        print(f"Batch scan: seen={seen}, pending={len(tasks)}, skipped_existing={skipped}")

        if args.dry_run or parallelism == 1 or len(tasks) <= 1:
            for task in tasks:
                print(f"\n[{task['short_bug']}] {task['card_path'].name}")
                ok = generate_one_cause_card(task["card_path"], task["output_dir"], client, MODEL,
                                             system_prompt,
                                             temperature=args.temperature,
                                             max_tool_rounds=args.max_tool_rounds,
                                             intervention_round=args.intervention_round,
                                             save_debug=args.save_debug,
                                             force=args.force,
                                             dry_run=args.dry_run)
                if ok:
                    processed += 1
                    if not args.dry_run:
                        time.sleep(1)
                else:
                    failed += 1
        else:
            print(f"Launching parallel batch with {parallelism} workers")

            def _worker(task_spec: Dict[str, Any]) -> Dict[str, Any]:
                worker_client = _get_thread_client()
                ok = generate_one_cause_card(
                    task_spec["card_path"],
                    task_spec["output_dir"],
                    worker_client,
                    MODEL,
                    system_prompt,
                    temperature=args.temperature,
                    max_tool_rounds=args.max_tool_rounds,
                    intervention_round=args.intervention_round,
                    save_debug=args.save_debug,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                return {
                    "ok": ok,
                    "task": task_spec,
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
                future_to_task = {}
                for task in tasks:
                    print(f"\n[QUEUE] [{task['short_bug']}] {task['card_path'].name}")
                    future = executor.submit(_worker, task)
                    future_to_task[future] = task

                completed = 0
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    completed += 1
                    try:
                        result = future.result()
                        if result["ok"]:
                            processed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        print(f"  [ERROR] Worker failed for {task['card_path'].name}: {e}")

                    print(f"[PROGRESS] completed={completed}/{len(tasks)}, success={processed}, failed={failed}, skipped_existing={skipped}")

        print(f"\n{'='*60}")
        print(f"Done. Processed: {processed}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
