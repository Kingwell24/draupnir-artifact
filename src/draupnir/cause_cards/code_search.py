#!/usr/bin/env python3
r"""
Kernel code searcher for LLM-assisted syzbot crash root-cause analysis.

Works against a locally cached Linux source tree (no network access needed).
Provides both a Python API (KernelCodeSearcher class) and a CLI.

Usage as CLI:
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src line --file sound/core/timer.c --line 818
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src function --symbol snd_timer_interrupt
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src callees --symbol snd_timer_interrupt --file sound/core/timer.c
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src callers --symbol snd_timer_close --paths 'sound/core/*.c'
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src grep --pattern 'spin_lock_irqsave.*timer' --paths 'sound/core/*.c'
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src struct --name snd_timer
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src field --struct snd_timer --field flags
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src macro --name CONFIG_SND_TIMER
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src containing --file sound/core/timer.c --line 818
  python -m draupnir.cause_cards.code_search --repo /path/to/linux_src batch --requests '[{"op":"function","symbol":"snd_timer_interrupt","file":"sound/core/timer.c"},{"op":"callers","symbol":"snd_timer_interrupt","paths":["sound/core/*.c","include/sound/*.h"]}]'

Usage as Python library:
  from draupnir.cause_cards.code_search import KernelCodeSearcher
  ks = KernelCodeSearcher("/path/to/linux_src")
  result = ks.get_function("snd_timer_interrupt", file="sound/core/timer.c")
  callers = ks.find_callers("snd_timer_interrupt", paths=["sound/core/*.c"])
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Core searcher class
# ---------------------------------------------------------------------------

class KernelCodeSearcher:
    """Search a locally cached Linux kernel source tree."""

    def __init__(self, repo: str):
        self.repo = pathlib.Path(repo).resolve()
        if not self.repo.is_dir():
            raise SystemExit(f"repo not found: {self.repo}")
        self._has_rg: Optional[bool] = None

    # -- helpers -----------------------------------------------------------

    def _resolve(self, file: str) -> pathlib.Path:
        """Resolve a file path relative to repo root; reject path traversal."""
        p = (self.repo / file).resolve()
        if self.repo not in [p] + list(p.parents):
            raise ValueError(f"path escapes repo: {file}")
        return p

    def _read_lines(self, file: str) -> List[str]:
        return self._resolve(file).read_text(errors="replace").splitlines()

    def _expand_paths(self, paths: List[str]) -> List[str]:
        """Expand glob patterns in paths relative to repo root."""
        out: List[str] = []
        for pat in paths:
            if any(ch in pat for ch in "*?["):
                matches = sorted(self.repo.glob(pat))
                out.extend(str(m.relative_to(self.repo)) for m in matches if m.is_file())
            else:
                out.append(pat)
        return out or paths

    def _try_external_grep(self, pattern: str, paths: List[str]) -> Optional[str]:
        """Try ripgrep or git grep; return None if neither is available."""
        if self._has_rg is None:
            try:
                subprocess.run(["rg", "--version"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=True)
                self._has_rg = True
            except (FileNotFoundError, subprocess.CalledProcessError):
                try:
                    subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=True)
                    self._has_rg = False
                except (FileNotFoundError, subprocess.CalledProcessError):
                    self._has_rg = False  # fall back to Python
        if self._has_rg is True:
            try:
                cmd = ["rg", "-n", "--no-heading", pattern] + paths
                p = subprocess.run(cmd, cwd=str(self.repo), text=True,
                                   encoding="utf-8", errors="replace",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=30)
                return p.stdout if p.returncode in (0, 1) else None
            except Exception:
                return None
        elif self._has_rg is False:
            try:
                cmd = ["git", "grep", "-n", "-E", pattern, "--"] + paths
                p = subprocess.run(cmd, cwd=str(self.repo), text=True,
                                   encoding="utf-8", errors="replace",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=30)
                return p.stdout if p.returncode in (0, 1) else None
            except Exception:
                return None
        return None

    @staticmethod
    def _posix_to_python(pattern: str) -> str:
        """Convert POSIX character classes to Python regex equivalents.

        Uses literal character sets (not \\s/\\d shorthand) so replacement
        is safe inside bracket expressions on all Python versions.
        """
        posix_map = {
            "space": r" \t\n\r\f\v",
            "blank": r" \t",
            "alnum": r"a-zA-Z0-9",
            "alpha": r"a-zA-Z",
            "digit": r"0-9",
            "upper": r"A-Z",
            "lower": r"a-z",
            "xdigit": r"0-9a-fA-F",
        }
        cls_names = "|".join(posix_map.keys())
        # Pass 1: standalone [[:class:]] — replace with bracket expression
        def replace_standalone(m):
            return "[" + posix_map.get(m.group(1), m.group(0)) + "]"
        pattern = re.sub(r'\[\[:(' + cls_names + r'):\]\]',
                         replace_standalone, pattern)
        # Pass 2: [:class:] inside bracket expressions (e.g. [^[:space:]\(])
        def replace_inner(m):
            full = m.group(0)
            for cls_name, py_val in posix_map.items():
                full = full.replace(f"[:{cls_name}:]", py_val)
            return full
        pattern = re.sub(r'\[[^\]]*\[:(' + cls_names + r'):\][^\]]*\]',
                         replace_inner, pattern)
        return pattern

    @staticmethod
    def _python_grep(pattern: str, files: List[str], repo_root: pathlib.Path,
                     limit: int = 20000) -> str:
        """Pure-Python regex grep fallback."""
        import warnings
        py_pattern = KernelCodeSearcher._posix_to_python(pattern)
        # \s inside [...] is valid in Python 3.12+ (FutureWarning in 3.10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            compiled = re.compile(py_pattern)
        out_lines: List[str] = []
        total_chars = 0
        for fpath_str in files:
            fpath = repo_root / fpath_str
            if not fpath.is_file():
                continue
            try:
                for lineno, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        out_lines.append(f"{fpath_str}:{lineno}:{line}")
                        total_chars += len(out_lines[-1]) + 1
                        if total_chars > limit:
                            out_lines.append("...[truncated]...")
                            return "\n".join(out_lines)
            except Exception:
                continue
        return "\n".join(out_lines)

    def _run_grep(self, pattern: str, paths: Optional[List[str]] = None,
                  limit: int = 20000) -> str:
        """Run ripgrep, git grep, or pure-Python fallback."""
        expanded = self._expand_paths(paths or ["*.c", "*.h"])
        out = self._try_external_grep(pattern, expanded)
        if out is None:
            out = self._python_grep(pattern, expanded, self.repo, limit)
        elif self._has_rg is False and not out.strip():
            # git grep on shallow clones can miss files not in its index;
            # fall back to Python so we don't silently return empty results
            out = self._python_grep(pattern, expanded, self.repo, limit)
        if len(out) > limit:
            out = out[:limit] + "\n...[truncated]...\n"
        return out

    # -- function search ---------------------------------------------------

    @staticmethod
    def _find_function_bounds(lines: List[str], symbol: str,
                              around_line: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """Locate a C function definition and return (start_line, end_line) 1-indexed."""
        # Pass 1: standard definition pattern  name(...)  with no semicolon before {
        candidates = []
        pat = re.compile(r"\b" + re.escape(symbol) + r"\s*\([^;]*$")
        in_block = False
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if stripped.startswith("/*"):
                in_block = True
            if in_block or stripped.startswith(("*", "//")):
                if "*/" in stripped:
                    in_block = False
                continue
            if "*/" in stripped:
                in_block = False
                continue
            if symbol in ln and pat.search(ln):
                candidates.append(i)
        # Pass 2: multi-line definition  name(\n...) {
        if not candidates:
            in_block = False
            for i, ln in enumerate(lines):
                stripped = ln.strip()
                if stripped.startswith("/*"):
                    in_block = True
                if in_block or stripped.startswith(("*", "//")):
                    if "*/" in stripped:
                        in_block = False
                    continue
                if "*/" in stripped:
                    in_block = False
                    continue
                if re.search(r"\b" + re.escape(symbol) + r"\s*\(", ln):
                    window = "\n".join(lines[i:min(len(lines), i + 10)])
                    if "{" in window and ";" not in window.split("{")[0]:
                        candidates.append(i)
        if not candidates:
            return None
        # Pick closest to around_line if given
        idx = candidates[0]
        if around_line:
            idx = min(candidates, key=lambda x: abs((x + 1) - around_line))
        # Find opening brace
        open_i = idx
        while open_i < len(lines) and "{" not in lines[open_i]:
            open_i += 1
        if open_i >= len(lines):
            return None
        # Match braces
        depth = 0
        started = False
        end = open_i
        for j in range(open_i, len(lines)):
            code = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[j])
            depth += code.count("{")
            if code.count("{"):
                started = True
            depth -= code.count("}")
            if started and depth <= 0:
                end = j
                break
        return idx + 1, end + 1

    def get_function(self, symbol: str, file: Optional[str] = None,
                     around_line: Optional[int] = None,
                     before: int = 5, after: int = 5,
                     max_files: int = 5) -> List[Dict[str, Any]]:
        """Return the source of a function definition."""
        results: List[Dict[str, Any]] = []
        files: List[str] = []
        if file:
            files = [file]
        else:
            grep_out = self._run_grep(r"\b" + re.escape(symbol) + r"\s*\(", ["*.c", "*.h"])
            for ln in grep_out.splitlines():
                f = ln.split(":", 1)[0]
                if f not in files:
                    files.append(f)
        for f in files[:max_files]:
            path = self._resolve(f)
            if not path.exists():
                continue
            lines = path.read_text(errors="replace").splitlines()
            bounds = self._find_function_bounds(lines, symbol, around_line)
            if bounds:
                start, end = bounds
                disp_start = max(1, start - before)
                disp_end = min(len(lines), end + after)
                body_lines = []
                for no in range(disp_start, disp_end + 1):
                    marker = ">" if no == start else " "
                    body_lines.append(f"{marker}{no:5d}: {lines[no - 1]}")
                results.append({
                    "file": f,
                    "start_line": start,
                    "end_line": end,
                    "display_start": disp_start,
                    "display_end": disp_end,
                    "source": "\n".join(body_lines)
                })
        return results

    # -- line context ------------------------------------------------------

    def get_lines(self, file: str, line: int,
                  before: int = 60, after: int = 80) -> Dict[str, Any]:
        """Return source lines around a given file:line."""
        lines = self._read_lines(file)
        start = max(1, line - before)
        end = min(len(lines), line + after)
        payload = []
        for no in range(start, end + 1):
            mark = "=>" if no == line else "  "
            payload.append(f"{mark} {no:5d}: {lines[no - 1]}")
        return {
            "file": file,
            "target_line": line,
            "display_start": start,
            "display_end": end,
            "source": "\n".join(payload)
        }

    # -- caller search -----------------------------------------------------

    def find_callers(self, symbol: str, paths: Optional[List[str]] = None,
                     limit: int = 20000) -> str:
        """Find all call sites of a function."""
        pattern = r"(^|[^A-Za-z0-9_])" + re.escape(symbol) + r"[[:space:]]*\("
        return self._run_grep(pattern, paths, limit=limit)

    # -- callee extraction -------------------------------------------------

    def find_callees(self, symbol: str, file: Optional[str] = None,
                     max_depth: int = 1) -> List[Dict[str, Any]]:
        """Extract function calls made within a function's body.

        This is a heuristic parser – it extracts identifiers followed by '('
        and filters out C keywords and common non-function patterns.
        Returns call targets grouped by approximate category.
        """
        C_KEYWORDS = {
            "if", "for", "while", "switch", "return", "sizeof", "typeof",
            "defined", "offsetof", "container_of", "BUILD_BUG_ON",
            "likely", "unlikely", "__must_check", "__printf",
            "case", "default", "goto", "break", "continue", "void",
            "char", "int", "long", "short", "float", "double", "unsigned",
            "signed", "const", "volatile", "static", "inline", "extern",
            "struct", "union", "enum", "typedef",
        }
        funcs = self.get_function(symbol, file=file, before=0, after=0)
        if not funcs:
            return [{"error": f"function '{symbol}' not found"}]
        result = []
        for f_info in funcs:
            source = f_info["source"]
            # Extract identifiers immediately before '('
            callee_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
            raw_callees: List[str] = []
            for m in callee_pattern.finditer(source):
                name = m.group(1)
                if name not in C_KEYWORDS and not name.startswith("__maybe"):
                    raw_callees.append(name)
            # Deduplicate, preserve order
            seen = set()
            unique = []
            for n in raw_callees:
                if n not in seen:
                    seen.add(n)
                    unique.append(n)
            result.append({
                "file": f_info["file"],
                "function": symbol,
                "line_range": f"{f_info['start_line']}-{f_info['end_line']}",
                "callees": unique,
                "callee_count": len(unique)
            })
        return result

    # -- struct definition -------------------------------------------------

    def get_struct_def(self, name: str, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Find a struct definition and return its body."""
        # Search for "struct name {" pattern
        pattern = r"struct[[:space:]]+" + re.escape(name) + r"[[:space:]]*\{"
        out = self._run_grep(pattern, paths or ["include/*.h", "include/*/*.h", "*.h", "*/*.h", "*.c", "*/*.c"])
        if not out.strip():
            return {"error": f"struct '{name}' not found", "name": name}
        # Parse the first match to extract the struct body
        first_line = out.splitlines()[0]
        file = first_line.split(":", 1)[0]
        line_no = int(first_line.split(":", 2)[1].split(":")[0])
        lines = self._read_lines(file)
        # Find the full struct body
        start = line_no - 1
        while start > 0 and "struct" not in lines[start - 1]:
            start -= 1
        # Count braces from the definition line
        idx = line_no - 1
        depth = 0
        started = False
        end_idx = idx
        for j in range(idx, len(lines)):
            code = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[j])
            depth += code.count("{")
            if code.count("{") and not started:
                started = True
            depth -= code.count("}")
            if started and depth <= 0:
                end_idx = j
                break
        # Extract fields
        body = []
        fields = []
        for j in range(idx, end_idx + 1):
            body.append(f"{j+1:5d}: {lines[j]}")
            stripped = lines[j].strip()
            if stripped and not stripped.startswith(("struct", "{", "}", "/*", "*", "//")) \
               and not stripped.startswith("#") and not stripped.endswith("\\") \
               and stripped.endswith(";"):
                fields.append(stripped.rstrip(";").strip())
        return {
            "name": name,
            "file": file,
            "lines": f"{idx+1}-{end_idx+1}",
            "field_count": len(fields),
            "fields": fields,
            "source": "\n".join(body)
        }

    # -- field usage -------------------------------------------------------

    def find_field_usage(self, struct_name: str, field_name: str,
                         paths: Optional[List[str]] = None,
                         limit: int = 20000) -> str:
        """Find code locations that access struct.field.

        Searches for patterns like:  ->field  or  .field  preceded by
        context that suggests the struct type."""
        # Broad search: just find field name preceded by -> or .
        pattern = r"(->|\\.)" + re.escape(field_name) + r"\b"
        return self._run_grep(pattern, paths, limit=limit)

    # -- macro definition --------------------------------------------------

    def get_macro_def(self, name: str, limit: int = 20000) -> Dict[str, Any]:
        """Find a #define macro definition."""
        pattern = r"#define[[:space:]]+" + re.escape(name) + r"[[:space:]\(]"
        out = self._run_grep(
            pattern,
            ["*.h", "*/*.h", "include/*.h", "include/*/*.h", "*.c", "*/*.c"],
            limit=limit,
        )
        results: List[Dict[str, str]] = []
        for ln in out.splitlines():
            if not ln.strip():
                continue
            parts = ln.split(":", 2)
            if len(parts) >= 3:
                results.append({"file": parts[0], "line": parts[1], "text": parts[2].strip()})
        return {"name": name, "matches": len(results), "definitions": results}

    # -- containing function -----------------------------------------------

    def find_containing_function(self, file: str, line: int) -> Dict[str, Any]:
        """Find the function that contains a given file:line."""
        lines = self._read_lines(file)
        if line > len(lines) or line < 1:
            return {"error": f"line {line} out of range (file has {len(lines)} lines)"}
        # Walk backwards to find function definition
        func_pat = re.compile(
            r'^[A-Za-z_][A-Za-z0-9_:*\s]+\s+'  # return type
            r'([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*$'  # function name ( ... not ending in ;
        )
        for i in range(line - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped.startswith(("/*", "*", "//", "#")):
                continue
            m = func_pat.match(stripped)
            if m:
                func_name = m.group(1)
                # Verify this is likely a function definition (not a call)
                if func_name in {"if", "for", "while", "switch", "return"}:
                    continue
                # Try to find full bounds
                bounds = self._find_function_bounds(lines, func_name, around_line=i + 1)
                if bounds and bounds[0] <= line <= bounds[1]:
                    return {
                        "function": func_name,
                        "file": file,
                        "func_start": bounds[0],
                        "func_end": bounds[1],
                        "target_line": line
                    }
                # Fallback: just return the name
                return {
                    "function": func_name,
                    "file": file,
                    "func_start": i + 1,
                    "func_end": None,
                    "target_line": line,
                    "note": "bounds not fully resolved"
                }
        return {"error": f"no containing function found for {file}:{line}"}

    # -- file outline ------------------------------------------------------

    def get_file_outline(self, file: str, around_line: Optional[int] = None,
                         radius: int = 220, max_macros: int = 60,
                         max_functions: int = 40) -> Dict[str, Any]:
        """Return a compact file-level outline near a target line.

        Useful when the model needs nearby macros and function layout without
        spending several single-purpose tool calls.
        """
        lines = self._read_lines(file)
        total_lines = len(lines)
        start = 1
        end = total_lines
        if around_line:
            start = max(1, around_line - radius)
            end = min(total_lines, around_line + radius)

        macro_re = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\b")
        macros: List[Dict[str, Any]] = []
        for no in range(start, end + 1):
            match = macro_re.match(lines[no - 1])
            if not match:
                continue
            macros.append({
                "line": no,
                "name": match.group(1),
                "text": lines[no - 1].strip(),
            })
            if len(macros) >= max_macros:
                break

        func_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*$")
        functions: List[Dict[str, Any]] = []
        seen_functions = set()
        if around_line:
            containing = self.find_containing_function(file, around_line)
            if isinstance(containing, dict) and containing.get("function"):
                marker = (
                    containing.get("function"),
                    containing.get("func_start"),
                    containing.get("func_end"),
                )
                seen_functions.add(marker)
                functions.append({
                    "name": containing.get("function"),
                    "start_line": containing.get("func_start"),
                    "end_line": containing.get("func_end"),
                    "contains_target": True,
                })
        idx = start - 1
        while idx < end:
            line = lines[idx]
            stripped = line.strip()
            if stripped.startswith(("/*", "*", "//", "#")):
                idx += 1
                continue
            if line[:1].isspace():
                idx += 1
                continue
            if "(" not in line or ";" in line:
                idx += 1
                continue
            match = func_re.search(line)
            if not match:
                idx += 1
                continue
            name = match.group(1)
            if name in {"if", "for", "while", "switch", "return"}:
                idx += 1
                continue
            window = "\n".join(lines[idx:min(len(lines), idx + 8)])
            if "{" not in window or ";" in window.split("{", 1)[0]:
                idx += 1
                continue
            bounds = self._find_function_bounds(lines, name, around_line=idx + 1)
            if not bounds:
                idx += 1
                continue
            f_start, f_end = bounds
            if f_end < start or f_start > end:
                idx += 1
                continue
            marker = (name, f_start, f_end)
            if marker in seen_functions:
                idx = max(idx + 1, f_end)
                continue
            seen_functions.add(marker)
            functions.append({
                "name": name,
                "start_line": f_start,
                "end_line": f_end,
                "contains_target": bool(around_line and f_start <= around_line <= f_end),
            })
            if len(functions) >= max_functions:
                break
            idx = max(idx + 1, f_end)

        return {
            "file": file,
            "target_line": around_line,
            "window_start": start,
            "window_end": end,
            "total_lines": total_lines,
            "macros": macros,
            "functions": functions,
        }

    # -- symbol search (narrower than grep) --------------------------------

    def search_symbol(self, symbol: str, paths: Optional[List[str]] = None,
                      limit: int = 20000) -> str:
        """Search for a symbol name as a word (not as substring)."""
        pattern = r"\b" + re.escape(symbol) + r"\b"
        return self._run_grep(pattern, paths, limit=limit)

    # -- grep (raw regex) --------------------------------------------------

    def grep(self, pattern: str, paths: Optional[List[str]] = None,
             limit: int = 20000) -> str:
        """Raw regex search across source files."""
        return self._run_grep(pattern, paths, limit=limit)

    # -- batch operations --------------------------------------------------

    def batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple operations in one call.

        Each request: {"op": "...", ...params...}
        Supported ops: line, function, callers, callees, struct, field,
                       macro, containing, grep, search_symbol, file_outline
        """
        results = []
        for req in requests:
            op = req.get("op", "")
            try:
                if op == "line":
                    r = self.get_lines(req["file"], req["line"],
                                       req.get("before", 60), req.get("after", 80))
                elif op == "function":
                    r = self.get_function(req["symbol"],
                                          file=req.get("file"),
                                          around_line=req.get("around_line"),
                                          before=req.get("before", 5),
                                          after=req.get("after", 5))
                elif op == "callers":
                    r = self.find_callers(req["symbol"], req.get("paths"),
                                          limit=req.get("limit", 20000))
                elif op == "callees":
                    r = self.find_callees(req["symbol"], req.get("file"))
                elif op == "struct":
                    r = self.get_struct_def(req["name"], req.get("paths"))
                elif op == "field":
                    r = self.find_field_usage(req["struct"], req["field"],
                                              req.get("paths"),
                                              limit=req.get("limit", 20000))
                elif op == "macro":
                    r = self.get_macro_def(req["name"], limit=req.get("limit", 20000))
                elif op == "containing":
                    r = self.find_containing_function(req["file"], req["line"])
                elif op == "file_outline":
                    r = self.get_file_outline(
                        req["file"],
                        around_line=req.get("around_line"),
                        radius=req.get("radius", 220),
                        max_macros=req.get("max_macros", 60),
                        max_functions=req.get("max_functions", 40),
                    )
                elif op == "grep":
                    r = self.grep(req["pattern"], req.get("paths"),
                                  limit=req.get("limit", 20000))
                elif op == "search_symbol":
                    r = self.search_symbol(req["symbol"], req.get("paths"),
                                           limit=req.get("limit", 20000))
                else:
                    r = {"error": f"unknown op: {op}"}
            except Exception as e:
                r = {"error": str(e), "op": op}
            if isinstance(r, list):
                r = {"op": op, "result": r}
            elif isinstance(r, str):
                r = {"op": op, "result": r}
            else:
                r["op"] = op
            results.append(r)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Kernel code searcher for LLM root-cause analysis")
    ap.add_argument("--repo", required=True, help="Path to cached Linux source tree")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # line
    p = sub.add_parser("line")
    p.add_argument("--file", required=True)
    p.add_argument("--line", type=int, required=True)
    p.add_argument("--before", type=int, default=60)
    p.add_argument("--after", type=int, default=80)
    p.set_defaults(op="line")

    # function
    p = sub.add_parser("function")
    p.add_argument("--symbol", required=True)
    p.add_argument("--file")
    p.add_argument("--around-line", type=int)
    p.add_argument("--before", type=int, default=5)
    p.add_argument("--after", type=int, default=5)
    p.add_argument("--max-files", type=int, default=5)
    p.set_defaults(op="function")

    # callers
    p = sub.add_parser("callers")
    p.add_argument("--symbol", required=True)
    p.add_argument("--paths", nargs="*")
    p.set_defaults(op="callers")

    # callees (NEW)
    p = sub.add_parser("callees")
    p.add_argument("--symbol", required=True)
    p.add_argument("--file")
    p.set_defaults(op="callees")

    # grep
    p = sub.add_parser("grep")
    p.add_argument("--pattern", required=True)
    p.add_argument("--paths", nargs="*")
    p.set_defaults(op="grep")

    # struct
    p = sub.add_parser("struct")
    p.add_argument("--name", required=True)
    p.add_argument("--paths", nargs="*")
    p.set_defaults(op="struct")

    # field (NEW)
    p = sub.add_parser("field")
    p.add_argument("--struct", required=True)
    p.add_argument("--field", required=True)
    p.add_argument("--paths", nargs="*")
    p.set_defaults(op="field")

    # macro (NEW)
    p = sub.add_parser("macro")
    p.add_argument("--name", required=True)
    p.set_defaults(op="macro")

    # containing (NEW)
    p = sub.add_parser("containing")
    p.add_argument("--file", required=True)
    p.add_argument("--line", type=int, required=True)
    p.set_defaults(op="containing")

    # search_symbol (NEW)
    p = sub.add_parser("search_symbol")
    p.add_argument("--symbol", required=True)
    p.add_argument("--paths", nargs="*")
    p.set_defaults(op="search_symbol")

    # batch (NEW)
    p = sub.add_parser("batch")
    p.add_argument("--requests", required=True, help="JSON array of request objects")
    p.set_defaults(op="batch")

    return ap


def _cli_main():
    ap = _parse_args()
    args = ap.parse_args()
    ks = KernelCodeSearcher(args.repo)

    op = args.op
    if op == "line":
        r = ks.get_lines(args.file, args.line, args.before, args.after)
        print(r["source"])
    elif op == "function":
        r = ks.get_function(args.symbol, file=args.file,
                            around_line=args.around_line,
                            before=args.before, after=args.after,
                            max_files=args.max_files)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif op == "callers":
        print(ks.find_callers(args.symbol, args.paths))
    elif op == "callees":
        r = ks.find_callees(args.symbol, file=args.file)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif op == "grep":
        print(ks.grep(args.pattern, args.paths))
    elif op == "struct":
        r = ks.get_struct_def(args.name, args.paths)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif op == "field":
        print(ks.find_field_usage(args.struct, args.field, args.paths))
    elif op == "macro":
        r = ks.get_macro_def(args.name)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif op == "containing":
        r = ks.find_containing_function(args.file, args.line)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif op == "search_symbol":
        print(ks.search_symbol(args.symbol, args.paths))
    elif op == "batch":
        requests = json.loads(args.requests)
        r = ks.batch(requests)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli_main()
