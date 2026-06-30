# Draupnir Syzbot Dataset

This directory contains the syzbot benchmark used by Draupnir's main experiment.

## Layout

- `crashes_by_bug/`: raw syzbot crash artifacts grouped by syzbot bug id. Each bug directory contains `bug_meta.json`, `crashes.jsonl`, and one directory per benchmark crash under `crashes/`.
- `ground_truth.csv`: mapping from each benchmark crash to its final root-cause label.

## Summary

- syzbot bug reports: 247
- benchmark crash rows/directories: 6,666
- final root-cause labels: 102
- 

## Crash Directory Names

Crash directories use the format `r<row_index>__<YYYY-MM-DD-HHMM>__<crash_uid>`. The `row_index` prefix is the row number from the crawled syzbot crash table for that bug. It makes the released directory names stable and unique even when multiple crashes from the same syzbot bug share the same minute-level timestamp and normalized crash title.

The experiment scripts use `ground_truth.csv` to map each `bug_id/case_id` crash key to its final root-cause label.
