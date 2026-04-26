"""A/B comparison CLI for Eureka runs.

Usage:
    python -m utils.compare_runs --run_a <timestamp_dir> --run_b <timestamp_dir>

Prints a side-by-side table of max successes, token costs, and wall-clock for
two Eureka run directories (single-mode vs multi-mode, or any two runs).
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np


def _read_summary_npz(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "summary.npz")
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _read_eureka_log(run_dir: str) -> str:
    path = os.path.join(run_dir, "eureka.log")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_final_success(log_text: str) -> Dict[str, Optional[float]]:
    m = re.search(
        r"Final Success Mean:\s*([\-0-9.eE]+),\s*Std:\s*([\-0-9.eE]+)",
        log_text,
    )
    if m is None:
        return {"final_success_mean": None, "final_success_std": None}
    return {
        "final_success_mean": float(m.group(1)),
        "final_success_std": float(m.group(2)),
    }


def _parse_token_totals_from_log(log_text: str) -> Dict[str, int]:
    total_prompt = 0
    total_completion = 0
    total = 0
    for line in log_text.splitlines():
        m = re.search(
            r"Prompt Tokens:\s*(\d+),\s*Completion Tokens:\s*(\d+),\s*Total Tokens:\s*(\d+)",
            line,
        )
        if m:
            total_prompt += int(m.group(1))
            total_completion += int(m.group(2))
            total += int(m.group(3))
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total,
    }


def _read_multi_agent_summary(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "multi_agent_summary.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _sum_multi_agent_tokens(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    total_tokens = 0
    total_wall = 0.0
    for e in entries:
        total_tokens += int(e.get("total_tokens", 0) or 0)
        total_wall += float(e.get("total_wall_clock_seconds", 0.0) or 0.0)
    return {"multi_agent_total_tokens": total_tokens, "multi_agent_total_wall": total_wall}


def _detect_mode(run_dir: str, multi_summary: List[Dict[str, Any]]) -> str:
    if multi_summary:
        return "multi"
    if os.path.isdir(os.path.join(run_dir, "multi_agent_trace")):
        return "multi"
    return "single"


def _estimate_wall_clock(run_dir: str) -> float:
    """Approximate wall clock from file mtimes if no explicit timer available."""
    log_path = os.path.join(run_dir, "eureka.log")
    if not os.path.exists(log_path):
        return 0.0
    newest = os.path.getmtime(log_path)
    oldest = newest
    for entry in os.scandir(run_dir):
        try:
            mtime = entry.stat().st_mtime
            if mtime < oldest:
                oldest = mtime
        except OSError:
            continue
    return max(newest - oldest, 0.0)


def summarize_run(run_dir: str) -> Dict[str, Any]:
    npz = _read_summary_npz(run_dir)
    log = _read_eureka_log(run_dir)
    final = _parse_final_success(log)
    token_totals = _parse_token_totals_from_log(log)
    multi_summary = _read_multi_agent_summary(run_dir)
    multi_totals = _sum_multi_agent_tokens(multi_summary)
    mode = _detect_mode(run_dir, multi_summary)

    max_successes = npz.get("max_successes")
    if max_successes is not None:
        max_successes_list = [float(x) for x in np.array(max_successes).tolist()]
    else:
        max_successes_list = []

    best_success = max(max_successes_list) if max_successes_list else 0.0

    total_tokens = token_totals["total_tokens"]
    if mode == "multi" and multi_totals["multi_agent_total_tokens"] > 0:
        total_tokens = multi_totals["multi_agent_total_tokens"]

    wall_clock = _estimate_wall_clock(run_dir)
    if mode == "multi" and multi_totals["multi_agent_total_wall"] > 0:
        wall_clock = multi_totals["multi_agent_total_wall"]

    tokens_per_success = (total_tokens / best_success) if best_success > 0 else float("inf")

    return {
        "run_dir": run_dir,
        "mode": mode,
        "iterations": len(max_successes_list),
        "max_successes_per_iter": max_successes_list,
        "best_success": best_success,
        "final_success_mean": final["final_success_mean"],
        "final_success_std": final["final_success_std"],
        "total_tokens": total_tokens,
        "wall_clock_seconds": wall_clock,
        "tokens_per_success": tokens_per_success,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v:.2f}"
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.1f}" for x in v) + "]"
    return str(v)


def print_comparison(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    rows = [
        ("run_dir", a["run_dir"], b["run_dir"]),
        ("mode", a["mode"], b["mode"]),
        ("iterations", a["iterations"], b["iterations"]),
        ("max_successes per iter", a["max_successes_per_iter"], b["max_successes_per_iter"]),
        ("best_success", a["best_success"], b["best_success"]),
        ("final_success_mean", a["final_success_mean"], b["final_success_mean"]),
        ("final_success_std", a["final_success_std"], b["final_success_std"]),
        ("total_tokens", a["total_tokens"], b["total_tokens"]),
        ("wall_clock_seconds", a["wall_clock_seconds"], b["wall_clock_seconds"]),
        ("tokens_per_success", a["tokens_per_success"], b["tokens_per_success"]),
    ]
    label_w = max(len(r[0]) for r in rows)
    a_w = max(len(_fmt(r[1])) for r in rows)
    b_w = max(len(_fmt(r[2])) for r in rows)
    a_w = max(a_w, len("run_a"))
    b_w = max(b_w, len("run_b"))

    header = f"{'':<{label_w}}  {'run_a':>{a_w}}  {'run_b':>{b_w}}"
    print(header)
    print("-" * len(header))
    for label, va, vb in rows:
        print(f"{label:<{label_w}}  {_fmt(va):>{a_w}}  {_fmt(vb):>{b_w}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Eureka run directories.")
    parser.add_argument("--run_a", required=True, help="First run directory (timestamp dir)")
    parser.add_argument("--run_b", required=True, help="Second run directory (timestamp dir)")
    args = parser.parse_args()

    a = summarize_run(args.run_a)
    b = summarize_run(args.run_b)
    print_comparison(a, b)


if __name__ == "__main__":
    main()
