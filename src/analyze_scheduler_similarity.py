#!/usr/bin/env python3
"""Scheduler similarity diagnostic report generator.

Usage:
  python src/analyze_scheduler_similarity.py --csv src/out_bev_ranges/rb_simulation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def to_float(v: str) -> float:
    return float(v.strip())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_if_needed(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Accept both summary and raw results CSV formats.

    - summary format: already has mean_* columns
    - results format: has per-run columns (e.g., total_throughput_bps)
      and needs aggregation by input_state+scheduler
    """
    if not rows:
        return rows

    fields = set(rows[0].keys())
    is_summary = "mean_total_throughput_bps" in fields and "mean_avg_delay_sec" in fields
    if is_summary:
        return rows

    is_results = "total_throughput_bps" in fields and "avg_delay_sec" in fields and "scheduler" in fields
    if not is_results:
        raise ValueError("Unsupported CSV schema. Provide rb_simulation_summary.csv or rb_simulation_results.csv")

    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        buckets[(r["input_state"], r["scheduler"])].append(r)

    agg_rows: list[dict[str, str]] = []
    for (state, scheduler), items in buckets.items():
        n = len(items)

        def m(col: str) -> float:
            return mean(float(x[col]) for x in items)

        agg_rows.append(
            {
                "input_state": state,
                "scheduler": scheduler,
                "count": str(n),
                "mean_total_throughput_bps": str(m("total_throughput_bps")),
                "mean_avg_delay_sec": str(m("avg_delay_sec")),
                "mean_fairness": str(m("fairness")),
                "mean_congestion_avg_throughput_bps": str(m("congestion_avg_throughput_bps")),
                "mean_normal_avg_throughput_bps": str(m("normal_avg_throughput_bps")),
                "mean_empty_avg_throughput_bps": str(m("empty_avg_throughput_bps")),
                "mean_active_users_per_slot": str(m("mean_active_users_per_slot")),
                "mean_requested_rb_per_slot": str(m("mean_requested_rb_per_slot")),
                "mean_allocated_rb_per_slot": str(m("mean_allocated_rb_per_slot")),
                "mean_unserved_users_per_slot": str(m("mean_unserved_users_per_slot")),
            }
        )

    return agg_rows


def group_by_state(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["input_state"]].append(r)
    return dict(grouped)




def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    w = pos - lo
    return v[lo] * (1 - w) + v[hi] * w


def report_state(rows: list[dict[str, str]], state: str) -> list[str]:
    lines: list[str] = []
    lines.append(f"\n[State] {state}")
    lines.append("scheduler      thr(Mbps)  delay(s)  fairness  req/alloc  unserved/active")

    normalized: list[dict[str, float | str]] = []
    for r in rows:
        req = to_float(r["mean_requested_rb_per_slot"])
        alloc = to_float(r["mean_allocated_rb_per_slot"])
        active = to_float(r["mean_active_users_per_slot"])
        unserved = to_float(r["mean_unserved_users_per_slot"])
        thr = to_float(r["mean_total_throughput_bps"]) / 1e6
        delay = to_float(r["mean_avg_delay_sec"])
        fair = to_float(r["mean_fairness"])
        req_alloc = req / alloc if alloc > 0 else float("inf")
        uns_act = unserved / active if active > 0 else 0.0
        normalized.append(
            {
                "scheduler": r["scheduler"],
                "thr": thr,
                "delay": delay,
                "fair": fair,
                "req_alloc": req_alloc,
                "uns_act": uns_act,
            }
        )

    normalized.sort(key=lambda x: float(x["thr"]), reverse=True)

    for r in normalized:
        lines.append(
            f"{r['scheduler']:<13}{float(r['thr']):>9.2f}{float(r['delay']):>10.3f}{float(r['fair']):>10.4f}"
            f"{float(r['req_alloc']):>11.1f}{float(r['uns_act']):>16.3f}"
        )

    req_alloc_all = [float(x["req_alloc"]) for x in normalized]
    uns_act_all = [float(x["uns_act"]) for x in normalized]
    lines.append(
        "- saturation check: "
        + (
            "HIGH (requested>>allocated and unserved share high)"
            if mean(req_alloc_all) > 10 and mean(uns_act_all) > 0.3
            else "LOW/MEDIUM"
        )
    )

    best_thr = float(normalized[0]["thr"])
    worst_thr = float(normalized[-1]["thr"])
    thr_gap_pct = ((best_thr - worst_thr) / max(worst_thr, 1e-9)) * 100
    best_delay = min(float(x["delay"]) for x in normalized)
    worst_delay = max(float(x["delay"]) for x in normalized)
    delay_gap_pct = ((worst_delay - best_delay) / max(worst_delay, 1e-9)) * 100

    lines.append(f"- spread(all): throughput gap={thr_gap_pct:.2f}% , delay gap={delay_gap_pct:.2f}%")

    # robust spread: exclude bottom-throughput scheduler (often MaxThroughput outlier)
    if len(normalized) >= 4:
        core = normalized[:-1]
        core_best_thr = float(core[0]["thr"])
        core_worst_thr = float(core[-1]["thr"])
        core_thr_gap_pct = ((core_best_thr - core_worst_thr) / max(core_worst_thr, 1e-9)) * 100
        core_best_delay = min(float(x["delay"]) for x in core)
        core_worst_delay = max(float(x["delay"]) for x in core)
        core_delay_gap_pct = ((core_worst_delay - core_best_delay) / max(core_worst_delay, 1e-9)) * 100
        lines.append(f"- spread(core): throughput gap={core_thr_gap_pct:.2f}% , delay gap={core_delay_gap_pct:.2f}%")

        similar = core_thr_gap_pct <= 3.0 and core_delay_gap_pct <= 10.0
        lines.append("- similarity(core): " + ("HIGH (정말 비슷)" if similar else "MEDIUM/LOW (차이 존재)"))

    uns_vals = [float(x["uns_act"]) for x in normalized]
    uns_p50 = percentile(uns_vals, 0.5)
    lines.append(f"- unserved/active median across schedulers: {uns_p50:.3f}")
    lines.append("- interpretation: if saturation is HIGH, validate with tail metrics (P95/P99, unserved streak).")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scheduler similarity from summary/results CSV")
    parser.add_argument("--csv", type=Path, default=None, help="Path to rb_simulation_summary.csv or rb_simulation_results.csv")
    args = parser.parse_args()

    if args.csv is None:
        candidates = [
            Path("src/out_bev_ranges/rb_simulation_summary.csv"),
            Path("src/out_bev_ranges/rb_simulation_results.csv"),
            Path(__file__).resolve().parent / "out_bev_ranges" / "rb_simulation_summary.csv",
            Path(__file__).resolve().parent / "out_bev_ranges" / "rb_simulation_results.csv",
        ]
        csv_path = next((c for c in candidates if c.exists()), None)
        if csv_path is None:
            raise SystemExit("No CSV path provided and no default file found. Use --csv <path>.")
    else:
        csv_path = args.csv

    raw_rows = load_rows(csv_path)
    rows = aggregate_if_needed(raw_rows)
    by_state = group_by_state(rows)

    print("=== Scheduler Similarity Diagnostic Report ===")
    print(f"csv: {csv_path}")
    for state, state_rows in sorted(by_state.items()):
        for line in report_state(state_rows, state):
            print(line)


if __name__ == "__main__":
    main()
