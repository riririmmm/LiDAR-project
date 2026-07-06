# rb_batch_from_scorevote.py
# 각 구간의 최종 상태 읽어 RR/PF/Ours 등 RB 스케줄러 시뮬레이션을 구간별로 돌리고 결과 CSV와 그래프 생성

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from complex_rb_simulator import simulate_once


@dataclass
class SimConfig:
    total_rb: int = 25
    n_vehicles: int = 20
    n_slots: int = 300
    seed: int = 42
    rb_per_slot: int | None = None
    input_state: str | None = None


STATE_LOAD_PRESET = {
    # tuple = (total_rb, n_vehicles)
    "Empty": (22, 28),
    "Normal": (20, 40),
    "Congestion": (14, 60),
}

SCHEDULERS = ["RR", "PF", "Ours", "OursPF", "MaxThroughput"]

DETAIL_FIELDS = [
    "sequence",
    "frame_range",
    "event_file",
    "input_state",
    "scheduler",
    "total_throughput_bps",
    "avg_delay_sec",
    "fairness",
    "timely_throughput_bps",
    "deadline_miss_rate",
    "p95_delay_sec",
    "p99_delay_sec",
    "max_unserved_streak",
    "congestion_avg_throughput_bps",
    "normal_avg_throughput_bps",
    "empty_avg_throughput_bps",
    "active_users_per_slot",
    "requested_rb_per_slot",
    "allocated_rb_per_slot",
    "unserved_users_per_slot",
]

SUMMARY_FIELDS = [
    "input_state",
    "scheduler",
    "count",
    "mean_total_throughput_bps",
    "mean_avg_delay_sec",
    "mean_fairness",
    "mean_timely_throughput_bps",
    "mean_deadline_miss_rate",
    "mean_p95_delay_sec",
    "mean_p99_delay_sec",
    "mean_max_unserved_streak",
    "mean_congestion_avg_throughput_bps",
    "mean_normal_avg_throughput_bps",
    "mean_empty_avg_throughput_bps",
    "mean_active_users_per_slot",
    "mean_requested_rb_per_slot",
    "mean_allocated_rb_per_slot",
    "mean_unserved_users_per_slot",
]


def find_scorevote_files(root: Path) -> List[Path]:
    return sorted(root.rglob("final_event_scorevote.txt"))


def extract_seq_and_range(scorevote_path: Path, root: Path) -> Tuple[str, str]:
    rel = scorevote_path.relative_to(root)
    parts = rel.parts
    seq = parts[0] if len(parts) >= 1 else ""
    frame_range = parts[1] if len(parts) >= 2 else ""
    return seq, frame_range


def read_final_event_type(scorevote_path: Path) -> str:
    with scorevote_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("final_event_type:"):
                return line.split(":", 1)[1].strip()
    raise ValueError(f"final_event_type not found in {scorevote_path}")


def normalize_state(state: str) -> str:
    s = state.strip().lower()
    if s in ("congestion", "jam", "trafficjam"):
        return "Congestion"
    if s == "normal":
        return "Normal"
    if s in ("empty", "light", "sparse"):
        return "Empty"
    return "Normal"


def parse_args():
    default_cfg = SimConfig()
    default_root = Path(__file__).resolve().parent / "out_bev_ranges"

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(default_root))
    p.add_argument("--total-rb", type=int, default=default_cfg.total_rb)
    p.add_argument("--n-vehicles", type=int, default=default_cfg.n_vehicles)
    p.add_argument("--n-slots", type=int, default=default_cfg.n_slots)
    p.add_argument("--n-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=default_cfg.seed)
    p.add_argument("--state-aware-load", dest="state_aware_load", action="store_true", default=True)
    p.add_argument("--no-state-aware-load", dest="state_aware_load", action="store_false")
    p.add_argument("--max-files-per-state", type=int, default=0)
    return p.parse_args()


def safe_open_for_write(target: Path):
    try:
        return target.open("w", newline="", encoding="utf-8-sig"), target
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = target.with_name(f"{target.stem}_{ts}{target.suffix}")
        print(f"[WARN] file locked: {target}")
        print(f"[WARN] writing to fallback: {fallback}")
        return fallback.open("w", newline="", encoding="utf-8-sig"), fallback


def save_detail_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f, actual_path = safe_open_for_write(out_csv)
    with f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {actual_path}")


def save_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    grouped: Dict[Tuple[str, str], Dict[str, float]] = {}
    numeric_fields = [field for field in DETAIL_FIELDS if field not in {"sequence", "frame_range", "event_file", "input_state", "scheduler"}]

    for row in rows:
        key = (row["input_state"], row["scheduler"])
        g = grouped.setdefault(key, {"count": 0.0, **{field: 0.0 for field in numeric_fields}})
        g["count"] += 1.0
        for field in numeric_fields:
            g[field] += float(row.get(field, 0.0) or 0.0)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f, actual_path = safe_open_for_write(out_csv)
    with f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for (state, scheduler), g in sorted(grouped.items()):
            count = max(g["count"], 1.0)
            writer.writerow(
                {
                    "input_state": state,
                    "scheduler": scheduler,
                    "count": int(g["count"]),
                    "mean_total_throughput_bps": g["total_throughput_bps"] / count,
                    "mean_avg_delay_sec": g["avg_delay_sec"] / count,
                    "mean_fairness": g["fairness"] / count,
                    "mean_timely_throughput_bps": g["timely_throughput_bps"] / count,
                    "mean_deadline_miss_rate": g["deadline_miss_rate"] / count,
                    "mean_p95_delay_sec": g["p95_delay_sec"] / count,
                    "mean_p99_delay_sec": g["p99_delay_sec"] / count,
                    "mean_max_unserved_streak": g["max_unserved_streak"] / count,
                    "mean_congestion_avg_throughput_bps": g["congestion_avg_throughput_bps"] / count,
                    "mean_normal_avg_throughput_bps": g["normal_avg_throughput_bps"] / count,
                    "mean_empty_avg_throughput_bps": g["empty_avg_throughput_bps"] / count,
                    "mean_active_users_per_slot": g["active_users_per_slot"] / count,
                    "mean_requested_rb_per_slot": g["requested_rb_per_slot"] / count,
                    "mean_allocated_rb_per_slot": g["allocated_rb_per_slot"] / count,
                    "mean_unserved_users_per_slot": g["unserved_users_per_slot"] / count,
                }
            )
    print(f"[SAVE] {actual_path}")


def metric_to_row(metrics, *, seq: str, frame_range: str, event_file: Path, input_state: str) -> Dict:
    metric_dict = asdict(metrics)
    return {
        "sequence": seq,
        "frame_range": frame_range,
        "event_file": str(event_file),
        "input_state": input_state,
        **metric_dict,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    scorevote_files = find_scorevote_files(root)
    print(f"[INFO] root: {root}")
    print(f"[INFO] scorevote files: {len(scorevote_files)}")

    rows: List[Dict] = []
    processed_by_state: Dict[str, int] = {}
    for file_idx, scorevote_path in enumerate(scorevote_files):
        try:
            input_state = normalize_state(read_final_event_type(scorevote_path))
            if args.max_files_per_state > 0:
                current_count = processed_by_state.get(input_state, 0)
                if current_count >= args.max_files_per_state:
                    continue
                processed_by_state[input_state] = current_count + 1
            seq, frame_range = extract_seq_and_range(scorevote_path, root)

            total_rb = args.total_rb
            n_vehicles = args.n_vehicles
            if args.state_aware_load:
                total_rb, n_vehicles = STATE_LOAD_PRESET.get(input_state, (total_rb, n_vehicles))

            for scheduler in SCHEDULERS:
                for run_idx in range(max(1, int(args.n_runs))):
                    cfg = SimConfig(
                        total_rb=total_rb,
                        n_vehicles=n_vehicles,
                        n_slots=args.n_slots,
                        seed=args.seed + file_idx * 1000,
                        rb_per_slot=total_rb,
                        input_state=input_state,
                    )
                    result = simulate_once(
                        scenario=scorevote_path,
                        scheduler=scheduler,
                        cfg=cfg,
                        run_idx=run_idx,
                    )
                    metrics = result[0] if isinstance(result, tuple) else result
                    rows.append(
                        metric_to_row(
                            metrics,
                            seq=seq,
                            frame_range=frame_range,
                            event_file=scorevote_path,
                            input_state=input_state,
                        )
                    )
        except Exception as exc:
            print(f"[FAIL] {scorevote_path}: {exc}")

    detail_csv = root / "rb_simulation_results.csv"
    summary_csv = root / "rb_simulation_summary.csv"
    save_detail_csv(rows, detail_csv)
    save_summary_csv(rows, summary_csv)
    print("[DONE]")


if __name__ == "__main__":
    main()
