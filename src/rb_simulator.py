from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


# =========================================================
# 1) 기본 설정
# =========================================================


@dataclass
class SimConfig:
    # 무선 자원 설정
    total_rb: int = 25
    bandwidth_per_rb_hz: float = 180_000.0
    slot_sec: float = 0.1
    n_slots: int = 300

    # 간단 채널 모델용 전력/잡음
    tx_power: float = 1.0
    noise_power: float = 1e-9

    # 트래픽 모델
    packet_size_bits: int = 12_000

    # 차량 수
    n_vehicles: int = 20

    # 반복 실험 수
    n_runs: int = 30

    # 난수
    seed: int = 42

    # PF 안정화 상수
    pf_epsilon: float = 1e-6

    # RB marginal efficiency decay
    # same user gets more RBs in a slot -> marginal gain decreases
    rb_efficiency_decay: float = 0.35

    # Limit for requested RB search
    rb_demand_cap_factor: float = 2.0

    # Ours coefficients
    ours_rate_coeff_cong: float = 0.50
    ours_queue_coeff_cong: float = 0.60
    ours_starve_coeff_cong: float = 0.60
    ours_wait_coeff_cong: float = 0.40

    ours_rate_coeff_normal: float = 0.60
    ours_queue_coeff_normal: float = 0.40
    ours_starve_coeff_normal: float = 0.35
    ours_wait_coeff_normal: float = 0.25

    ours_rate_coeff_empty: float = 0.75
    ours_queue_coeff_empty: float = 0.15
    ours_starve_coeff_empty: float = 0.10
    ours_wait_coeff_empty: float = 0.05

    # OursPF coefficients
    ourspf_queue_coeff_cong: float = 0.65
    ourspf_starve_coeff_cong: float = 0.95
    ourspf_wait_coeff_cong: float = 0.75

    ourspf_queue_coeff_normal: float = 0.30
    ourspf_starve_coeff_normal: float = 0.35
    ourspf_wait_coeff_normal: float = 0.20

    ourspf_queue_coeff_empty: float = 0.10
    ourspf_starve_coeff_empty: float = 0.10
    ourspf_wait_coeff_empty: float = 0.05

    # 이번 슬롯 내 과몰빵 방지
    intra_slot_decay_ours: float = 0.78
    intra_slot_decay_ourspf: float = 0.20

    # 디버그 옵션
    debug_slots: int = 10
    debug_every: int = 0
    save_debug_csv: bool = True


@dataclass
class Vehicle:
    vid: int
    state: str  # 환경 생성용으로만 유지
    long_gain: float

    queue_bits: float = 0.0
    served_bits_total: float = 0.0
    delay_sum_bits_sec: float = 0.0

    selected_count: int = 0
    allocated_rb_total: int = 0
    unserved_slot_count: int = 0


@dataclass
class Metrics:
    scheduler: str
    scenario: str
    run_idx: int

    total_throughput_bps: float
    avg_delay_sec: float
    fairness: float

    congestion_avg_throughput_bps: float
    normal_avg_throughput_bps: float
    empty_avg_throughput_bps: float

    mean_active_users_per_slot: float
    mean_requested_rb_per_slot: float
    mean_allocated_rb_per_slot: float
    mean_unserved_users_per_slot: float


# =========================================================
# 2) 상태/시나리오 유틸
# =========================================================


def normalize_state_name(state: str) -> str:
    s = state.strip().lower()
    if s in ("congestion", "jam", "trafficjam"):
        return "Congestion"
    if s in ("normal",):
        return "Normal"
    if s in ("empty", "light", "sparse"):
        return "Empty"
    raise ValueError(f"Unknown state: {state}")


def normalize_scenario_name(scenario: str) -> str:
    """
    standalone 시뮬레이터 preset scenario를
    Ours/OursPF 계수 선택용 3상태로 매핑
    """
    s = scenario.strip().lower()

    if s in ("congestion", "jam", "trafficjam"):
        return "Congestion"
    if s in ("normal",):
        return "Normal"
    if s in ("empty", "light", "sparse"):
        return "Empty"

    if s in ("congestion_heavy",):
        return "Congestion"
    if s in ("normal_heavy", "balanced"):
        return "Normal"
    if s in ("empty_heavy",):
        return "Empty"

    raise ValueError(f"Unknown scenario: {scenario}")


STATE_ENV: Dict[str, Dict[str, float]] = {
    # Higher traffic demand + worse average channel in congestion
    "Congestion": {
        "arrival_pkts": 10.0,
        "base_gain": 0.18,
        "shadow_sigma_db": 5.0,
    },
    "Normal": {
        "arrival_pkts": 7.0,
        "base_gain": 0.45,
        "shadow_sigma_db": 4.0,
    },
    "Empty": {
        "arrival_pkts": 3.0,
        "base_gain": 0.90,
        "shadow_sigma_db": 3.0,
    },
}


def get_scenario_coeffs_ours(scenario: str, cfg: SimConfig) -> Tuple[float, float, float, float]:
    scenario = normalize_scenario_name(scenario)

    if scenario == "Congestion":
        return (
            cfg.ours_rate_coeff_cong,
            cfg.ours_queue_coeff_cong,
            cfg.ours_starve_coeff_cong,
            cfg.ours_wait_coeff_cong,
        )
    if scenario == "Normal":
        return (
            cfg.ours_rate_coeff_normal,
            cfg.ours_queue_coeff_normal,
            cfg.ours_starve_coeff_normal,
            cfg.ours_wait_coeff_normal,
        )
    return (
        cfg.ours_rate_coeff_empty,
        cfg.ours_queue_coeff_empty,
        cfg.ours_starve_coeff_empty,
        cfg.ours_wait_coeff_empty,
    )


def get_scenario_coeffs_ourspf(scenario: str, cfg: SimConfig) -> Tuple[float, float, float]:
    scenario = normalize_scenario_name(scenario)

    if scenario == "Congestion":
        return (
            cfg.ourspf_queue_coeff_cong,
            cfg.ourspf_starve_coeff_cong,
            cfg.ourspf_wait_coeff_cong,
        )
    if scenario == "Normal":
        return (
            cfg.ourspf_queue_coeff_normal,
            cfg.ourspf_starve_coeff_normal,
            cfg.ourspf_wait_coeff_normal,
        )
    return (
        cfg.ourspf_queue_coeff_empty,
        cfg.ourspf_starve_coeff_empty,
        cfg.ourspf_wait_coeff_empty,
    )


# =========================================================
# 3) 시나리오 생성
# =========================================================


def build_vehicle_states(scenario: str, n_vehicles: int, rng: np.random.Generator) -> List[str]:
    raw = scenario.strip().lower()

    # 직접 상태가 들어오면 "혼합 생성"하지 말고 전 차량 동일 상태로 고정
    if raw in ("congestion", "jam", "trafficjam"):
        return ["Congestion"] * n_vehicles
    elif raw in ("normal",):
        return ["Normal"] * n_vehicles
    elif raw in ("empty", "light", "sparse"):
        return ["Empty"] * n_vehicles

    # 아래는 standalone preset scenario에서만 혼합 비율 사용
    if raw == "balanced":
        weights = [1 / 3, 1 / 3, 1 / 3]
    elif raw == "congestion_heavy":
        weights = [0.6, 0.3, 0.1]
    elif raw == "normal_heavy":
        weights = [0.2, 0.6, 0.2]
    elif raw == "empty_heavy":
        weights = [0.1, 0.3, 0.6]
    else:
        raise ValueError(
            "scenario must be one of: balanced, congestion_heavy, normal_heavy, "
            "empty_heavy, or direct states: Empty, Normal, Congestion"
        )

    states = rng.choice(
        ["Congestion", "Normal", "Empty"],
        size=n_vehicles,
        p=weights,
    )
    return [str(x) for x in states]


# =========================================================
# 4) 채널 / 전송률
# =========================================================


def make_vehicles(states: List[str], rng: np.random.Generator) -> List[Vehicle]:
    vehicles: List[Vehicle] = []

    for i, state in enumerate(states):
        env = STATE_ENV[state]

        base_gain = float(env["base_gain"])
        sigma_db = float(env["shadow_sigma_db"])

        # Long-term shadowing
        shadow_db = rng.normal(loc=0.0, scale=sigma_db)
        shadow_lin = 10.0 ** (shadow_db / 10.0)
        long_gain = max(base_gain * shadow_lin, 1e-6)

        vehicles.append(
            Vehicle(
                vid=i,
                state=state,
                long_gain=float(long_gain),
            )
        )

    return vehicles


def sample_channel_gains(vehicles: List[Vehicle], rng: np.random.Generator) -> np.ndarray:
    """
    Simple paper-friendly channel model:
    - per-vehicle long-term gain
    - small fast fading per slot
    """
    gains = np.zeros(len(vehicles), dtype=np.float64)

    for i, v in enumerate(vehicles):
        fast_db = rng.normal(loc=0.0, scale=2.0)
        fast_lin = 10.0 ** (fast_db / 10.0)
        gains[i] = max(v.long_gain * fast_lin, 1e-6)

    return gains


def shannon_rate_bits_per_rb(gains: np.ndarray, cfg: SimConfig) -> np.ndarray:
    snr = (cfg.tx_power * gains) / max(cfg.noise_power, 1e-12)
    spectral_eff = np.log2(1.0 + snr)
    return (cfg.bandwidth_per_rb_hz * spectral_eff * cfg.slot_sec).astype(np.float64)


def marginal_rb_capacity(rate_per_rb: float, rb_index: int, cfg: SimConfig) -> float:
    decay = 1.0 + cfg.rb_efficiency_decay * rb_index
    return float(rate_per_rb / decay)


def total_capacity_for_alloc(rate_per_rb: float, alloc_rb: int, cfg: SimConfig) -> float:
    if alloc_rb <= 0:
        return 0.0
    return float(sum(marginal_rb_capacity(rate_per_rb, k, cfg) for k in range(alloc_rb)))


def estimate_requested_rb(queue_bits: float, rate_per_rb: float, cfg: SimConfig) -> int:
    if queue_bits <= 0:
        return 0

    max_rb = max(1, int(cfg.total_rb * cfg.rb_demand_cap_factor))
    acc = 0.0

    for rb in range(1, max_rb + 1):
        acc += marginal_rb_capacity(rate_per_rb, rb - 1, cfg)
        if acc >= queue_bits:
            return rb

    return max_rb


# =========================================================
# 5) 트래픽 도착 / 큐 처리
# =========================================================


def update_arrivals(vehicles: List[Vehicle], rng: np.random.Generator) -> None:
    for v in vehicles:
        lam = STATE_ENV[v.state]["arrival_pkts"]
        arrivals = rng.poisson(lam=lam)
        v.queue_bits += float(arrivals * 12_000)


def serve_queues(
    vehicles: List[Vehicle],
    alloc_rb: np.ndarray,
    rate_per_rb: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    served_bits_arr = np.zeros(len(vehicles), dtype=np.float64)

    for i, v in enumerate(vehicles):
        capacity_bits = total_capacity_for_alloc(float(rate_per_rb[i]), int(alloc_rb[i]), cfg)
        served = min(v.queue_bits, capacity_bits)

        if alloc_rb[i] > 0:
            v.selected_count += 1
            v.allocated_rb_total += int(alloc_rb[i])

        if v.queue_bits > 0 and alloc_rb[i] == 0:
            v.unserved_slot_count += 1
        elif alloc_rb[i] > 0:
            v.unserved_slot_count = max(0, v.unserved_slot_count - 1)

        if served > 0:
            v.queue_bits -= served
            v.served_bits_total += served
            served_bits_arr[i] = served

        if v.queue_bits < 1e-9:
            v.queue_bits = 0.0

        v.delay_sum_bits_sec += v.queue_bits * cfg.slot_sec

    return served_bits_arr


# =========================================================
# 6) 공개 정보 기반 보조 feature
# =========================================================


def normalized_queue_pressure(vehicles: List[Vehicle]) -> np.ndarray:
    q = np.array([v.queue_bits for v in vehicles], dtype=np.float64)
    mx = float(np.max(q)) if np.max(q) > 0 else 1.0
    return q / mx


def normalized_starvation_pressure(vehicles: List[Vehicle]) -> np.ndarray:
    s = np.array([v.unserved_slot_count for v in vehicles], dtype=np.float64)
    mx = float(np.max(s)) if np.max(s) > 0 else 1.0
    return s / mx


def normalized_wait_pressure(vehicles: List[Vehicle], cfg: SimConfig) -> np.ndarray:
    w = np.array([v.queue_bits / max(cfg.packet_size_bits, 1.0) for v in vehicles], dtype=np.float64)
    mx = float(np.max(w)) if np.max(w) > 0 else 1.0
    return w / mx


# =========================================================
# 7) 스케줄러
# =========================================================


def alloc_round_robin(
    total_rb: int,
    rr_cursor: int,
    n_users: int,
    active_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    alloc = np.zeros(n_users, dtype=np.int32)
    if not np.any(active_mask):
        return alloc, rr_cursor

    for k in range(total_rb):
        idx = (rr_cursor + k) % n_users
        search_count = 0

        while not active_mask[idx] and search_count < n_users:
            idx = (idx + 1) % n_users
            search_count += 1

        if search_count >= n_users:
            break

        alloc[idx] += 1

    new_cursor = (rr_cursor + total_rb) % n_users
    return alloc, new_cursor


def alloc_max_throughput(
    total_rb: int,
    rate_per_rb: np.ndarray,
    active_mask: np.ndarray,
    requested_rb: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    alloc = np.zeros_like(requested_rb, dtype=np.int32)
    if not np.any(active_mask):
        return alloc

    remaining_req = requested_rb.astype(np.int32).copy()

    for _ in range(total_rb):
        feasible = active_mask & (remaining_req > 0)
        if not np.any(feasible):
            break

        metric = np.full(len(rate_per_rb), -1.0, dtype=np.float64)

        for i in range(len(rate_per_rb)):
            if feasible[i]:
                metric[i] = marginal_rb_capacity(float(rate_per_rb[i]), int(alloc[i]), cfg)

        best = int(np.argmax(metric))
        if metric[best] < 0:
            break

        alloc[best] += 1
        remaining_req[best] -= 1

    return alloc


def alloc_proportional_fair(
    total_rb: int,
    rate_per_rb: np.ndarray,
    avg_thr: np.ndarray,
    epsilon: float,
    active_mask: np.ndarray,
    requested_rb: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    alloc = np.zeros_like(requested_rb, dtype=np.int32)
    if not np.any(active_mask):
        return alloc

    remaining_req = requested_rb.astype(np.int32).copy()

    for _ in range(total_rb):
        feasible = active_mask & (remaining_req > 0)
        if not np.any(feasible):
            break

        metric = np.full(len(rate_per_rb), -1.0, dtype=np.float64)

        for i in range(len(rate_per_rb)):
            if feasible[i]:
                marginal_rate = marginal_rb_capacity(float(rate_per_rb[i]), int(alloc[i]), cfg)
                metric[i] = marginal_rate / max(avg_thr[i], epsilon)

        best = int(np.argmax(metric))
        if metric[best] < 0:
            break

        alloc[best] += 1
        remaining_req[best] -= 1

    return alloc


def alloc_ours_weighted_by_state(
    total_rb: int,
    vehicles: List[Vehicle],
    rate_per_rb: np.ndarray,
    cfg: SimConfig,
    scenario: str,   # 인터페이스 유지용, 내부에서는 사용 안 함
    requested_rb: np.ndarray,
) -> np.ndarray:
    alloc = np.zeros(len(vehicles), dtype=np.int32)
    active_mask = np.array([v.queue_bits > 0 for v in vehicles], dtype=bool)
    if not np.any(active_mask):
        return alloc

    remaining_req = requested_rb.astype(np.int32).copy()
    q_norm = normalized_queue_pressure(vehicles)
    s_norm = normalized_starvation_pressure(vehicles)
    w_norm = normalized_wait_pressure(vehicles, cfg)

    max_rate = max(float(np.max(rate_per_rb)), 1.0)

    for _ in range(total_rb):
        feasible = active_mask & (remaining_req > 0)
        if not np.any(feasible):
            break

        scores = np.full(len(vehicles), -1.0, dtype=np.float64)

        for i in range(len(vehicles)):
            if not feasible[i]:
                continue

            # 차량별 state 기준 계수 적용
            a_rate, a_queue, a_starve, a_wait = get_scenario_coeffs_ours(vehicles[i].state, cfg)

            marginal_rate = marginal_rb_capacity(float(rate_per_rb[i]), int(alloc[i]), cfg)
            marginal_rate_norm = marginal_rate / max_rate

            score = (
                a_rate * marginal_rate_norm
                + a_queue * q_norm[i]
                + a_starve * s_norm[i]
                + a_wait * w_norm[i]
            )
            score /= (1.0 + cfg.intra_slot_decay_ours * alloc[i])
            scores[i] = score

        best = int(np.argmax(scores))
        if scores[best] < 0:
            break

        alloc[best] += 1
        remaining_req[best] -= 1

    return alloc


def alloc_ours_pf_hybrid(
    total_rb: int,
    vehicles: List[Vehicle],
    rate_per_rb: np.ndarray,
    avg_thr: np.ndarray,
    cfg: SimConfig,
    scenario: str,   # 인터페이스 유지용, 내부에서는 사용 안 함
    requested_rb: np.ndarray,
) -> np.ndarray:
    alloc = np.zeros(len(vehicles), dtype=np.int32)
    active_mask = np.array([v.queue_bits > 0 for v in vehicles], dtype=bool)
    if not np.any(active_mask):
        return alloc

    remaining_req = requested_rb.astype(np.int32).copy()
    q_norm = normalized_queue_pressure(vehicles)
    s_norm = normalized_starvation_pressure(vehicles)
    w_norm = normalized_wait_pressure(vehicles, cfg)

    for _ in range(total_rb):
        feasible = active_mask & (remaining_req > 0)
        if not np.any(feasible):
            break

        scores = np.full(len(vehicles), -1.0, dtype=np.float64)

        for i in range(len(vehicles)):
            if not feasible[i]:
                continue

            # 차량별 state 기준 계수 적용
            b_queue, b_starve, b_wait = get_scenario_coeffs_ourspf(vehicles[i].state, cfg)

            marginal_rate = marginal_rb_capacity(float(rate_per_rb[i]), int(alloc[i]), cfg)
            pf_term = marginal_rate / max(avg_thr[i], cfg.pf_epsilon)
            bonus = 1.0 + b_queue * q_norm[i] + b_starve * s_norm[i] + b_wait * w_norm[i]

            score = pf_term * bonus
            score /= (1.0 + cfg.intra_slot_decay_ourspf * alloc[i])
            scores[i] = score

        best = int(np.argmax(scores))
        if scores[best] < 0:
            break

        alloc[best] += 1
        remaining_req[best] -= 1

    return alloc


# =========================================================
# 8) 평가 지표
# =========================================================


def jain_fairness(x: np.ndarray) -> float:
    denom = len(x) * np.sum(x ** 2)
    if denom <= 0:
        return 0.0
    return float((np.sum(x) ** 2) / denom)


def build_metrics(
    vehicles: List[Vehicle],
    scheduler: str,
    scenario: str,
    run_idx: int,
    cfg: SimConfig,
    mean_active_users_per_slot: float,
    mean_requested_rb_per_slot: float,
    mean_allocated_rb_per_slot: float,
    mean_unserved_users_per_slot: float,
) -> Metrics:
    total_time = cfg.n_slots * cfg.slot_sec
    user_thr = np.array([v.served_bits_total / total_time for v in vehicles], dtype=np.float64)
    fairness = jain_fairness(user_thr)

    total_served_bits = float(sum(v.served_bits_total for v in vehicles))
    total_delay_bits_sec = float(sum(v.delay_sum_bits_sec for v in vehicles))
    avg_delay = total_delay_bits_sec / max(total_served_bits, 1e-9)

    def mean_thr_for_state(target: str) -> float:
        vals = [v.served_bits_total / total_time for v in vehicles if v.state == target]
        return float(np.mean(vals)) if vals else 0.0

    return Metrics(
        scheduler=scheduler,
        scenario=scenario,
        run_idx=run_idx,
        total_throughput_bps=float(np.sum(user_thr)),
        avg_delay_sec=float(avg_delay),
        fairness=fairness,
        congestion_avg_throughput_bps=mean_thr_for_state("Congestion"),
        normal_avg_throughput_bps=mean_thr_for_state("Normal"),
        empty_avg_throughput_bps=mean_thr_for_state("Empty"),
        mean_active_users_per_slot=float(mean_active_users_per_slot),
        mean_requested_rb_per_slot=float(mean_requested_rb_per_slot),
        mean_allocated_rb_per_slot=float(mean_allocated_rb_per_slot),
        mean_unserved_users_per_slot=float(mean_unserved_users_per_slot),
    )


# =========================================================
# 9) 단일 실행 / 반복 실험
# =========================================================


def simulate_once(
    scheduler_name: str,
    scenario: str,
    cfg: SimConfig,
    run_idx: int,
) -> Tuple[Metrics, List[Dict[str, float | int | str]]]:
    rng = np.random.default_rng(cfg.seed + run_idx)

    states = build_vehicle_states(scenario, cfg.n_vehicles, rng)
    vehicles = make_vehicles(states, rng)

    n_users = len(vehicles)
    rr_cursor = 0
    avg_thr = np.full(n_users, 1.0, dtype=np.float64)

    debug_rows: List[Dict[str, float | int | str]] = []

    sum_active_users = 0.0
    sum_requested_rb = 0.0
    sum_allocated_rb = 0.0
    sum_unserved_users = 0.0

    for slot in range(cfg.n_slots):
        update_arrivals(vehicles, rng)

        queue_before = np.array([v.queue_bits for v in vehicles], dtype=np.float64)
        gains = sample_channel_gains(vehicles, rng)
        rate_per_rb = shannon_rate_bits_per_rb(gains, cfg)
        active_mask = np.array([v.queue_bits > 0 for v in vehicles], dtype=bool)

        requested_rb = np.zeros(n_users, dtype=np.int32)
        for i in range(n_users):
            if active_mask[i]:
                requested_rb[i] = estimate_requested_rb(float(queue_before[i]), float(rate_per_rb[i]), cfg)

        if scheduler_name == "RR":
            alloc_rb, rr_cursor = alloc_round_robin(cfg.total_rb, rr_cursor, n_users, active_mask)

        elif scheduler_name == "MaxThroughput":
            alloc_rb = alloc_max_throughput(
                cfg.total_rb,
                rate_per_rb,
                active_mask,
                requested_rb,
                cfg,
            )

        elif scheduler_name == "PF":
            alloc_rb = alloc_proportional_fair(
                cfg.total_rb,
                rate_per_rb,
                avg_thr,
                cfg.pf_epsilon,
                active_mask,
                requested_rb,
                cfg,
            )

        elif scheduler_name == "Ours":
            alloc_rb = alloc_ours_weighted_by_state(
                cfg.total_rb,
                vehicles,
                rate_per_rb,
                cfg,
                scenario,
                requested_rb,
            )

        elif scheduler_name == "OursPF":
            alloc_rb = alloc_ours_pf_hybrid(
                cfg.total_rb,
                vehicles,
                rate_per_rb,
                avg_thr,
                cfg,
                scenario,
                requested_rb,
            )

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        served_bits_arr = serve_queues(vehicles, alloc_rb, rate_per_rb, cfg)
        queue_after = np.array([v.queue_bits for v in vehicles], dtype=np.float64)

        active_users = int(np.sum(active_mask))
        requested_total = int(np.sum(requested_rb))
        allocated_total = int(np.sum(alloc_rb))
        unserved_users = int(np.sum(active_mask & (alloc_rb == 0)))

        sum_active_users += active_users
        sum_requested_rb += requested_total
        sum_allocated_rb += allocated_total
        sum_unserved_users += unserved_users

        do_debug = (slot < cfg.debug_slots) or (cfg.debug_every > 0 and slot % cfg.debug_every == 0)
        if do_debug:
            for i, v in enumerate(vehicles):
                debug_rows.append(
                    {
                        "scheduler": scheduler_name,
                        "scenario": scenario,
                        "run_idx": run_idx,
                        "slot": slot,
                        "vehicle_id": v.vid,
                        "vehicle_state_env_only": v.state,
                        "queue_bits_before": float(queue_before[i]),
                        "channel_gain": float(gains[i]),
                        "rate_per_rb": float(rate_per_rb[i]),
                        "requested_rb": int(requested_rb[i]),
                        "allocated_rb": int(alloc_rb[i]),
                        "served_bits": float(served_bits_arr[i]),
                        "queue_bits_after": float(queue_after[i]),
                        "active_flag": int(active_mask[i]),
                        "selected_flag": int(alloc_rb[i] > 0),
                        "unserved_slot_count": int(v.unserved_slot_count),
                    }
                )

        inst_thr = served_bits_arr / cfg.slot_sec
        avg_thr = 0.9 * avg_thr + 0.1 * inst_thr

    metrics = build_metrics(
        vehicles=vehicles,
        scheduler=scheduler_name,
        scenario=scenario,
        run_idx=run_idx,
        cfg=cfg,
        mean_active_users_per_slot=sum_active_users / max(cfg.n_slots, 1),
        mean_requested_rb_per_slot=sum_requested_rb / max(cfg.n_slots, 1),
        mean_allocated_rb_per_slot=sum_allocated_rb / max(cfg.n_slots, 1),
        mean_unserved_users_per_slot=sum_unserved_users / max(cfg.n_slots, 1),
    )

    return metrics, debug_rows


def run_experiments(
    cfg: SimConfig,
    scenarios: List[str],
    schedulers: List[str],
) -> Tuple[List[Metrics], List[Dict[str, float | int | str]]]:
    results: List[Metrics] = []
    debug_rows_all: List[Dict[str, float | int | str]] = []

    for scenario in scenarios:
        for scheduler in schedulers:
            for run_idx in range(cfg.n_runs):
                metrics, debug_rows = simulate_once(scheduler, scenario, cfg, run_idx)
                results.append(metrics)
                debug_rows_all.extend(debug_rows)

    return results, debug_rows_all


# =========================================================
# 10) 결과 집계 / 저장
# =========================================================


def metrics_to_rows(metrics: List[Metrics]) -> List[Dict[str, float | str | int]]:
    return [asdict(m) for m in metrics]


def save_csv_rows(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(metrics: List[Metrics]) -> Dict[str, Dict[str, Dict[str, float]]]:
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}

    scenarios = sorted(set(m.scenario for m in metrics))
    schedulers = sorted(set(m.scheduler for m in metrics))

    for scenario in scenarios:
        agg[scenario] = {}
        for scheduler in schedulers:
            subset = [m for m in metrics if m.scenario == scenario and m.scheduler == scheduler]
            agg[scenario][scheduler] = {
                "total_throughput_bps": float(np.mean([x.total_throughput_bps for x in subset])),
                "avg_delay_sec": float(np.mean([x.avg_delay_sec for x in subset])),
                "fairness": float(np.mean([x.fairness for x in subset])),
                "congestion_avg_throughput_bps": float(np.mean([x.congestion_avg_throughput_bps for x in subset])),
                "normal_avg_throughput_bps": float(np.mean([x.normal_avg_throughput_bps for x in subset])),
                "empty_avg_throughput_bps": float(np.mean([x.empty_avg_throughput_bps for x in subset])),
            }

    return agg


def save_summary_txt(agg: Dict[str, Dict[str, Dict[str, float]]], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    for scenario, by_scheduler in agg.items():
        lines.append(f"[Scenario] {scenario}")
        header = (
            f"{'Scheduler':<16}"
            f"{'Throughput(bps)':>18}"
            f"{'Delay(s)':>14}"
            f"{'Fairness':>12}"
            f"{'CongThr':>14}"
            f"{'NormThr':>14}"
            f"{'EmptyThr':>14}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for scheduler, vals in by_scheduler.items():
            lines.append(
                f"{scheduler:<16}"
                f"{vals['total_throughput_bps']:>18.2f}"
                f"{vals['avg_delay_sec']:>14.6f}"
                f"{vals['fairness']:>12.4f}"
                f"{vals['congestion_avg_throughput_bps']:>14.2f}"
                f"{vals['normal_avg_throughput_bps']:>14.2f}"
                f"{vals['empty_avg_throughput_bps']:>14.2f}"
            )
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# 11) 그래프
# =========================================================


PLOT_EXCLUDED_SCHEDULERS = {"OursPF"}


def get_plot_schedulers(schedulers) -> List[str]:
    return [scheduler for scheduler in schedulers if scheduler not in PLOT_EXCLUDED_SCHEDULERS]


def plot_metric_bars(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    metric_name: str,
    ylabel: str,
    out_path: Path,
) -> None:
    scenarios = list(agg.keys())
    schedulers = get_plot_schedulers(next(iter(agg.values())).keys())

    x = np.arange(len(scenarios))
    width = 0.16

    plt.figure(figsize=(10, 5))
    for i, scheduler in enumerate(schedulers):
        vals = [agg[sc][scheduler][metric_name] for sc in scenarios]
        plt.bar(x + i * width - width * (len(schedulers) - 1) / 2, vals, width=width, label=scheduler)

    plt.xticks(x, scenarios)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by scenario")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_state_throughput_bars(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    target_state: str,
    out_path: Path,
) -> None:
    key_map = {
        "Congestion": "congestion_avg_throughput_bps",
        "Normal": "normal_avg_throughput_bps",
        "Empty": "empty_avg_throughput_bps",
    }
    metric_name = key_map[target_state]

    scenarios = list(agg.keys())
    schedulers = get_plot_schedulers(next(iter(agg.values())).keys())

    x = np.arange(len(scenarios))
    width = 0.16

    plt.figure(figsize=(10, 5))
    for i, scheduler in enumerate(schedulers):
        vals = [agg[sc][scheduler][metric_name] for sc in scenarios]
        plt.bar(x + i * width - width * (len(schedulers) - 1) / 2, vals, width=width, label=scheduler)

    plt.xticks(x, scenarios)
    plt.ylabel("Average throughput (bps)")
    plt.title(f"Average throughput of {target_state} vehicles")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()




def load_sim_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_sim_config_path(cfg_path: str | None) -> Path:
    if cfg_path:
        p = Path(cfg_path).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)

    candidates = [
        Path("src/rb_simulator_config.yaml"),
        Path(__file__).resolve().parent / "rb_simulator_config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("rb_simulator_config.yaml not found. Use --config <path>.")

# =========================================================
# 12) CLI
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean RB scheduler simulator")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--total-rb", type=int, default=None)
    parser.add_argument("--n-vehicles", type=int, default=None)
    parser.add_argument("--n-slots", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--preset", type=str, default=None, choices=[None, "", "low_load", "mid_load", "high_load"])
    return parser.parse_args()


# =========================================================
# 13) 메인 실행
# =========================================================


def main() -> None:
    args = parse_args()

    cfg_path = resolve_sim_config_path(args.config)
    raw = load_sim_yaml(cfg_path)

    sim_raw = raw.get("sim", {})
    total_rb = int(sim_raw.get("total_rb", 12))
    n_vehicles = int(sim_raw.get("n_vehicles", 60))
    n_slots = int(sim_raw.get("n_slots", 300))
    n_runs = int(sim_raw.get("n_runs", 30))
    seed = int(sim_raw.get("seed", 42))

    out_dir_val = raw.get("output", {}).get("out_dir", "sim_out")
    preset = raw.get("preset", "")

    if args.total_rb is not None:
        total_rb = args.total_rb
    if args.n_vehicles is not None:
        n_vehicles = args.n_vehicles
    if args.n_slots is not None:
        n_slots = args.n_slots
    if args.n_runs is not None:
        n_runs = args.n_runs
    if args.seed is not None:
        seed = args.seed
    if args.out_dir is not None:
        out_dir_val = args.out_dir
    if args.preset is not None:
        preset = args.preset

    if preset == "low_load":
        total_rb, n_vehicles = 24, 30
    elif preset == "mid_load":
        total_rb, n_vehicles = 12, 60
    elif preset == "high_load":
        total_rb, n_vehicles = 8, 90

    cfg = SimConfig(
        total_rb=total_rb,
        n_vehicles=n_vehicles,
        n_slots=n_slots,
        n_runs=n_runs,
        seed=seed,
    )

    scenarios = raw.get("experiment", {}).get(
        "scenarios",
        ["balanced", "congestion_heavy", "normal_heavy", "empty_heavy"],
    )
    schedulers = raw.get("experiment", {}).get(
        "schedulers",
        ["RR", "MaxThroughput", "PF", "Ours", "OursPF"],
    )

    out_dir = Path(out_dir_val)
    plot_dir = out_dir / "plots"

    print("[INFO] Start simulation")
    print(
        f"[INFO] total_rb={cfg.total_rb}, "
        f"n_vehicles={cfg.n_vehicles}, "
        f"n_slots={cfg.n_slots}, "
        f"n_runs={cfg.n_runs}"
    )

    metrics, debug_rows = run_experiments(cfg, scenarios, schedulers)
    agg = aggregate_results(metrics)

    save_csv_rows(metrics_to_rows(metrics), out_dir / "rb_sim_results.csv")

    if cfg.save_debug_csv:
        save_csv_rows(debug_rows, out_dir / "rb_sim_debug.csv")

    save_summary_txt(agg, out_dir / "rb_sim_summary.txt")

    plot_metric_bars(agg, "total_throughput_bps", "Total throughput (bps)", plot_dir / "total_throughput.png")
    plot_metric_bars(agg, "avg_delay_sec", "Average delay (sec)", plot_dir / "average_delay.png")
    plot_metric_bars(agg, "fairness", "Jain fairness", plot_dir / "fairness.png")

    plot_state_throughput_bars(agg, "Congestion", plot_dir / "throughput_congestion_vehicles.png")
    plot_state_throughput_bars(agg, "Normal", plot_dir / "throughput_normal_vehicles.png")
    plot_state_throughput_bars(agg, "Empty", plot_dir / "throughput_empty_vehicles.png")

    print(f"[SAVE] CSV     : {out_dir / 'rb_sim_results.csv'}")
    print(f"[SAVE] DEBUG   : {out_dir / 'rb_sim_debug.csv'}")
    print(f"[SAVE] SUMMARY : {out_dir / 'rb_sim_summary.txt'}")
    print(f"[SAVE] PLOTS   : {plot_dir}")


if __name__ == "__main__":
    main()
