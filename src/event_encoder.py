# src/event_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class EncoderConfig:
    # Density 정규화용 (셀당 unique 차량 수를 이 값으로 나눠 0~1로 클립)
    density_cap: float = 5.0

    # 속도 기준 (m/s)
    v_low: float = 2.0     # 정체로 보일 만큼 느림
    v_ok: float = 6.0      # 정상 흐름으로 보일 만큼 빠름

    # Occupancy(체류) 기준 (0~1 정규화 후)
    occ_high: float = 0.6

    # 요약 방식: density 상위 k% 셀만 보고 요약
    topk_ratio: float = 0.05


def _safe_nanmean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _topk_mask(score_map: np.ndarray, ratio: float) -> np.ndarray:
    """score_map에서 값이 큰 상위 ratio만 True 마스크"""
    s = score_map.astype(np.float32).copy()
    s[~np.isfinite(s)] = np.nan
    flat = s[np.isfinite(s)]
    if flat.size == 0:
        return np.zeros_like(s, dtype=bool)

    k = max(1, int(np.ceil(flat.size * ratio)))
    thr = np.partition(flat, -k)[-k]
    return np.isfinite(s) & (s >= thr)


def encode_event_type(
    unique_cnt_map: np.ndarray,   # (H,W) 셀별 unique 차량 수
    mean_speed_map: np.ndarray,   # (H,W) 셀별 평균 속도 (m/s), NaN 많을 수 있음
    dwell_map: np.ndarray,        # (H,W) 셀별 체류 프레임 수
    *,
    cfg: EncoderConfig = EncoderConfig(),
    static_dwell_map: Optional[np.ndarray] = None,   # (H,W) 정적 체류(occlusion 보조)
    static_change_rate: Optional[np.ndarray] = None, # (H,W) 정적 변화율(ego-stop/occlusion 보조)
) -> Tuple[str, Dict[str, float]]:
    """
    히트맵 기반 EventType 결정 (룰 기반 인코더)

    Semantic features:
      - density_mean   : 밀집(0~1)
      - speed_mean     : 속도(m/s)
      - occupancy_mean : 체류(0~1)

    보조 신호:
      - static_dwell / static_change_rate로 "고정/가려짐" 성격을 강화
    """

    # 1) Density (0~1)
    density = np.clip(unique_cnt_map.astype(np.float32) / float(cfg.density_cap), 0.0, 1.0)

    # 2) Occupancy (0~1): dwell을 전체에서 가장 큰 dwell 기준으로 정규화
    dwell_f = dwell_map.astype(np.float32)
    dwell_max = float(np.nanmax(dwell_f)) if np.isfinite(dwell_f).any() else 0.0
    if dwell_max <= 0:
        occupancy = np.zeros_like(dwell_f, dtype=np.float32)
    else:
        occupancy = np.clip(dwell_f / dwell_max, 0.0, 1.0)

    # 3) 관심영역: density 상위 topk 셀만 본다
    focus = _topk_mask(density, cfg.topk_ratio)

    if not focus.any():
        # 차량 거의 없음 -> Normal
        feats = {
            "density_mean": 0.0,
            "speed_mean": _safe_nanmean(mean_speed_map),
            "occupancy_mean": 0.0,
            "occlusion_score": 0.0,
            "congestion_score": 0.0,
        }
        return "Normal", feats

    dens_mean = _safe_nanmean(density[focus])
    speed_mean = _safe_nanmean(mean_speed_map.astype(np.float32)[focus])
    occ_mean = _safe_nanmean(occupancy[focus])

    # ------------------------------------------------------------------
    # 점수 기반 판별
    # ------------------------------------------------------------------

    # A) Congestion: 밀집↑ + 속도↓ + 체류↑
    congestion_score = 0.0
    if np.isfinite(dens_mean) and dens_mean >= 0.6:
        congestion_score += 1.0
    if np.isfinite(speed_mean) and speed_mean <= cfg.v_low:
        congestion_score += 1.0
    if np.isfinite(occ_mean) and occ_mean >= cfg.occ_high:
        congestion_score += 1.0

    # B) Occlusion: "고정된 패턴" (체류↑, 속도↓) + 정적 신호로 강화
    occlusion_score = 0.0

    # 핵심 패턴: 체류 높고 속도 거의 없음
    if np.isfinite(occ_mean) and occ_mean >= cfg.occ_high:
        occlusion_score += 1.0
    if (not np.isfinite(speed_mean)) or (np.isfinite(speed_mean) and speed_mean <= cfg.v_low):
        occlusion_score += 0.5

    # 밀집이 낮은데 체류만 높으면 "정체"보단 "고정" 쪽(가려짐/정지물) 가능성이 커짐
    if np.isfinite(dens_mean) and dens_mean <= 0.2:
        occlusion_score += 0.5

    # 보조: 정적 체류/변화율 (있으면 강력한 힌트)
    if static_dwell_map is not None:
        sd = static_dwell_map.astype(np.float32)
        sd_focus_mean = _safe_nanmean(sd[focus])
        if np.isfinite(sd_focus_mean) and sd_focus_mean > 0:
            occlusion_score += 0.5

    if static_change_rate is not None:
        scr = static_change_rate.astype(np.float32)
        scr_focus_mean = _safe_nanmean(scr[focus])
        # 변화율 낮음 = 프레임간 거의 안 바뀜 = 고정
        if np.isfinite(scr_focus_mean) and scr_focus_mean < 0.2:
            occlusion_score += 0.5

    # ------------------------------------------------------------------
    # 최종 결정: Occlusion > Congestion > Normal
    # ------------------------------------------------------------------
    if occlusion_score >= 1.5:
        event = "Occlusion"
    elif congestion_score >= 2.0:
        event = "Congestion"
    else:
        event = "Normal"

    feats = {
        "density_mean": float(dens_mean) if np.isfinite(dens_mean) else float("nan"),
        "speed_mean": float(speed_mean) if np.isfinite(speed_mean) else float("nan"),
        "occupancy_mean": float(occ_mean) if np.isfinite(occ_mean) else float("nan"),
        "occlusion_score": float(occlusion_score),
        "congestion_score": float(congestion_score),
    }
    return event, feats
