from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label, uniform_filter

from megalith_lidar.catalog import (
    CLASS_NAMES,
    ENCLOSURE_CLASS_NAMES,
    ROMAN_CLASS_NAMES,
    MonumentCatalog,
    bng_to_latlon,
    bng_to_os_grid_ref,
)
from megalith_lidar.lidar_wcs import fetch_or_load_paired_tile, fetch_or_load_region_windows
from megalith_lidar.model import MegalithUNet
from megalith_lidar.osm_context import (
    annotate_candidates_with_spatial_context,
    fetch_or_load_osm_context,
)
from megalith_lidar.stats import PlattCalibrator
from megalith_lidar.terrain_features import NUM_TERRAIN_CHANNELS, compute_terrain_features


def compute_concentric_ring_observables(
    raw_features: np.ndarray,
    cx_px: float,
    cy_px: float,
    area_m2: float,
) -> Dict[str, float]:
    """
    Compute radial concentric ring-ditch and annular ring-cairn observables around a candidate
    centroid `(cx_px, cy_px)` on the 4 m (`raw_features[0]`) and 16 m (`raw_features[0] + raw_features[1]`)
    scale-space relief bands.
    """
    rel4 = raw_features[0]
    rel16 = raw_features[0] + raw_features[1]
    h, w = rel4.shape
    r_eff = float(np.clip(math.sqrt(max(area_m2, 25.0) / math.pi), 6.0, 28.0))
    rad = int(math.ceil(3.2 * r_eff)) + 2
    ix, iy = int(round(cx_px)), int(round(cy_px))
    x0, x1 = max(0, ix - rad), min(w, ix + rad + 1)
    y0, y1 = max(0, iy - rad), min(h, iy + rad + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.hypot(xx - cx_px, yy - cy_px)
    angles = np.arctan2(yy - cy_px, xx - cx_px)
    r4_sub = rel4[y0:y1, x0:x1]
    r16_sub = rel16[y0:y1, x0:x1]

    m_inner = rr <= 0.55 * r_eff
    m_rim = (rr >= 0.65 * r_eff) & (rr <= 1.15 * r_eff)
    m_d4_a = (rr >= 1.05 * r_eff) & (rr <= 1.75 * r_eff)
    m_d4_b = (rr >= 1.45 * r_eff) & (rr <= 2.35 * r_eff)
    m_ditch16 = (rr >= 1.45 * r_eff) & (rr <= 2.45 * r_eff)
    m_reb_a = (rr >= 1.75 * r_eff) & (rr <= 2.55 * r_eff)
    m_reb_b = (rr >= 2.35 * r_eff) & (rr <= 3.15 * r_eff)

    inner4 = float(np.mean(r4_sub[m_inner])) if np.any(m_inner) else 0.0
    rim4 = float(np.mean(r4_sub[m_rim])) if np.any(m_rim) else 0.0
    d4_a = float(np.mean(r4_sub[m_d4_a])) if np.any(m_d4_a) else 0.0
    d4_b = float(np.mean(r4_sub[m_d4_b])) if np.any(m_d4_b) else 0.0
    if d4_b < d4_a:
        ditch4 = d4_b
        m_ditch4 = m_d4_b
        reb4 = (float(np.mean(r4_sub[m_reb_b])) - ditch4) if np.any(m_reb_b) else 0.0
    else:
        ditch4 = d4_a
        m_ditch4 = m_d4_a
        reb4 = (float(np.mean(r4_sub[m_reb_a])) - ditch4) if np.any(m_reb_a) else 0.0
    ditch16 = float(np.mean(r16_sub[m_ditch16])) if np.any(m_ditch16) else 0.0

    neg_sec = 0
    rim_sec = 0
    for s in range(12):
        a0 = -math.pi + s * (2.0 * math.pi / 12.0)
        a1 = a0 + (2.0 * math.pi / 12.0)
        ang_m = (angles >= a0) & (angles < a1)
        sm4 = m_ditch4 & ang_m
        sm16 = m_ditch16 & ang_m
        sm_rim = m_rim & ang_m
        c4 = float(np.mean(r4_sub[sm4])) < -0.007 if np.any(sm4) else False
        c16 = float(np.mean(r16_sub[sm16])) < -0.040 if np.any(sm16) else False
        if c4 or c16:
            neg_sec += 1
        if np.any(sm_rim) and float(np.mean(r4_sub[sm_rim])) > max(0.015, inner4 + 0.006):
            rim_sec += 1

    return {
        "ring_inner4_m": round(inner4, 4),
        "ring_rim4_m": round(rim4, 4),
        "ring_ditch4_m": round(ditch4, 4),
        "ring_ditch16_m": round(ditch16, 4),
        "ring_reb4_m": round(reb4, 4),
        "ring_neg_sectors": int(neg_sec),
        "ring_rim_sectors": int(rim_sec),
    }


def evaluate_concentric_ring_score(cand: Dict) -> None:
    """
    Evaluate whether a candidate exhibits a concentric prehistoric ring-ditch (`ring_ditch_barrow`)
    or annular ring-cairn (`annular_ring_cairn`) and compute `ring_barrow_score` in `[0, 1]`.
    """
    inner4 = float(cand.get("ring_inner4_m") or 0.0)
    rim4 = float(cand.get("ring_rim4_m") or 0.0)
    ditch4 = float(cand.get("ring_ditch4_m") or 0.0)
    ditch16 = float(cand.get("ring_ditch16_m") or 0.0)
    reb4 = float(cand.get("ring_reb4_m") or 0.0)
    neg_sec = int(cand.get("ring_neg_sectors") or 0)
    rim_sec = int(cand.get("ring_rim_sectors") or 0)

    adj_chan = float(cand.get("adjacent_channel_span_m") or 0.0)
    pond_pk = float(cand.get("pond_unet_local_peak") or 0.0)
    exg = float(cand.get("cropmark_ring_exg_diff") or 0.0)
    lum = float(cand.get("soilmark_core_lum_diff") or 0.0)
    elong = float(cand.get("elongation") or 1.0)
    patch_min = float(cand.get("patch_min_relief_m") or 0.0)

    is_pond_like = (
        (adj_chan >= 35.0)
        or (pond_pk >= 0.55 and exg <= -0.020 and lum <= -0.018)
        or (pond_pk >= 0.75 and patch_min <= -0.30)
        or (pond_pk >= 0.85)
        or (ditch4 <= -0.042 and patch_min <= -0.30)
    )
    ring_type: Optional[str] = None
    ring_score = 0.0
    if not is_pond_like and elong <= 1.65 and cand.get("target_domain", "prehistoric") == "prehistoric":
        if (
            inner4 >= 0.055
            and -0.040 <= ditch4 <= -0.009
            and reb4 >= 0.0025
            and neg_sec >= 7
            and ((ditch16 <= -0.040 and neg_sec >= 8) or (reb4 >= 0.012 and exg >= 0.020))
        ):
            s_d4 = float(np.clip((-ditch4 - 0.007) / 0.015, 0.0, 1.0))
            s_d16 = (
                float(np.clip((-ditch16 - 0.035) / 0.040, 0.0, 1.0))
                if ditch16 < 0
                else float(np.clip(reb4 / 0.018, 0.0, 1.0))
            )
            s_sec = float(np.clip((neg_sec - 6.0) / 4.0, 0.0, 1.0))
            s_reb = float(np.clip(reb4 / 0.012, 0.0, 1.0))
            s_crop = float(np.clip(exg / 0.05, 0.0, 1.0)) if exg > 0 else 0.0
            ring_score = float(0.30 * s_d4 + 0.25 * s_d16 + 0.18 * s_sec + 0.12 * s_reb + 0.15 * s_crop)
            ring_type = "ring_ditch_barrow"
        if rim4 >= 0.028 and (rim4 - inner4) >= 0.014 and rim_sec >= 7 and -0.035 <= ditch4 <= 0.002 and reb4 >= 0.006:
            s_rim = float(np.clip((rim4 - inner4 - 0.010) / 0.025, 0.0, 1.0))
            s_rsec = float(np.clip((rim_sec - 6.0) / 4.0, 0.0, 1.0))
            s_reb = float(np.clip(reb4 / 0.014, 0.0, 1.0))
            sc2 = float(0.45 * s_rim + 0.30 * s_rsec + 0.25 * s_reb)
            if sc2 > ring_score:
                ring_score = sc2
                ring_type = "annular_ring_cairn"

    cand["has_concentric_ring"] = bool(ring_score > 0.0)
    cand["ring_type"] = ring_type
    cand["ring_barrow_score"] = round(ring_score, 4)
    raw_conf = float(cand.get("raw_unet_confidence", cand.get("confidence", 0.0)))
    cand["raw_unet_confidence"] = round(raw_conf, 4)
    if (
        ring_score > 0.0
        and cand.get("context_category", "unidentified_candidate") == "unidentified_candidate"
        and cand.get("vlm_verdict_tier", "PLAUSIBLE_MONUMENT") == "PLAUSIBLE_MONUMENT"
    ):
        cand["confidence"] = round(min(0.997, raw_conf + (0.997 - raw_conf) * 0.55 * ring_score), 4)


def extract_candidates_from_tile(
    prob_map: np.ndarray,
    cls_prob_map: np.ndarray,
    raw_features: np.ndarray,
    e_min: float,
    e_max: float,
    n_min: float,
    n_max: float,
    catalog: MonumentCatalog,
    prob_threshold: float = 0.50,
    min_area_m2: float = 9.0,
    border_margin_px: int = 8,
    known_match_radius_m: float = 45.0,
    target_domain: str = "prehistoric",
) -> List[Dict]:
    """
    Extract connected-component candidate detections purely from the model's predicted
    probability map above `prob_threshold` (with zero ad-hoc topographic post-filters or
    artificial score bonuses). Topographic measurements are reported strictly as descriptive
    observables.
    """
    h, w = prob_map.shape
    scale_e = (e_max - e_min) / float(w)
    scale_n = (n_max - n_min) / float(h)
    pixel_area_m2 = scale_e * scale_n

    smoothed = gaussian_filter(prob_map.astype(np.float32), sigma=1.0)
    binary = smoothed >= prob_threshold
    if border_margin_px > 0 and h > 2 * border_margin_px and w > 2 * border_margin_px:
        binary[:border_margin_px, :] = False
        binary[-border_margin_px:, :] = False
        binary[:, :border_margin_px] = False
        binary[:, -border_margin_px:] = False

    labeled, num_comp = label(binary)
    if num_comp == 0:
        return []

    slope_arr = raw_features[3]
    slope_mean = uniform_filter(slope_arr, size=9)
    slope_std = np.sqrt(np.maximum(uniform_filter(slope_arr * slope_arr, size=9) - slope_mean * slope_mean, 0.0))
    tile_tin_px = int(np.sum((slope_arr > 0.03) & (slope_std < 0.0022)))

    full_rel_16 = raw_features[0] + raw_features[1]
    labeled_bar, _ = label(full_rel_16 >= 0.18)
    labeled_ch, _ = label(full_rel_16 <= -0.18)
    yy_grid, xx_grid = np.ogrid[:h, :w]

    candidates: List[Dict] = []
    for comp_id in range(1, num_comp + 1):
        ys, xs = np.where(labeled == comp_id)
        area_m2 = float(len(xs)) * pixel_area_m2
        if area_m2 < min_area_m2:
            continue

        weights = smoothed[ys, xs]
        w_sum = float(np.sum(weights)) + 1e-8
        cx_px = float(np.sum(xs * weights) / w_sum)
        cy_px = float(np.sum(ys * weights) / w_sum)

        peak_prob = float(np.max(prob_map[ys, xs]))
        mean_prob = float(np.mean(prob_map[ys, xs]))

        # Second-moment morphological elongation (descriptive only, not used to filter)
        cov_xx = float(np.mean((xs - cx_px) ** 2)) + 1.0
        cov_yy = float(np.mean((ys - cy_px) ** 2)) + 1.0
        cov_xy = float(np.mean((xs - cx_px) * (ys - cy_px)))
        trace = cov_xx + cov_yy
        det = max(cov_xx * cov_yy - cov_xy * cov_xy, 1e-4)
        disc = math.sqrt(max(trace * trace - 4.0 * det, 0.0))
        lam1 = 0.5 * (trace + disc)
        lam2 = max(0.5 * (trace - disc), 0.25)
        elongation = math.sqrt(lam1 / lam2)

        if cls_prob_map.shape[0] >= 4:
            pos_cls_scores = [
                float(np.sum(cls_prob_map[c, ys, xs] * weights) / w_sum) for c in (1, 2, 3)
            ]
            best_cls_idx = int(np.argmax(pos_cls_scores)) + 1
        else:
            best_cls_idx = 1

        easting = float(e_min + (cx_px + 0.5) * scale_e)
        northing = float(n_max - (cy_px + 0.5) * scale_n)
        lat, lon = bng_to_latlon(easting, northing)
        grid_ref = bng_to_os_grid_ref(easting, northing)

        # Descriptive physical observables from raw (unstandardized) features:
        # band_fine_4m + band_med_16m = z - G_16(z) in metres
        local_relief_16m = full_rel_16[ys, xs]
        signed_relief_m = float(local_relief_16m[np.argmax(np.abs(local_relief_16m))])
        max_pos_relief_m = float(np.max(local_relief_16m))
        cy_i = min(h - 1, max(0, int(round(cy_px))))
        cx_i = min(w - 1, max(0, int(round(cx_px))))
        centroid_relief_m = float(full_rel_16[cy_i, cx_i])
        ry0, ry1 = max(0, cy_i - 14), min(h, cy_i + 15)
        rx0, rx1 = max(0, cx_i - 14), min(w, cx_i + 15)
        patch_min_relief_m = float(np.min(full_rel_16[ry0:ry1, rx0:rx1]))
        grad_vals = raw_features[3, ys, xs]
        ndsm_m = float(np.expm1(np.max(raw_features[5, ys, xs])))

        # Underlying terrain ridge & channel continuity observables (detects river meander scroll-bars / palaeochannels)
        dist_sq = (yy_grid - cy_i) ** 2 + (xx_grid - cx_i) ** 2
        bar_ids = set(np.unique(labeled_bar[dist_sq <= 144])) - {0}
        ridge_span_m = 0.0
        ridge_elong = 1.0
        for bid in bar_ids:
            by_px, bx_px = np.where(labeled_bar == bid)
            b_span = float(np.hypot((bx_px.max() - bx_px.min()) * scale_e, (by_px.max() - by_px.min()) * scale_n))
            if len(by_px) >= 16:
                b_cov = np.cov(bx_px.astype(float), by_px.astype(float))
                b_ev = np.linalg.eigvalsh(b_cov)
                b_el = float(np.sqrt(max(b_ev[1], 1e-3) / max(b_ev[0], 1e-3)))
            else:
                b_el = 1.0
            if b_span > ridge_span_m:
                ridge_span_m = b_span
                ridge_elong = b_el

        ch_ids = set(np.unique(labeled_ch[dist_sq <= 576])) - {0}
        channel_span_m = 0.0
        for cid in ch_ids:
            chy_px, chx_px = np.where(labeled_ch == cid)
            c_span = float(np.hypot((chx_px.max() - chx_px.min()) * scale_e, (chy_px.max() - chy_px.min()) * scale_n))
            if c_span > channel_span_m:
                channel_span_m = c_span

        # Condition subtype class assignment on 2D component footprint geometry:
        # A circular/sub-circular mound (elongation < 1.65) cannot physically be a Long Barrow.
        if target_domain == "prehistoric" and best_cls_idx == 1:
            if elongation < 1.65 or area_m2 < 220.0:
                if centroid_relief_m < 0.25 * max(max_pos_relief_m, 0.1) and area_m2 >= 260.0:
                    best_cls_idx = 2  # stone_circle_henge
                else:
                    best_cls_idx = 3  # round_barrow_cairn
        elif target_domain == "enclosure":
            if area_m2 <= 950.0:
                best_cls_idx = 1  # hut_circle_pound_settlement
            elif elongation >= 2.20:
                best_cls_idx = 3  # cross_ridge_dyke_field_system
            else:
                best_cls_idx = 2  # iron_age_hillfort_banjo_enclosure
        elif target_domain == "roman":
            if elongation >= 2.25:
                best_cls_idx = 2  # roman_road_agger_trackway
            elif area_m2 <= 1450.0:
                best_cls_idx = 1  # roman_villa_building
            else:
                best_cls_idx = 3  # roman_fort_camp_settlement

        py0, py1 = max(0, int(cy_px) - 48), min(h, int(cy_px) + 48)
        px0, px1 = max(0, int(cx_px) - 48), min(w, int(cx_px) + 48)
        patch_tin_px = int(np.sum(slope_std[py0:py1, px0:px1] < 0.0018))
        has_tin_facet = bool(tile_tin_px >= 6000 or patch_tin_px >= 120)

        nearby = catalog.find_nearby(easting, northing, max_distance_m=known_match_radius_m)
        is_known = len(nearby) > 0
        matched_info = None
        if target_domain == "roman":
            cls_map_dict = ROMAN_CLASS_NAMES
        elif target_domain == "enclosure":
            cls_map_dict = ENCLOSURE_CLASS_NAMES
        else:
            cls_map_dict = CLASS_NAMES
        if is_known:
            rec, dist_m = nearby[0]
            if target_domain == "roman" and rec.roman_cls > 0:
                matched_cls_id = rec.roman_cls
                matched_cls_dict = ROMAN_CLASS_NAMES
            elif target_domain == "enclosure" and rec.enclosure_cls > 0:
                matched_cls_id = rec.enclosure_cls
                matched_cls_dict = ENCLOSURE_CLASS_NAMES
            else:
                matched_cls_id = rec.cls
                matched_cls_dict = CLASS_NAMES
            matched_info = {
                "list_entry": rec.list_entry,
                "name": rec.name,
                "class": matched_cls_dict.get(matched_cls_id, "unknown"),
                "distance_m": round(float(dist_m), 1),
            }

        # Pure model probability before concentric ring evaluation
        score = float(0.75 * peak_prob + 0.25 * mean_prob)
        ring_obs = compute_concentric_ring_observables(raw_features, cx_px, cy_px, area_m2)

        cand_dict = {
            "easting": round(easting, 1),
            "northing": round(northing, 1),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "os_grid_ref": grid_ref,
            "confidence": round(score, 4),
            "raw_unet_confidence": round(score, 4),
            "peak_prob": round(peak_prob, 4),
            "mean_prob": round(mean_prob, 4),
            "predicted_class_id": best_cls_idx,
            "predicted_class": cls_map_dict.get(best_cls_idx, "megalithic_monument"),
            "target_domain": target_domain,
            "area_m2": round(area_m2, 1),
            "elongation": round(elongation, 2),
            "est_local_relief_m": round(signed_relief_m, 2),
            "max_positive_relief_m": round(max_pos_relief_m, 2),
            "centroid_local_relief_m": round(centroid_relief_m, 2),
            "patch_min_relief_m": round(patch_min_relief_m, 2),
            "est_surface_prominence_m": round(ndsm_m, 2),
            "mean_slope_tangent": round(float(np.mean(grad_vals)), 3),
            "ridge_continuation_span_m": round(ridge_span_m, 1),
            "ridge_continuation_elongation": round(ridge_elong, 2),
            "adjacent_channel_span_m": round(channel_span_m, 1),
            "has_lidar_tin_facet": has_tin_facet,
            "is_known_monument": is_known,
            "matched_monument": matched_info,
        }
        cand_dict.update(ring_obs)
        candidates.append(cand_dict)

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


def nms_deduplicate_candidates(candidates: List[Dict], merge_radius_m: float = 28.0) -> List[Dict]:
    """Spatial Non-Maximum Suppression (NMS) across overlapping scan windows."""
    sorted_cands = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    kept: List[Dict] = []
    for cand in sorted_cands:
        ce, cn = cand["easting"], cand["northing"]
        duplicate = False
        for k in kept:
            if math.hypot(ce - k["easting"], cn - k["northing"]) <= merge_radius_m:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


@torch.no_grad()
def predict_tile_with_d4_tta(
    model: MegalithUNet,
    standardized_features: np.ndarray,
    device: torch.device,
    calibrator: Optional[PlattCalibrator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run inference over the full 8-element Dihedral Group D4 (4 rotations x 2 reflections)
    so predictions are strictly invariant to compass orientation and chirality.
    """
    x = torch.from_numpy(standardized_features).unsqueeze(0).to(device=device, dtype=torch.float32)
    xf = torch.flip(x, [3])
    aug_inputs = [
        x,
        torch.rot90(x, 1, [2, 3]),
        torch.rot90(x, 2, [2, 3]),
        torch.rot90(x, 3, [2, 3]),
        xf,
        torch.rot90(xf, 1, [2, 3]),
        torch.rot90(xf, 2, [2, 3]),
        torch.rot90(xf, 3, [2, 3]),
    ]
    batch = torch.cat(aug_inputs, dim=0)  # (8, C, H, W)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model(batch)
        seg_logits = out["seg_logits"].float()  # (8, 1, H, W)
        cls_probs = torch.softmax(out["cls_seg_logits"].float(), dim=1)  # (8, 4, H, W)
        tile_logits = out["tile_logits"].float().squeeze(1)  # (8,)

    # Inverse D4 transforms on seg_logits and cls_probs
    inv_seg = []
    inv_cls = []
    for k in range(4):
        inv_seg.append(torch.rot90(seg_logits[k], -k, [1, 2]))
        inv_cls.append(torch.rot90(cls_probs[k], -k, [1, 2]))
    for k in range(4):
        s_unrot = torch.rot90(seg_logits[4 + k], -k, [1, 2])
        c_unrot = torch.rot90(cls_probs[4 + k], -k, [1, 2])
        inv_seg.append(torch.flip(s_unrot, [2]))
        inv_cls.append(torch.flip(c_unrot, [2]))

    mean_seg_logit = torch.stack(inv_seg, dim=0).mean(dim=0).squeeze(0).cpu().numpy()
    seg_prob_map = (1.0 / (1.0 + np.exp(-np.clip(mean_seg_logit, -40.0, 40.0)))).astype(np.float32)
    cls_mean = torch.stack(inv_cls, dim=0).mean(dim=0).cpu().numpy()

    mean_tile_logit = float(tile_logits.mean().item())
    if calibrator is not None:
        cal_tile_prob = float(calibrator.predict_proba(np.array([mean_tile_logit]))[0])
    else:
        cal_tile_prob = float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, mean_tile_logit)))))

    return seg_prob_map, cls_mean, mean_tile_logit, cal_tile_prob


def save_candidate_diagnostic_png(
    raw_features: np.ndarray,
    prob_map: np.ndarray,
    e_min: float,
    e_max: float,
    n_min: float,
    n_max: float,
    candidates: List[Dict],
    out_path: Path,
) -> None:
    """Render a 4-panel visual diagnostic PNG: [Isotropic Slope | Local Relief 16m | Log nDSM | Posterior Probability]."""
    h, w = prob_map.shape
    scale_e = w / (e_max - e_min)
    scale_n = h / (n_max - n_min)

    def to_u8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    # Panel 1: Inverted isotropic gradient magnitude (flat=bright, steep scarp=dark, zero directional bias)
    slope_inv_u8 = 255 - to_u8(raw_features[3], 0.0, 0.45)
    p1 = np.stack([slope_inv_u8, slope_inv_u8, slope_inv_u8], axis=-1)

    # Panel 2: Local Relief (z - G_16(z) = band_fine + band_med, in [-1.5m, +1.5m])
    rel_16 = raw_features[0] + raw_features[1]
    rel_u8 = to_u8(rel_16, -1.5, 1.5)
    p2 = np.stack([rel_u8, to_u8(rel_16, -1.0, 1.0), 255 - rel_u8], axis=-1)

    # Panel 3: Log nDSM
    ndsm_u8 = to_u8(raw_features[5], 0.0, 1.8)
    p3 = np.stack([ndsm_u8, (ndsm_u8 * 0.8).astype(np.uint8), (ndsm_u8 * 0.4).astype(np.uint8)], axis=-1)

    # Panel 4: Probability overlay
    prob_u8 = to_u8(prob_map, 0.0, 1.0)
    p4 = np.stack(
        [
            np.maximum(slope_inv_u8, prob_u8),
            (slope_inv_u8 * (1.0 - 0.65 * prob_map)).astype(np.uint8),
            (slope_inv_u8 * (1.0 - 0.4 * prob_map) + 0.4 * prob_u8).astype(np.uint8),
        ],
        axis=-1,
    )

    canvas = Image.new("RGB", (w * 4 + 12, h), (20, 20, 24))
    for idx, panel in enumerate((p1, p2, p3, p4)):
        im_p = Image.fromarray(panel)
        draw = ImageDraw.Draw(im_p)
        for c in candidates:
            cx = (c["easting"] - e_min) * scale_e
            cy = (n_max - c["northing"]) * scale_n
            r = max(10.0, math.sqrt(max(c["area_m2"], 16.0) / math.pi) + 6.0)
            cat = c.get("context_category")
            if cat == "rediscovered_nhle_scheduled" or c.get("is_known_monument"):
                color = (0, 230, 118)
            elif cat == "rediscovered_osm_archaeology":
                color = (0, 210, 255)
            elif cat == "flagged_modern_or_karst_confounder":
                color = (255, 110, 64)
            else:
                color = (255, 214, 0)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        canvas.paste(im_p, (idx * (w + 4), 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


_POND_UNET_CACHE: Dict[str, Tuple[MegalithUNet, np.ndarray, np.ndarray, PlattCalibrator, Dict[int, str]]] = {}


def _get_or_load_pond_unet(
    ckpt_path: Path, device: torch.device
) -> Optional[Tuple[MegalithUNet, np.ndarray, np.ndarray, PlattCalibrator, Dict[int, str]]]:
    if not ckpt_path.exists():
        return None
    key = f"{ckpt_path}:{device}"
    if key in _POND_UNET_CACHE:
        return _POND_UNET_CACHE[key]
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        p_model = MegalithUNet(in_channels=NUM_TERRAIN_CHANNELS, num_classes=4, pretrained=False).to(device)
        p_model.load_state_dict(ckpt.get("model_state_dict") or ckpt["model_state"])
        p_model.eval()
        p_mean = np.asarray(ckpt["channel_mean"], dtype=np.float32).reshape(-1, 1, 1)
        p_std = np.asarray(ckpt["channel_std"], dtype=np.float32).reshape(-1, 1, 1)
        p_cal = PlattCalibrator.from_dict(ckpt["calibrator"])
        p_cls = {int(k): v for k, v in ckpt.get("class_names", {}).items()}
        bundle = (p_model, p_mean, p_std, p_cal, p_cls)
        _POND_UNET_CACHE[key] = bundle
        return bundle
    except Exception:
        return None


def scan_region(
    model: MegalithUNet,
    catalog: MonumentCatalog,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    calibrator: PlattCalibrator,
    tile_prob_threshold: float,
    pixel_prob_threshold: float,
    e_min: int,
    e_max: int,
    n_min: int,
    n_max: int,
    device: torch.device,
    tile_cache_dir: Path,
    output_dir: Path,
    region_name: str = "uk_scan",
    window_size: int = 256,
    stride: int = 192,
    osm_cache_dir: Optional[Path] = None,
    target_domain: str = "prehistoric",
) -> Dict:
    """
    Scan a contiguous EPSG:27700 bounding box using strictly validation-calibrated
    thresholds, full D4 Test-Time Augmentation, and automated OpenStreetMap spatial
    context cross-checking.
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if osm_cache_dir is None:
        osm_cache_dir = Path(tile_cache_dir).parent / "osm_cache"

    osm_index = fetch_or_load_osm_context(
        e_min=e_min,
        n_min=n_min,
        e_max=e_max,
        n_max=n_max,
        osm_cache_dir=osm_cache_dir,
    )

    ch_mean = np.asarray(channel_mean, dtype=np.float32).reshape(-1, 1, 1)
    ch_std = np.asarray(channel_std, dtype=np.float32).reshape(-1, 1, 1)
    include_cropmarks = ch_mean.shape[0] >= NUM_TERRAIN_CHANNELS + 2
    aerial_cache_dir = Path(tile_cache_dir).parent / "aerial_tiles"

    es = list(range(e_min, max(e_min + 1, e_max - window_size + 1), stride))
    if es[-1] + window_size < e_max:
        es.append(e_max - window_size)
    ns = list(range(n_min, max(n_min + 1, n_max - window_size + 1), stride))
    if ns[-1] + window_size < n_max:
        ns.append(n_max - window_size)

    coords = [(e0, e0 + window_size, n0, n0 + window_size) for n0 in ns for e0 in es]

    loaded_tiles = fetch_or_load_region_windows(
        e_min=e_min,
        e_max=e_max,
        n_min=n_min,
        n_max=n_max,
        coords=coords,
        cache_dir=tile_cache_dir,
    )

    raw_candidates: List[Dict] = []
    window_records: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, List[Dict]]] = {}
    windows_above_tile_th = 0

    for key in sorted(loaded_tiles.keys()):
        e0, e1, n0, n1 = key
        dtm, dsm = loaded_tiles[key]
        raw_feats = compute_terrain_features(dtm, dsm)
        # Fast Stage-1 LiDAR pre-screen: only fetch aerial orthophotos on windows that have
        # non-trivial LiDAR terrain plausibility (or are already cached on disk).
        std_lidar = (raw_feats - ch_mean[:NUM_TERRAIN_CHANNELS]) / ch_std[:NUM_TERRAIN_CHANNELS]
        prob_map, cls_map, raw_tile_logit, cal_tile_prob = predict_tile_with_d4_tta(
            model, std_lidar, device, calibrator=calibrator
        )
        if include_cropmarks and cal_tile_prob >= tile_prob_threshold * 0.45:
            from megalith_lidar.aerial_cropmarks import (
                compute_candidate_cropmark_metrics,
                compute_cropmark_features,
                fetch_or_load_aerial_tile,
            )

            rgb = fetch_or_load_aerial_tile(e0, e1, n0, n1, cache_dir=aerial_cache_dir, size=dtm.shape[0])
            if rgb is not None:
                crop_feats = compute_cropmark_features(rgb, dtm=dtm, dsm=dsm)
                full_feats = np.concatenate([raw_feats, crop_feats], axis=0)
                std_feats = (full_feats - ch_mean) / ch_std
                prob_map, cls_map, raw_tile_logit, cal_tile_prob = predict_tile_with_d4_tta(
                    model, std_feats, device, calibrator=calibrator
                )
        else:
            rgb = None
        if cal_tile_prob < tile_prob_threshold:
            continue
        windows_above_tile_th += 1

        cands = extract_candidates_from_tile(
            prob_map=prob_map,
            cls_prob_map=cls_map,
            raw_features=raw_feats,
            e_min=e0,
            e_max=e1,
            n_min=n0,
            n_max=n1,
            catalog=catalog,
            prob_threshold=pixel_prob_threshold,
            target_domain=target_domain,
        )
        if rgb is None and cands:
            from megalith_lidar.aerial_cropmarks import (
                compute_candidate_cropmark_metrics,
                fetch_or_load_aerial_tile,
            )

            rgb = fetch_or_load_aerial_tile(e0, e1, n0, n1, cache_dir=aerial_cache_dir, size=dtm.shape[0])
        if cands:
            pond_ckpt_path = Path(tile_cache_dir).parent / "checkpoints" / "pond_unet_best.pt"
            pond_bundle = _get_or_load_pond_unet(pond_ckpt_path, device)
            if pond_bundle is not None:
                p_model, p_mean, p_std, p_cal, p_cls_names = pond_bundle
                if rgb is not None:
                    from megalith_lidar.aerial_cropmarks import compute_cropmark_features

                    p_crop = compute_cropmark_features(rgb, dtm=dtm, dsm=dsm)
                else:
                    p_crop = np.zeros((2, dtm.shape[0], dtm.shape[1]), dtype=np.float32)
                p_full = np.concatenate([raw_feats[:NUM_TERRAIN_CHANNELS], p_crop], axis=0)
                p_std_feats = (p_full - p_mean) / p_std
                p_prob_map, p_cls_map, _, p_tile_prob = predict_tile_with_d4_tta(
                    p_model, p_std_feats, device, calibrator=p_cal
                )
                yy_p, xx_p = np.ogrid[: dtm.shape[0], : dtm.shape[1]]
                for c in cands:
                    cx_p = int(np.clip(round(c["easting"] - e0), 4, dtm.shape[1] - 5))
                    cy_p = int(np.clip(round(n1 - c["northing"]), 4, dtm.shape[0] - 5))
                    r_p = int(np.clip(round(math.sqrt(max(float(c.get("area_m2", 400.0)), 36.0) / math.pi)), 6, 40))
                    disc_p = ((xx_p - cx_p) ** 2 + (yy_p - cy_p) ** 2) <= r_p**2
                    c["pond_unet_tile_prob"] = round(float(p_tile_prob), 4)
                    c["pond_unet_local_peak"] = round(float(np.percentile(p_prob_map[disc_p], 90.0)), 4)
                    c["pond_unet_local_mean"] = round(float(np.mean(p_prob_map[disc_p])), 4)
                    c_scores = [float(np.mean(p_cls_map[cid][disc_p])) for cid in (1, 2, 3)]
                    c["pond_unet_predicted_class"] = p_cls_names.get(int(np.argmax(c_scores)) + 1, "pond_lagoon_water_basin")
        for c in cands:
            c["calibrated_tile_prob"] = round(cal_tile_prob, 4)
            c["raw_tile_logit"] = round(raw_tile_logit, 4)
            c["window_bbox"] = [e0, n0, e1, n1]
            c["region_name"] = region_name
            if rgb is not None:
                cx_px = float((c["easting"] - e0) * (dtm.shape[1] / max(e1 - e0, 1.0)))
                cy_px = float((n1 - c["northing"]) * (dtm.shape[0] / max(n1 - n0, 1.0)))
                cm = compute_candidate_cropmark_metrics(rgb, cx_px=cx_px, cy_px=cy_px, area_m2=float(c.get("area_m2", 300.0)))
                c.update(cm)
        annotate_candidates_with_spatial_context(cands, osm_index, target_domain=target_domain)
        for c in cands:
            evaluate_concentric_ring_score(c)
        raw_candidates.extend(cands)
        if cands:
            window_records[key] = (raw_feats, prob_map, cands)

    deduped = nms_deduplicate_candidates(raw_candidates, merge_radius_m=28.0)
    known_hits = [c for c in deduped if c["context_category"] == "rediscovered_nhle_scheduled"]
    osm_arch_hits = [c for c in deduped if c["context_category"] == "rediscovered_osm_archaeology"]
    confounder_hits = [c for c in deduped if c["context_category"] == "flagged_modern_or_karst_confounder"]
    unidentified = [c for c in deduped if c["context_category"] == "unidentified_candidate"]

    diag_dir = output_dir / "diagnostics"
    saved_pngs: List[str] = []
    window_to_png: Dict[Tuple[int, int, int, int], str] = {}

    unid_windows = []
    known_windows = []
    for key, (raw_feats, prob_map, cands) in window_records.items():
        unid_c = [c for c in cands if c["context_category"] == "unidentified_candidate"]
        arch_c = [
            c
            for c in cands
            if c["context_category"] in {"rediscovered_nhle_scheduled", "rediscovered_osm_archaeology"}
        ]
        if unid_c:
            unid_windows.append((max(c["confidence"] for c in unid_c), key, raw_feats, prob_map, cands))
        if arch_c:
            known_windows.append((max(c["confidence"] for c in arch_c), key, raw_feats, prob_map, cands))

    unid_windows.sort(key=lambda item: item[0], reverse=True)
    known_windows.sort(key=lambda item: item[0], reverse=True)

    for idx, (best_conf, key, raw_feats, prob_map, cands) in enumerate(unid_windows[:6]):
        e0, e1, n0, n1 = key
        png_path = diag_dir / f"{region_name}_unidentified_{idx + 1:02d}_E{e0}_N{n0}_conf{int(best_conf * 1000)}.png"
        save_candidate_diagnostic_png(raw_feats, prob_map, e0, e1, n0, n1, cands, png_path)
        saved_pngs.append(str(png_path))
        window_to_png[key] = png_path.name

    for idx, (best_conf, key, raw_feats, prob_map, cands) in enumerate(known_windows[:4]):
        e0, e1, n0, n1 = key
        png_path = diag_dir / f"{region_name}_rediscovered_{idx + 1:02d}_E{e0}_N{n0}_conf{int(best_conf * 1000)}.png"
        save_candidate_diagnostic_png(raw_feats, prob_map, e0, e1, n0, n1, cands, png_path)
        saved_pngs.append(str(png_path))
        window_to_png.setdefault(key, png_path.name)

    for c in deduped:
        wb = c.get("window_bbox")
        if wb and len(wb) == 4:
            key = (wb[0], wb[2], wb[1], wb[3])
            if key in window_to_png:
                c["diagnostic_png"] = window_to_png[key]

    geojson_features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
            "properties": c,
        }
        for c in deduped
    ]
    geojson_path = output_dir / f"{region_name}_candidates.geojson"
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": geojson_features}, indent=2),
        encoding="utf-8",
    )

    area_km2 = round(((e_max - e_min) * (n_max - n_min)) / 1e6, 3)
    summary = {
        "region_name": region_name,
        "bbox_epsg27700": [e_min, n_min, e_max, n_max],
        "area_km2": area_km2,
        "tile_prob_threshold_applied": tile_prob_threshold,
        "pixel_prob_threshold_applied": pixel_prob_threshold,
        "osm_context_available": osm_index is not None,
        "windows_requested": len(coords),
        "windows_with_valid_1m_lidar": len(loaded_tiles),
        "windows_above_tile_threshold": windows_above_tile_th,
        "total_detections_after_nms": len(deduped),
        "rediscovered_scheduled_monuments_count": len(known_hits),
        "rediscovered_osm_archaeology_count": len(osm_arch_hits),
        "flagged_modern_or_karst_confounders_count": len(confounder_hits),
        "unidentified_candidates_count": len(unidentified),
        "unidentified_density_per_km2": round(len(unidentified) / max(area_km2, 1e-6), 3),
        "rediscovered_scheduled_monuments": known_hits,
        "rediscovered_osm_archaeology": osm_arch_hits,
        "flagged_modern_or_karst_confounders": confounder_hits,
        "unidentified_candidates": unidentified,
        "diagnostic_images": saved_pngs,
        "geojson_path": str(geojson_path),
    }
    report_path = output_dir / f"{region_name}_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
