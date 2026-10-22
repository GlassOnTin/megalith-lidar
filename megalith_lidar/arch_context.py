from __future__ import annotations

import base64
import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from megalith_lidar.catalog import (
    CLASS_BARROW_CAIRN,
    CLASS_CIRCLE_HENGE_STONE,
    CLASS_MEGALITHIC_TOMB_LONG_BARROW,
    MonumentCatalog,
)

COMPASS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]

NHLE_CLASS_LABELS = {
    CLASS_MEGALITHIC_TOMB_LONG_BARROW: "Neolithic long barrow / chambered tomb",
    CLASS_CIRCLE_HENGE_STONE: "Henge / stone circle / ceremonial alignment",
    CLASS_BARROW_CAIRN: "Bronze Age round barrow / cairn",
    4: "Scheduled enclosure / settlement / earthwork",
}


def compass_bearing_from_Site(cand_e: float, cand_n: float, mon_e: float, mon_n: float) -> str:
    """Return 16-wind compass direction from candidate (cand_e, cand_n) to monument (mon_e, mon_n)."""
    dx = mon_e - cand_e
    dy = mon_n - cand_n
    deg = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    idx = int(round(deg / 22.5)) % 16
    return COMPASS_16[idx]


def infer_landscape_character_area(e: float, n: float) -> str:
    """Return a concise British landscape/physiographic descriptor from EPSG:27700 coordinates."""
    if 415000 <= e <= 440000 and 475000 <= n <= 505000:
        return "Ure Valley & Mowbray / Masham fluvio-glacial terrace (North Yorkshire)"
    if 400000 <= e <= 435000 and 210000 <= n <= 245000:
        return "Cotswold oolitic limestone dip-slope (Gloucestershire / West Oxfordshire)"
    if 395000 <= e <= 430000 and 115000 <= n <= 155000:
        return "Wessex Chalk — Salisbury Plain, Wylye & Bourne valleys (Wiltshire)"
    if 475000 <= e <= 520000 and 100000 <= n <= 125000:
        return "South Downs Upper Chalk escarpment & dip-slope (West Sussex)"
    if 195000 <= e <= 230000 and 70000 <= n <= 105000:
        return "Bodmin Moor granite upland & moorland fringe (Cornwall)"
    if 250000 <= e <= 285000 and 60000 <= n <= 98000:
        return "Dartmoor granite upland & moorland fringe (Devon)"
    if 340000 <= e <= 375000 and 140000 <= n <= 165000:
        return "Mendip Carboniferous limestone plateau (Somerset)"
    if 270000 <= e <= 305000 and 130000 <= n <= 155000:
        return "Exmoor Devonian sandstone moorland (Somerset / North Devon)"
    if 335000 <= e <= 365000 and 505000 <= n <= 545000:
        return "Eden Valley & North Cumbrian upland fringe (Cumbria)"
    if 385000 <= e <= 425000 and 500000 <= n <= 640000:
        return "North Pennine & Northumberland sandstone/whin upland"
    if n >= 450000:
        return "Northern England upland / valley terrace landscape"
    if n <= 160000 and e <= 360000:
        return "South-West Peninsula upland / valley landscape"
    if n <= 190000 and e > 360000:
        return "Southern English Chalk & lowlands"
    return "English lowland / river terrace landscape"


class NHLESpatialContextExtractor:
    """Fast vectorized nearest-neighbour lookup over the 20,003-record NHLE Scheduled Monument catalog."""

    def __init__(self, catalog: MonumentCatalog) -> None:
        self.records = list(catalog.records)
        self.eastings = np.array([r.easting for r in self.records], dtype=np.float64)
        self.northings = np.array([r.northing for r in self.records], dtype=np.float64)
        self.classes = np.array([r.cls for r in self.records], dtype=np.int32)

    def query_context(
        self,
        e: float,
        n: float,
        max_radius_m: float = 4000.0,
        top_k: int = 4,
    ) -> Tuple[List[Dict], Dict]:
        dx = self.eastings - e
        dy = self.northings - n
        d2 = dx * dx + dy * dy

        within_5k = np.where(d2 <= 5000.0 * 5000.0)[0]
        within_2k = np.where(d2 <= 2000.0 * 2000.0)[0]

        prehist_mask_2k = np.isin(self.classes[within_2k], [1, 2, 3]) if len(within_2k) else np.array([], dtype=bool)
        prehist_mask_5k = np.isin(self.classes[within_5k], [1, 2, 3]) if len(within_5k) else np.array([], dtype=bool)

        cluster_summary = {
            "landscape_area": infer_landscape_character_area(e, n),
            "nhle_within_2km": int(len(within_2k)),
            "prehistoric_within_2km": int(np.sum(prehist_mask_2k)),
            "nhle_within_5km": int(len(within_5k)),
            "prehistoric_within_5km": int(np.sum(prehist_mask_5k)),
        }

        within_r = np.where(d2 <= max_radius_m * max_radius_m)[0]
        if len(within_r) == 0:
            # Take single absolute nearest monument in England so the user still gets a reference distance
            nearest_idx = int(np.argmin(d2))
            within_r = np.array([nearest_idx], dtype=np.int64)

        order = within_r[np.argsort(d2[within_r])[:top_k]]
        nearby_list: List[Dict] = []
        for idx in order:
            rec = self.records[int(idx)]
            dist_m = float(math.sqrt(d2[int(idx)]))
            bearing = compass_bearing_from_Site(e, n, rec.easting, rec.northing)
            nearby_list.append(
                {
                    "list_entry": int(rec.list_entry),
                    "name": rec.name,
                    "cls_id": int(rec.cls),
                    "cls_label": NHLE_CLASS_LABELS.get(int(rec.cls), "Scheduled monument"),
                    "distance_m": round(dist_m, 0),
                    "bearing": bearing,
                }
            )
        return nearby_list, cluster_summary


def build_deterministic_academic_context(cand: Dict, nearby_nhle: List[Dict], cluster: Dict) -> str:
    """
    Generate a rigorous, non-hallucinated academic archaeological & 1m LiDAR morphological
    assessment from exact NHLE spatial relationships and measured terrain observables.
    """
    cat = cand.get("context_category", "unidentified_candidate")
    cls_name = cand.get("predicted_class", "round_barrow_cairn")
    rel_m = float(cand.get("est_local_relief_m", 0.5))
    slope = float(cand.get("mean_slope_tangent", 0.08))
    slope_deg = math.degrees(math.atan(slope))
    area = float(cand.get("area_m2", 250.0))
    eq_diam = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
    elong = float(cand.get("elongation", 1.15))
    ndsm = float(cand.get("est_surface_prominence_m", 0.0))
    landscape = cluster.get("landscape_area", "English landscape")
    n2k = cluster.get("nhle_within_2km", 0)
    p2k = cluster.get("prehistoric_within_2km", 0)
    n5k = cluster.get("nhle_within_5km", 0)

    if cls_name == "megalithic_tomb_long_barrow":
        form_phrase = f"an elongated or oval mound ({elong:.2f}× axial elongation, {area:.0f} m² footprint, ~{eq_diam:.0f} m equivalent span)"
        arch_parallel = "ploughed-down Neolithic long barrow, oval barrow, or bank-and-ditch mortuary enclosure"
    elif cls_name == "stone_circle_henge_alignment":
        form_phrase = f"a ring-bank or annular earthwork ({area:.0f} m² footprint, ~{eq_diam:.0f} m diameter, {elong:.2f}× elongation)"
        arch_parallel = "Neolithic/Early Bronze Age henge bank, ring cairn, or enclosed cremation cemetery"
    else:
        form_phrase = f"a sub-circular mound ({elong:.2f}× elongation, {area:.0f} m² footprint, ~{eq_diam:.0f} m diameter)"
        arch_parallel = "Bronze Age bowl/bell barrow or upland funerary cairn"

    nearest_str = ""
    if nearby_nhle:
        top_n = nearby_nhle[:2]
        parts = [f"{m['name']} (NHLE #{m['list_entry']}, {int(m['distance_m'])} m {m['bearing']})" for m in top_n]
        nearest_str = " and ".join(parts)

    if cat == "rediscovered_nhle_scheduled" and cand.get("matched_monument"):
        mm = cand["matched_monument"]
        return (
            f"Situated within the {landscape}, this detection independently recovers the scheduled monument "
            f"'{mm['name']}' (NHLE #{mm['list_entry']}, centroid offset {mm['distance_m']:.1f} m) in a held-out test square. "
            f"In 1 m bare-earth LiDAR, the earthwork survives as {form_phrase} with {rel_m:+.2f} m of 16 m local relief "
            f"and a mean scarp gradient of {slope:.3f} ({slope_deg:.1f}°; surface canopy nDSM {ndsm:.2f} m). "
            f"The site lies within a recorded monument cluster containing {n2k} scheduled monuments ({p2k} prehistoric) "
            f"within 2 km and {n5k} within 5 km."
        )

    if cat == "rediscovered_osm_archaeology" and cand.get("osm_archaeology_matches"):
        om = cand["osm_archaeology_matches"][0]
        return (
            f"Located in the {landscape}, this detection corresponds to an unscheduled archaeological feature "
            f"recorded in OpenStreetMap ({om.get('name') or om.get('subtype')}, {om['osm_id']}) outside Historic England's "
            f"Scheduled Monument polygon register. The 1 m DTM resolves {form_phrase} with {rel_m:+.2f} m local relief "
            f"and {slope:.3f} ({slope_deg:.1f}°) mean scarp slope (nDSM {ndsm:.2f} m). "
            f"Nearest scheduled parallels include {nearest_str or 'regional barrow groups'} ({n5k} scheduled monuments within 5 km)."
        )

    if cat == "flagged_modern_or_karst_confounder":
        cm = (cand.get("osm_confounder_matches") or [{}])[0]
        reason = cm.get("name") or cm.get("subtype") or "modern/geological confounder"
        return (
            f"Located in the {landscape}, this {area:.0f} m² topographic anomaly ({rel_m:+.2f} m local relief, "
            f"{slope:.3f} / {slope_deg:.1f}° mean scarp slope, nDSM {ndsm:.2f} m) mimics prehistoric earthwork geometry "
            f"in 1 m LiDAR but is automatically separated by the spatial/geometric filter as '{reason}'. "
            f"Steep machine-graded scarps, negative-relief extraction pits, or mapped modern infrastructure distinguish it "
            f"from weathered prehistoric mounds despite proximity to {nearest_str or 'regional monuments'}."
        )

    # Unidentified candidate
    cluster_clause = (
        f"within an active monument landscape ({n2k} scheduled monuments within 2 km and {n5k} within 5 km, "
        f"closest being {nearest_str})"
        if n5k > 0 and nearest_str
        else f"in a lightly scheduled sector of the {landscape} (nearest scheduled monument: {nearest_str})"
    )
    canopy_clause = (
        f"Bare-earth/surface comparison shows minimal above-ground vegetation or structure (nDSM {ndsm:.2f} m), "
        f"confirming that the anomaly is a genuine ground-surface earthwork rather than scrub canopy."
        if ndsm < 0.25
        else f"Moderate surface roughness (nDSM {ndsm:.2f} m) warrants checking for low gorse/brash cover or stone cairn exposure."
    )
    return (
        f"Situated in the {landscape} {cluster_clause}, the 1 m DTM isolates {form_phrase} rising {rel_m:+.2f} m "
        f"above its 16 m local trend surface with weathered scarp gradients of {slope:.3f} ({slope_deg:.1f}°), "
        f"morphologically consistent with a {arch_parallel}. {canopy_clause} "
        f"Recommended HER follow-up: cross-check First Edition 25-inch OS mapping and historic aerial photography "
        f"to rule out post-medieval marl/quarry spoil or field-clearance mounds before ground-truthing."
    )


def query_local_vlm_academic_context(
    cand: Dict,
    nearby_nhle: List[Dict],
    cluster: Dict,
    diag_png_path: Optional[Path] = None,
    endpoint: str = "http://localhost:8090/v1/chat/completions",
    timeout: float = 18.0,
) -> Optional[str]:
    """
    Query the local Qwen3.8-Flash-Next vision-language server (`localhost:8090`) with the exact
    NHLE spatial fact sheet + 4-panel 1m LiDAR PNG to produce a 3-sentence academic assessment.
    """
    nearest_lines = [
        f"- {m['name']} (NHLE #{m['list_entry']}, {m['cls_label']}): {int(m['distance_m'])} m {m['bearing']}"
        for m in nearby_nhle[:3]
    ]
    nearest_block = "\n".join(nearest_lines) if nearest_lines else "- None within 4 km"

    slope = float(cand.get("mean_slope_tangent", 0.08))
    slope_deg = math.degrees(math.atan(slope))
    prompt_text = (
        f"Site OS Grid Ref: {cand.get('os_grid_ref')} (BNG E {cand.get('easting'):.0f}, N {cand.get('northing'):.0f})\n"
        f"Landscape Character Area: {cluster.get('landscape_area')}\n"
        f"Detection Category: {cand.get('context_category')}\n"
        f"Model Nearest Form Class: {cand.get('predicted_class')} (score={cand.get('confidence'):.3f}, calibrated tile p={cand.get('calibrated_tile_prob', 0.0):.3f})\n"
        f"1m LiDAR Measurements: 16m bare-earth relief = {cand.get('est_local_relief_m'):+.2f} m, "
        f"mean scarp slope = {slope:.3f} ({slope_deg:.1f} deg), footprint = {cand.get('area_m2')} m2, "
        f"axial elongation = {cand.get('elongation')}x, surface canopy nDSM = {cand.get('est_surface_prominence_m')} m.\n"
        f"Scheduled Monuments within 2 km: {cluster.get('nhle_within_2km')} ({cluster.get('prehistoric_within_2km')} prehistoric); "
        f"within 5 km: {cluster.get('nhle_within_5km')}.\n"
        f"Nearest Historic England Scheduled Monuments:\n{nearest_block}\n\n"
        "Write a concise 3-sentence academic landscape-archaeology and 1m LiDAR morphological assessment of this site. "
        "Sentence 1: Landscape setting and spatial relationship to the nearest named NHLE Scheduled Monuments (cite their exact distances and NHLE IDs). "
        "Sentence 2: Morphological reading of the 1m LiDAR bare-earth relief, slope angle, footprint, and nDSM canopy (and the 4-panel image if attached). "
        "Sentence 3: Specific archaeological interpretation and what historic map / HER / field check should be performed to distinguish a prehistoric monument from post-medieval agricultural or extractive activity. "
        "Use plain, rigorous British archaeological prose with no bullet points, no markdown headers, and no invented facts."
    )

    user_content: object
    if diag_png_path is not None and diag_png_path.exists():
        b64 = base64.b64encode(diag_png_path.read_bytes()).decode("ascii")
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]
    else:
        user_content = prompt_text

    payload = {
        "model": "default",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior British landscape archaeologist and LiDAR specialist. "
                    "You write concise, empirically grounded 3-sentence site assessments using only the supplied NHLE records and 1m LiDAR measurements."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 240,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if len(text) > 60:
                return " ".join(text.split())
    except Exception:
        return None
    return None


def enrich_candidates_with_archaeological_context(
    candidates: List[Dict],
    extractor: NHLESpatialContextExtractor,
    cache_path: Path,
    docs_diag_dir: Optional[Path] = None,
    max_vlm_calls: int = 25,
) -> int:
    """
    Populate `nearby_nhle_monuments`, `nhle_cluster_summary`, `arch_context_note`,
    and `arch_context_source` ('qwen3.8_vlm' or 'nhle_grounded_synthesis') on every candidate.
    """
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    vlm_calls = 0
    cache_dirty = False

    # Prioritize Strict 1% FPR unidentified candidates with PNGs for VLM calls first
    sorted_indices = sorted(
        range(len(candidates)),
        key=lambda i: (
            1 if (candidates[i].get("context_category") == "unidentified_candidate" and candidates[i].get("passes_fpr_0_01")) else 0,
            1 if candidates[i].get("diagnostic_png") else 0,
            float(candidates[i].get("confidence", 0.0)),
        ),
        reverse=True,
    )

    for idx in sorted_indices:
        cand = candidates[idx]
        e = float(cand.get("easting", 0.0))
        n = float(cand.get("northing", 0.0))
        cat = cand.get("context_category", "unidentified_candidate")
        ckey = f"{cand.get('os_grid_ref')}_{int(round(e))}_{int(round(n))}_{cat}"

        nearby_nhle, cluster = extractor.query_context(e, n)
        cand["nearby_nhle_monuments"] = nearby_nhle
        cand["nhle_cluster_summary"] = cluster

        cached_entry = cache.get(ckey)
        if cached_entry and cached_entry.get("note"):
            cand["arch_context_note"] = cached_entry["note"]
            cand["arch_context_source"] = cached_entry.get("source", "nhle_grounded_synthesis")
            continue

        is_priority_unid = (
            cat == "unidentified_candidate"
            and bool(cand.get("passes_fpr_0_01"))
        )
        if is_priority_unid and vlm_calls < max_vlm_calls:
            diag_p: Optional[Path] = None
            if vlm_calls < 8 and docs_diag_dir is not None and cand.get("diagnostic_png"):
                p_try = Path(docs_diag_dir) / str(cand["diagnostic_png"])
                if p_try.exists():
                    diag_p = p_try
            vlm_note = query_local_vlm_academic_context(cand, nearby_nhle, cluster, diag_png_path=diag_p)
            vlm_calls += 1
            if vlm_note:
                src = "qwen3.8_vlm_vision" if diag_p is not None else "qwen3.8_llm"
                cand["arch_context_note"] = vlm_note
                cand["arch_context_source"] = src
                cache[ckey] = {"note": vlm_note, "source": src}
                cache_dirty = True
                continue

        det_note = build_deterministic_academic_context(cand, nearby_nhle, cluster)
        cand["arch_context_note"] = det_note
        cand["arch_context_source"] = "nhle_grounded_synthesis"

    if cache_dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return vlm_calls
