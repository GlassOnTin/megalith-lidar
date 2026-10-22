from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

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
    elif cls_name == "roman_villa_building":
        form_phrase = f"a rectilinear building platform or courtyard enclosure ({area:.0f} m² footprint, ~{eq_diam:.0f} m span, {elong:.2f}× axial elongation)"
        arch_parallel = "Romano-British villa range, aisled building platform, or temple temenos earthwork"
    elif cls_name == "roman_road_agger_trackway":
        form_phrase = f"a cambered linear embankment or hollow-way terrace ({elong:.2f}× axial elongation, {area:.0f} m² footprint)"
        arch_parallel = "Roman road agger with flanking quarry-scoop ditches or ancient ridgeway trackway"
    elif cls_name == "roman_fort_camp_settlement":
        form_phrase = f"a rectilinear rampart-and-ditch enclosure or settlement earthwork ({area:.0f} m² footprint, ~{eq_diam:.0f} m span, {elong:.2f}× elongation)"
        arch_parallel = "Roman fortlet, marching-camp vallum, signal station, or Romano-British enclosed farmstead"
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


def make_zoomed_crop_b64(png_path: Path) -> str:
    """
    Create a 3x-magnified 80m x 80m close-up crop of [Panel 1: Bare-Earth Slope | Panel 2: 16m Local Relief]
    centered on the detection circle so the VLM can clearly inspect corners, scarps, and interior hollows.
    """
    im = Image.open(png_path)
    arr = np.array(im)[:, :256, :]
    yellow = (arr[:, :, 0] > 230) & (arr[:, :, 1] > 190) & (arr[:, :, 2] < 30)
    green = (arr[:, :, 0] < 30) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 90)
    ys, xs = np.where(yellow | green)
    if len(xs) > 0:
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
    else:
        cx, cy = 128, 128
    r = 40
    x0, x1 = max(0, min(256 - 2 * r, cx - r)), min(256, max(2 * r, cx + r))
    y0, y1 = max(0, min(256 - 2 * r, cy - r)), min(256, max(2 * r, cy + r))
    p1_crop = im.crop((x0, y0, x1, y1)).resize((256, 256), Image.Resampling.NEAREST)
    p2_crop = im.crop((260 + x0, y0, 260 + x1, y1)).resize((256, 256), Image.Resampling.NEAREST)
    out = Image.new("RGB", (516, 256), (20, 20, 24))
    out.paste(p1_crop, (0, 0))
    out.paste(p2_crop, (260, 0))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_VLM_OBVIOUS_CONFOUNDER_PAT = (
    "modern military emplacement",
    "covered reservoir",
    "modern reservoir",
    "modern utility",
    "modern spoil",
    "modern forestry spoil",
    "field clearance ring",
    "post-medieval chalk extraction",
    "post-medieval marl pit",
    "extractive pit",
    "quarry scoop",
    "fighter pen",
)


def classify_vlm_note_downrating(
    note: str,
    shape_tag: str,
    cand: Dict,
) -> Tuple[str, float, Optional[str]]:
    """
    Evaluate Qwen3.8's visual/text assessment and physical morphology to determine:
      - `verdict_tier`: 'PLAUSIBLE_MONUMENT' (1.0x), 'UNCERTAIN_OR_DEGRADED' (0.82x),
        or 'OBVIOUS_FALSE_POSITIVE' (0.35x and routed to Tab 3 confounders).
      - `multiplier`: factor applied to `raw_unet_confidence` to compute `confidence`.
      - `reason`: concise explanation when downrated or routed to Tab 3.
    """
    low = (note or "").lower()
    is_roman = cand.get("target_domain") == "roman"
    pred_cls = cand.get("predicted_class", "")
    elong = float(cand.get("elongation", 1.15))

    if not is_roman and shape_tag in {"RECTILINEAR_OR_U_BUND", "OBVIOUS_MODERN_CONFOUNDER"}:
        return (
            "OBVIOUS_FALSE_POSITIVE",
            0.35,
            "VLM Visual Audit: Square/rectilinear embankment or U-shaped cut-and-fill bund",
        )

    for phrase in _VLM_OBVIOUS_CONFOUNDER_PAT:
        if phrase in low and ("more likely" in low or "strongly indicates" in low or "likely represents" in low):
            # If the note explicitly concludes it is a modern military emplacement, reservoir, or spoil/extraction pit
            if phrase in {"modern military emplacement", "covered reservoir", "modern reservoir", "modern utility", "fighter pen"}:
                return (
                    "OBVIOUS_FALSE_POSITIVE",
                    0.35,
                    f"VLM Assessment: Flagged as {phrase}",
                )

    # Check if VLM notes a contradiction between long-barrow prediction and circular/low-amplitude morphology,
    # or concludes it is likely natural scree/gravel hummock/forestry spoil
    if (
        (
            pred_cls == "megalithic_tomb_long_barrow"
            and elong < 1.35
            and (
                "despite the model" in low
                or "contradicts" in low
                or "rejected" in low
                or "lacking the axial elongation" in low
                or "lacking any rectilinear" in low
                or "absence of a defined long-barrow" in low
                or "although the model" in low
            )
        )
        or "natural fluvio-glacial" in low
        or "natural chalk scree" in low
        or "natural limestone scree" in low
        or "natural granite outcrop" in low
        or "modern forestry" in low
        or "field clearance ring" in low
        or "more likely a post-medieval" in low
    ):
        return (
            "UNCERTAIN_OR_DEGRADED",
            0.84,
            "VLM Assessment: Downrated (circular/low-relief morphology vs long-barrow class or likely natural/post-medieval feature)",
        )

    return ("PLAUSIBLE_MONUMENT", 1.0, None)


def query_local_vlm_academic_context(
    cand: Dict,
    nearby_nhle: List[Dict],
    cluster: Dict,
    diag_png_path: Optional[Path] = None,
    endpoint: str = "http://localhost:8090/v1/chat/completions",
    timeout: float = 25.0,
) -> Optional[Tuple[str, str]]:
    """
    Query the local Qwen3.8-Flash-Next vision-language server (`localhost:8090`) with the exact
    NHLE spatial fact sheet + 3x-magnified 80m x 80m 1m LiDAR close-up to produce both a shape/confounder
    verdict ('RECTILINEAR_OR_U_BUND', 'OBVIOUS_MODERN_CONFOUNDER', or 'ROUNDED_MOUND_OR_RING') and a
    3-sentence academic assessment.
    """
    nearest_lines = [
        f"- {m['name']} (NHLE #{m['list_entry']}, {m['cls_label']}): located {int(m['distance_m'])} m {m['bearing']} of this candidate"
        for m in nearby_nhle[:3]
    ]
    nearest_block = "\n".join(nearest_lines) if nearest_lines else "- None within 4 km"

    slope = float(cand.get("mean_slope_tangent", 0.08))
    slope_deg = math.degrees(math.atan(slope))
    has_img = diag_png_path is not None and diag_png_path.exists()
    shape_prefix_instr = (
        "First output one tag: [SHAPE: RECTILINEAR_OR_U_BUND] (if the earthwork inside the circle in the 80m x 80m close-up "
        "has straight/square 90-degree corners, a flat rectangular reservoir/building platform, or a U-shaped bank wrapping a dark blue sunken pit), "
        "[SHAPE: OBVIOUS_MODERN_CONFOUNDER] (if it is clearly a modern road/track embankment, quarry scoop, military gun-pit, or utility feature), "
        "or [SHAPE: ROUNDED_MOUND_OR_RING] (if it is a smooth circular/oval mound or ring). "
        if has_img
        else ""
    )
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
        f"{shape_prefix_instr}"
        "Then write a concise 3-sentence (maximum 90 words total) academic landscape-archaeology and 1m LiDAR morphological assessment of this site. "
        "Sentence 1: Landscape setting and spatial relationship to the nearest named NHLE Scheduled Monuments (cite their exact distances and NHLE IDs). "
        "Sentence 2: Critical morphological reading of the 1m LiDAR close-up (Left=Bare-Earth Slope, Right=16m Local Relief where yellow is raised and blue is sunken) and measurements. "
        "Sentence 3: Specific archaeological or modern-confounder interpretation and what historic map / HER / field check is required. "
        "Use plain, rigorous British archaeological prose with no bullet points, no markdown headers, and no invented facts."
    )

    user_content: object
    if has_img and diag_png_path is not None:
        b64 = make_zoomed_crop_b64(diag_png_path)
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
                    "You are a senior British landscape archaeologist and critical 1m LiDAR specialist inspecting a 3x-magnified 80m x 80m close-up "
                    "(Left=Bare-Earth Slope where dark is steep scarp; Right=16m Local Relief where yellow is raised bank/mound and blue is sunken pit/hollow). "
                    "You write concise, complete 3-sentence site assessments (under 90 words) using only the supplied NHLE records and 1m LiDAR imagery/measurements."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 420,
        "temperature": 0.15,
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
            raw_text = " ".join((data["choices"][0]["message"].get("content") or "").strip().split())
            shape_tag = "ROUNDED_MOUND_OR_RING"
            if "[SHAPE: RECTILINEAR_OR_U_BUND]" in raw_text:
                shape_tag = "RECTILINEAR_OR_U_BUND"
                raw_text = raw_text.replace("[SHAPE: RECTILINEAR_OR_U_BUND]", "").strip()
            elif "[SHAPE: OBVIOUS_MODERN_CONFOUNDER]" in raw_text:
                shape_tag = "OBVIOUS_MODERN_CONFOUNDER"
                raw_text = raw_text.replace("[SHAPE: OBVIOUS_MODERN_CONFOUNDER]", "").strip()
            elif "[SHAPE: ROUNDED_MOUND_OR_RING]" in raw_text:
                shape_tag = "ROUNDED_MOUND_OR_RING"
                raw_text = raw_text.replace("[SHAPE: ROUNDED_MOUND_OR_RING]", "").strip()
            if len(raw_text) > 60 and raw_text.endswith((".", "!", "?")):
                return raw_text, shape_tag
    except Exception:
        return None
    return None


def enrich_candidates_with_archaeological_context(
    candidates: List[Dict],
    extractor: NHLESpatialContextExtractor,
    cache_path: Path,
    docs_diag_dir: Optional[Path] = None,
    max_vlm_calls: int = 80,
) -> int:
    """
    Populate `nearby_nhle_monuments`, `nhle_cluster_summary`, `arch_context_note`,
    `arch_context_source` ('qwen3.8_vlm_vision', 'qwen3.8_llm', or 'nhle_grounded_synthesis'),
    `raw_unet_confidence`, and `vlm_verdict_tier` on every candidate.
    Downrates candidates where Qwen3.8 identifies obvious false-positive morphology or
    class-vs-shape contradiction, and routes obvious false positives to `flagged_modern_or_karst_confounder`.
    """
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    vlm_calls = 0
    cache_dirty = False

    def _apply_vlm_verdict(c: Dict, note: str, shape_tag: str) -> None:
        raw_conf = float(c.get("raw_unet_confidence", c.get("confidence", 0.0)))
        c["raw_unet_confidence"] = round(raw_conf, 4)
        tier, mult, reason = classify_vlm_note_downrating(note, shape_tag, c)
        c["vlm_verdict_tier"] = tier
        c["vlm_downrate_reason"] = reason
        if c.get("context_category") in {"unidentified_candidate", "flagged_modern_or_karst_confounder"}:
            c["confidence"] = round(raw_conf * mult, 4)
        if tier == "OBVIOUS_FALSE_POSITIVE" and c.get("context_category") == "unidentified_candidate":
            c["context_category"] = "flagged_modern_or_karst_confounder"
            confs = list(c.get("osm_confounder_matches") or [])
            confs.insert(
                0,
                {
                    "osm_id": "vlm/visual_audit",
                    "subtype": "vlm_obvious_false_positive",
                    "name": reason or "VLM Visual Audit: Obvious modern/rectilinear confounder",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                },
            )
            c["osm_confounder_matches"] = confs

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
        ckey = f"v2_{cand.get('os_grid_ref')}_{int(round(e))}_{int(round(n))}"

        nearby_nhle, cluster = extractor.query_context(e, n)
        cand["nearby_nhle_monuments"] = nearby_nhle
        cand["nhle_cluster_summary"] = cluster

        cached_entry = cache.get(ckey)
        if cached_entry and cached_entry.get("note") and str(cached_entry["note"]).strip().endswith((".", "!", "?")):
            cand["arch_context_note"] = cached_entry["note"]
            cand["arch_context_source"] = cached_entry.get("source", "nhle_grounded_synthesis")
            cand["vlm_shape_verdict"] = cached_entry.get("shape_tag", "ROUNDED_MOUND_OR_RING")
            _apply_vlm_verdict(cand, cand["arch_context_note"], cand["vlm_shape_verdict"])
            continue

        is_priority_unid = (
            cat == "unidentified_candidate"
            and bool(cand.get("passes_fpr_0_01"))
        )
        if is_priority_unid and vlm_calls < max_vlm_calls:
            diag_p: Optional[Path] = None
            if docs_diag_dir is not None and cand.get("diagnostic_png"):
                p_try = Path(docs_diag_dir) / str(cand["diagnostic_png"])
                if p_try.exists():
                    diag_p = p_try
            res_vlm = query_local_vlm_academic_context(cand, nearby_nhle, cluster, diag_png_path=diag_p)
            vlm_calls += 1
            if res_vlm:
                vlm_note, shape_tag = res_vlm
                src = "qwen3.8_vlm_vision" if diag_p is not None else "qwen3.8_llm"
                cand["arch_context_note"] = vlm_note
                cand["arch_context_source"] = src
                cand["vlm_shape_verdict"] = shape_tag
                _apply_vlm_verdict(cand, vlm_note, shape_tag)
                cache[ckey] = {"note": vlm_note, "source": src, "shape_tag": shape_tag}
                cache_dirty = True
                continue

        det_note = build_deterministic_academic_context(cand, nearby_nhle, cluster)
        cand["arch_context_note"] = det_note
        cand["arch_context_source"] = "nhle_grounded_synthesis"
        cand["raw_unet_confidence"] = round(float(cand.get("raw_unet_confidence", cand.get("confidence", 0.0))), 4)
        cand["vlm_verdict_tier"] = "PLAUSIBLE_MONUMENT"

    if cache_dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return vlm_calls
