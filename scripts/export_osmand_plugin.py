#!/usr/bin/env python3
"""Export OsmAnd GPX waypoints and one-tap OsmAnd Custom Plugin (.osf) package.

Produces:
  1. docs/megalith_lidar_osmand.gpx
     Standalone OsmAnd / Garmin / Organic Maps / QGIS GPX file containing all
     unidentified Prehistoric, Iron Age, and Roman monument candidates with
     OsmAnd colour/icon groups, earthwork walkability telemetry, 1888 OS status,
     and clickable deep-links back to the 7-panel LiDAR web inspector.
  2. docs/megalith-lidar.osf
     One-tap OsmAnd Custom Plugin archive (ZIP format, SettingsHelper VERSION=3)
     bundling:
       - Custom plugin metadata (pluginId: "uk.megalith.lidar")
       - Categorised monument Favourites (/favourites.gpx)
       - Searchable GPX tracks (/tracks/UK_Megalith_LiDAR_Candidates.gpx and
         /tracks/UK_Roman_and_Ancient_Roads.gpx)
       - Custom Online Tile Sources (UK 1888–1913 OS 6-Inch Map & Digital Atlas
         of the Roman Empire DARE)
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from xml.sax.saxutils import escape
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megalith_lidar.arch_context import _VLM_HISTORIC_OS_ANTIQUITY_PHRASES


def has_historic_os_antiquity_evidence(note: str) -> bool:
    low = (note or "").lower()
    return any(p in low for p in _VLM_HISTORIC_OS_ANTIQUITY_PHRASES)

SITE_BASE_URL = "https://glassontin.github.io/megalith-lidar/"

CLASS_LABELS: Dict[str, str] = {
    "megalithic_tomb_long_barrow": "Long barrow / chambered mound",
    "stone_circle_henge_alignment": "Henge / banked circle / alignment",
    "round_barrow_cairn": "Round barrow / cairn mound",
    "roman_villa_building": "Roman villa / building platform",
    "roman_road_agger_trackway": "Roman road agger / ancient trackway",
    "roman_fort_camp_settlement": "Roman fort / camp / RB settlement",
    "hut_circle_pound_settlement": "Hut circle / pound settlement",
    "iron_age_hillfort_banjo_enclosure": "Iron Age hillfort / banjo enclosure",
    "cross_ridge_dyke_field_system": "Cross-ridge dyke / prehistoric field system",
}

SHORT_CLASS_LABELS: Dict[str, str] = {
    "megalithic_tomb_long_barrow": "Long Barrow",
    "stone_circle_henge_alignment": "Henge / Circle",
    "round_barrow_cairn": "Round Barrow",
    "roman_villa_building": "Roman Villa / Platform",
    "roman_road_agger_trackway": "Roman Agger",
    "roman_fort_camp_settlement": "Roman Camp / Fort",
    "hut_circle_pound_settlement": "Hut Circle / Settlement",
    "iron_age_hillfort_banjo_enclosure": "Iron Age Enclosure",
    "cross_ridge_dyke_field_system": "Cross-Ridge Dyke",
}

OSMAND_GROUPS: List[Dict[str, str]] = [
    {
        "name": "1888 OS Confirmed Antiquities ★",
        "color": "#15803d",
        "icon": "special_star",
        "background": "circle",
    },
    {
        "name": "Prehistoric Candidates ●",
        "color": "#d97706",
        "icon": "special_monument",
        "background": "circle",
    },
    {
        "name": "Iron Age & Settlement ◆",
        "color": "#0284c7",
        "icon": "special_arch",
        "background": "octagon",
    },
    {
        "name": "Roman Candidates ■",
        "color": "#7e22ce",
        "icon": "special_house",
        "background": "square",
    },
]

GROUP_BY_NAME: Dict[str, Dict[str, str]] = {g["name"]: g for g in OSMAND_GROUPS}


def _clean_grid_id(os_grid_ref: str) -> str:
    return "".join(str(os_grid_ref or "").upper().split())


def _walkability_summary(p: Dict[str, Any]) -> str:
    relief = abs(float(p.get("est_local_relief_m") or 0.0))
    has_crop = bool(p.get("has_distinct_cropmark"))
    if relief >= 0.75:
        earthwork = f"Prominent earthwork ({float(p.get('est_local_relief_m') or 0.0):+.2f} m relief — visible on foot)"
    elif relief >= 0.35:
        earthwork = f"Moderate earthwork ({float(p.get('est_local_relief_m') or 0.0):+.2f} m low-angle bank/mound)"
    elif has_crop:
        earthwork = f"Subtle plough-levelled / cropmark site ({float(p.get('est_local_relief_m') or 0.0):+.2f} m relief)"
    else:
        earthwork = f"Low-relief micro-topography ({float(p.get('est_local_relief_m') or 0.0):+.2f} m relief)"

    hints = " ".join(str(h) for h in (p.get("environmental_confounder_hints") or [])).lower()
    note_low = str(p.get("arch_context_note") or "").lower()
    if "canopy" in hints or "woodland" in hints or "plantation" in note_low or "tree cover" in note_low:
        canopy = "Woodland / scrub cover noted in nDSM"
    else:
        canopy = "Open ground / low canopy in 1 m LiDAR"
    return f"{earthwork} · {canopy}"


def _classify_group(p: Dict[str, Any], domain: str) -> str:
    note = str(p.get("arch_context_note") or "")
    status = str(p.get("user_survey_status") or "")
    tier = str(p.get("vlm_downrating_tier") or "")
    if (
        tier == "REDISCOVERED_1888_ANTIQUITY"
        or has_historic_os_antiquity_evidence(note)
        or status in ("field_verified_monument", "historic_survey_confirmed")
    ):
        return "1888 OS Confirmed Antiquities ★"
    if domain == "roman":
        return "Roman Candidates ■"
    if domain == "enclosure":
        return "Iron Age & Settlement ◆"
    return "Prehistoric Candidates ●"


def _load_candidates(docs_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    ann_by_grid: Dict[str, Dict[str, Any]] = {}
    ann_path = docs_dir / "community_annotations.json"
    if ann_path.exists():
        try:
            ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
            for item in ann_data.get("annotations", []):
                gref = _clean_grid_id(item.get("os_grid_ref", ""))
                if gref:
                    ann_by_grid[gref] = item
        except Exception:
            pass

    domain_files = [
        ("prehistoric", docs_dir / "candidates.geojson"),
        ("enclosure", docs_dir / "enclosure_candidates.geojson"),
        ("roman", docs_dir / "roman_candidates.geojson"),
    ]
    out: List[Tuple[str, Dict[str, Any]]] = []
    seen_keys = set()
    for domain, path in domain_files:
        if not path.exists():
            continue
        gj = json.loads(path.read_text(encoding="utf-8"))
        for feat in gj.get("features", []):
            p = dict(feat.get("properties") or {})
            if p.get("context_category") != "unidentified_candidate":
                continue
            gref_id = _clean_grid_id(p.get("os_grid_ref", ""))
            ann = ann_by_grid.get(gref_id)
            if ann:
                st = str(ann.get("status") or "")
                if st == "field_rejected_confounder":
                    continue
                p["user_survey_status"] = st
                p["user_survey_comment"] = ann.get("comment") or ""
                p["user_survey_observer"] = ann.get("observer") or ""
            dedup_key = (domain, gref_id or f"{p.get('easting')}_{p.get('northing')}")
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            out.append((domain, p))

    # Sort: 1888 OS confirmed antiquities first, then strict 1% FPR, then confidence descending
    out.sort(
        key=lambda item: (
            1 if _classify_group(item[1], item[0]) == "1888 OS Confirmed Antiquities ★" else 0,
            1 if item[1].get("passes_fpr_0_01") else 0,
            float(item[1].get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return out


def build_candidates_gpx_xml(candidates: List[Tuple[str, Dict[str, Any]]], updated_utc: str) -> str:
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>',
        '<gpx version="1.1" creator="UK 1 m LiDAR &amp; Aerial Cropmark Monument Survey (https://glassontin.github.io/megalith-lidar/)" '
        'xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:osmand="https://osmand.net" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">',
        "  <metadata>",
        "    <name>UK 1 m LiDAR Monument Survey Candidates</name>",
        "    <desc>Unrecorded Prehistoric, Iron Age, and Roman earthwork candidates detected by 1 m LiDAR &amp; 16 m aerial cropmarks across England, cross-checked against 1888 OS 6-Inch maps, Historic England NHLE, OpenStreetMap, and Megalithic.co.uk.</desc>",
        f'    <link href="{SITE_BASE_URL}"><text>UK 1 m LiDAR Interactive Web Inspector</text></link>',
        f"    <time>{escape(updated_utc)}</time>",
        "  </metadata>",
    ]

    for domain, p in candidates:
        lat = float(p.get("lat") or 0.0)
        lon = float(p.get("lon") or 0.0)
        gref = str(p.get("os_grid_ref") or "").strip()
        gref_id = _clean_grid_id(gref)
        cls_key = str(p.get("predicted_class") or "")
        cls_full = CLASS_LABELS.get(cls_key, cls_key.replace("_", " "))
        cls_short = SHORT_CLASS_LABELS.get(cls_key, cls_full)
        conf = float(p.get("confidence") or 0.0)
        relief = float(p.get("est_local_relief_m") or 0.0)
        area_m2 = int(round(float(p.get("area_m2") or 0.0)))
        county = str(p.get("county") or "England")
        easting = int(round(float(p.get("easting") or 0.0)))
        northing = int(round(float(p.get("northing") or 0.0)))
        group_name = _classify_group(p, domain)
        grp = GROUP_BY_NAME[group_name]
        is_os1888 = group_name == "1888 OS Confirmed Antiquities ★"

        prefix = "★ " if is_os1888 else ""
        wpt_name = f"{prefix}{gref} · {cls_short} ({conf:.3f})"

        deep_link = f"{SITE_BASE_URL}#site={gref_id}"
        nls_url = f"https://maps.nls.uk/geo/explore/#zoom=17&lat={lat:.5f}&lon={lon:.5f}&layers=6&b=1&o=100"
        diag_png = str(p.get("diagnostic_png") or "").strip()
        diag_url = f"{SITE_BASE_URL}docs/diagnostics/{diag_png}" if diag_png else ""

        mp_matches = p.get("megalithic_portal_matches") or []
        top_mp = mp_matches[0] if mp_matches else None

        op_tier = "Strict 1% FPR (LR+ 61.1×)" if p.get("passes_fpr_0_01") else "Standard 5% FPR (LR+ 16.4×)"
        walkability = _walkability_summary(p)

        desc_parts = [
            f"{cls_full} ({county}) | Score: {conf:.3f} [{op_tier}]",
            f"Relief: {relief:+.2f} m | Area: {area_m2} m² | BNG E {easting}, N {northing} ({lat:.5f}, {lon:.5f})",
            f"Field Walkability: {walkability}",
        ]
        if is_os1888:
            desc_parts.append(
                "1888 OS 6-Inch Map: Explicitly marked 'Tumulus' / 'Camp' / Hachured Antiquity (Unrecorded in NHLE & OSM)"
            )
        if top_mp:
            desc_parts.append(
                f"Nearest Megalithic.co.uk: {top_mp.get('name')} (#{top_mp.get('sid')}, {top_mp.get('site_type')}, {top_mp.get('distance_m')} m) — {top_mp.get('url')}"
            )
        if p.get("user_survey_comment"):
            obs = f" ({p.get('user_survey_observer')})" if p.get("user_survey_observer") else ""
            desc_parts.append(f"Field Survey Log{obs}: {p.get('user_survey_comment')}")
        if p.get("arch_context_note"):
            desc_parts.append(f"Assessment: {p.get('arch_context_note')}")
        desc_parts.append(f"7-Panel LiDAR & Cropmark Inspector: {deep_link}")
        if diag_url:
            desc_parts.append(f"4-Panel 1 m LiDAR Diagnostic PNG: {diag_url}")
        desc_parts.append(f"1888 OS 6-Inch Map (NLS): {nls_url}")

        desc_text = "\n".join(desc_parts)

        lines.append(f'  <wpt lat="{lat:.6f}" lon="{lon:.6f}">')
        lines.append(f"    <name>{escape(wpt_name)}</name>")
        lines.append(f"    <desc>{escape(desc_text)}</desc>")
        lines.append(f'    <link href="{escape(deep_link)}"><text>Open 7-Panel LiDAR Inspector ({escape(gref)})</text></link>')
        if diag_url:
            lines.append(f'    <link href="{escape(diag_url)}"><text>4-Panel 1 m LiDAR Diagnostic PNG</text></link>')
        lines.append(f"    <type>{escape(group_name)}</type>")
        lines.append("    <extensions>")
        lines.append(f"      <osmand:color>{escape(grp['color'])}</osmand:color>")
        lines.append(f"      <osmand:icon>{escape(grp['icon'])}</osmand:icon>")
        lines.append(f"      <osmand:background>{escape(grp['background'])}</osmand:background>")
        lines.append("    </extensions>")
        lines.append("  </wpt>")

    lines.append("  <extensions>")
    lines.append("    <osmand:points_groups>")
    for grp in OSMAND_GROUPS:
        lines.append(
            f'      <group name="{escape(grp["name"])}" '
            f'color="{escape(grp["color"])}" '
            f'icon="{escape(grp["icon"])}" '
            f'background="{escape(grp["background"])}" />'
        )
    lines.append("    </osmand:points_groups>")
    lines.append("  </extensions>")
    lines.append("</gpx>")
    return "\n".join(lines) + "\n"


def _to_xy_km(pt: List[float]) -> Tuple[float, float]:
    return (float(pt[0]) * 69.0, float(pt[1]) * 111.0)


def _unit_vec(p1: List[float], p2: List[float]) -> Tuple[float, float]:
    import math

    dx = (float(p2[0]) - float(p1[0])) * 69.0
    dy = (float(p2[1]) - float(p1[1])) * 111.0
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)


def _angle_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    import math

    if v1 == (0.0, 0.0) or v2 == (0.0, 0.0):
        return 0.0
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
    return math.degrees(math.acos(dot))


def _curvature_aware_rdp(
    pts: List[List[float]],
    straight_eps_km: float = 0.045,
    curve_eps_km: float = 0.010,
) -> List[List[float]]:
    """Curvature-adaptive RDP simplification.

    Scales allowable perpendicular chord error by local turning angle θ:
      - 45 m tolerance on straight alignments (θ ≈ 0°) to strip redundant collinear OSM nodes
      - 10 m tolerance on topographic bends and Roman sighting-point inflections (θ ≥ 15°)
    """
    if len(pts) <= 2:
        return pts
    import math

    x1, y1 = _to_xy_km(pts[0])
    x2, y2 = _to_xy_km(pts[-1])
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    max_score = -1.0
    idx = 0
    for i in range(1, len(pts) - 1):
        px, py = _to_xy_km(pts[i])
        d = (
            math.hypot(px - x1, py - y1)
            if norm == 0.0
            else abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
        )
        v_in = _unit_vec(pts[i - 1], pts[i])
        v_out = _unit_vec(pts[i], pts[i + 1])
        turn_deg = _angle_deg(v_in, v_out)
        local_eps = curve_eps_km + (straight_eps_km - curve_eps_km) * math.exp(
            -((turn_deg / 15.0) ** 2)
        )
        score = d / local_eps
        if score > max_score:
            max_score = score
            idx = i
    if max_score > 1.0:
        left = _curvature_aware_rdp(pts[: idx + 1], straight_eps_km, curve_eps_km)
        right = _curvature_aware_rdp(pts[idx:], straight_eps_km, curve_eps_km)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _can_join_directed(
    seg_a: List[List[float]],
    seg_b: List[List[float]],
    d_km: float,
    max_turn_deg: float = 32.0,
) -> bool:
    va = _unit_vec(seg_a[max(0, len(seg_a) - 3)], seg_a[-1])
    vb = _unit_vec(seg_b[0], seg_b[min(len(seg_b) - 1, 2)])
    if _angle_deg(va, vb) > (50.0 if d_km < 0.04 else max_turn_deg):
        return False
    if d_km >= 0.04:
        v_bridge = _unit_vec(seg_a[-1], seg_b[0])
        if _angle_deg(va, v_bridge) > max_turn_deg or _angle_deg(v_bridge, vb) > max_turn_deg:
            return False
    return True


def _simplify_and_chain_road_segments(
    segs: List[List[List[float]]], is_named_route: bool = True
) -> List[List[List[float]]]:
    """Deduplicate parallel dual-carriageway ways, chain tangent-aligned segments, and curvature-simplify."""
    import math

    def dist_km(a: List[float], b: List[float]) -> float:
        return math.hypot((a[0] - b[0]) * 69.0, (a[1] - b[1]) * 111.0)

    def seg_len_km(s: List[List[float]]) -> float:
        return sum(dist_km(s[i], s[i + 1]) for i in range(len(s) - 1))

    uk_segs: List[List[List[float]]] = []
    for s in segs:
        s_uk = [p for p in s if len(p) >= 2 and -6.5 <= float(p[0]) <= 1.85 and 49.9 <= float(p[1]) <= 59.0]
        if len(s_uk) >= 2:
            uk_segs.append(s_uk)
    if not uk_segs:
        return []

    by_cell: Dict[Tuple[int, int], Tuple[float, List[List[float]]]] = {}
    for s in uk_segs:
        mid = s[len(s) // 2]
        cell = (round(float(mid[0]) * 170), round(float(mid[1]) * 275))
        length = seg_len_km(s)
        if cell not in by_cell or length > by_cell[cell][0]:
            by_cell[cell] = (length, s)

    pool = [s for _, s in by_cell.values()]
    max_gap = 12.0 if is_named_route else 3.0
    merged: List[List[List[float]]] = []
    while pool:
        cur = pool.pop(0)
        changed = True
        while changed:
            changed = False
            best_i = -1
            best_d = max_gap
            best_cand: List[List[float]] = []
            best_prepend = False
            for i, cand in enumerate(pool):
                rc = list(reversed(cand))
                options = [
                    (dist_km(cur[-1], cand[0]), cur, cand, False),
                    (dist_km(cur[-1], rc[0]), cur, rc, False),
                    (dist_km(cand[-1], cur[0]), cand, cur, True),
                    (dist_km(rc[-1], cur[0]), rc, cur, True),
                ]
                for d, sa, sb, is_prep in options:
                    if d < best_d and _can_join_directed(sa, sb, d):
                        best_d = d
                        best_i = i
                        best_cand = sa if is_prep else sb
                        best_prepend = is_prep
            if best_i >= 0 and best_cand:
                pool.pop(best_i)
                if best_prepend:
                    cur = best_cand + (cur[1:] if best_d < 0.04 else cur)
                else:
                    cur = cur + (best_cand[1:] if best_d < 0.04 else best_cand)
                changed = True
        if seg_len_km(cur) >= 3.0:
            merged.append(_curvature_aware_rdp(cur))
    return merged


def build_ancient_roads_gpx_xml(roads_geojson_path: Path, updated_utc: str) -> str:
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>',
        '<gpx version="1.1" creator="UK 1 m LiDAR &amp; Aerial Cropmark Monument Survey" '
        'xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:osmand="https://osmand.net">',
        "  <metadata>",
        "    <name>UK Roman Roads &amp; Prehistoric Ridgeways</name>",
        "    <desc>British Roman road network (Margary routes, Watling Street, Ermine Street, Fosse Way, Dere Street, Stanegate, Akeman Street, Peddars Way) and prehistoric ridgeways (The Ridgeway, Icknield Way).</desc>",
        f"    <time>{escape(updated_utc)}</time>",
        "  </metadata>",
    ]
    if roads_geojson_path.exists():
        gj = json.loads(roads_geojson_path.read_text(encoding="utf-8"))
        for feat in gj.get("features", []):
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            if gtype == "LineString":
                raw_segments = [coords]
            elif gtype == "MultiLineString":
                raw_segments = coords
            else:
                continue
            name = str(props.get("name") or "Roman Road")
            if any(fr in name for fr in ("Boulogne", "Arras", "Amiens", "Cassel", "Thérouanne", "Bavay")):
                continue
            is_named = name not in ("Roman Road", "ROMAN ROAD", "Roman Road (mapped course)")
            segments = _simplify_and_chain_road_segments(raw_segments, is_named_route=is_named)
            if not segments:
                continue
            rtype = str(props.get("road_type") or "roman_road")
            marg = str(props.get("margary") or "")
            label = f"{name} (Margary {marg})" if marg else name
            desc = "Prehistoric Ridgeway / Ancient Trackway" if rtype == "prehistoric_trackway" else "Roman Road"
            lines.append("  <trk>")
            lines.append(f"    <name>{escape(label)}</name>")
            lines.append(f"    <desc>{escape(desc)}</desc>")
            for seg in segments:
                if not seg:
                    continue
                lines.append("    <trkseg>")
                for pt in seg:
                    if len(pt) >= 2:
                        lines.append(f'      <trkpt lat="{float(pt[1]):.5f}" lon="{float(pt[0]):.5f}" />')
                lines.append("    </trkseg>")
            lines.append("  </trk>")

    lines.append("  <extensions>")
    lines.append("    <osmand:color>#7e22ce</osmand:color>")
    lines.append("    <osmand:width>medium</osmand:width>")
    lines.append("    <osmand:show_arrows>false</osmand:show_arrows>")
    lines.append("    <osmand:show_start_finish>false</osmand:show_start_finish>")
    lines.append("  </extensions>")
    lines.append("</gpx>")
    return "\n".join(lines) + "\n"


def _build_plugin_graphics() -> Tuple[bytes, bytes]:
    """Generate a 96x96 plugin icon PNG and 640x240 LiDAR hillshade banner PNG."""
    import io
    import math
    from PIL import Image, ImageDraw

    # 1. 96x96 Icon: concentric amber/gold henge & barrow reticle on dark slate circle
    icon = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    draw.ellipse((4, 4, 92, 92), fill=(15, 23, 42, 255), outline=(217, 119, 6, 255), width=4)
    draw.ellipse((18, 18, 78, 78), outline=(245, 158, 11, 230), width=4)
    draw.ellipse((32, 32, 64, 64), outline=(34, 197, 94, 240), width=3)
    draw.ellipse((43, 43, 53, 53), fill=(245, 158, 11, 255))
    icon_buf = io.BytesIO()
    icon.save(icon_buf, format="PNG", optimize=True)

    # 2. 640x240 Banner: synthetic multi-azimuth LiDAR hillshade of a concentric henge & barrow
    banner = Image.new("RGB", (640, 240), (20, 29, 43))
    px = banner.load()
    for y in range(240):
        for x in range(640):
            dx = (x - 320.0) / 85.0
            dy = (y - 120.0) / 85.0
            r = math.hypot(dx, dy)
            # Concentric bank + ditch + central mound relief profile derivative (NW illumination)
            grad = (
                -0.45 * (dx + dy) * math.exp(-((r - 1.05) ** 2) / 0.045)
                + 0.35 * (dx + dy) * math.exp(-((r - 0.72) ** 2) / 0.035)
                - 0.55 * (dx + dy) * math.exp(-(r**2) / 0.08)
            )
            shade = max(18, min(235, int(72 + 125 * grad)))
            px[x, y] = (int(shade * 0.88), int(shade * 0.94), min(255, int(shade * 1.06)))
    bdraw = ImageDraw.Draw(banner)
    bdraw.ellipse((320 - 115, 120 - 115, 320 + 115, 120 + 115), outline=(245, 158, 11), width=2)
    bdraw.line((320 - 130, 120, 320 - 100, 120), fill=(245, 158, 11), width=2)
    bdraw.line((320 + 100, 120, 320 + 130, 120), fill=(245, 158, 11), width=2)
    bdraw.line((320, 120 - 130, 320, 120 - 100), fill=(245, 158, 11), width=2)
    bdraw.line((320, 120 + 100, 320, 120 + 130), fill=(245, 158, 11), width=2)
    banner_buf = io.BytesIO()
    banner.save(banner_buf, format="PNG", optimize=True)
    return icon_buf.getvalue(), banner_buf.getvalue()


def _osm_tags_for_candidate(domain: str, p: Dict[str, Any]) -> Dict[str, str]:
    cls_key = str(p.get("predicted_class") or "")
    tags: Dict[str, str] = {"historic": "archaeological_site"}
    if cls_key == "megalithic_tomb_long_barrow":
        tags["archaeological_site"] = "tumulus"
        tags["tumulus"] = "long_barrow"
    elif cls_key == "round_barrow_cairn":
        tags["archaeological_site"] = "tumulus"
        tags["tumulus"] = "round_barrow"
    elif cls_key == "stone_circle_henge_alignment":
        tags["archaeological_site"] = "henge"
    elif cls_key == "roman_villa_building":
        tags["archaeological_site"] = "roman_villa"
        tags["historic:civilization"] = "ancient_roman"
    elif cls_key == "roman_road_agger_trackway":
        tags["historic"] = "roman_road"
        tags["historic:civilization"] = "ancient_roman"
    elif cls_key == "roman_fort_camp_settlement":
        tags["archaeological_site"] = "fortification"
        tags["historic:civilization"] = "ancient_roman"
    elif cls_key == "hut_circle_pound_settlement":
        tags["archaeological_site"] = "settlement"
    elif cls_key == "iron_age_hillfort_banjo_enclosure" or domain == "enclosure":
        tags["archaeological_site"] = "enclosure"
    elif cls_key == "cross_ridge_dyke_field_system":
        tags["archaeological_site"] = "earthwork"
    else:
        lbl_low = str(p.get("predicted_class_label") or "").lower()
        if "barrow" in lbl_low or "cairn" in lbl_low:
            tags["archaeological_site"] = "tumulus"
        elif "enclosure" in lbl_low or "hillfort" in lbl_low:
            tags["archaeological_site"] = "enclosure"
        else:
            tags["archaeological_site"] = "earthwork"

    gref = str(p.get("os_grid_ref") or "").strip()
    gref_id = _clean_grid_id(gref)
    if gref:
        tags["ref:gb:os_grid"] = gref
    tags["source"] = "Environment Agency 1m LiDAR; OS Six-Inch 1888-1913"
    tags["url"] = f"{SITE_BASE_URL}?site={gref_id}&domain={domain}"
    note_txt = str(p.get("arch_context_note") or "").strip()
    if note_txt:
        tags["note"] = note_txt[:250]
    return tags


def build_os1888_osm_exports(
    candidates: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[Dict[str, Any], str, int]:
    """Build MapRoulette GeoJSON and JOSM .osm XML for 1888 OS Confirmed Antiquities missing from OSM."""
    features: List[Dict[str, Any]] = []
    osm_lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<osm version="0.6" upload="discouraged" generator="megalith-lidar">',
    ]
    node_id = -1
    for domain, p in candidates:
        if _classify_group(p, domain) != "1888 OS Confirmed Antiquities ★":
            continue
        lat = float(p.get("lat") or 0.0)
        lon = float(p.get("lon") or 0.0)
        tags = _osm_tags_for_candidate(domain, p)
        gref = str(p.get("os_grid_ref") or "").strip()
        props = dict(tags)
        props["os_grid_ref"] = gref
        props["county"] = str(p.get("county") or "")
        props["confidence"] = round(float(p.get("confidence") or 0.0), 3)
        props["est_local_relief_m"] = round(float(p.get("est_local_relief_m") or 0.0), 2)
        props["id_editor_url"] = (
            f"https://www.openstreetmap.org/edit?editor=id#map=18/{lat:.6f}/{lon:.6f}"
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                "properties": props,
            }
        )
        osm_lines.append(f'  <node id="{node_id}" lat="{lat:.6f}" lon="{lon:.6f}" version="1">')
        for k, v in tags.items():
            osm_lines.append(f'    <tag k="{escape(k)}" v="{escape(str(v))}" />')
        osm_lines.append("  </node>")
        node_id -= 1

    osm_lines.append("</osm>")
    mr_fc = {"type": "FeatureCollection", "features": features}
    return mr_fc, "\n".join(osm_lines) + "\n", len(features)


def export_osmand_files(docs_dir: Path) -> Dict[str, Any]:
    updated_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    candidates = _load_candidates(docs_dir)
    gpx_xml = build_candidates_gpx_xml(candidates, updated_utc)

    gpx_path = docs_dir / "megalith_lidar_osmand.gpx"
    gpx_path.write_text(gpx_xml, encoding="utf-8")

    mr_fc, josm_xml, os1888_count = build_os1888_osm_exports(candidates)
    mr_path = docs_dir / "maproulette_os1888_antiquities.geojson"
    mr_path.write_text(json.dumps(mr_fc, indent=2, ensure_ascii=False), encoding="utf-8")
    josm_path = docs_dir / "josm_os1888_antiquities.osm"
    josm_path.write_text(josm_xml, encoding="utf-8")

    roads_xml = build_ancient_roads_gpx_xml(docs_dir / "roman_and_ancient_roads.geojson", updated_utc)
    icon_png, banner_png = _build_plugin_graphics()

    # Split candidates into modular, individually toggleable GPX track layers under tracks/UK_Megalith_LiDAR/
    # so personal Favourites are never polluted.
    track_buckets: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
        "tracks/UK_Megalith_LiDAR/1888_OS_Confirmed_Antiquities.gpx": [],
        "tracks/UK_Megalith_LiDAR/Prehistoric_High_Confidence_1pct_FPR.gpx": [],
        "tracks/UK_Megalith_LiDAR/Prehistoric_Extended_5pct_FPR.gpx": [],
        "tracks/UK_Megalith_LiDAR/Iron_Age_and_Settlement.gpx": [],
        "tracks/UK_Megalith_LiDAR/Roman_Candidates.gpx": [],
    }
    for dom, p in candidates:
        grp = _classify_group(p, dom)
        if grp == "1888 OS Confirmed Antiquities ★":
            track_buckets["tracks/UK_Megalith_LiDAR/1888_OS_Confirmed_Antiquities.gpx"].append((dom, p))
        elif grp == "Iron Age & Settlement ◆":
            track_buckets["tracks/UK_Megalith_LiDAR/Iron_Age_and_Settlement.gpx"].append((dom, p))
        elif grp == "Roman Candidates ■":
            track_buckets["tracks/UK_Megalith_LiDAR/Roman_Candidates.gpx"].append((dom, p))
        elif p.get("passes_fpr_0_01"):
            track_buckets["tracks/UK_Megalith_LiDAR/Prehistoric_High_Confidence_1pct_FPR.gpx"].append((dom, p))
        else:
            track_buckets["tracks/UK_Megalith_LiDAR/Prehistoric_Extended_5pct_FPR.gpx"].append((dom, p))

    gpx_items_manifest: List[Dict[str, Any]] = []
    gpx_payloads: Dict[str, str] = {}
    for rel_file, bucket in track_buckets.items():
        gpx_payloads[rel_file] = build_candidates_gpx_xml(bucket, updated_utc)
        gpx_items_manifest.append({"type": "GPX", "file": rel_file})

    roads_rel = "tracks/UK_Megalith_LiDAR/UK_Roman_and_Ancient_Roads.gpx"
    gpx_payloads[roads_rel] = roads_xml
    gpx_items_manifest.append(
        {
            "type": "GPX",
            "file": roads_rel,
            "color": "#7e22ce",
            "width": "medium",
            "show_arrows": False,
            "show_start_finish": False,
        }
    )

    items_manifest = {
        "version": 3,
        "items": [
            {
                "type": "PLUGIN",
                "pluginId": "uk.megalith.lidar",
                "version": 1,
                "icon": {"": "@icon.png"},
                "image": {"": "@image.png"},
                "name": {"": "UK 1 m LiDAR Monument Survey"},
                "description": {
                    "": (
                        "Unrecorded Prehistoric, Iron Age, and Roman earthwork candidates detected by "
                        "1 m LiDAR & 16 m aerial cropmarks across England, cross-checked against "
                        "1888–1913 Historic OS 6-Inch maps, Historic England NHLE, OpenStreetMap, "
                        "and Megalithic.co.uk. Installs modular monument GPX layers under Tracks → "
                        "UK_Megalith_LiDAR (leaving personal Favourites untouched), Roman & Ancient "
                        "Road tracks, live 1 m EA LiDAR Hillshade, and 1888 OS / Roman Empire overlays."
                    )
                },
            },
            {
                "type": "RESOURCES",
                "pluginId": "uk.megalith.lidar",
                "file": "res",
            },
            {
                "type": "MAP_SOURCES",
                "pluginId": "uk.megalith.lidar",
                "items": [
                    {
                        "name": "🇬🇧 ⛰ England LiDAR Terrain 1m (EA)",
                        "minZoom": 11,
                        "maxZoom": 19,
                        "url": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=Lidar_Composite_Hillshade_DTM_1m&STYLES=&SRS=EPSG:4326&BBOX={bbox}&WIDTH=256&HEIGHT=256&FORMAT=image/png",
                        "ext": ".png",
                        "tileSize": 256,
                        "bitDensity": 16,
                        "avgSize": 32000,
                        "ellipsoid": False,
                        "inverted_y": False,
                        "randoms": "",
                        "rule": "",
                        "sql": False,
                    },
                    {
                        "name": "🇬🇧 ⛰ England LiDAR Surface 1m (EA)",
                        "minZoom": 11,
                        "maxZoom": 19,
                        "url": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=Lidar_Composite_Hillshade_LZ_DSM_1m&STYLES=&SRS=EPSG:4326&BBOX={bbox}&WIDTH=256&HEIGHT=256&FORMAT=image/png",
                        "ext": ".png",
                        "tileSize": 256,
                        "bitDensity": 16,
                        "avgSize": 32000,
                        "ellipsoid": False,
                        "inverted_y": False,
                        "randoms": "",
                        "rule": "",
                        "sql": False,
                    },
                    {
                        "name": "UK 1888 OS 6-Inch Map",
                        "minZoom": 6,
                        "maxZoom": 17,
                        "url": "https://api.maptiler.com/tiles/uk-osgb1888/{0}/{1}/{2}?key=7Y0Q1ck46BnB8cXXXg8X",
                        "ext": ".png",
                        "tileSize": 256,
                        "bitDensity": 16,
                        "avgSize": 32000,
                        "ellipsoid": False,
                        "inverted_y": False,
                        "randoms": "",
                        "rule": "",
                        "sql": False,
                    },
                    {
                        "name": "Roman Empire Atlas (DARE)",
                        "minZoom": 6,
                        "maxZoom": 11,
                        "url": "https://dh.gu.se/tiles/imperium/{0}/{1}/{2}.png",
                        "ext": ".png",
                        "tileSize": 256,
                        "bitDensity": 16,
                        "avgSize": 38000,
                        "ellipsoid": False,
                        "inverted_y": False,
                        "randoms": "",
                        "rule": "",
                        "sql": False,
                    },
                ],
            },
            *gpx_items_manifest,
        ],
    }

    osf_path = docs_dir / "megalith-lidar.osf"
    with zipfile.ZipFile(osf_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("items.json", json.dumps(items_manifest, indent=2, ensure_ascii=False))
        zf.writestr("res/icon.png", icon_png)
        zf.writestr("res/image.png", banner_png)
        for rel_file, xml_str in gpx_payloads.items():
            zf.writestr(rel_file, xml_str)

    qr_svg_path = docs_dir / "osmand_qr.svg"
    if not qr_svg_path.exists():
        import shutil
        import subprocess

        if shutil.which("qrencode"):
            subprocess.run(
                [
                    "qrencode",
                    "-t",
                    "SVG",
                    "-m",
                    "2",
                    "-o",
                    str(qr_svg_path),
                    f"{SITE_BASE_URL}docs/megalith-lidar.osf",
                ],
                check=False,
            )

    counts_by_group: Dict[str, int] = {}
    for dom, p in candidates:
        grp_name = _classify_group(p, dom)
        counts_by_group[grp_name] = counts_by_group.get(grp_name, 0) + 1

    return {
        "total_waypoints": len(candidates),
        "counts_by_group": counts_by_group,
        "os1888_osm_export_count": os1888_count,
        "gpx_path": str(gpx_path),
        "gpx_size_kb": round(gpx_path.stat().st_size / 1024.0, 1),
        "osf_path": str(osf_path),
        "osf_size_kb": round(osf_path.stat().st_size / 1024.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OsmAnd GPX and .osf Custom Plugin package.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=REPO_ROOT / "docs",
        help="Path to docs directory containing candidates GeoJSONs.",
    )
    args = parser.parse_args()
    res = export_osmand_files(args.docs_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
