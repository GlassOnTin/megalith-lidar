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
                segments = [coords]
            elif gtype == "MultiLineString":
                segments = coords
            else:
                continue
            name = str(props.get("name") or "Roman Road")
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
    lines.append("  </extensions>")
    lines.append("</gpx>")
    return "\n".join(lines) + "\n"


def export_osmand_files(docs_dir: Path) -> Dict[str, Any]:
    updated_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    candidates = _load_candidates(docs_dir)
    gpx_xml = build_candidates_gpx_xml(candidates, updated_utc)

    gpx_path = docs_dir / "megalith_lidar_osmand.gpx"
    gpx_path.write_text(gpx_xml, encoding="utf-8")

    roads_xml = build_ancient_roads_gpx_xml(docs_dir / "roman_and_ancient_roads.geojson", updated_utc)

    items_manifest = {
        "version": 3,
        "items": [
            {
                "type": "PLUGIN",
                "pluginId": "uk.megalith.lidar",
                "version": 1,
                "name": {"": "UK 1 m LiDAR Monument Survey"},
                "description": {
                    "": (
                        "Unrecorded Prehistoric, Iron Age, and Roman earthwork candidates detected by "
                        "1 m LiDAR & 16 m aerial cropmarks across England, cross-checked against "
                        "1888–1913 Historic OS 6-Inch maps, Historic England NHLE, OpenStreetMap, "
                        "and Megalithic.co.uk. Includes monument Favourites, Roman & Ancient Road "
                        "tracks, and 1888 OS / Roman Empire map overlays."
                    )
                },
            },
            {
                "type": "MAP_SOURCES",
                "pluginId": "uk.megalith.lidar",
                "items": [
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
            {
                "type": "FAVOURITES",
                "file": "favourites.gpx",
            },
            {
                "type": "GPX",
                "file": "tracks/UK_Megalith_LiDAR_Candidates.gpx",
            },
            {
                "type": "GPX",
                "file": "tracks/UK_Roman_and_Ancient_Roads.gpx",
                "color": "#7e22ce",
                "width": "medium",
                "show_arrows": False,
            },
        ],
    }

    osf_path = docs_dir / "megalith-lidar.osf"
    with zipfile.ZipFile(osf_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("items.json", json.dumps(items_manifest, indent=2, ensure_ascii=False))
        zf.writestr("favourites.gpx", gpx_xml)
        zf.writestr("tracks/UK_Megalith_LiDAR_Candidates.gpx", gpx_xml)
        zf.writestr("tracks/UK_Roman_and_Ancient_Roads.gpx", roads_xml)

    counts_by_group: Dict[str, int] = {}
    for dom, p in candidates:
        grp_name = _classify_group(p, dom)
        counts_by_group[grp_name] = counts_by_group.get(grp_name, 0) + 1

    return {
        "total_waypoints": len(candidates),
        "counts_by_group": counts_by_group,
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
