#!/usr/bin/env python3
"""Deterministically prepare bundled offline web assets for the Android APK.

Populates android/app/src/main/assets/web/ from the repository's source HTML,
vendored Leaflet 1.9.4 assets, and survey JSON/GeoJSON datasets, while:
  - Replacing CDN Leaflet tags with local bundled leaflet/leaflet.css & leaflet.js
  - Stripping the web-only GoatCounter analytics script tag
  - Deduplicating repeated NHLE/OSM features across domain GeoJSONs so the
    bundled Android dataset loads quickly offline with identical tab counts
  - Generating a pure-source OsmAnd plugin (.osf) and GPX waypoint export
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_osmand_plugin import export_osmand_files  # noqa: E402


def _clean_grid_id(val: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(val or "").upper())


def _dedup_key(p: Dict[str, Any], mode: str) -> str:
    k = _clean_grid_id(p.get("os_grid_ref"))
    if mode == "known":
        mm = p.get("matched_monument")
        if isinstance(mm, dict) and mm.get("list_entry"):
            return f"nhle_{mm['list_entry']}"
        osm = p.get("osm_archaeology_matches")
        if isinstance(osm, list) and osm and isinstance(osm[0], dict) and osm[0].get("osm_id"):
            return f"osm_{osm[0]['osm_id']}"
    if not k:
        k = f"{p.get('os_grid_ref')}_{p.get('easting')}_{p.get('northing')}_{p.get('predicted_class', '')}"
    return k


def _deduplicate_features(features: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for f in features:
        p = (f or {}).get("properties") or {}
        dk = _dedup_key(p, mode)
        prev = by_id.get(dk)
        if prev is None:
            by_id[dk] = f
        else:
            prev_p = prev.get("properties") or {}
            prev_png = 1 if prev_p.get("diagnostic_png") else 0
            cur_png = 1 if p.get("diagnostic_png") else 0
            if cur_png > prev_png or (
                cur_png == prev_png
                and float(p.get("confidence") or 0.0) > float(prev_p.get("confidence") or 0.0)
            ):
                by_id[dk] = f
    return list(by_id.values())


def _compact_feature(f: Dict[str, Any]) -> Dict[str, Any]:
    p = dict((f or {}).get("properties") or {})
    if isinstance(p.get("nearby_nhle_monuments"), list):
        p["nearby_nhle_monuments"] = p["nearby_nhle_monuments"][:3]
    if isinstance(p.get("megalithic_portal_matches"), list):
        p["megalithic_portal_matches"] = p["megalithic_portal_matches"][:3]
    if isinstance(p.get("osm_archaeology_matches"), list):
        p["osm_archaeology_matches"] = p["osm_archaeology_matches"][:1]
    if isinstance(p.get("osm_confounder_matches"), list):
        p["osm_confounder_matches"] = p["osm_confounder_matches"][:1]
    if isinstance(p.get("environmental_confounder_hints"), list):
        p["environmental_confounder_hints"] = p["environmental_confounder_hints"][:2]
    return {
        "type": "Feature",
        "geometry": f.get("geometry"),
        "properties": p,
    }


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(summary)
    if isinstance(out.get("scanned_squares"), list):
        keep_sq_keys = (
            "square_id",
            "sampling_stream",
            "os_grid_ref",
            "bounds_latlon",
            "center_lat",
            "center_lon",
            "area_km2",
            "rediscovered_nhle_scheduled_count",
            "flagged_modern_or_karst_confounders_count",
            "unidentified_candidates_count",
            "unidentified_candidates_fpr_0_01_count",
        )
        out["scanned_squares"] = [
            {k: sq[k] for k in keep_sq_keys if k in sq}
            for sq in out["scanned_squares"]
            if isinstance(sq, dict)
        ]
    return out


def _write_compact_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_android_assets() -> None:
    docs_src = REPO_ROOT / "docs"
    leaflet_src = REPO_ROOT / "android" / "app" / "src" / "main" / "leaflet"
    web_out = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "web"
    docs_out = web_out / "docs"
    leaflet_out = web_out / "leaflet"

    if web_out.exists():
        shutil.rmtree(web_out)
    docs_out.mkdir(parents=True, exist_ok=True)
    leaflet_out.mkdir(parents=True, exist_ok=True)

    # 1. Copy vendored Leaflet 1.9.4 assets
    shutil.copyfile(leaflet_src / "leaflet.css", leaflet_out / "leaflet.css")
    shutil.copyfile(leaflet_src / "leaflet.js", leaflet_out / "leaflet.js")

    # 2. Transform index.html for local offline WebView execution (no CDN, no GoatCounter)
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    html = re.sub(
        r'<link\s+rel="stylesheet"\s+href="https://unpkg\.com/leaflet@1\.9\.4/dist/leaflet\.css"[^>]*/>',
        '<link rel="stylesheet" href="leaflet/leaflet.css" />',
        html,
    )
    html = re.sub(
        r'<script\s+src="https://unpkg\.com/leaflet@1\.9\.4/dist/leaflet\.js"[^>]*></script>',
        '<script src="leaflet/leaflet.js"></script>',
        html,
    )
    html = re.sub(
        r'\s*<script[^>]*data-goatcounter=[^>]*></script>',
        "",
        html,
    )
    (web_out / "index.html").write_text(html, encoding="utf-8")

    # 3. Compact and copy summary JSONs and static overlays
    summary_raw = json.loads((docs_src / "survey_summary.json").read_text(encoding="utf-8"))
    updated_utc = str(summary_raw.get("updated_at_utc") or "2026-01-01T00:00:00Z").strip()
    _write_compact_json(docs_out / "survey_summary.json", _compact_summary(summary_raw))

    for opt_json in ("roman_summary.json", "enclosure_summary.json"):
        src_p = docs_src / opt_json
        if src_p.exists():
            obj = json.loads(src_p.read_text(encoding="utf-8"))
            _write_compact_json(docs_out / opt_json, _compact_summary(obj))

    for copy_json in (
        "uk_counties.geojson",
        "roman_and_ancient_roads.geojson",
        "community_annotations.json",
    ):
        src_p = docs_src / copy_json
        if src_p.exists():
            obj = json.loads(src_p.read_text(encoding="utf-8"))
            _write_compact_json(docs_out / copy_json, obj)

    if (docs_src / "osmand_qr.svg").exists():
        shutil.copyfile(docs_src / "osmand_qr.svg", docs_out / "osmand_qr.svg")

    # 4. Deduplicate & compact candidate GeoJSONs while preserving 100% of tab features
    c_gj = json.loads((docs_src / "candidates.geojson").read_text(encoding="utf-8"))
    r_gj = json.loads((docs_src / "roman_candidates.geojson").read_text(encoding="utf-8"))
    e_gj = json.loads((docs_src / "enclosure_candidates.geojson").read_text(encoding="utf-8"))

    c_feats = [_compact_feature(f) for f in (c_gj.get("features") or [])]
    r_feats = [_compact_feature(f) for f in (r_gj.get("features") or [])]
    e_feats = [_compact_feature(f) for f in (e_gj.get("features") or [])]
    all_feats = c_feats + r_feats + e_feats

    known_winners = _deduplicate_features(
        [
            f
            for f in all_feats
            if (f.get("properties") or {}).get("context_category")
            in ("rediscovered_nhle_scheduled", "rediscovered_osm_archaeology")
        ],
        "known",
    )
    conf_winners = _deduplicate_features(
        [
            f
            for f in all_feats
            if (f.get("properties") or {}).get("context_category")
            == "flagged_modern_or_karst_confounder"
        ],
        "confounders",
    )

    c_unid = [
        f
        for f in c_feats
        if (f.get("properties") or {}).get("context_category") == "unidentified_candidate"
    ]
    r_unid = [
        f
        for f in r_feats
        if (f.get("properties") or {}).get("context_category") == "unidentified_candidate"
    ]
    e_unid = [
        f
        for f in e_feats
        if (f.get("properties") or {}).get("context_category") == "unidentified_candidate"
    ]

    _write_compact_json(
        docs_out / "candidates.geojson",
        {"type": "FeatureCollection", "features": c_unid + known_winners + conf_winners},
    )
    _write_compact_json(
        docs_out / "roman_candidates.geojson",
        {"type": "FeatureCollection", "features": r_unid},
    )
    _write_compact_json(
        docs_out / "enclosure_candidates.geojson",
        {"type": "FeatureCollection", "features": e_unid},
    )

    # 5. Generate pure-source OsmAnd .osf, .gpx, MapRoulette GeoJSON, and JOSM .osm in docs_out
    export_osmand_files(docs_out, updated_utc=updated_utc, include_obf=False)


if __name__ == "__main__":
    prepare_android_assets()
