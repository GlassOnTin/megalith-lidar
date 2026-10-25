#!/usr/bin/env python3
"""
Import and apply field-survey or historic-survey annotations (from a JSON or GeoJSON file,
or directly from `docs/community_annotations.json`) onto the public GeoJSON datasets.

Usage:
    .venv/bin/python3 scripts/import_survey_annotations.py [path/to/new_survey.json]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megalith_lidar.arch_context import (
    _norm_grid_ref,
    apply_community_annotations_to_candidates,
    load_community_annotations,
    parse_survey_annotations_payload,
)


def merge_annotations(existing: list[dict], incoming: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    ordered_keys: list[str] = []
    for item in existing + incoming:
        gref = _norm_grid_ref(item.get("os_grid_ref"))
        if not gref and "easting" in item and "northing" in item:
            gref = f"E{int(round(float(item['easting'])))}_N{int(round(float(item['northing'])))}"
        if not gref and "lat" in item and "lon" in item:
            gref = f"LAT{float(item['lat']):.5f}_LON{float(item['lon']):.5f}"
        if not gref:
            continue
        if gref not in by_key:
            ordered_keys.append(gref)
            by_key[gref] = dict(item)
        else:
            merged = dict(by_key[gref])
            for k, v in item.items():
                if v not in (None, ""):
                    merged[k] = v
            by_key[gref] = merged
    return [by_key[k] for k in ordered_keys]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and apply field/historic survey annotations.")
    parser.add_argument(
        "input_files",
        nargs="*",
        help="Optional JSON or GeoJSON files containing field survey notes or historic survey records to merge into docs/community_annotations.json",
    )
    parser.add_argument(
        "--community-file",
        default=str(ROOT / "docs" / "community_annotations.json"),
        help="Path to docs/community_annotations.json",
    )
    args = parser.parse_args()

    comm_path = Path(args.community_file)
    existing = load_community_annotations(comm_path)
    incoming_all: list[dict] = []
    for fp in args.input_files:
        p = Path(fp)
        payload = json.loads(p.read_text(encoding="utf-8"))
        parsed = parse_survey_annotations_payload(payload)
        print(f"Loaded {len(parsed)} annotations from {p}")
        incoming_all.extend(parsed)

    merged = merge_annotations(existing, incoming_all)
    if incoming_all or not comm_path.exists():
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out_payload = {
            "schema_version": "1.0",
            "updated_at_utc": now_utc,
            "description": (
                "Community field-survey observations, county HER cross-references, and historic survey "
                "annotations for the UK 1 m LiDAR & Aerial Cropmark Monument Survey. Users can load/save "
                "this JSON in the web viewer or submit additions via GitHub Pull Request."
            ),
            "annotations": merged,
        }
        comm_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(merged)} total annotations to {comm_path}")

    for gj_rel in (
        "docs/candidates.geojson",
        "docs/roman_candidates.geojson",
        "docs/enclosure_candidates.geojson",
    ):
        gj_path = ROOT / gj_rel
        if not gj_path.exists():
            continue
        gj = json.loads(gj_path.read_text(encoding="utf-8"))
        cand_props = [f["properties"] for f in gj.get("features", [])]
        matched = apply_community_annotations_to_candidates(cand_props, merged)
        if matched > 0:
            gj_path.write_text(json.dumps(gj), encoding="utf-8")
        print(f"{gj_rel}: applied {matched} matching annotations across {len(cand_props)} features")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
