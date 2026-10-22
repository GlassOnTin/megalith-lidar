#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megalith_lidar.arch_context import (
    NHLESpatialContextExtractor,
    enrich_candidates_with_archaeological_context,
)
from megalith_lidar.catalog import download_or_load_nhle_catalog
from megalith_lidar.model import MegalithUNet
from megalith_lidar.scanner import scan_region
from megalith_lidar.stats import PlattCalibrator


def run_enclosure_scan(
    data_dir: Path,
    enclosure_ckpt_path: Path,
    docs_dir: Path,
    device_str: str = "cuda:0",
    max_squares: int = 60,
) -> Dict:
    survey_dir = data_dir / "national_survey"
    enclosure_survey_dir = data_dir / "enclosure_survey"
    enclosure_survey_dir.mkdir(parents=True, exist_ok=True)
    queue_path = survey_dir / "queue_state.json"
    eval_metrics_path = enclosure_ckpt_path.parent / "enclosure_evaluation_metrics.json"

    catalog = download_or_load_nhle_catalog(data_dir / "catalog" / "nhle_scheduled_monuments_27700.json")
    tile_cache_dir = data_dir / "tiles"
    osm_cache_dir = data_dir / "osm_cache"

    queue_state = json.loads(queue_path.read_text(encoding="utf-8"))
    completed = [q for q in queue_state["queue"] if q.get("status") == "completed"]

    # Prioritize completed squares that contain known Iron Age enclosures, hut circle settlements, or dykes
    def _sq_priority(q: Dict) -> int:
        hits = catalog.query_bbox(q["e_min"], q["n_min"], q["e_max"], q["n_max"])
        has_enc = any(h.is_enclosure_or_settlement for h in hits)
        return 0 if has_enc else 1

    completed.sort(key=lambda q: (_sq_priority(q), q["square_id"]))
    target_squares = completed[:max_squares]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(enclosure_ckpt_path, map_location=device, weights_only=False)
    model = MegalithUNet(in_channels=6, num_classes=4, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    calibrator = PlattCalibrator.from_dict(ckpt["calibrator"])
    op_points = ckpt["val_thresholds"]
    th_005 = float(op_points["fpr_0.05"]["threshold"])
    th_001 = float(op_points["fpr_0.01"]["threshold"])
    pixel_th_005 = min(float(ckpt.get("val_pixel_threshold_fpr05", 0.72)), 0.78)
    pixel_th_001 = max(pixel_th_005, 0.82)

    channel_mean = np.array(ckpt["channel_mean"], dtype=np.float32)
    channel_std = np.array(ckpt["channel_std"], dtype=np.float32)

    docs_diag_dir = docs_dir / "diagnostics"
    docs_diag_dir.mkdir(parents=True, exist_ok=True)
    copied_pngs: Set[str] = set()
    all_enclosure_features: List[Dict] = []

    for idx, q in enumerate(target_squares, 1):
        sq_id = f"enclosure_{q['square_id']}"
        out_dir = enclosure_survey_dir / sq_id
        gj_path = out_dir / f"{sq_id}_candidates.geojson"
        if not gj_path.exists():
            t0 = time.time()
            summary = scan_region(
                model=model,
                catalog=catalog,
                channel_mean=channel_mean,
                channel_std=channel_std,
                calibrator=calibrator,
                tile_prob_threshold=th_005,
                pixel_prob_threshold=pixel_th_005,
                e_min=q["e_min"],
                e_max=q["e_max"],
                n_min=q["n_min"],
                n_max=q["n_max"],
                device=device,
                tile_cache_dir=tile_cache_dir,
                output_dir=out_dir,
                region_name=sq_id,
                osm_cache_dir=osm_cache_dir,
                target_domain="enclosure",
            )
            if gj_path.exists():
                gj_data = json.loads(gj_path.read_text(encoding="utf-8"))
                for feat in gj_data.get("features", []):
                    p = feat.get("properties", {})
                    p["passes_fpr_0_01"] = bool(
                        float(p.get("calibrated_tile_prob", 0.0)) >= th_001
                        and float(p.get("peak_prob", 0.0)) >= pixel_th_001
                    )
                gj_path.write_text(json.dumps(gj_data, indent=2), encoding="utf-8")
            print(
                f"[enclosure {idx}/{len(target_squares)}] {sq_id} ({time.time() - t0:.1f}s): "
                f"NHLE={summary['rediscovered_scheduled_monuments_count']}, "
                f"Unid={summary['unidentified_candidates_count']}"
            )

        if gj_path.exists():
            gj_data = json.loads(gj_path.read_text(encoding="utf-8"))
            for feat in gj_data.get("features", []):
                p = feat["properties"]
                p["square_id"] = q["square_id"]
                p["sampling_stream"] = q["sampling_stream"]
                p["split"] = q.get("split", "test")
                p["target_domain"] = "enclosure"
                diag_name = p.get("diagnostic_png")
                if diag_name:
                    src_png = out_dir / "diagnostics" / diag_name
                    if src_png.exists() and diag_name not in copied_pngs:
                        shutil.copy2(src_png, docs_diag_dir / diag_name)
                        copied_pngs.add(diag_name)
                all_enclosure_features.append(feat)

    extractor = NHLESpatialContextExtractor(catalog)
    cand_props = [f["properties"] for f in all_enclosure_features]
    enrich_candidates_with_archaeological_context(
        candidates=cand_props,
        extractor=extractor,
        cache_path=data_dir / "enclosure_arch_context_cache.json",
        docs_diag_dir=docs_diag_dir,
        max_vlm_calls=45,
    )

    eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8")) if eval_metrics_path.exists() else {}
    enclosure_summary = {
        "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "squares_scanned": len(target_squares),
        "area_km2": round(len(target_squares) * 4.0, 1),
        "total_detections": len(all_enclosure_features),
        "rediscovered_nhle_scheduled": sum(
            1 for f in all_enclosure_features if f["properties"].get("context_category") == "rediscovered_nhle_scheduled"
        ),
        "rediscovered_osm_archaeology": sum(
            1 for f in all_enclosure_features if f["properties"].get("context_category") == "rediscovered_osm_archaeology"
        ),
        "unidentified_candidates": sum(
            1 for f in all_enclosure_features if f["properties"].get("context_category") == "unidentified_candidate"
        ),
        "unidentified_candidates_fpr_0_01": sum(
            1
            for f in all_enclosure_features
            if f["properties"].get("context_category") == "unidentified_candidate"
            and f["properties"].get("passes_fpr_0_01")
        ),
        "heldout_test_evaluation": eval_metrics.get("model_test_metrics", {}),
        "subgroup_and_confounding_analysis": eval_metrics.get("model_test_subgroup_and_confounding_analysis", {}),
    }

    (docs_dir / "enclosure_candidates.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": all_enclosure_features}, indent=2),
        encoding="utf-8",
    )
    (docs_dir / "enclosure_summary.json").write_text(
        json.dumps(enclosure_summary, indent=2),
        encoding="utf-8",
    )
    return enclosure_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan held-out 2km x 2km test squares with EnclosureUNet")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/bulk/megalith-lidar"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/bulk/megalith-lidar/checkpoints/enclosure_unet_best.pt"),
    )
    parser.add_argument("--docs-dir", type=Path, default=Path("/home/ian/Code/megalith-lidar/docs"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-squares", type=int, default=60)
    args = parser.parse_args()
    res = run_enclosure_scan(args.data_dir, args.checkpoint, args.docs_dir, args.device, args.max_squares)
    print("Enclosure scan summary:", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
