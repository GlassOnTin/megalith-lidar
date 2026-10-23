#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megalith_lidar.catalog import (
    CLASS_BARROW_CAIRN,
    CLASS_CIRCLE_HENGE_STONE,
    CLASS_MEGALITHIC_TOMB_LONG_BARROW,
    CLASS_NAMES,
    ENCLOSURE_CLASS_DYKE_FIELD_SYSTEM,
    ENCLOSURE_CLASS_HILLFORT_BANJO_ENCLOSURE,
    ENCLOSURE_CLASS_HUT_CIRCLE_POUND,
    ENCLOSURE_CLASS_NAMES,
    ROMAN_CLASS_FORT_CAMP_SETTLEMENT,
    ROMAN_CLASS_NAMES,
    ROMAN_CLASS_ROAD_AGGER_TRACKWAY,
    ROMAN_CLASS_VILLA_BUILDING,
    MonumentCatalog,
    download_or_load_nhle_catalog,
)
from megalith_lidar.dataset import (
    MegalithLidarDataset,
    assign_spatial_split,
    compute_train_channel_stats,
)
from megalith_lidar.lidar_wcs import fetch_or_load_paired_tile
from megalith_lidar.model import MegalithLoss, MegalithUNet
from megalith_lidar.stats import (
    PlattCalibrator,
    evaluate_predictions_statistically,
    select_validation_thresholds,
)
from megalith_lidar.terrain_features import compute_terrain_features
from scripts.train import collect_split_predictions, compute_detailed_subgroup_analysis
from scripts.train_enclosure import compute_enclosure_subgroup_analysis
from scripts.train_roman import compute_roman_subgroup_analysis


def _verify_positive_tile(
    spec: Dict,
    catalog: MonumentCatalog,
    tile_cache_dir: Path,
    target_domain: str,
    min_p95_relief: float = 0.16,
) -> Optional[Dict]:
    res = fetch_or_load_paired_tile(
        spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"], tile_cache_dir
    )
    if res is None:
        return None
    dtm, dsm = res
    if dtm.shape != (256, 256) or dsm.shape != (256, 256):
        return None

    if target_domain == "enclosure":
        mask = catalog.rasterize_enclosure_tile(
            spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"], size=256
        )
    elif target_domain == "roman":
        mask = catalog.rasterize_roman_tile(
            spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"], size=256
        )
    else:
        mask = catalog.rasterize_tile(
            spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"], size=256
        )
    pos_px = (mask > 0) & (mask != 255)
    n_pos_px = int(np.sum(pos_px))
    if n_pos_px < 16:
        return None

    feats = compute_terrain_features(dtm, dsm)
    rel16 = np.abs(feats[0] + feats[1])
    p95_rel = float(np.percentile(rel16[pos_px], 95.0))
    if p95_rel < min_p95_relief:
        return None

    spec["positive_pixels"] = n_pos_px
    spec["relief_p95_m"] = round(p95_rel, 3)
    spec["monument_relief_p95_m"] = round(p95_rel, 3)
    return spec


def _verify_negative_tile(
    spec: Dict,
    catalog: MonumentCatalog,
    tile_cache_dir: Path,
    target_domain: str,
    require_hard_relief: bool = False,
) -> Optional[Dict]:
    hits = catalog.query_bbox(
        spec["e_min"] - 80, spec["n_min"] - 80, spec["e_max"] + 80, spec["n_max"] + 80
    )
    if target_domain == "prehistoric":
        if any(h.cls in (1, 2, 3) for h in hits):
            return None
    elif target_domain == "roman":
        if any(h.is_roman_or_road for h in hits):
            return None
    elif target_domain == "enclosure":
        if any(h.is_enclosure_or_settlement for h in hits):
            return None

    res = fetch_or_load_paired_tile(
        spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"], tile_cache_dir
    )
    if res is None:
        return None
    dtm, dsm = res
    if dtm.shape != (256, 256) or dsm.shape != (256, 256):
        return None

    if require_hard_relief:
        feats = compute_terrain_features(dtm, dsm)
        rel16 = feats[0] + feats[1]
        p995_rel = float(np.percentile(np.abs(rel16), 99.5))
        if p995_rel < 0.28:
            return None
    spec["positive_pixels"] = 0
    return spec


def expand_train_split_manifest(
    base_manifest_path: Path,
    out_manifest_path: Path,
    catalog: MonumentCatalog,
    tile_cache_dir: Path,
    target_domain: str,
    all_manifests: List[Path],
    seed: int = 20261023,
) -> List[Dict]:
    """
    Expand ONLY the `train` split of `base_manifest_path` while keeping `val` and `test`
    splits 100% identical to `base_manifest_path` for strict apples-to-apples evaluation.
    """
    if out_manifest_path.exists():
        cached = json.loads(out_manifest_path.read_text(encoding="utf-8"))
        if len(cached) > 2600:
            return cached

    base_items = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    half = 128

    existing_coords: Set[Tuple[int, int]] = {
        (int(m["e_min"]), int(m["n_min"])) for m in base_items
    }
    existing_pos_entries: Set[int] = {
        int(m.get("list_entry", 0)) for m in base_items if m.get("is_positive") and m.get("list_entry")
    }

    # 1. Cross-pollinate already-cached train-split negatives from the other two manifests
    cross_neg_candidates: List[Dict] = []
    for mp in all_manifests:
        if not mp.exists() or mp == base_manifest_path:
            continue
        other_items = json.loads(mp.read_text(encoding="utf-8"))
        for m in other_items:
            if m.get("split") != "train":
                continue
            coord = (int(m["e_min"]), int(m["n_min"]))
            if coord in existing_coords:
                continue
            ce = (int(m["e_min"]) + int(m["e_max"])) // 2
            cn = (int(m["n_min"]) + int(m["n_max"])) // 2
            if assign_spatial_split(ce, cn) != "train":
                continue

            if not m.get("is_positive"):
                existing_coords.add(coord)
                sub = m.get("negative_subtype", "matched_environment")
                if sub in ("anthropogenic_non_target", "anthropogenic_non_roman", "anthropogenic_non_enclosure"):
                    sub = (
                        "anthropogenic_non_target"
                        if target_domain == "prehistoric"
                        else (f"anthropogenic_non_{target_domain}")
                    )
                cross_neg_candidates.append(
                    {
                        "sample_id": f"v3_xneg_{target_domain}_{coord[0]}_{coord[1]}",
                        "list_entry": 0,
                        "name": m.get("name", f"Cross Hard Negative ({sub})"),
                        "primary_class": 0,
                        "is_positive": False,
                        "negative_subtype": sub,
                        "e_min": int(m["e_min"]),
                        "e_max": int(m["e_max"]),
                        "n_min": int(m["n_min"]),
                        "n_max": int(m["n_max"]),
                        "split": "train",
                    }
                )
            else:
                # Positive in another domain -> hard anthropogenic negative if outside target_domain
                existing_coords.add(coord)
                sub = (
                    "anthropogenic_non_target"
                    if target_domain == "prehistoric"
                    else f"anthropogenic_non_{target_domain}"
                )
                cross_neg_candidates.append(
                    {
                        "sample_id": f"v3_anthro_{target_domain}_{coord[0]}_{coord[1]}",
                        "list_entry": 0,
                        "name": f"Cross-Domain Earthwork Negative: {m.get('name', '')}",
                        "primary_class": 0,
                        "is_positive": False,
                        "negative_subtype": sub,
                        "e_min": int(m["e_min"]),
                        "e_max": int(m["e_max"]),
                        "n_min": int(m["n_min"]),
                        "n_max": int(m["n_max"]),
                        "split": "train",
                    }
                )

    # 2. Gather additional train-split positive candidates (unused NHLE train monuments + offset crops)
    new_pos_candidates: List[Dict] = []
    if target_domain == "prehistoric":
        train_barrows = [
            r
            for r in catalog.records
            if r.cls == CLASS_BARROW_CAIRN
            and r.list_entry not in existing_pos_entries
            and assign_spatial_split(r.easting, r.northing) == "train"
        ]
        train_barrows.sort(key=lambda r: (int(r.easting // 40000), int(r.northing // 40000), r.list_entry))
        if len(train_barrows) > 850:
            idx = np.linspace(0, len(train_barrows) - 1, 850, dtype=int)
            train_barrows = [train_barrows[i] for i in idx]

        # Also add a shifted second crop (+/-56m) for train-split Long Barrows (cls 1) & Stone Circles/Henges (cls 2)
        train_rare = [
            r
            for r in catalog.records
            if r.cls in (CLASS_MEGALITHIC_TOMB_LONG_BARROW, CLASS_CIRCLE_HENGE_STONE)
            and assign_spatial_split(r.easting, r.northing) == "train"
        ]
        for rec in train_barrows:
            je, jn = int(rng.integers(-48, 49)), int(rng.integers(-48, 49))
            ce, cn = int(round(rec.easting)) + je, int(round(rec.northing)) + jn
            if assign_spatial_split(ce, cn) != "train":
                continue
            coord = (ce - half, cn - half)
            if coord in existing_coords:
                continue
            existing_coords.add(coord)
            new_pos_candidates.append(
                {
                    "sample_id": f"v3_pre_pos_{rec.list_entry}",
                    "list_entry": rec.list_entry,
                    "name": rec.name,
                    "primary_class": int(rec.cls),
                    "is_positive": True,
                    "negative_subtype": None,
                    "e_min": ce - half,
                    "e_max": ce + half,
                    "n_min": cn - half,
                    "n_max": cn + half,
                    "split": "train",
                }
            )
        for rec in train_rare:
            de = int(rng.choice([-56, 56]))
            dn = int(rng.choice([-56, 56]))
            ce, cn = int(round(rec.easting)) + de, int(round(rec.northing)) + dn
            if assign_spatial_split(ce, cn) != "train":
                continue
            coord = (ce - half, cn - half)
            if coord in existing_coords:
                continue
            existing_coords.add(coord)
            new_pos_candidates.append(
                {
                    "sample_id": f"v3_pre_shift_{rec.list_entry}",
                    "list_entry": rec.list_entry,
                    "name": rec.name,
                    "primary_class": int(rec.cls),
                    "is_positive": True,
                    "negative_subtype": None,
                    "e_min": ce - half,
                    "e_max": ce + half,
                    "n_min": cn - half,
                    "n_max": cn + half,
                    "split": "train",
                }
            )

    elif target_domain == "roman":
        train_roman = [
            r
            for r in catalog.records
            if r.is_roman_or_road and assign_spatial_split(r.easting, r.northing) == "train"
        ]
        for rec in train_roman:
            # 1) Include any train-split Roman monument not yet in manifest
            if rec.list_entry not in existing_pos_entries:
                je, jn = int(rng.integers(-44, 45)), int(rng.integers(-44, 45))
                ce, cn = int(round(rec.easting)) + je, int(round(rec.northing)) + jn
                if assign_spatial_split(ce, cn) == "train":
                    coord = (ce - half, cn - half)
                    if coord not in existing_coords:
                        existing_coords.add(coord)
                        new_pos_candidates.append(
                            {
                                "sample_id": f"v3_rom_pos_{rec.list_entry}",
                                "list_entry": rec.list_entry,
                                "name": rec.name,
                                "primary_class": int(rec.roman_cls),
                                "is_positive": True,
                                "negative_subtype": None,
                                "e_min": ce - half,
                                "e_max": ce + half,
                                "n_min": cn - half,
                                "n_max": cn + half,
                                "split": "train",
                            }
                        )
            # 2) Add an offset window (+/-56m) for train-split Roman roads & villas/forts
            de = int(rng.choice([-56, 56]))
            dn = int(rng.choice([-56, 56]))
            ce, cn = int(round(rec.easting)) + de, int(round(rec.northing)) + dn
            if assign_spatial_split(ce, cn) == "train":
                coord = (ce - half, cn - half)
                if coord not in existing_coords:
                    existing_coords.add(coord)
                    new_pos_candidates.append(
                        {
                            "sample_id": f"v3_rom_shift_{rec.list_entry}",
                            "list_entry": rec.list_entry,
                            "name": rec.name,
                            "primary_class": int(rec.roman_cls),
                            "is_positive": True,
                            "negative_subtype": None,
                            "e_min": ce - half,
                            "e_max": ce + half,
                            "n_min": cn - half,
                            "n_max": cn + half,
                            "split": "train",
                        }
                    )

    elif target_domain == "enclosure":
        train_enc = [
            r
            for r in catalog.records
            if r.is_enclosure_or_settlement and assign_spatial_split(r.easting, r.northing) == "train"
        ]
        for rec in train_enc:
            if rec.list_entry not in existing_pos_entries:
                je, jn = int(rng.integers(-44, 45)), int(rng.integers(-44, 45))
                ce, cn = int(round(rec.easting)) + je, int(round(rec.northing)) + jn
                if assign_spatial_split(ce, cn) == "train":
                    coord = (ce - half, cn - half)
                    if coord not in existing_coords:
                        existing_coords.add(coord)
                        new_pos_candidates.append(
                            {
                                "sample_id": f"v3_enc_pos_{rec.list_entry}",
                                "list_entry": rec.list_entry,
                                "name": rec.name,
                                "primary_class": int(rec.enclosure_cls),
                                "is_positive": True,
                                "negative_subtype": None,
                                "e_min": ce - half,
                                "e_max": ce + half,
                                "n_min": cn - half,
                                "n_max": cn + half,
                                "split": "train",
                            }
                        )
            elif rec.enclosure_cls in (ENCLOSURE_CLASS_HILLFORT_BANJO_ENCLOSURE, ENCLOSURE_CLASS_DYKE_FIELD_SYSTEM):
                de = int(rng.choice([-56, 56]))
                dn = int(rng.choice([-56, 56]))
                ce, cn = int(round(rec.easting)) + de, int(round(rec.northing)) + dn
                if assign_spatial_split(ce, cn) == "train":
                    coord = (ce - half, cn - half)
                    if coord not in existing_coords:
                        existing_coords.add(coord)
                        new_pos_candidates.append(
                            {
                                "sample_id": f"v3_enc_shift_{rec.list_entry}",
                                "list_entry": rec.list_entry,
                                "name": rec.name,
                                "primary_class": int(rec.enclosure_cls),
                                "is_positive": True,
                                "negative_subtype": None,
                                "e_min": ce - half,
                                "e_max": ce + half,
                                "n_min": cn - half,
                                "n_max": cn + half,
                                "split": "train",
                            }
                        )

    print(
        f"[{target_domain} v3] Verifying {len(new_pos_candidates)} new train positives "
        f"and {len(cross_neg_candidates)} cross-domain hard negatives..."
    )

    verified_new_pos: List[Dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [
            pool.submit(_verify_positive_tile, c, catalog, tile_cache_dir, target_domain, 0.16)
            for c in new_pos_candidates
        ]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                verified_new_pos.append(res)

    rng.shuffle(cross_neg_candidates)
    cross_neg_candidates = cross_neg_candidates[:950]
    verified_new_neg: List[Dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [
            pool.submit(_verify_negative_tile, c, catalog, tile_cache_dir, target_domain, False)
            for c in cross_neg_candidates
        ]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                verified_new_neg.append(res)

    merged = list(base_items) + verified_new_pos + verified_new_neg
    merged.sort(key=lambda m: m["sample_id"])
    out_manifest_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(
        f"[{target_domain} v3] Saved {out_manifest_path.name}: {len(base_items)} -> {len(merged)} tiles "
        f"(+{len(verified_new_pos)} train pos, +{len(verified_new_neg)} train hard neg; val & test unchanged)."
    )
    return merged


def train_and_evaluate_v3_domain(
    target_domain: str,
    manifest_v3: List[Dict],
    catalog: MonumentCatalog,
    tile_cache_dir: Path,
    feat_cache_dir: Path,
    ckpt_dir: Path,
    prod_ckpt_name: str,
    prod_metrics_name: str,
    class_names: Dict[int, str],
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 6e-4,
) -> Tuple[bool, Dict]:
    train_specs = [m for m in manifest_v3 if m["split"] == "train"]
    val_specs = [m for m in manifest_v3 if m["split"] == "val"]
    test_specs = [m for m in manifest_v3 if m["split"] == "test"]
    manifest_by_id = {m["sample_id"]: m for m in manifest_v3}

    train_ds = MegalithLidarDataset(
        train_specs, catalog, tile_cache_dir, augment=True, feature_cache_dir=feat_cache_dir, target_domain=target_domain
    )
    val_ds = MegalithLidarDataset(
        val_specs, catalog, tile_cache_dir, augment=False, feature_cache_dir=feat_cache_dir, target_domain=target_domain
    )
    test_ds = MegalithLidarDataset(
        test_specs, catalog, tile_cache_dir, augment=False, feature_cache_dir=feat_cache_dir, target_domain=target_domain
    )

    ch_mean, ch_std = compute_train_channel_stats(train_ds)
    train_ds.set_normalization(ch_mean, ch_std)
    val_ds.set_normalization(ch_mean, ch_std)
    test_ds.set_normalization(ch_mean, ch_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    n_pos_train = sum(1 for m in train_specs if m["is_positive"])
    n_neg_train = len(train_specs) - n_pos_train
    pos_tile_weight = float(n_neg_train) / max(float(n_pos_train), 1.0)
    print(
        f"\n[{target_domain} v3] Training on {len(train_specs)} train tiles "
        f"({n_pos_train} pos, {n_neg_train} neg; val={len(val_specs)}, test={len(test_specs)})..."
    )

    model = MegalithUNet(pretrained=True).to(device)
    criterion = MegalithLoss(pos_pixel_weight=2.5, pos_tile_weight=pos_tile_weight).to(device)

    enc_modules = (model.stem, model.enc1, model.enc2, model.enc3, model.enc4)
    enc_param_ids = {id(p) for m in enc_modules for p in m.parameters()}
    enc_params = [p for p in model.parameters() if id(p) in enc_param_ids]
    dec_params = [p for p in model.parameters() if id(p) not in enc_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr * 0.25},
            {"params": dec_params, "lr": lr},
        ],
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    v3_ckpt_path = ckpt_dir / prod_ckpt_name.replace("_best.pt", "_v3_best.pt")
    best_val_sel = -1.0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_loader, 1):
            x = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            tile_label = batch["tile_label"].to(device, non_blocking=True)
            type_label = batch["type_label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(x)
                loss, _ = criterion(out, mask, tile_label, type_label)
                scaled_loss = loss / float(grad_accum)

            scaled_loss.backward()
            loss_sum += float(loss.item())
            steps += 1
            if step_idx % grad_accum == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        val_out = collect_split_predictions(model, val_loader, criterion, device, use_d4_tta=False)
        val_auc = float(roc_auc_score(val_out["y_true"], val_out["raw_logits"]))
        sel = val_auc + 0.15 * val_out["pos_seg_dice"]
        marker = ""
        if sel > best_val_sel:
            best_val_sel = sel
            torch.save(
                {
                    "epoch": epoch,
                    "target_domain": target_domain,
                    "class_names": class_names,
                    "model_state_dict": model.state_dict(),
                    "channel_mean": ch_mean.tolist(),
                    "channel_std": ch_std.tolist(),
                },
                v3_ckpt_path,
            )
            marker = " *BEST*"
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(loss_sum / max(steps, 1), 4),
                "val_loss": val_out["loss"],
                "val_roc_auc": round(val_auc, 4),
                "val_pos_seg_dice": val_out["pos_seg_dice"],
            }
        )
        print(
            f"  [{target_domain} v3] Epoch {epoch:02d}/{epochs:02d} ({time.time() - t0:.1f}s) | "
            f"train_loss={loss_sum / max(steps, 1):.4f} | val_auc={val_auc:.4f} dice={val_out['pos_seg_dice']:.4f}{marker}"
        )

    ckpt = torch.load(v3_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_d4 = collect_split_predictions(model, val_loader, criterion, device, use_d4_tta=True)
    calibrator = PlattCalibrator().fit(val_d4["raw_logits"], val_d4["y_true"])
    val_cal_probs = calibrator.predict_proba(val_d4["raw_logits"])
    val_thresholds = select_validation_thresholds(val_cal_probs, val_d4["y_true"], target_fprs=(0.01, 0.05, 0.10))
    neg_val_peaks = val_d4["seg_peak_probs"][val_d4["y_true"] == 0]
    pixel_th_fpr05 = float(np.quantile(neg_val_peaks, 0.95))

    ckpt["calibrator"] = calibrator.to_dict()
    ckpt["val_thresholds"] = val_thresholds
    ckpt["val_pixel_threshold_fpr05"] = round(pixel_th_fpr05, 4)
    torch.save(ckpt, v3_ckpt_path)

    test_d4 = collect_split_predictions(model, test_loader, criterion, device, use_d4_tta=True)
    val_stats = evaluate_predictions_statistically(val_d4["y_true"], val_d4["raw_logits"], calibrator, val_thresholds, n_boot=1000)
    test_stats = evaluate_predictions_statistically(test_d4["y_true"], test_d4["raw_logits"], calibrator, val_thresholds, n_boot=1000)

    if target_domain == "prehistoric":
        test_subgroups = compute_detailed_subgroup_analysis(test_d4, calibrator, val_thresholds, manifest_by_id)
    elif target_domain == "roman":
        test_subgroups = compute_roman_subgroup_analysis(test_d4, calibrator, val_thresholds, manifest_by_id)
    else:
        test_subgroups = compute_enclosure_subgroup_analysis(test_d4, calibrator, val_thresholds, manifest_by_id)

    prod_metrics_path = ckpt_dir / prod_metrics_name
    old_metrics = json.loads(prod_metrics_path.read_text(encoding="utf-8")) if prod_metrics_path.exists() else {}
    old_test_auc = float(old_metrics.get("model_test_metrics", {}).get("roc_auc", 0.0))
    new_test_auc = float(test_stats["roc_auc"])

    old_op01 = old_metrics.get("model_test_metrics", {}).get("operating_points", {}).get("fpr_0.01", {})
    new_op01 = test_stats.get("operating_points", {}).get("fpr_0.01", {})
    old_lr01 = float(old_op01.get("likelihood_ratio_pos", 0.0))
    new_lr01 = float(new_op01.get("likelihood_ratio_pos", 0.0))

    print(
        f"\n[{target_domain} v2 vs v3 Held-Out Test Comparison] "
        f"Test ROC-AUC: {old_test_auc:.4f} -> {new_test_auc:.4f} | "
        f"Strict 1% Op LR+: {old_lr01:.1f}x -> {new_lr01:.1f}x"
    )

    split_counts = Counter(m["split"] for m in manifest_v3)
    cls_counts = Counter(m["primary_class"] for m in manifest_v3 if m["is_positive"])
    neg_counts = Counter(m["negative_subtype"] for m in manifest_v3 if not m["is_positive"])

    new_summary = dict(old_metrics)
    new_summary.update(
        {
            "dataset_total_tiles": len(manifest_v3),
            "split_counts": dict(split_counts),
            "class_counts": dict(cls_counts),
            "negative_null_counts": dict(neg_counts),
            "best_epoch": int(ckpt["epoch"]),
            "train_channel_mean": ch_mean.tolist(),
            "train_channel_std": ch_std.tolist(),
            "platt_calibrator": calibrator.to_dict(),
            "val_selected_thresholds": val_thresholds,
            "val_pixel_threshold_fpr05": round(pixel_th_fpr05, 4),
            "model_validation_metrics": val_stats,
            "model_test_metrics": test_stats,
            "model_test_pos_seg_dice": test_d4["pos_seg_dice"],
            "model_test_subgroup_and_confounding_analysis": test_subgroups,
            "history": history,
        }
    )

    v3_metrics_path = ckpt_dir / prod_metrics_name.replace(".json", "_v3.json")
    v3_metrics_path.write_text(json.dumps(new_summary, indent=2), encoding="utf-8")

    # Promote v3 if Held-Out Test ROC-AUC improves (or stays within 0.001 while improving Strict 1% Op LR+)
    improved = (new_test_auc > old_test_auc + 0.001) or (
        new_test_auc >= old_test_auc - 0.001 and new_lr01 > old_lr01 * 1.05
    )
    if improved:
        prod_ckpt_path = ckpt_dir / prod_ckpt_name
        if prod_ckpt_path.exists():
            shutil.copy2(prod_ckpt_path, ckpt_dir / prod_ckpt_name.replace("_best.pt", "_v2_backup.pt"))
        shutil.copy2(v3_ckpt_path, prod_ckpt_path)
        prod_metrics_path.write_text(json.dumps(new_summary, indent=2), encoding="utf-8")
        print(f"  >>> PROMOTED {target_domain} v3 to production ({prod_ckpt_path.name})!")
    else:
        print(f"  --- Kept {target_domain} v2 in production (v3 saved at {v3_ckpt_path.name}).")

    return improved, new_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand train manifests (v3), train on cuda:0, and upgrade/rescore if improved.")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/bulk/megalith-lidar"))
    parser.add_argument("--docs-dir", type=Path, default=Path("/home/ian/Code/megalith-lidar/docs"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    catalog = download_or_load_nhle_catalog(args.data_dir / "catalog" / "nhle_scheduled_monuments_27700.json")
    tile_cache_dir = args.data_dir / "tiles"
    feat_cache_dir = args.data_dir / "features_cache"
    ckpt_dir = args.data_dir / "checkpoints"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    m_pre = args.data_dir / "manifest_uk_megaliths.json"
    m_rom = args.data_dir / "manifest_uk_roman.json"
    m_enc = args.data_dir / "manifest_uk_enclosure.json"
    all_m = [m_pre, m_rom, m_enc]

    configs = [
        (
            "enclosure",
            m_enc,
            args.data_dir / "manifest_uk_enclosure_v3.json",
            "enclosure_unet_best.pt",
            "enclosure_evaluation_metrics.json",
            ENCLOSURE_CLASS_NAMES,
        ),
        (
            "roman",
            m_rom,
            args.data_dir / "manifest_uk_roman_v3.json",
            "roman_unet_best.pt",
            "roman_evaluation_metrics.json",
            ROMAN_CLASS_NAMES,
        ),
        (
            "prehistoric",
            m_pre,
            args.data_dir / "manifest_uk_megaliths_v3.json",
            "megalith_unet_best.pt",
            "evaluation_metrics_v2_unbiased.json",
            CLASS_NAMES,
        ),
    ]

    for domain, base_mp, v3_mp, ckpt_name, metrics_name, cls_names in configs:
        manifest_v3 = expand_train_split_manifest(
            base_manifest_path=base_mp,
            out_manifest_path=v3_mp,
            catalog=catalog,
            tile_cache_dir=tile_cache_dir,
            target_domain=domain,
            all_manifests=all_m,
        )
        improved, new_sum = train_and_evaluate_v3_domain(
            target_domain=domain,
            manifest_v3=manifest_v3,
            catalog=catalog,
            tile_cache_dir=tile_cache_dir,
            feat_cache_dir=feat_cache_dir,
            ckpt_dir=ckpt_dir,
            prod_ckpt_name=ckpt_name,
            prod_metrics_name=metrics_name,
            class_names=cls_names,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
        if improved:
            if domain == "enclosure":
                # Wipe existing enclosure_survey geojsons so next scan cycle re-scores all completed squares with v3
                for gj in (args.data_dir / "enclosure_survey").glob("*/*_candidates.geojson"):
                    gj.unlink(missing_ok=True)
                from scripts.scan_enclosure_batch import run_enclosure_scan
                run_enclosure_scan(args.data_dir, ckpt_dir / ckpt_name, args.docs_dir, args.device, max_squares=100)
            elif domain == "roman":
                for gj in (args.data_dir / "roman_survey").glob("*/*_candidates.geojson"):
                    gj.unlink(missing_ok=True)
                from scripts.scan_roman_batch import run_roman_scan
                run_roman_scan(args.data_dir, ckpt_dir / ckpt_name, args.docs_dir, args.device, max_squares=100)
            elif domain == "prehistoric":
                shutil.copy2(ckpt_dir / metrics_name, args.docs_dir / metrics_name)
                repo_root = Path(__file__).resolve().parents[1]
                subprocess.run(
                    [
                        str(repo_root / ".venv" / "bin" / "python"),
                        str(repo_root / "scripts" / "run_national_batch.py"),
                        "--rescore-completed",
                        "--device",
                        args.device,
                        "--git-push",
                    ],
                    cwd=repo_root,
                    check=False,
                )


if __name__ == "__main__":
    main()
