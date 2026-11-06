#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megalith_lidar.catalog import (
    ROMAN_CLASS_BACKGROUND,
    ROMAN_CLASS_FORT_CAMP_SETTLEMENT,
    ROMAN_CLASS_NAMES,
    ROMAN_CLASS_ROAD_AGGER_TRACKWAY,
    ROMAN_CLASS_VILLA_BUILDING,
    MonumentCatalog,
    MonumentRecord,
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
    bootstrap_metric_ci,
    evaluate_predictions_statistically,
    select_validation_thresholds,
)
from megalith_lidar.terrain_features import compute_terrain_features

ROMAN_NEGATIVE_SUBTYPES = (
    "random_national",
    "matched_environment",
    "anthropogenic_non_roman",
    "woodland_canopy_and_pit",
    "trackside_and_field_boundary",
)


def build_or_update_roman_manifest(
    manifest_path: Path,
    megalith_manifest_path: Path,
    catalog: MonumentCatalog,
    tile_cache_dir: Path,
    min_positive_relief_p95_m: float = 0.18,
    seed: int = 20261022,
) -> List[Dict]:
    """
    Assemble the UK Roman Villa, Camp/Settlement & Lost Roman Road dataset manifest:
      1. Positives: All Roman Scheduled Monuments in NHLE Layer 6 across classes 1..3:
           - ROMAN_CLASS_VILLA_BUILDING (1): Roman villas, bath-houses, temples, mausolea
           - ROMAN_CLASS_ROAD_AGGER_TRACKWAY (2): Roman road aggers, causeways, ancient trackways
           - ROMAN_CLASS_FORT_CAMP_SETTLEMENT (3): Roman forts, marching camps, signal stations, RB settlements
         Verified against 1m EA LiDAR coverage and filtered so sites with measurable 1m bare-earth
         topographic relief (`p95 |z - G_16(z)| >= 0.18 m` inside the polygon) are used for positive
         supervision (preventing zero-relief cropmark-only arable fields from corrupting the elevation model).
      2. Negatives:
           - Reuses verified monument-free negatives from `manifest_uk_megaliths.json` ('random_national',
             'matched_environment', 'woodland_canopy_and_pit', 'trackside_and_field_boundary') after
             verifying they do not intersect any Roman monument within 80 m.
           - Adds 'matched_environment' negatives sampled 500..1800 m from Roman positives.
           - Adds 'anthropogenic_non_roman' negatives (prehistoric barrows, medieval moats/abbeys/castles,
             industrial earthworks) so RomanUNet discriminates Roman geometry from prehistoric/medieval earthworks.
    """
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(cached) >= 1200:
                return cached
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    half = 128

    roman_records = [r for r in catalog.records if r.is_roman_or_road]
    roman_records.sort(key=lambda r: (r.roman_cls, int(r.easting // 50000), int(r.northing // 50000), r.list_entry))

    # Cap class 3 (forts + RB settlements) at 360 geographically stratified sites so villas (294) and
    # Roman roads/trackways (183) have balanced representation
    villas = [r for r in roman_records if r.roman_cls == ROMAN_CLASS_VILLA_BUILDING]
    roads = [r for r in roman_records if r.roman_cls == ROMAN_CLASS_ROAD_AGGER_TRACKWAY]
    forts_all = [r for r in roman_records if r.roman_cls == ROMAN_CLASS_FORT_CAMP_SETTLEMENT]
    if len(forts_all) > 360:
        idx = np.linspace(0, len(forts_all) - 1, 360, dtype=int)
        forts = [forts_all[i] for i in idx]
    else:
        forts = forts_all

    pos_candidates: List[Dict] = []
    for rec in villas + roads + forts:
        jitter_e = int(rng.integers(-48, 49))
        jitter_n = int(rng.integers(-48, 49))
        ce = int(round(rec.easting)) + jitter_e
        cn = int(round(rec.northing)) + jitter_n
        e_min, e_max = ce - half, ce + half
        n_min, n_max = cn - half, cn + half
        pos_candidates.append(
            {
                "sample_id": f"roman_pos_{rec.list_entry}",
                "list_entry": rec.list_entry,
                "name": rec.name,
                "primary_class": int(rec.roman_cls),
                "is_positive": True,
                "negative_subtype": None,
                "e_min": e_min,
                "e_max": e_max,
                "n_min": n_min,
                "n_max": n_max,
                "split": assign_spatial_split(ce, cn),
            }
        )

    print(f"      Fetching & verifying 1m LiDAR bare-earth relief for {len(pos_candidates)} Roman monument candidates...")

    def _verify_pos(spec: Dict) -> Optional[Dict]:
        res = fetch_or_load_paired_tile(
            spec["e_min"],
            spec["e_max"],
            spec["n_min"],
            spec["n_max"],
            cache_dir=tile_cache_dir,
        )
        if res is None:
            return None
        dtm, dsm = res
        if dtm.shape != (256, 256):
            return None
        mask = catalog.rasterize_roman_tile(spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"], size=256)
        pos_px = mask > 0
        if int(np.sum(pos_px)) < 16:
            return None
        feats = compute_terrain_features(dtm, dsm)
        rel16 = np.abs(feats[0] + feats[1])
        p95_rel = float(np.percentile(rel16[pos_px], 95))
        if p95_rel < min_positive_relief_p95_m:
            return None
        spec["relief_p95_m"] = round(p95_rel, 3)
        return spec

    verified_pos: List[Dict] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(_verify_pos, s) for s in pos_candidates]
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                verified_pos.append(r)
    verified_pos.sort(key=lambda x: x["sample_id"])
    print(f"      Verified {len(verified_pos)} Roman positive tiles with measurable 1m LiDAR relief (>= {min_positive_relief_p95_m}m).")

    # Reuse monument-free negatives and non-Roman earthwork negatives from manifest_uk_megaliths.json
    neg_list: List[Dict] = []
    used_bboxes = {(int(m["e_min"]), int(m["e_max"]), int(m["n_min"]), int(m["n_max"])) for m in verified_pos}

    if megalith_manifest_path.exists():
        meg_manifest = json.loads(megalith_manifest_path.read_text(encoding="utf-8"))
        non_roman_pos_pool = []
        for item in meg_manifest:
            e0, e1, n0, n1 = int(item["e_min"]), int(item["e_max"]), int(item["n_min"]), int(item["n_max"])
            if (e0, e1, n0, n1) in used_bboxes:
                continue
            hits = catalog.query_bbox(e0 - 80, n0 - 80, e1 + 80, n1 + 80)
            if any(h.is_roman_or_road for h in hits):
                continue
            ce, cn = (e0 + e1) // 2, (n0 + n1) // 2
            sp = assign_spatial_split(ce, cn)
            if not item["is_positive"]:
                sub = item.get("negative_subtype") or "matched_environment"
                if sub == "anthropogenic_non_target":
                    sub = "anthropogenic_non_roman"
                neg_list.append(
                    {
                        "sample_id": item["sample_id"],
                        "list_entry": item.get("list_entry", 0),
                        "name": item["name"],
                        "primary_class": ROMAN_CLASS_BACKGROUND,
                        "is_positive": False,
                        "negative_subtype": sub,
                        "e_min": e0,
                        "e_max": e1,
                        "n_min": n0,
                        "n_max": n1,
                        "split": sp,
                    }
                )
                used_bboxes.add((e0, e1, n0, n1))
            else:
                non_roman_pos_pool.append(item)

        # Also sample 140 prehistoric barrow/henge/tomb tiles as hard 'anthropogenic_non_roman' negatives
        # so RomanUNet explicitly learns NOT to fire on round barrows or stone circles
        rng.shuffle(non_roman_pos_pool)
        for item in non_roman_pos_pool[:140]:
            e0, e1, n0, n1 = int(item["e_min"]), int(item["e_max"]), int(item["n_min"]), int(item["n_max"])
            ce, cn = (e0 + e1) // 2, (n0 + n1) // 2
            neg_list.append(
                {
                    "sample_id": f"neg_prehist_{item['list_entry']}",
                    "list_entry": item["list_entry"],
                    "name": f"Non-Roman prehistoric monument: {item['name']}",
                    "primary_class": ROMAN_CLASS_BACKGROUND,
                    "is_positive": False,
                    "negative_subtype": "anthropogenic_non_roman",
                    "e_min": e0,
                    "e_max": e1,
                    "n_min": n0,
                    "n_max": n1,
                    "split": assign_spatial_split(ce, cn),
                }
            )
            used_bboxes.add((e0, e1, n0, n1))

    manifest = verified_pos + neg_list
    manifest.sort(key=lambda x: x["sample_id"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@torch.no_grad()
def collect_split_predictions(
    model: MegalithUNet,
    loader: DataLoader,
    criterion: MegalithLoss,
    device: torch.device,
    use_d4_tta: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    y_true: List[int] = []
    raw_logits: List[float] = []
    seg_peak_probs: List[float] = []
    y_types: List[int] = []
    relief_p95: List[float] = []
    sample_ids: List[str] = []

    pos_pixel_inter = 0.0
    pos_pixel_union = 0.0

    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        tile_label = batch["tile_label"].to(device, non_blocking=True)
        type_label = batch["type_label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(x)
            loss, _ = criterion(out, mask, tile_label, type_label)
            seg_logit = out["seg_logits"].float().squeeze(1)
            tile_logit = out["tile_logits"].float().squeeze(1)

            if use_d4_tta:
                xf = torch.flip(x, [3])
                aug_list = [
                    (torch.rot90(x, 1, [2, 3]), lambda s: torch.rot90(s, -1, [1, 2])),
                    (torch.rot90(x, 2, [2, 3]), lambda s: torch.rot90(s, -2, [1, 2])),
                    (torch.rot90(x, 3, [2, 3]), lambda s: torch.rot90(s, -3, [1, 2])),
                    (xf, lambda s: torch.flip(s, [2])),
                    (torch.rot90(xf, 1, [2, 3]), lambda s: torch.flip(torch.rot90(s, -1, [1, 2]), [2])),
                    (torch.rot90(xf, 2, [2, 3]), lambda s: torch.flip(torch.rot90(s, -2, [1, 2]), [2])),
                    (torch.rot90(xf, 3, [2, 3]), lambda s: torch.flip(torch.rot90(s, -3, [1, 2]), [2])),
                ]
                for xa, inv_fn in aug_list:
                    outa = model(xa)
                    seg_logit = seg_logit + inv_fn(outa["seg_logits"].float().squeeze(1))
                    tile_logit = tile_logit + outa["tile_logits"].float().squeeze(1)
                seg_logit = seg_logit / 8.0
                tile_logit = tile_logit / 8.0

        total_loss += float(loss.item())
        n_batches += 1

        seg_probs = torch.sigmoid(seg_logit)
        valid_px = (mask != 255).float()
        bin_mask = ((mask > 0) & (mask != 255)).float()
        pos_mask = tile_label > 0.5
        if torch.any(pos_mask):
            pred_bin = (seg_probs[pos_mask] >= 0.5).float() * valid_px[pos_mask]
            gt_bin = bin_mask[pos_mask]
            pos_pixel_inter += float((pred_bin * gt_bin).sum().item())
            pos_pixel_union += float((pred_bin.sum() + gt_bin.sum()).item())

        flat_seg = seg_probs.view(seg_probs.shape[0], -1)
        top64 = torch.topk(flat_seg, k=min(64, flat_seg.shape[1]), dim=1)[0].mean(dim=1)

        y_true.extend(tile_label.cpu().numpy().astype(int).tolist())
        raw_logits.extend(tile_logit.cpu().numpy().tolist())
        seg_peak_probs.extend(top64.cpu().numpy().tolist())
        y_types.extend(type_label.cpu().numpy().astype(int).tolist())
        relief_p95.extend(batch["relief_p95_m"].numpy().tolist())
        sample_ids.extend(batch["sample_id"])

    return {
        "loss": round(total_loss / max(n_batches, 1), 4),
        "pos_seg_dice": round(float((2.0 * pos_pixel_inter + 1.0) / (pos_pixel_union + 1.0)), 4),
        "y_true": np.array(y_true, dtype=np.int32),
        "raw_logits": np.array(raw_logits, dtype=np.float64),
        "seg_peak_probs": np.array(seg_peak_probs, dtype=np.float64),
        "y_types": np.array(y_types, dtype=np.int32),
        "relief_p95": np.array(relief_p95, dtype=np.float64),
        "sample_ids": sample_ids,
    }


def compute_roman_subgroup_analysis(
    preds_dict: Dict[str, np.ndarray],
    calibrator: PlattCalibrator,
    val_thresholds: Dict[str, Dict[str, float]],
    manifest_by_id: Dict[str, Dict],
) -> Dict:
    y = preds_dict["y_true"]
    probs = calibrator.predict_proba(preds_dict["raw_logits"])
    sids = preds_dict["sample_ids"]
    y_types = preds_dict["y_types"]
    relief = preds_dict["relief_p95"]

    pos_mask = y == 1

    null_roc_aucs = {}
    for subtype in ROMAN_NEGATIVE_SUBTYPES:
        neg_sub_mask = np.array(
            [not manifest_by_id[sid]["is_positive"] and manifest_by_id[sid].get("negative_subtype") == subtype for sid in sids]
        )
        if np.any(neg_sub_mask):
            sub_idx = np.where(pos_mask | neg_sub_mask)[0]
            est, lo, hi = bootstrap_metric_ci(
                y[sub_idx],
                probs[sub_idx],
                lambda yt, yp: float(roc_auc_score(yt, yp)),
                n_boot=1000,
            )
            null_roc_aucs[subtype] = {
                "n_negatives": int(np.sum(neg_sub_mask)),
                "roc_auc": est,
                "ci95": [lo, hi],
            }

    by_threshold = {}
    for th_name in ("fpr_0.01", "fpr_0.05", "fpr_0.10", "max_youden_j"):
        th = val_thresholds[th_name]["threshold"]
        pred_pos = probs >= th

        cls_tpr = {}
        for cls_id in (1, 2, 3):
            cm = y_types == cls_id
            if np.any(cm):
                cls_tpr[ROMAN_CLASS_NAMES[cls_id]] = round(float(np.mean(pred_pos[cm])), 4)

        neg_fpr = {}
        for subtype in ROMAN_NEGATIVE_SUBTYPES:
            nm = np.array(
                [not manifest_by_id[sid]["is_positive"] and manifest_by_id[sid].get("negative_subtype") == subtype for sid in sids]
            )
            if np.any(nm):
                neg_fpr[subtype] = round(float(np.mean(pred_pos[nm])), 4)

        snr_tiers = {
            "low_moderate_relief_0.18_to_0.45m": (pos_mask) & (relief < 0.45),
            "prominent_relief_0.45_to_0.90m": (pos_mask) & (relief >= 0.45) & (relief < 0.90),
            "high_relief_ge_0.90m": (pos_mask) & (relief >= 0.90),
        }
        snr_recall = {}
        for t_label, tm in snr_tiers.items():
            if np.any(tm):
                snr_recall[t_label] = {
                    "n_sites": int(np.sum(tm)),
                    "tpr_recall": round(float(np.mean(pred_pos[tm])), 4),
                }

        by_threshold[th_name] = {
            "threshold": th,
            "tpr_by_monument_class": cls_tpr,
            "fpr_by_negative_null": neg_fpr,
            "tpr_by_physical_relief_snr": snr_recall,
        }

    return {
        "discrimination_by_negative_null": null_roc_aucs,
        "operating_point_breakdowns": by_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistically Rigorous UK LiDAR Roman Villa, Fort & Lost Road Training")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/bulk/megalith-lidar"))
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    catalog_path = data_dir / "catalog" / "nhle_scheduled_monuments_27700.json"
    tile_cache_dir = data_dir / "tiles"
    feat_cache_dir = data_dir / "features_cache"
    manifest_path = data_dir / "manifest_uk_roman.json"
    megalith_manifest_path = data_dir / "manifest_uk_megaliths.json"
    ckpt_dir = data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading Historic England Scheduled Monuments catalog (EPSG:27700)...")
    catalog = download_or_load_nhle_catalog(catalog_path)

    print("[2/6] Building & verifying Roman Villa, Fort & Lost Road dataset manifest...")
    manifest = build_or_update_roman_manifest(
        manifest_path=manifest_path,
        megalith_manifest_path=megalith_manifest_path,
        catalog=catalog,
        tile_cache_dir=tile_cache_dir,
    )
    manifest_by_id = {m["sample_id"]: m for m in manifest}

    split_counts = Counter(m["split"] for m in manifest)
    cls_counts = Counter(ROMAN_CLASS_NAMES[m["primary_class"]] for m in manifest)
    neg_counts = Counter(m["negative_subtype"] for m in manifest if not m["is_positive"])
    print("      Total verified tiles:", len(manifest))
    print("      Splits (pure SHA-256 20km block hash):", dict(split_counts))
    print("      Classes:", dict(cls_counts))
    print("      Negative Null Subtypes:", dict(neg_counts))

    train_specs = [m for m in manifest if m["split"] == "train"]
    val_specs = [m for m in manifest if m["split"] == "val"]
    test_specs = [m for m in manifest if m["split"] == "test"]

    print("[3/6] Pre-computing 6-channel isotropic scale-space features & Train-only Z-score stats...")
    t_feat = time.time()
    all_ds = MegalithLidarDataset(
        manifest, catalog, tile_cache_dir, augment=False, feature_cache_dir=feat_cache_dir, target_domain="roman"
    )
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(all_ds._load_features_and_mask, manifest))

    train_ds = MegalithLidarDataset(
        train_specs, catalog, tile_cache_dir, augment=True, feature_cache_dir=feat_cache_dir, target_domain="roman"
    )
    val_ds = MegalithLidarDataset(
        val_specs, catalog, tile_cache_dir, augment=False, feature_cache_dir=feat_cache_dir, target_domain="roman"
    )
    test_ds = MegalithLidarDataset(
        test_specs, catalog, tile_cache_dir, augment=False, feature_cache_dir=feat_cache_dir, target_domain="roman"
    )

    ch_mean, ch_std = compute_train_channel_stats(train_ds)
    train_ds.set_normalization(ch_mean, ch_std)
    val_ds.set_normalization(ch_mean, ch_std)
    test_ds.set_normalization(ch_mean, ch_std)
    print(f"      Feature cache & Train Z-score stats ready in {time.time() - t_feat:.1f}s.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    n_pos_train = sum(1 for m in train_specs if m["is_positive"])
    n_neg_train = len(train_specs) - n_pos_train
    pos_tile_weight = float(n_neg_train) / max(float(n_pos_train), 1.0)
    print(f"      Train Positives={n_pos_train}, Negatives={n_neg_train} -> prevalence-neutral pos_tile_weight={pos_tile_weight:.4f}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[4/6] Training RomanUNet on {device} (reflection padding, MIL negative-peak suppression)...")
    model = MegalithUNet(pretrained=True).to(device)
    criterion = MegalithLoss(pos_pixel_weight=2.5, pos_tile_weight=pos_tile_weight).to(device)

    enc_modules = (model.stem, model.enc1, model.enc2, model.enc3, model.enc4)
    enc_param_ids = {id(p) for m in enc_modules for p in m.parameters()}
    enc_params = [p for p in model.parameters() if id(p) in enc_param_ids]
    dec_params = [p for p in model.parameters() if id(p) not in enc_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr * 0.25},
            {"params": dec_params, "lr": args.lr},
        ],
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_auc = -1.0
    best_ckpt_path = ckpt_dir / "roman_unet_best.pt"
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        train_loss_sum = 0.0
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
                scaled_loss = loss / float(args.grad_accum)

            scaled_loss.backward()
            train_loss_sum += float(loss.item())
            steps += 1

            if step_idx % args.grad_accum == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        val_out = collect_split_predictions(model, val_loader, criterion, device, use_d4_tta=False)
        val_auc = float(roc_auc_score(val_out["y_true"], val_out["raw_logits"]))
        ep_time = time.time() - t_ep

        sel_criterion = val_auc + 0.15 * val_out["pos_seg_dice"]
        marker = ""
        if sel_criterion > best_val_auc:
            best_val_auc = sel_criterion
            torch.save(
                {
                    "epoch": epoch,
                    "target_domain": "roman",
                    "class_names": ROMAN_CLASS_NAMES,
                    "model_state_dict": model.state_dict(),
                    "channel_mean": ch_mean.tolist(),
                    "channel_std": ch_std.tolist(),
                },
                best_ckpt_path,
            )
            marker = " *BEST*"

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss_sum / max(steps, 1), 4),
                "val_loss": val_out["loss"],
                "val_roc_auc": round(val_auc, 4),
                "val_pos_seg_dice": val_out["pos_seg_dice"],
            }
        )
        print(
            f"  Epoch {epoch:02d}/{args.epochs:02d} ({ep_time:.1f}s) | "
            f"train_loss={train_loss_sum / max(steps, 1):.4f} | "
            f"val_auc={val_auc:.4f} val_pos_dice={val_out['pos_seg_dice']:.4f}{marker}"
        )

    print("\n[5/6] Fitting Platt Calibrator & selecting operating thresholds STRICTLY on Validation split...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
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
    torch.save(ckpt, best_ckpt_path)

    print("      Platt Calibrator (fitted on Val):", calibrator.to_dict())
    print("      Validation-selected thresholds  :", json.dumps(val_thresholds, indent=2))
    print(f"      Validation-selected pixel threshold (95th pct negative peak): {pixel_th_fpr05:.4f}")

    print("\n[6/6] Evaluating once on untouched Held-Out Test Split (with 1,000-resample 95% Bootstrap CIs)...")
    test_d4 = collect_split_predictions(model, test_loader, criterion, device, use_d4_tta=True)
    val_stats = evaluate_predictions_statistically(val_d4["y_true"], val_d4["raw_logits"], calibrator, val_thresholds, n_boot=1000)
    test_stats = evaluate_predictions_statistically(test_d4["y_true"], test_d4["raw_logits"], calibrator, val_thresholds, n_boot=1000)
    test_subgroups = compute_roman_subgroup_analysis(test_d4, calibrator, val_thresholds, manifest_by_id)

    summary_out = {
        "target_domain": "roman_villas_camps_and_lost_roads",
        "methodology": {
            "spatial_partitioning": "20km x 20km BNG blocks assigned strictly by SHA-256 hash (zero manual overrides)",
            "feature_representation": "6-channel isotropic scale-space & differential invariants (exact D4 equivariance, Train-only Z-score standardization)",
            "sub_canopy_supervision": "Bare-earth 1m DTM relief pixels (|z - G_16(z)| >= 0.16m) inside NHLE polygons retained under woodland canopy (nDSM >= 2.5m)",
            "calibration_and_thresholding": "Platt scaling and operating thresholds (FPR=1%, 5%, 10%, Max Youden J) fitted strictly on Validation split",
        },
        "dataset_total_tiles": len(manifest),
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
    out_file = ckpt_dir / "roman_evaluation_metrics.json"
    out_file.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    print("\n=== ROMAN UNET HELD-OUT TEST SPLIT STATISTICAL SUMMARY (95% BOOTSTRAP CI) ===")
    print(f"RomanUNet Test ROC-AUC        : {test_stats['roc_auc']} {test_stats['roc_auc_ci95']}")
    print(f"RomanUNet Test Brier (uncal->cal): {test_stats['brier_score_uncalibrated']} -> {test_stats['brier_score_calibrated']}")
    print("Sub-Null ROC-AUCs (Confounding Test):", json.dumps(test_subgroups["discrimination_by_negative_null"], indent=2))
    print("Subgroup Recall by Roman Class (Test):", json.dumps(test_subgroups["operating_point_breakdowns"], indent=2))


if __name__ == "__main__":
    main()
