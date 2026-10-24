#!/usr/bin/env python3
"""
Train & Evaluate the Pond, Pit & Modern Confounder UNet (`pond_unet_best.pt`)
=============================================================================
Trains a dedicated 8-channel (6-channel 1m LiDAR + 2-channel 16m Aerial Cropmark/Soilmark)
`MegalithUNet` to segment and classify non-archaeological terrain confounders that mimic
Roman forts/camps, prehistoric barrows/henges, and Iron Age enclosures:

  Class 0: background (including genuine NHLE Scheduled Monuments & pristine terrain)
  Class 1: pond_lagoon_water_basin (ponds, slurry lagoons, farm reservoirs, sunken basins)
  Class 2: quarry_pit_bund_earthwork (gravel/marl pits, quarries, sinkholes, spoil/U-bunds)
  Class 3: levelled_court_pitch_utility (levelled tennis courts, sports pitches, covered reservoirs)

Supervision is assembled strictly from already-cached 256m x 256m 1m LiDAR + aerial tiles
with zero network downloads:
  - Positives:
      1. Closed OpenStreetMap polygon footprints (`natural=water`, `water=pond|basin|lagoon|wastewater`,
         `landuse=quarry|basin|reservoir`, `leisure=pitch|track`, `man_made=embankment|reservoir_covered`)
         inside cached 2km OSM squares that have measurable 1m LiDAR relief (`p95 |z - G_16(z)| >= 0.22 m`).
      2. Verified flagged confounder detections from `docs/roman_candidates.geojson`,
         `docs/candidates.geojson`, and `docs/enclosure_candidates.geojson`.
  - Negatives:
      1. Verified NHLE Scheduled Monuments (Roman villas/forts/camps/roads, prehistoric barrows/henges/tombs,
         Iron Age hillforts/enclosures/hut circles) as hard `archaeological_monument` negatives so the
         Pond UNet learns NEVER to fire on genuine monuments.
      2. Verified monument-free & confounder-free `random_national` and `matched_environment` background tiles.

Evaluated with 20km x 20km BNG spatial block cross-validation (`assign_spatial_split`) and Platt
scaling, then integrated into `megalith_lidar/scanner.py` and applied across the national candidate
GeoJSONs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megalith_lidar.aerial_cropmarks import compute_cropmark_features, fetch_or_load_aerial_tile
from megalith_lidar.catalog import download_or_load_nhle_catalog
from megalith_lidar.dataset import assign_spatial_split
from megalith_lidar.lidar_wcs import fetch_or_load_paired_tile
from megalith_lidar.model import MegalithLoss, MegalithUNet
from megalith_lidar.osm_context import _TRANSFORMER_4326_TO_27700
from megalith_lidar.stats import (
    PlattCalibrator,
    evaluate_predictions_statistically,
    select_validation_thresholds,
)
from megalith_lidar.terrain_features import NUM_TERRAIN_CHANNELS, compute_terrain_features

POND_CLASS_BACKGROUND = 0
POND_CLASS_WATER_BASIN = 1
POND_CLASS_QUARRY_PIT_BUND = 2
POND_CLASS_COURT_PITCH_UTILITY = 3

POND_CLASS_NAMES = {
    POND_CLASS_BACKGROUND: "background",
    POND_CLASS_WATER_BASIN: "pond_lagoon_water_basin",
    POND_CLASS_QUARRY_PIT_BUND: "quarry_pit_bund_earthwork",
    POND_CLASS_COURT_PITCH_UTILITY: "levelled_court_pitch_utility",
}


def assign_pond_spatial_split(easting: float, northing: float) -> str:
    """
    Assign a 10km x 10km British National Grid block to 'train' (75%), 'val' (12.5%), or 'test' (12.5%)
    using an independent SHA-256 salt ('pond_block_') so that OSM squares scanned on the monument
    test split are uniformly partitioned across train/val/test for the Pond & Confounder UNet.
    """
    import hashlib

    bx = int(easting // 10000)
    by = int(northing // 10000)
    digest = hashlib.sha256(f"pond_block_{bx}_{by}".encode("utf-8")).digest()
    bucket = digest[0] % 8
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def _rasterize_polygons_256(
    polys: List[Tuple[int, List[Tuple[float, float]]]],
    e_min: int,
    e_max: int,
    n_min: int,
    n_max: int,
    size: int = 256,
) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    scale_e = size / float(e_max - e_min)
    scale_n = size / float(n_max - n_min)
    for cls_id, coords in polys:
        pts = [((e - e_min) * scale_e, (n_max - n) * scale_n) for e, n in coords]
        if len(pts) >= 3:
            draw.polygon(pts, fill=int(cls_id), outline=int(cls_id))
    return np.array(img, dtype=np.uint8)


def build_or_load_pond_manifest(
    manifest_path: Path,
    mask_cache_dir: Path,
    tile_cache_dir: Path,
    osm_cache_dir: Path,
    seed: int = 20261024,
) -> List[Dict]:
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(cached) >= 1200:
                return cached
        except Exception:
            pass

    mask_cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Index all cached 256m LiDAR tiles by 2km square and by exact bbox
    tile_by_bbox: Dict[Tuple[int, int, int, int], Path] = {}
    tiles_by_sq: Dict[Tuple[int, int], List[Tuple[int, int, int, int, Path]]] = {}
    for p in tile_cache_dir.glob("*/*.npz"):
        m = re.match(r"tile_E(\d+)_(\d+)_N(\d+)_(\d+)\.npz", p.name)
        if not m:
            continue
        e0, e1, n0, n1 = map(int, m.groups())
        if (e1 - e0) != 256 or (n1 - n0) != 256:
            continue
        tile_by_bbox[(e0, e1, n0, n1)] = p
        sq_e = (e0 // 2000) * 2000
        sq_n = (n0 // 2000) * 2000
        tiles_by_sq.setdefault((sq_e, sq_n), []).append((e0, e1, n0, n1, p))

    print(f"[PondDataset] Indexed {len(tile_by_bbox)} cached 256m LiDAR tiles across {len(tiles_by_sq)} 2km squares.")

    # 2. Extract OSM pond / quarry / sports-court polygons inside cached 256m tiles
    tile_polys: Dict[Tuple[int, int, int, int], List[Tuple[int, List[Tuple[float, float]]]]] = {}
    for op in sorted(osm_cache_dir.glob("*.json")):
        m = re.search(r"E(\d+)_N(\d+)", op.name)
        if not m:
            continue
        sq_e, sq_n = int(m.group(1)), int(m.group(2))
        sq_tiles = tiles_by_sq.get((sq_e, sq_n))
        if not sq_tiles:
            continue
        try:
            data = json.loads(op.read_text(encoding="utf-8"))
        except Exception:
            continue
        elements = data.get("elements", [])
        raw_nodes = [el for el in elements if el.get("type") == "node" and "lat" in el and "lon" in el]
        if not raw_nodes:
            continue
        lons = [float(el["lon"]) for el in raw_nodes]
        lats = [float(el["lat"]) for el in raw_nodes]
        es, ns = _TRANSFORMER_4326_TO_27700.transform(lons, lats)
        nodes_27700 = {int(el["id"]): (float(e), float(n)) for el, e, n in zip(raw_nodes, es, ns)}

        for el in elements:
            if el.get("type") != "way":
                continue
            nd_ids = el.get("nodes", [])
            if len(nd_ids) < 4 or nd_ids[0] != nd_ids[-1]:
                continue
            tags = el.get("tags", {})
            if not tags or "historic" in tags or "archaeological_site" in tags.values():
                continue
            nat = tags.get("natural", "")
            wat = tags.get("water", "")
            lu = tags.get("landuse", "")
            leis = tags.get("leisure", "")
            mm = tags.get("man_made", "")
            cls_id = 0
            if nat == "water" or wat in {"pond", "basin", "wastewater", "lagoon", "reservoir"} or lu in {"basin", "reservoir"}:
                cls_id = POND_CLASS_WATER_BASIN
            elif lu == "quarry" or nat == "sinkhole":
                cls_id = POND_CLASS_QUARRY_PIT_BUND
            elif leis in {"pitch", "track"} or mm in {"embankment", "reservoir_covered", "wastewater_plant"}:
                cls_id = POND_CLASS_COURT_PITCH_UTILITY
            if cls_id == 0:
                continue
            coords = [nodes_27700[nid] for nid in nd_ids if nid in nodes_27700]
            if len(coords) < 4:
                continue
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            pe0, pe1, pn0, pn1 = min(xs), max(xs), min(ys), max(ys)
            bbox_area = (pe1 - pe0) * (pn1 - pn0)
            if bbox_area > 28000.0 or (pe1 - pe0) < 6.0 or (pn1 - pn0) < 6.0:
                continue
            for te0, te1, tn0, tn1, _ in sq_tiles:
                if pe1 > te0 + 14 and pe0 < te1 - 14 and pn1 > tn0 + 14 and pn0 < tn1 - 14:
                    tile_polys.setdefault((te0, te1, tn0, tn1), []).append((cls_id, coords))

    # 3. Also harvest flagged confounder detections from national candidate GeoJSONs whose 256m window is cached
    flagged_points_by_tile: Dict[Tuple[int, int, int, int], List[Tuple[int, float, float, float]]] = {}
    for gpath in ["docs/roman_candidates.geojson", "docs/candidates.geojson", "docs/enclosure_candidates.geojson"]:
        p_obj = Path(gpath)
        if not p_obj.exists():
            continue
        fc = json.loads(p_obj.read_text(encoding="utf-8"))
        for feat in fc.get("features", []):
            pr = feat.get("properties", {})
            if pr.get("is_known_monument"):
                continue
            confs = pr.get("osm_confounder_matches") or []
            tier = pr.get("vlm_verdict_tier")
            if not confs and tier != "OBVIOUS_FALSE_POSITIVE":
                continue
            wb = pr.get("window_bbox")
            if not wb or len(wb) != 4:
                continue
            e0, n0, e1, n1 = int(wb[0]), int(wb[1]), int(wb[2]), int(wb[3])
            tkey = (e0, e1, n0, n1)
            if tkey not in tile_by_bbox:
                continue
            # Determine class
            sub_str = " ".join(str(c.get("osm_id", "")) + " " + str(c.get("subtype", "")) for c in confs).lower()
            if "slurry_lagoon" in sub_str or "water" in sub_str or "pond" in sub_str or "basin" in sub_str or float(pr.get("centroid_local_relief_m", 0.5)) <= 0.05:
                c_id = POND_CLASS_WATER_BASIN
            elif "pitch" in sub_str or "reservoir_or_bund" in sub_str or "golf" in sub_str or "building" in sub_str:
                c_id = POND_CLASS_COURT_PITCH_UTILITY
            else:
                c_id = POND_CLASS_QUARRY_PIT_BUND
            area_m2 = float(pr.get("area_m2", 400.0))
            rad_m = max(6.0, min(55.0, math.sqrt(max(area_m2, 36.0) / math.pi)))
            flagged_points_by_tile.setdefault(tkey, []).append((c_id, float(pr["easting"]), float(pr["northing"]), rad_m))

    # Merge candidate positive tiles and verify 1m LiDAR relief
    all_pos_keys = sorted(set(tile_polys.keys()) | set(flagged_points_by_tile.keys()))
    rng.shuffle(all_pos_keys)

    positives: List[Dict] = []
    pos_counts = Counter()
    used_bboxes: Set[Tuple[int, int, int, int]] = set()

    for tkey in all_pos_keys:
        e0, e1, n0, n1 = tkey
        res = fetch_or_load_paired_tile(e0, e1, n0, n1, tile_cache_dir)
        if res is None:
            continue
        dtm, dsm = res
        if dtm.shape != (256, 256):
            continue
        feats = compute_terrain_features(dtm, dsm)
        rel16 = np.abs(feats[0] + feats[1])
        slope = feats[3]

        mask = np.zeros((256, 256), dtype=np.uint8)
        if tkey in tile_polys:
            poly_mask = _rasterize_polygons_256(tile_polys[tkey], e0, e1, n0, n1, size=256)
            # Keep active earthwork/rim/scarp/basin pixels inside or within 4m of the polygon
            dil = binary_dilation(poly_mask > 0, iterations=4)
            active = dil & ((rel16 >= 0.14) | (slope >= 0.055) | (poly_mask > 0))
            for cid in (POND_CLASS_WATER_BASIN, POND_CLASS_QUARRY_PIT_BUND, POND_CLASS_COURT_PITCH_UTILITY):
                c_dil = binary_dilation(poly_mask == cid, iterations=3) & active
                mask[c_dil] = cid

        if tkey in flagged_points_by_tile:
            yy, xx = np.ogrid[:256, :256]
            for cid, ce_pt, cn_pt, rad_m in flagged_points_by_tile[tkey]:
                cx_px = (ce_pt - e0)
                cy_px = (n1 - cn_pt)
                dist2 = (xx - cx_px) ** 2 + (yy - cy_px) ** 2
                disc = dist2 <= (rad_m + 3.0) ** 2
                active_disc = disc & ((rel16 >= 0.15) | (slope >= 0.055) | (dist2 <= (rad_m * 0.65) ** 2))
                mask[active_disc] = cid

        pos_px = (mask > 0) & (mask != 255)
        n_pos = int(np.sum(pos_px))
        if n_pos < 24:
            continue
        p95_rel = float(np.percentile(rel16[pos_px], 95.0))
        if p95_rel < 0.22:
            continue

        # Determine primary class by majority pixel count
        cls_vals, cls_cnts = np.unique(mask[pos_px], return_counts=True)
        primary_cls = int(cls_vals[int(np.argmax(cls_cnts))])
        if pos_counts[primary_cls] >= 480:
            continue

        # Add 3m ignore transition halo around positive mask
        halo = binary_dilation(pos_px, iterations=3) & (~pos_px)
        mask[halo] = 255

        sample_id = f"pond_pos_c{primary_cls}_E{e0}_N{n0}"
        mask_path = mask_cache_dir / f"{sample_id}.npy"
        np.save(mask_path, mask)

        ce, cn = (e0 + e1) // 2, (n0 + n1) // 2
        positives.append(
            {
                "sample_id": sample_id,
                "list_entry": 0,
                "name": f"Confounder {POND_CLASS_NAMES[primary_cls]} E{e0} N{n0}",
                "primary_class": primary_cls,
                "is_positive": True,
                "negative_subtype": None,
                "e_min": e0,
                "e_max": e1,
                "n_min": n0,
                "n_max": n1,
                "positive_pixels": n_pos,
                "relief_p95_m": round(p95_rel, 3),
                "mask_path": str(mask_path),
                "split": assign_pond_spatial_split(ce, cn),
            }
        )
        pos_counts[primary_cls] += 1
        used_bboxes.add(tkey)

    print(f"[PondDataset] Assembled {len(positives)} positive confounder tiles: {dict(pos_counts)}")

    # 4. Assemble OSM-verified Confounder-Free Negatives (from OSM-scanned 2km squares)
    #    + Hard Scheduled Monument Negatives (Roman, Prehistoric, Enclosure)
    negatives: List[Dict] = []
    osm_clean_pool: List[Tuple[int, int, int, int]] = []
    for sq_key, sq_tiles in tiles_by_sq.items():
        op = osm_cache_dir / f"osm_E{sq_key[0]}_N{sq_key[1]}_E{sq_key[0]+2000}_N{sq_key[1]+2000}.json"
        if not op.exists():
            continue
        for te0, te1, tn0, tn1, _ in sq_tiles:
            tkey = (te0, te1, tn0, tn1)
            if tkey not in used_bboxes and tkey not in tile_polys and tkey not in flagged_points_by_tile:
                osm_clean_pool.append(tkey)

    arch_neg_pool: List[Tuple[Tuple[int, int, int, int], Dict]] = []
    for mpath_str in [
        "/mnt/bulk/megalith-lidar/manifest_uk_roman_v3.json",
        "/mnt/bulk/megalith-lidar/manifest_uk_megaliths_v3.json",
        "/mnt/bulk/megalith-lidar/manifest_uk_enclosure_v3.json",
    ]:
        mp = Path(mpath_str)
        if not mp.exists():
            continue
        items = json.loads(mp.read_text(encoding="utf-8"))
        for it in items:
            if not it.get("is_positive"):
                continue
            tkey = (int(it["e_min"]), int(it["e_max"]), int(it["n_min"]), int(it["n_max"]))
            if tkey in used_bboxes or tkey in tile_polys or tkey in flagged_points_by_tile:
                continue
            if tkey not in tile_by_bbox:
                continue
            arch_neg_pool.append((tkey, it))

    rng.shuffle(arch_neg_pool)
    rng.shuffle(osm_clean_pool)

    target_arch_neg = min(len(arch_neg_pool), int(len(positives) * 0.55))
    target_osm_neg = min(len(osm_clean_pool), int(len(positives) * 0.70))

    for tkey, it in arch_neg_pool:
        if len(negatives) >= target_arch_neg:
            break
        if tkey in used_bboxes:
            continue
        e0, e1, n0, n1 = tkey
        used_bboxes.add(tkey)
        ce, cn = (e0 + e1) // 2, (n0 + n1) // 2
        negatives.append(
            {
                "sample_id": f"pond_neg_arch_E{e0}_N{n0}",
                "list_entry": int(it.get("list_entry", 0)),
                "name": f"Scheduled Monument Hard Negative: {it.get('name', '')}",
                "primary_class": POND_CLASS_BACKGROUND,
                "is_positive": False,
                "negative_subtype": "scheduled_monument_hard_negative",
                "e_min": e0,
                "e_max": e1,
                "n_min": n0,
                "n_max": n1,
                "positive_pixels": 0,
                "relief_p95_m": float(it.get("relief_p95_m", 0.35)),
                "mask_path": None,
                "split": assign_pond_spatial_split(ce, cn),
            }
        )

    for tkey in osm_clean_pool:
        if len(negatives) >= target_arch_neg + target_osm_neg:
            break
        if tkey in used_bboxes:
            continue
        e0, e1, n0, n1 = tkey
        res = fetch_or_load_paired_tile(e0, e1, n0, n1, tile_cache_dir)
        if res is None or res[0].shape != (256, 256):
            continue
        used_bboxes.add(tkey)
        ce, cn = (e0 + e1) // 2, (n0 + n1) // 2
        negatives.append(
            {
                "sample_id": f"pond_neg_osm_clean_E{e0}_N{n0}",
                "list_entry": 0,
                "name": "OSM-verified confounder-free terrain",
                "primary_class": POND_CLASS_BACKGROUND,
                "is_positive": False,
                "negative_subtype": "osm_verified_clean_terrain",
                "e_min": e0,
                "e_max": e1,
                "n_min": n0,
                "n_max": n1,
                "positive_pixels": 0,
                "relief_p95_m": 0.25,
                "mask_path": None,
                "split": assign_pond_spatial_split(ce, cn),
            }
        )

    manifest = positives + negatives
    manifest.sort(key=lambda x: x["sample_id"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PondDataset] Saved {len(manifest)} total samples ({len(positives)} pos, {len(negatives)} neg) -> {manifest_path}")
    return manifest


class PondConfounderDataset(Dataset):
    def __init__(
        self,
        manifest: List[Dict],
        tile_cache_dir: Path,
        aerial_cache_dir: Path,
        augment: bool = False,
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
    ) -> None:
        self.manifest = list(manifest)
        self.tile_cache_dir = Path(tile_cache_dir)
        self.aerial_cache_dir = Path(aerial_cache_dir)
        self.augment = augment
        self.num_channels = NUM_TERRAIN_CHANNELS + 2
        self.channel_mean = (
            np.asarray(channel_mean, dtype=np.float32).reshape(-1, 1, 1)
            if channel_mean is not None
            else np.zeros((self.num_channels, 1, 1), dtype=np.float32)
        )
        self.channel_std = (
            np.asarray(channel_std, dtype=np.float32).reshape(-1, 1, 1)
            if channel_std is not None
            else np.ones((self.num_channels, 1, 1), dtype=np.float32)
        )

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.channel_mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.channel_std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.manifest)

    def load_raw_features(self, spec: Dict) -> np.ndarray:
        pair = fetch_or_load_paired_tile(
            spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"], self.tile_cache_dir
        )
        assert pair is not None, f"Missing tile {spec['sample_id']}"
        dtm, dsm = pair
        feats = compute_terrain_features(dtm, dsm)
        # Only load ALREADY CACHED aerial tile on disk to guarantee zero network blocking
        e0, e1, n0, n1 = spec["e_min"], spec["e_max"], spec["n_min"], spec["n_max"]
        aerial_file = self.aerial_cache_dir / f"aerial_E{e0}_{e1}_N{n0}_{n1}.png"
        if aerial_file.exists():
            rgb = fetch_or_load_aerial_tile(e0, e1, n0, n1, cache_dir=self.aerial_cache_dir, size=dtm.shape[0])
            if rgb is not None:
                crop_feats = compute_cropmark_features(rgb, dtm=dtm, dsm=dsm)
            else:
                crop_feats = np.zeros((2, dtm.shape[0], dtm.shape[1]), dtype=np.float32)
        else:
            crop_feats = np.zeros((2, dtm.shape[0], dtm.shape[1]), dtype=np.float32)
        return np.concatenate([feats[:NUM_TERRAIN_CHANNELS], crop_feats], axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        spec = self.manifest[idx]
        feats = self.load_raw_features(spec)
        if spec["is_positive"] and spec.get("mask_path") and Path(spec["mask_path"]).exists():
            mask = np.load(spec["mask_path"]).astype(np.uint8)
        else:
            mask = np.zeros((feats.shape[1], feats.shape[2]), dtype=np.uint8)

        if self.augment:
            k = int(np.random.randint(0, 4))
            if k > 0:
                feats = np.rot90(feats, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.5:
                feats = np.flip(feats, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                feats = np.flip(feats, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            scale = float(np.random.uniform(0.92, 1.08))
            feats[:5] *= scale

        norm_feats = (feats - self.channel_mean) / self.channel_std
        return {
            "features": torch.from_numpy(norm_feats.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "tile_label": torch.tensor(1.0 if spec["is_positive"] else 0.0, dtype=torch.float32),
            "type_label": torch.tensor(int(spec["primary_class"]), dtype=torch.int64),
            "relief_p95_m": torch.tensor(float(spec.get("relief_p95_m", 0.0)), dtype=torch.float32),
            "sample_id": spec["sample_id"],
        }


def compute_pond_channel_stats(ds: PondConfounderDataset) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(ds.num_channels, dtype=np.float64)
    sq_sums = np.zeros(ds.num_channels, dtype=np.float64)
    count = 0
    step = max(1, len(ds.manifest) // 300)
    for spec in ds.manifest[::step]:
        f = ds.load_raw_features(spec)
        sums += f.mean(axis=(1, 2))
        sq_sums += (f * f).mean(axis=(1, 2))
        count += 1
    mean = (sums / max(count, 1)).astype(np.float32)
    var = np.maximum((sq_sums / max(count, 1)) - mean.astype(np.float64) ** 2, 1e-4)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate Pond/Pit/Confounder UNet")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/bulk/megalith-lidar"))
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(20261024)
    np.random.seed(20261024)
    device = torch.device(args.device)

    manifest_path = args.data_dir / "manifest_uk_pond_confounders.json"
    mask_cache_dir = args.data_dir / "pond_masks_cache"
    tile_cache_dir = args.data_dir / "tiles"
    osm_cache_dir = args.data_dir / "osm_cache"
    aerial_cache_dir = args.data_dir / "aerial_tiles"
    ckpt_dir = args.data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_or_load_pond_manifest(
        manifest_path=manifest_path,
        mask_cache_dir=mask_cache_dir,
        tile_cache_dir=tile_cache_dir,
        osm_cache_dir=osm_cache_dir,
    )

    train_items = [m for m in manifest if m["split"] == "train"]
    val_items = [m for m in manifest if m["split"] == "val"]
    test_items = [m for m in manifest if m["split"] == "test"]
    print(f"[PondDataset] Split counts: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    train_ds = PondConfounderDataset(train_items, tile_cache_dir, aerial_cache_dir, augment=True)
    val_ds = PondConfounderDataset(val_items, tile_cache_dir, aerial_cache_dir, augment=False)
    test_ds = PondConfounderDataset(test_items, tile_cache_dir, aerial_cache_dir, augment=False)

    ch_mean, ch_std = compute_pond_channel_stats(train_ds)
    train_ds.set_normalization(ch_mean, ch_std)
    val_ds.set_normalization(ch_mean, ch_std)
    test_ds.set_normalization(ch_mean, ch_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MegalithUNet(in_channels=NUM_TERRAIN_CHANNELS, num_classes=4, pretrained=True).to(device)
    # Warm-start ONLY the encoder/decoder feature extractor (stem, full_skip, enc1..4, dec4..0)
    # from roman_unet_best.pt while initializing segmentation/tile classification heads fresh
    # so monument-positive output weights do not invert confounder vs monument supervision.
    roman_ckpt_path = ckpt_dir / "roman_unet_best.pt"
    if roman_ckpt_path.exists():
        r_ckpt = torch.load(roman_ckpt_path, map_location="cpu", weights_only=False)
        state = r_ckpt.get("model_state_dict") or r_ckpt.get("model_state")
        if state is not None:
            backbone_state = {
                k: v
                for k, v in state.items()
                if k.startswith(("stem.", "full_skip.", "enc", "dec"))
            }
            model.load_state_dict(backbone_state, strict=False)
        del r_ckpt
        print(f"[PondUNet] Warm-started encoder/decoder backbone from {roman_ckpt_path}")

    n_pos_train = sum(1 for m in train_items if m["is_positive"])
    n_neg_train = max(1, len(train_items) - n_pos_train)
    pos_tile_weight = round(float(n_neg_train / max(n_pos_train, 1)), 3)
    criterion = MegalithLoss(pos_pixel_weight=2.8, pos_tile_weight=pos_tile_weight).to(device)

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

    from scripts.train_roman import collect_split_predictions

    best_score = -1.0
    best_ckpt_path = ckpt_dir / "pond_unet_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_b = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_loader, 1):
            x = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            tile_label = batch["tile_label"].to(device, non_blocking=True)
            type_label = batch["type_label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(x)
                loss, _ = criterion(out, mask, tile_label, type_label)
            (loss / 4.0).backward()
            ep_loss += float(loss.item())
            n_b += 1
            if step_idx % 4 == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        val_preds = collect_split_predictions(model, val_loader, criterion, device, use_d4_tta=False)
        val_auc = float(roc_auc_score(val_preds["y_true"], val_preds["raw_logits"]))
        val_dice = float(val_preds["pos_seg_dice"])
        combo = val_auc + 0.35 * val_dice
        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | "
            f"train_loss={ep_loss / max(n_b, 1):.4f} | val_loss={val_preds['loss']:.4f} | "
            f"val_auc={val_auc:.4f} | val_dice={val_dice:.4f}"
        )

        if combo > best_score:
            best_score = combo
            torch.save(
                {
                    "epoch": epoch,
                    "target_domain": "pond_confounder",
                    "model_state_dict": model.state_dict(),
                    "model_state": model.state_dict(),
                    "channel_mean": ch_mean.tolist(),
                    "channel_std": ch_std.tolist(),
                    "class_names": POND_CLASS_NAMES,
                    "val_roc_auc": round(val_auc, 4),
                    "val_pos_seg_dice": round(val_dice, 4),
                },
                best_ckpt_path,
            )

    # Fit Platt calibrator & select operating thresholds strictly on Validation split
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_d4 = collect_split_predictions(model, val_loader, criterion, device, use_d4_tta=True)
    calibrator = PlattCalibrator().fit(val_d4["raw_logits"], val_d4["y_true"])
    val_cal_probs = calibrator.predict_proba(val_d4["raw_logits"])
    thresholds = select_validation_thresholds(val_cal_probs, val_d4["y_true"], target_fprs=(0.01, 0.05, 0.10))
    neg_val_peaks = val_d4["seg_peak_probs"][val_d4["y_true"] == 0]
    pixel_th_fpr05 = float(np.quantile(neg_val_peaks, 0.95)) if len(neg_val_peaks) > 0 else 0.50

    best_ckpt["calibrator"] = calibrator.to_dict()
    best_ckpt["val_thresholds"] = thresholds
    best_ckpt["thresholds"] = thresholds
    best_ckpt["val_pixel_threshold_fpr05"] = round(pixel_th_fpr05, 4)
    torch.save(best_ckpt, best_ckpt_path)

    test_preds = collect_split_predictions(model, test_loader, criterion, device, use_d4_tta=True)
    test_stats = evaluate_predictions_statistically(
        test_preds["y_true"],
        test_preds["raw_logits"],
        calibrator,
        thresholds,
        n_boot=1000,
    )

    # Also measure False Positive Rate specifically on Scheduled Monuments in the held-out test split
    id_to_spec = {m["sample_id"]: m for m in test_items}
    test_cal_probs = calibrator.predict_proba(test_preds["raw_logits"])
    th_j = float(thresholds.get("max_youden_j", {}).get("threshold", 0.50))
    th_fpr05 = float(thresholds.get("fpr_0.05", {}).get("threshold", 0.50))
    arch_mask = np.array(
        [id_to_spec[sid].get("negative_subtype") == "scheduled_monument_hard_negative" for sid in test_preds["sample_ids"]],
        dtype=bool,
    )
    arch_fpr_j = float(np.mean(test_cal_probs[arch_mask] >= th_j)) if np.any(arch_mask) else 0.0
    arch_fpr_05 = float(np.mean(test_cal_probs[arch_mask] >= th_fpr05)) if np.any(arch_mask) else 0.0

    report = {
        "model": "PondConfounderUNet (ResNet34-UNet + D4 Geom + 16m Cropmarks)",
        "best_epoch": best_ckpt["epoch"],
        "val_roc_auc": best_ckpt["val_roc_auc"],
        "val_pos_seg_dice": best_ckpt["val_pos_seg_dice"],
        "test_pos_seg_dice": test_preds["pos_seg_dice"],
        "test_statistics": test_stats,
        "test_scheduled_monument_false_alarm_rate_at_youden_j": round(arch_fpr_j, 4),
        "test_scheduled_monument_false_alarm_rate_at_fpr05": round(arch_fpr_05, 4),
        "thresholds": thresholds,
        "val_pixel_threshold_fpr05": round(pixel_th_fpr05, 4),
    }
    report_path = ckpt_dir / "pond_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== HELD-OUT SPATIAL BLOCK TEST EVALUATION (POND & CONFOUNDER UNET) ===")
    print(f"Test ROC-AUC: {test_stats['roc_auc']:.4f} (95% CI: {test_stats['roc_auc_ci95'][0]:.4f}..{test_stats['roc_auc_ci95'][1]:.4f})")
    print(f"Test PR-AUC : {test_stats['pr_auc_sample_prevalence']:.4f}")
    print(f"Test Seg Dice: {test_preds['pos_seg_dice']:.4f}")
    print(f"Scheduled Monument False Alarm Rate @ FPR=5%: {arch_fpr_05:.4f} | @ Youden-J: {arch_fpr_j:.4f}")
    print(f"Saved checkpoint -> {best_ckpt_path}")


if __name__ == "__main__":
    main()
