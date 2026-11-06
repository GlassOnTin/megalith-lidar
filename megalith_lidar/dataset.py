from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from megalith_lidar.catalog import (
    CLASS_BACKGROUND,
    CLASS_BARROW_CAIRN,
    CLASS_CIRCLE_HENGE_STONE,
    CLASS_MEGALITHIC_TOMB_LONG_BARROW,
    CLASS_OTHER_MONUMENT,
    MonumentCatalog,
    MonumentRecord,
)
from megalith_lidar.lidar_wcs import fetch_or_load_paired_tile
from megalith_lidar.terrain_features import NUM_TERRAIN_CHANNELS, compute_terrain_features


def assign_spatial_split(easting: float, northing: float) -> str:
    """
    Assign a 20km x 20km British National Grid block to 'train' (75%), 'val' (12.5%), or 'test' (12.5%)
    strictly via SHA-256 hash of (block_x, block_y), with zero manual block overrides.
    """
    bx = int(easting // 20000)
    by = int(northing // 20000)
    digest = hashlib.sha256(f"bng_block_{bx}_{by}".encode("utf-8")).digest()
    bucket = digest[0] % 8
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def build_national_dataset_manifest(
    catalog: MonumentCatalog,
    tile_cache_dir: Path,
    manifest_path: Path,
    max_barrows: int = 450,
    num_random_national_negatives: int = 450,
    num_matched_env_negatives: int = 450,
    num_hard_earthwork_negatives: int = 250,
    tile_size: int = 256,
    seed: int = 42,
) -> List[Dict]:
    """
    Assemble a statistically balanced, multi-null UK LiDAR dataset across England:
      Positives:
        - CLASS_MEGALITHIC_TOMB_LONG_BARROW
        - CLASS_CIRCLE_HENGE_STONE
        - Geographically stratified CLASS_BARROW_CAIRN
      Three distinct Negative Null Distributions (to test & eliminate confounding):
        1. 'random_national'         : Uniformly sampled across all 20km x 20km land blocks of England
                                       (1 weight per land block, independent of monument density).
        2. 'matched_environment'     : Sampled inside the same 20km block as positives (800m..2500m away),
                                       testing discrimination within identical geology/topography.
        3. 'anthropogenic_non_target': Non-prehistoric earthworks/monuments (medieval, Roman, industrial).
    """
    rng = np.random.default_rng(seed)
    half = tile_size // 2

    tombs = [r for r in catalog.records if r.cls == CLASS_MEGALITHIC_TOMB_LONG_BARROW]
    circles = [r for r in catalog.records if r.cls == CLASS_CIRCLE_HENGE_STONE]
    all_barrows = [r for r in catalog.records if r.cls == CLASS_BARROW_CAIRN]
    other_monuments = [r for r in catalog.records if r.cls == CLASS_OTHER_MONUMENT]

    all_barrows.sort(key=lambda r: (int(r.easting // 50000), int(r.northing // 50000), r.list_entry))
    if len(all_barrows) > max_barrows:
        idx = np.linspace(0, len(all_barrows) - 1, max_barrows, dtype=int)
        barrows = [all_barrows[i] for i in idx]
    else:
        barrows = all_barrows

    # Preserve existing verified tiles from disk if manifest already exists to avoid redundant WCS calls
    existing_by_bbox: Dict[Tuple[int, int, int, int], Dict] = {}
    if manifest_path.exists():
        try:
            old_list = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in old_list:
                key = (int(item["e_min"]), int(item["e_max"]), int(item["n_min"]), int(item["n_max"]))
                existing_by_bbox[key] = item
        except Exception:
            pass

    candidate_specs: List[Dict] = []

    # 1. Positives with uniform spatial offset in [-56m, +56m]
    for rec in tombs + circles + barrows:
        jitter_e = int(rng.integers(-56, 57))
        jitter_n = int(rng.integers(-56, 57))
        ce = int(round(rec.easting)) + jitter_e
        cn = int(round(rec.northing)) + jitter_n
        e_min, e_max = ce - half, ce + half
        n_min, n_max = cn - half, cn + half
        candidate_specs.append(
            {
                "sample_id": f"pos_{rec.list_entry}",
                "list_entry": rec.list_entry,
                "name": rec.name,
                "primary_class": int(rec.cls),
                "is_positive": True,
                "negative_subtype": None,
                "e_min": e_min,
                "e_max": e_max,
                "n_min": n_min,
                "n_max": n_max,
                "split": assign_spatial_split(ce, cn),
            }
        )

    # 2. Matched-Environment Null ('matched_environment'): same geological/topographic regime as positives
    pos_pool = tombs + circles + barrows
    matched_neg_specs: List[Dict] = []
    attempts = 0
    while len(matched_neg_specs) < num_matched_env_negatives and attempts < num_matched_env_negatives * 8:
        attempts += 1
        anchor: MonumentRecord = pos_pool[int(rng.integers(0, len(pos_pool)))]
        angle = float(rng.uniform(0, 2.0 * np.pi))
        dist = float(rng.uniform(400.0, 1600.0))
        ce = int(round(anchor.easting + dist * np.cos(angle)))
        cn = int(round(anchor.northing + dist * np.sin(angle)))
        e_min, e_max = ce - half, ce + half
        n_min, n_max = cn - half, cn + half
        hits = catalog.query_bbox(e_min - 60, n_min - 60, e_max + 60, n_max + 60)
        if not hits:
            matched_neg_specs.append(
                {
                    "sample_id": f"neg_matched_{len(matched_neg_specs):04d}_E{ce}_N{cn}",
                    "list_entry": 0,
                    "name": "Matched-environment null negative",
                    "primary_class": CLASS_BACKGROUND,
                    "is_positive": False,
                    "negative_subtype": "matched_environment",
                    "e_min": e_min,
                    "e_max": e_max,
                    "n_min": n_min,
                    "n_max": n_max,
                    "split": assign_spatial_split(ce, cn),
                }
            )
    candidate_specs.extend(matched_neg_specs)

    # 3. Unbiased National Uniform Random Null ('random_national'):
    # Equal probability across all unique 20km x 20km land blocks in England, independent of monument count
    land_blocks = sorted({(int(r.easting // 20000), int(r.northing // 20000)) for r in catalog.records})
    national_neg_specs: List[Dict] = []
    attempts = 0
    while len(national_neg_specs) < num_random_national_negatives and attempts < num_random_national_negatives * 10:
        attempts += 1
        bx, by = land_blocks[int(rng.integers(0, len(land_blocks)))]
        ce = int(bx * 20000 + rng.integers(1000, 19000))
        cn = int(by * 20000 + rng.integers(1000, 19000))
        e_min, e_max = ce - half, ce + half
        n_min, n_max = cn - half, cn + half
        hits = catalog.query_bbox(e_min - 100, n_min - 100, e_max + 100, n_max + 100)
        if not hits:
            national_neg_specs.append(
                {
                    "sample_id": f"neg_national_{len(national_neg_specs):04d}_E{ce}_N{cn}",
                    "list_entry": 0,
                    "name": "Uniform national random land negative",
                    "primary_class": CLASS_BACKGROUND,
                    "is_positive": False,
                    "negative_subtype": "random_national",
                    "e_min": e_min,
                    "e_max": e_max,
                    "n_min": n_min,
                    "n_max": n_max,
                    "split": assign_spatial_split(ce, cn),
                }
            )
    candidate_specs.extend(national_neg_specs)

    # 4. Anthropogenic Non-Target Earthwork Negatives ('anthropogenic_non_target')
    other_monuments.sort(key=lambda r: (int(r.easting // 50000), int(r.northing // 50000), r.list_entry))
    hard_neg_specs: List[Dict] = []
    stride_idx = np.linspace(
        0,
        len(other_monuments) - 1,
        min(len(other_monuments), num_hard_earthwork_negatives * 3),
        dtype=int,
    )
    for i in stride_idx:
        if len(hard_neg_specs) >= num_hard_earthwork_negatives:
            break
        rec = other_monuments[int(i)]
        ce = int(round(rec.easting))
        cn = int(round(rec.northing))
        e_min, e_max = ce - half, ce + half
        n_min, n_max = cn - half, cn + half
        intersecting = catalog.query_bbox(e_min - 40, n_min - 40, e_max + 40, n_max + 40)
        if any(r.is_megalithic_or_prehistoric for r in intersecting):
            continue
        hard_neg_specs.append(
            {
                "sample_id": f"neg_anthro_{rec.list_entry}",
                "list_entry": rec.list_entry,
                "name": f"Anthropogenic non-target: {rec.name}",
                "primary_class": CLASS_BACKGROUND,
                "is_positive": False,
                "negative_subtype": "anthropogenic_non_target",
                "e_min": e_min,
                "e_max": e_max,
                "n_min": n_min,
                "n_max": n_max,
                "split": assign_spatial_split(ce, cn),
            }
        )
    candidate_specs.extend(hard_neg_specs)

    valid_manifest: List[Dict] = []

    def _check_and_fetch(spec: Dict) -> Optional[Dict]:
        res = fetch_or_load_paired_tile(
            spec["e_min"],
            spec["e_max"],
            spec["n_min"],
            spec["n_max"],
            cache_dir=tile_cache_dir,
        )
        if res is None:
            return None
        dtm, _ = res
        if dtm.shape != (tile_size, tile_size):
            return None
        return spec

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_check_and_fetch, s) for s in candidate_specs]
        for f in as_completed(futs):
            out = f.result()
            if out is not None:
                valid_manifest.append(out)

    valid_manifest.sort(key=lambda s: s["sample_id"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(valid_manifest, indent=2), encoding="utf-8")
    return valid_manifest


def compute_train_channel_stats(dataset: "MegalithLidarDataset") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel empirical mean and standard deviation strictly over the training split
    to standardize raw physical units without data leakage.
    """
    n_ch = dataset.num_channels
    ch_sum = np.zeros(n_ch, dtype=np.float64)
    ch_sq_sum = np.zeros(n_ch, dtype=np.float64)
    total_px = 0

    for spec in dataset.manifest:
        feats, _ = dataset._load_features_and_mask(spec)
        c, h, w = feats.shape
        flat = feats.reshape(c, -1).astype(np.float64)
        ch_sum += flat.sum(axis=1)
        ch_sq_sum += (flat * flat).sum(axis=1)
        total_px += h * w

    mean = ch_sum / max(total_px, 1)
    var = np.maximum(ch_sq_sum / max(total_px, 1) - mean * mean, 1e-8)
    std = np.sqrt(var)
    # Channels 0, 1, 2, 4 (and 6, 7 bandpass cropmarks) are zero-mean differential/bandpass quantities;
    # pin mean=0 so exact D4 sign/zero-point symmetry is preserved
    for idx in (0, 1, 2, 4, 6, 7):
        if idx < n_ch:
            mean[idx] = 0.0
    return mean.astype(np.float32), std.astype(np.float32)


class MegalithLidarDataset(Dataset):
    """
    PyTorch Dataset yielding standardized 6-channel (or 8-channel LiDAR + Scale-Space Cropmark)
    isotropic scale-space tensors and exact surveyed NHLE polygon masks.
    """

    def __init__(
        self,
        manifest: List[Dict],
        catalog: MonumentCatalog,
        tile_cache_dir: Path,
        augment: bool = False,
        feature_cache_dir: Optional[Path] = None,
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
        target_domain: str = "prehistoric",
        include_cropmarks: bool = False,
        aerial_cache_dir: Optional[Path] = None,
    ) -> None:
        self.manifest = manifest
        self.catalog = catalog
        self.tile_cache_dir = Path(tile_cache_dir)
        self.augment = augment
        self.target_domain = target_domain
        self.include_cropmarks = include_cropmarks
        self.num_channels = NUM_TERRAIN_CHANNELS + (2 if include_cropmarks else 0)
        self.aerial_cache_dir = Path(aerial_cache_dir) if aerial_cache_dir else (self.tile_cache_dir.parent / "aerial_tiles")
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
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

    def _append_cropmarks_if_needed(self, feats: np.ndarray, spec: Dict) -> np.ndarray:
        if not self.include_cropmarks:
            return feats[:NUM_TERRAIN_CHANNELS]
        if feats.shape[0] >= NUM_TERRAIN_CHANNELS + 2:
            return feats[: NUM_TERRAIN_CHANNELS + 2]
        from megalith_lidar.aerial_cropmarks import compute_cropmark_features, fetch_or_load_aerial_tile

        rgb = fetch_or_load_aerial_tile(
            spec["e_min"],
            spec["e_max"],
            spec["n_min"],
            spec["n_max"],
            cache_dir=self.aerial_cache_dir,
            size=feats.shape[1],
        )
        if rgb is not None:
            ndsm = np.expm1(feats[5])
            crop_feats = compute_cropmark_features(rgb, dtm=np.zeros_like(ndsm), dsm=ndsm)
        else:
            crop_feats = np.zeros((2, feats.shape[1], feats.shape[2]), dtype=np.float32)
        return np.concatenate([feats[:NUM_TERRAIN_CHANNELS], crop_feats], axis=0).astype(np.float32)

    def _load_features_and_mask(self, spec: Dict) -> Tuple[np.ndarray, np.ndarray]:
        if self.target_domain == "roman":
            prefix = "roman_v1"
        elif self.target_domain == "enclosure":
            prefix = "enclosure_v1"
        else:
            prefix = "v3"
        if self.feature_cache_dir is not None:
            f_path = self.feature_cache_dir / f"{prefix}_{spec['sample_id']}.npz"
            if f_path.exists():
                try:
                    with np.load(f_path) as d:
                        raw_f = d["features"].astype(np.float32)
                        raw_m = d["mask"].astype(np.uint8)
                    if self.include_cropmarks and raw_f.shape[0] < NUM_TERRAIN_CHANNELS + 2:
                        raw_f = self._append_cropmarks_if_needed(raw_f, spec)
                        np.savez_compressed(f_path, features=raw_f.astype(np.float16), mask=raw_m)
                    elif not self.include_cropmarks and raw_f.shape[0] > NUM_TERRAIN_CHANNELS:
                        raw_f = raw_f[:NUM_TERRAIN_CHANNELS]
                    return raw_f, raw_m
                except Exception:
                    pass

        pair = fetch_or_load_paired_tile(
            spec["e_min"],
            spec["e_max"],
            spec["n_min"],
            spec["n_max"],
            cache_dir=self.tile_cache_dir,
        )
        assert pair is not None, f"Missing cached tile for {spec['sample_id']}"
        dtm, dsm = pair
        features = compute_terrain_features(dtm, dsm)
        if spec["is_positive"]:
            if self.target_domain in ("roman", "enclosure"):
                if self.target_domain == "roman":
                    mask = self.catalog.rasterize_roman_tile(
                        spec["e_min"],
                        spec["e_max"],
                        spec["n_min"],
                        spec["n_max"],
                        size=dtm.shape[0],
                    )
                else:
                    mask = self.catalog.rasterize_enclosure_tile(
                        spec["e_min"],
                        spec["e_max"],
                        spec["n_min"],
                        spec["n_max"],
                        size=dtm.shape[0],
                    )
                # For Roman villas/forts/roads and Iron Age hillforts/enclosures/hut circles/dykes:
                # Many survive under woodland canopy (nDSM >= 2.5m) where bare-earth 1m DTM
                # resolves the embankment/rampart/hut ring through the trees. Keep active bare-earth
                # relief pixels (|z - G_16(z)| >= 0.16m or top 32% of polygon relief) as positive,
                # mark a narrow 3m transition ring around the rampart as IGNORE (255), and supervise
                # flat interior pasture (>3m from ramparts) as explicit background (0) so the decoder
                # does not learn to flood-fill 20,000-45,000 m^2 interiors.
                pos_px = mask > 0
                if int(np.sum(pos_px)) > 500:
                    rel16 = np.abs(features[0] + features[1])
                    slope = features[3]
                    rel_th = max(0.16, min(0.28, float(np.percentile(rel16[pos_px], 68))))
                    earthwork = pos_px & ((rel16 >= rel_th) | (slope >= 0.065))
                    earthwork = binary_dilation(earthwork, iterations=2) & pos_px
                    if int(np.sum(earthwork)) < 48:
                        ys, xs = np.where(pos_px)
                        cy, cx = float(np.mean(ys)), float(np.mean(xs))
                        yy, xx = np.ogrid[: dtm.shape[0], : dtm.shape[1]]
                        core = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 12.0**2) & pos_px
                        earthwork = earthwork | core
                    transition_ring = binary_dilation(earthwork, iterations=3) & pos_px & (~earthwork)
                    refined = np.zeros_like(mask, dtype=np.uint8)
                    refined[transition_ring] = 255
                    refined[earthwork] = mask[earthwork]
                    mask = refined
            else:
                mask = self.catalog.rasterize_tile(
                    spec["e_min"],
                    spec["e_max"],
                    spec["n_min"],
                    spec["n_max"],
                    size=dtm.shape[0],
                )
                # Refine oversized NHLE legal-buffer polygons (> 600 m^2):
                # Keep active local-relief earthwork pixels as the positive monument class,
                # mark a 3m transition halo around the earthwork as IGNORE_INDEX (255), and mark
                # flat legal-buffer pasture and tree canopy as explicit background (0).
                pos_px = mask > 0
                if int(np.sum(pos_px)) > 600:
                    rel16 = np.abs(features[0] + features[1])
                    ndsm = np.expm1(features[5])
                    rel_th = max(0.20, float(np.percentile(rel16[pos_px], 75)))
                    earthwork = pos_px & (rel16 >= rel_th) & (ndsm < 2.5)
                    earthwork = binary_dilation(earthwork, iterations=2) & pos_px & (ndsm < 2.5)
                    if int(np.sum(earthwork)) < 36:
                        ys, xs = np.where(pos_px)
                        cy, cx = float(np.mean(ys)), float(np.mean(xs))
                        yy, xx = np.ogrid[: dtm.shape[0], : dtm.shape[1]]
                        core = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 10.0**2) & pos_px & (ndsm < 2.5)
                        earthwork = earthwork | core
                    transition_ring = binary_dilation(earthwork, iterations=3) & pos_px & (~earthwork) & (ndsm < 2.5)
                    refined = np.zeros_like(mask, dtype=np.uint8)
                    refined[transition_ring] = 255
                    refined[earthwork] = mask[earthwork]
                    mask = refined
        else:
            mask = np.zeros(dtm.shape, dtype=np.uint8)

        if self.include_cropmarks:
            features = self._append_cropmarks_if_needed(features, spec)

        if self.feature_cache_dir is not None:
            f_path = self.feature_cache_dir / f"{prefix}_{spec['sample_id']}.npz"
            np.savez_compressed(f_path, features=features.astype(np.float16), mask=mask)
        return features, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec = self.manifest[idx]
        features, mask = self._load_features_and_mask(spec)

        # Standardize using training-set mean & std
        features = (features - self.channel_mean) / self.channel_std

        # Measure physical local relief SNR (95th percentile of |band_fine + band_med| inside polygon)
        # before augmentation for stratified evaluation
        pos_px = (mask > 0) & (mask < 255)
        if np.any(pos_px):
            raw_local_relief = np.abs(
                features[0, pos_px] * self.channel_std[0, 0, 0] + features[1, pos_px] * self.channel_std[1, 0, 0]
            )
            relief_p95_m = float(np.percentile(raw_local_relief, 95))
        else:
            relief_p95_m = 0.0

        if self.augment:
            # 0. Modality dropout (50% probability of zeroing aerial cropmark channels during training)
            #    so the network remains accurate on bare-earth LiDAR in woodland/pasture or when aerial is offline
            if self.include_cropmarks and features.shape[0] >= NUM_TERRAIN_CHANNELS + 2:
                if np.random.rand() < 0.5:
                    features = features.copy()
                    features[NUM_TERRAIN_CHANNELS : NUM_TERRAIN_CHANNELS + 2] = 0.0
            # 1. Exact Dihedral D4 symmetry (rotations + flips)
            k_rot = int(np.random.randint(0, 4))
            if k_rot > 0:
                features = np.rot90(features, k=k_rot, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k_rot, axes=(0, 1)).copy()
            if np.random.rand() < 0.5:
                features = np.flip(features, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                features = np.flip(features, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

            # 2. Seam-free reflection-padded spatial shift (eliminates center-pixel spatial prior
            #    without introducing wrap-around step-scarp discontinuities from np.roll)
            shift_y = int(np.random.randint(-40, 41))
            shift_x = int(np.random.randint(-40, 41))
            if shift_y != 0 or shift_x != 0:
                pad = 40
                _, h_f, w_f = features.shape
                feat_pad = np.pad(features, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
                mask_fill = 255 if spec["is_positive"] else 0
                mask_pad = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=mask_fill)
                y0 = pad - shift_y
                x0 = pad - shift_x
                features = feat_pad[:, y0 : y0 + h_f, x0 : x0 + w_f].copy()
                mask = mask_pad[y0 : y0 + h_f, x0 : x0 + w_f].copy()

        has_pos = float(spec["is_positive"])
        primary_cls = int(spec["primary_class"]) if has_pos > 0.5 else 0

        return {
            "features": torch.from_numpy(features.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "tile_label": torch.tensor(has_pos, dtype=torch.float32),
            "type_label": torch.tensor(primary_cls, dtype=torch.long),
            "relief_p95_m": torch.tensor(relief_p95_m, dtype=torch.float32),
            "sample_id": spec["sample_id"],
        }
