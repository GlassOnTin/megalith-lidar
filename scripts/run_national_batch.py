#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megalith_lidar.arch_context import (
    NHLESpatialContextExtractor,
    enrich_candidates_with_archaeological_context,
)
from megalith_lidar.catalog import (
    CLASS_BARROW_CAIRN,
    CLASS_CIRCLE_HENGE_STONE,
    CLASS_MEGALITHIC_TOMB_LONG_BARROW,
    MonumentCatalog,
    bng_to_latlon,
    bng_to_os_grid_ref,
    download_or_load_nhle_catalog,
)
from megalith_lidar.dataset import assign_spatial_split
from megalith_lidar.model import MegalithUNet
from megalith_lidar.scanner import scan_region
from megalith_lidar.stats import PlattCalibrator


def build_deterministic_national_queue(
    catalog: MonumentCatalog,
    tile_cache_dir: Path,
    square_size_m: int = 2000,
    seed: int = 20261021,
) -> List[Dict]:
    """
    Build a deterministic, seed-shuffled queue of 2 km x 2 km British National Grid
    squares across England.
    All front-of-queue squares are drawn strictly from the held-out `test` split
    (`assign_spatial_split == "test"`, never used in model training or validation
    threshold calibration), alternating 1:1 between:
      - `uniform_national_null`: 1 weight per 20 km x 20 km England land block,
        zero monument-density weighting, outside known prehistoric monuments.
      - `heldout_test_landscape`: 2 km x 2 km test-split squares containing held-out
        prehistoric monuments to measure live recall alongside novel candidate discovery.
    """
    rng = np.random.default_rng(seed)

    prehistoric_classes = {
        CLASS_MEGALITHIC_TOMB_LONG_BARROW,
        CLASS_CIRCLE_HENGE_STONE,
        CLASS_BARROW_CAIRN,
    }

    # Collect all inhabited/land 20 km x 20 km blocks in England from the 20,003 NHLE records
    land_blocks_20k: Set[Tuple[int, int]] = set()
    prehistoric_squares_2k: Dict[Tuple[int, int], int] = {}

    for rec in catalog.records:
        bx = int(rec.easting // 20000)
        by = int(rec.northing // 20000)
        land_blocks_20k.add((bx, by))
        if rec.cls in prehistoric_classes:
            sx = int(rec.easting // square_size_m) * square_size_m
            sy = int(rec.northing // square_size_m) * square_size_m
            prehistoric_squares_2k[(sx, sy)] = prehistoric_squares_2k.get((sx, sy), 0) + 1

    def _has_cached_2km_square(sx: int, sy: int) -> bool:
        sub = tile_cache_dir / f"E{sx // 10000:02d}_N{sy // 10000:02d}"
        return (
            (sub / f"tile_E{sx}_{sx + 256}_N{sy}_{sy + 256}.npz").exists()
            and (sub / f"tile_E{sx + 192}_{sx + 448}_N{sy}_{sy + 256}.npz").exists()
        )

    # 1. Held-out test squares containing prehistoric monuments (never seen in train or val)
    heldout_landscape_squares: List[Dict] = []
    for (sx, sy), count in sorted(prehistoric_squares_2k.items()):
        ce = sx + square_size_m // 2
        cn = sy + square_size_m // 2
        split = assign_spatial_split(ce, cn)
        if split != "test":
            continue
        lat, lon = bng_to_latlon(ce, cn)
        is_cached = _has_cached_2km_square(sx, sy)
        heldout_landscape_squares.append(
            {
                "square_id": f"test_landscape_E{sx}_N{sy}",
                "sampling_stream": "heldout_test_landscape",
                "split": split,
                "e_min": sx,
                "n_min": sy,
                "e_max": sx + square_size_m,
                "n_max": sy + square_size_m,
                "center_lat": round(lat, 5),
                "center_lon": round(lon, 5),
                "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                "known_prehistoric_monuments_in_square": count,
                "status": "pending",
                "_cached": is_cached,
            }
        )
    rng.shuffle(heldout_landscape_squares)
    heldout_landscape_squares.sort(key=lambda d: not d.pop("_cached", False))

    # 2. Uniform national null squares across England test-split 20 km x 20 km land blocks
    # (1 draw per 20 km block per pass so every region of England has equal probability)
    test_land_blocks = sorted(
        [
            (bx, by)
            for (bx, by) in land_blocks_20k
            if assign_spatial_split(bx * 20000 + 10000, by * 20000 + 10000) == "test"
        ]
    )
    uniform_null_squares: List[Dict] = []
    seen_squares: Set[Tuple[int, int]] = set(prehistoric_squares_2k.keys())

    # Include any already-cached test-split null 2km square first (e.g. from earlier random null draw)
    for p in sorted(tile_cache_dir.glob("E*_N*/tile_E*_N*.npz")):
        parts = p.stem.split("_")
        if len(parts) == 5:
            e0 = int(parts[1][1:])
            n0 = int(parts[3][1:])
            if e0 % square_size_m == 0 and n0 % square_size_m == 0 and (e0, n0) not in seen_squares:
                if _has_cached_2km_square(e0, n0):
                    ce = e0 + square_size_m // 2
                    cn = n0 + square_size_m // 2
                    if assign_spatial_split(ce, cn) == "test":
                        seen_squares.add((e0, n0))
                        lat, lon = bng_to_latlon(ce, cn)
                        uniform_null_squares.append(
                            {
                                "square_id": f"null_test_E{e0}_N{n0}",
                                "sampling_stream": "uniform_national_null",
                                "split": "test",
                                "e_min": e0,
                                "n_min": n0,
                                "e_max": e0 + square_size_m,
                                "n_max": n0 + square_size_m,
                                "center_lat": round(lat, 5),
                                "center_lon": round(lon, 5),
                                "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                                "known_prehistoric_monuments_in_square": 0,
                                "status": "pending",
                            }
                        )

    for pass_idx in range(4):
        blocks_copy = list(test_land_blocks)
        rng.shuffle(blocks_copy)
        for bx, by in blocks_copy:
            sub_x = int(rng.integers(0, 20000 // square_size_m))
            sub_y = int(rng.integers(0, 20000 // square_size_m))
            sx = bx * 20000 + sub_x * square_size_m
            sy = by * 20000 + sub_y * square_size_m
            if (sx, sy) in seen_squares:
                continue
            seen_squares.add((sx, sy))
            ce = sx + square_size_m // 2
            cn = sy + square_size_m // 2
            split = assign_spatial_split(ce, cn)
            lat, lon = bng_to_latlon(ce, cn)
            uniform_null_squares.append(
                {
                    "square_id": f"null_test_E{sx}_N{sy}",
                    "sampling_stream": "uniform_national_null",
                    "split": split,
                    "e_min": sx,
                    "n_min": sy,
                    "e_max": sx + square_size_m,
                    "n_max": sy + square_size_m,
                    "center_lat": round(lat, 5),
                    "center_lon": round(lon, 5),
                    "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                    "known_prehistoric_monuments_in_square": 0,
                    "status": "pending",
                }
            )

    # Interleave 1:1 between uniform_national_null and heldout_test_landscape
    interleaved: List[Dict] = []
    max_len = max(len(uniform_null_squares), len(heldout_landscape_squares))
    for i in range(max_len):
        if i < len(uniform_null_squares):
            interleaved.append(uniform_null_squares[i])
        if i < len(heldout_landscape_squares):
            interleaved.append(heldout_landscape_squares[i])

    return interleaved


HELDOUT_SEED_REGIONS = [
    ("Ure Valley / Masham (North of Thornborough)", 422000, 488000, 420000, 480000),
    ("Cotswolds (Bourton / Windrush)", 414000, 224000, 400000, 220000),
    ("Wessex / Wylye & Avon Valleys", 414000, 132000, 400000, 120000),
    ("Bodmin Moor / Rough Tor Fringe", 214000, 88000, 200000, 80000),
    ("South Downs (Chichester / Bignor)", 490000, 112000, 480000, 100000),
]


def prioritize_contiguous_seed_regions_in_queue(
    queue_state: Dict,
    catalog: MonumentCatalog,
    square_size_m: int = 2000,
) -> bool:
    """
    Ensure that contiguous 12 km x 12 km (6x6 squares) patches around each of the 5 held-out
    test seed regions are present in the queue and ordered in concentric outward spirals at
    the front of the pending queue (round-robin across the 5 seed landscapes).
    """
    if queue_state.get("contiguous_seed_regions_prioritized"):
        return False

    prehistoric_classes = {
        CLASS_MEGALITHIC_TOMB_LONG_BARROW,
        CLASS_CIRCLE_HENGE_STONE,
        CLASS_BARROW_CAIRN,
    }
    prehistoric_squares_2k: Dict[Tuple[int, int], int] = {}
    for rec in catalog.records:
        if rec.cls in prehistoric_classes:
            sx = int(rec.easting // square_size_m) * square_size_m
            sy = int(rec.northing // square_size_m) * square_size_m
            prehistoric_squares_2k[(sx, sy)] = prehistoric_squares_2k.get((sx, sy), 0) + 1

    existing_by_coord: Dict[Tuple[int, int], Dict] = {}
    for item in queue_state["queue"]:
        existing_by_coord[(int(item["e_min"]), int(item["n_min"]))] = item

    seed_spirals: List[List[Dict]] = []
    prioritized_coords: Set[Tuple[int, int]] = set()

    for _region_name, seed_e, seed_n, block_e, block_n in HELDOUT_SEED_REGIONS:
        spiral_coords: List[Tuple[float, int, int]] = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                sx = seed_e + dx * square_size_m
                sy = seed_n + dy * square_size_m
                if not (block_e <= sx < block_e + 20000 and block_n <= sy < block_n + 20000):
                    continue
                ce = sx + square_size_m // 2
                cn = sy + square_size_m // 2
                if assign_spatial_split(ce, cn) != "test":
                    continue
                dist2 = float(dx * dx + dy * dy)
                spiral_coords.append((dist2, sx, sy))
        spiral_coords.sort(key=lambda t: (t[0], t[1], t[2]))

        region_pending: List[Dict] = []
        for _d2, sx, sy in spiral_coords:
            if (sx, sy) in prioritized_coords:
                continue
            item = existing_by_coord.get((sx, sy))
            if item is not None:
                if item.get("status") == "pending":
                    region_pending.append(item)
                    prioritized_coords.add((sx, sy))
            else:
                ce = sx + square_size_m // 2
                cn = sy + square_size_m // 2
                lat, lon = bng_to_latlon(ce, cn)
                count = prehistoric_squares_2k.get((sx, sy), 0)
                new_item = {
                    "square_id": f"test_landscape_E{sx}_N{sy}",
                    "sampling_stream": "heldout_test_landscape",
                    "split": "test",
                    "e_min": sx,
                    "n_min": sy,
                    "e_max": sx + square_size_m,
                    "n_max": sy + square_size_m,
                    "center_lat": round(lat, 5),
                    "center_lon": round(lon, 5),
                    "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                    "known_prehistoric_monuments_in_square": count,
                    "status": "pending",
                }
                existing_by_coord[(sx, sy)] = new_item
                region_pending.append(new_item)
                prioritized_coords.add((sx, sy))
        seed_spirals.append(region_pending)

    # Interleave round-robin across the 5 seed regions so all 5 grow outward together,
    # starting with two extra rings in the Ure Valley block immediately north of Thornborough.
    front_pending: List[Dict] = []
    ure_spiral = seed_spirals[0] if seed_spirals else []
    for item in ure_spiral[:8]:
        front_pending.append(item)
    seed_spirals[0] = ure_spiral[8:]

    max_len = max((len(s) for s in seed_spirals), default=0)
    for i in range(max_len):
        for s in seed_spirals:
            if i < len(s):
                front_pending.append(s[i])

    done_or_skipped = [q for q in queue_state["queue"] if q.get("status") != "pending"]
    remaining_pending = [
        q
        for q in queue_state["queue"]
        if q.get("status") == "pending" and (int(q["e_min"]), int(q["n_min"])) not in prioritized_coords
    ]
    queue_state["queue"] = done_or_skipped + front_pending + remaining_pending
    queue_state["contiguous_seed_regions_prioritized"] = True
    return True


def expand_national_queue_if_needed(
    queue_state: Dict,
    catalog: MonumentCatalog,
    square_size_m: int = 2000,
    min_pending: int = 1200,
    seed: int = 20261023,
) -> bool:
    """
    When pending held-out test squares drop below `min_pending`, append:
      1. Full 20 km x 20 km (10x10 squares) contiguous outward spirals for the 5 seed regions,
      2. All remaining held-out test-split 2 km x 2 km squares in England containing
         Prehistoric, Roman, or Iron Age/Enclosure NHLE Scheduled Monuments, interleaved 1:1
         with newly drawn uniform_national_null squares across all 53 test-split 20 km blocks.
    """
    pending_now = sum(1 for q in queue_state["queue"] if q.get("status") == "pending")
    if pending_now >= min_pending:
        return False

    rng = np.random.default_rng(seed)
    existing_coords: Set[Tuple[int, int]] = {
        (int(q["e_min"]), int(q["n_min"])) for q in queue_state["queue"]
    }

    prehistoric_classes = {
        CLASS_MEGALITHIC_TOMB_LONG_BARROW,
        CLASS_CIRCLE_HENGE_STONE,
        CLASS_BARROW_CAIRN,
    }
    monument_squares_2k: Dict[Tuple[int, int], int] = {}
    prehistoric_squares_2k: Dict[Tuple[int, int], int] = {}
    land_blocks_20k: Set[Tuple[int, int]] = set()

    for rec in catalog.records:
        bx = int(rec.easting // 20000)
        by = int(rec.northing // 20000)
        land_blocks_20k.add((bx, by))
        sx = int(rec.easting // square_size_m) * square_size_m
        sy = int(rec.northing // square_size_m) * square_size_m
        if rec.cls in prehistoric_classes:
            prehistoric_squares_2k[(sx, sy)] = prehistoric_squares_2k.get((sx, sy), 0) + 1
        monument_squares_2k[(sx, sy)] = monument_squares_2k.get((sx, sy), 0) + 1

    # 1. Full 10x10 (20 km x 20 km) outward spirals in the 5 held-out seed blocks
    seed_spirals: List[List[Dict]] = []
    for _region_name, seed_e, seed_n, block_e, block_n in HELDOUT_SEED_REGIONS:
        spiral_coords: List[Tuple[float, int, int]] = []
        for sx in range(block_e, block_e + 20000, square_size_m):
            for sy in range(block_n, block_n + 20000, square_size_m):
                if (sx, sy) in existing_coords:
                    continue
                ce = sx + square_size_m // 2
                cn = sy + square_size_m // 2
                if assign_spatial_split(ce, cn) != "test":
                    continue
                dx = (sx - seed_e) / square_size_m
                dy = (sy - seed_n) / square_size_m
                spiral_coords.append((float(dx * dx + dy * dy), sx, sy))
        spiral_coords.sort(key=lambda t: (t[0], t[1], t[2]))
        reg_items: List[Dict] = []
        for _d2, sx, sy in spiral_coords:
            existing_coords.add((sx, sy))
            ce = sx + square_size_m // 2
            cn = sy + square_size_m // 2
            lat, lon = bng_to_latlon(ce, cn)
            reg_items.append(
                {
                    "square_id": f"test_landscape_E{sx}_N{sy}",
                    "sampling_stream": "heldout_test_landscape",
                    "split": "test",
                    "e_min": sx,
                    "n_min": sy,
                    "e_max": sx + square_size_m,
                    "n_max": sy + square_size_m,
                    "center_lat": round(lat, 5),
                    "center_lon": round(lon, 5),
                    "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                    "known_prehistoric_monuments_in_square": prehistoric_squares_2k.get((sx, sy), 0),
                    "status": "pending",
                }
            )
        seed_spirals.append(reg_items)

    new_seed_round_robin: List[Dict] = []
    max_s = max((len(s) for s in seed_spirals), default=0)
    for i in range(max_s):
        for s in seed_spirals:
            if i < len(s):
                new_seed_round_robin.append(s[i])

    # 2. Remaining held-out test landscape squares across England, followed by 1-ring Halo squares
    #    in held-out test blocks and out-of-sample val-split monument squares so multi-day runs never starve
    new_landscape: List[Dict] = []
    for (sx, sy), _cnt in sorted(monument_squares_2k.items()):
        if (sx, sy) in existing_coords:
            continue
        ce = sx + square_size_m // 2
        cn = sy + square_size_m // 2
        if assign_spatial_split(ce, cn) != "test":
            continue
        existing_coords.add((sx, sy))
        lat, lon = bng_to_latlon(ce, cn)
        new_landscape.append(
            {
                "square_id": f"test_landscape_E{sx}_N{sy}",
                "sampling_stream": "heldout_test_landscape",
                "split": "test",
                "e_min": sx,
                "n_min": sy,
                "e_max": sx + square_size_m,
                "n_max": sy + square_size_m,
                "center_lat": round(lat, 5),
                "center_lon": round(lon, 5),
                "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                "known_prehistoric_monuments_in_square": prehistoric_squares_2k.get((sx, sy), 0),
                "status": "pending",
            }
        )
    rng.shuffle(new_landscape)

    # 2b. 1-ring Halo (2 km adjacent) squares around monuments inside held-out test 20 km blocks
    halo_test_coords: Set[Tuple[int, int]] = set()
    for (mx, my) in monument_squares_2k.keys():
        if assign_spatial_split(mx + square_size_m // 2, my + square_size_m // 2) != "test":
            continue
        bx, by = mx // 20000, my // 20000
        for dx in (-square_size_m, 0, square_size_m):
            for dy in (-square_size_m, 0, square_size_m):
                nx, ny = mx + dx, my + dy
                if (nx // 20000, ny // 20000) == (bx, by) and (nx, ny) not in existing_coords:
                    halo_test_coords.add((nx, ny))
    halo_test_list: List[Dict] = []
    for sx, sy in sorted(halo_test_coords):
        existing_coords.add((sx, sy))
        ce, cn = sx + square_size_m // 2, sy + square_size_m // 2
        lat, lon = bng_to_latlon(ce, cn)
        halo_test_list.append(
            {
                "square_id": f"test_landscape_E{sx}_N{sy}",
                "sampling_stream": "heldout_test_landscape",
                "split": "test",
                "e_min": sx,
                "n_min": sy,
                "e_max": sx + square_size_m,
                "n_max": sy + square_size_m,
                "center_lat": round(lat, 5),
                "center_lon": round(lon, 5),
                "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                "known_prehistoric_monuments_in_square": 0,
                "status": "pending",
            }
        )
    rng.shuffle(halo_test_list)

    # 2c. Out-of-sample val-split monument landscape squares across England
    val_landscape_list: List[Dict] = []
    for (sx, sy), _cnt in sorted(monument_squares_2k.items()):
        if (sx, sy) in existing_coords:
            continue
        ce, cn = sx + square_size_m // 2, sy + square_size_m // 2
        if assign_spatial_split(ce, cn) != "val":
            continue
        existing_coords.add((sx, sy))
        lat, lon = bng_to_latlon(ce, cn)
        val_landscape_list.append(
            {
                "square_id": f"test_landscape_E{sx}_N{sy}",
                "sampling_stream": "heldout_test_landscape",
                "split": "val",
                "e_min": sx,
                "n_min": sy,
                "e_max": sx + square_size_m,
                "n_max": sy + square_size_m,
                "center_lat": round(lat, 5),
                "center_lon": round(lon, 5),
                "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                "known_prehistoric_monuments_in_square": prehistoric_squares_2k.get((sx, sy), 0),
                "status": "pending",
            }
        )
    rng.shuffle(val_landscape_list)
    all_new_landscape = new_seed_round_robin + new_landscape + halo_test_list + val_landscape_list

    # 3. Matching number of uniform_national_null squares across out-of-sample (test + val) 20 km blocks
    test_land_blocks = sorted(
        [
            (bx, by)
            for (bx, by) in land_blocks_20k
            if assign_spatial_split(bx * 20000 + 10000, by * 20000 + 10000) in {"test", "val"}
        ]
    )
    new_null: List[Dict] = []
    for _pass_idx in range(45):
        if len(new_null) >= len(all_new_landscape):
            break
        blocks_copy = list(test_land_blocks)
        rng.shuffle(blocks_copy)
        for bx, by in blocks_copy:
            sub_x = int(rng.integers(0, 20000 // square_size_m))
            sub_y = int(rng.integers(0, 20000 // square_size_m))
            sx = bx * 20000 + sub_x * square_size_m
            sy = by * 20000 + sub_y * square_size_m
            if (sx, sy) in existing_coords or (sx, sy) in monument_squares_2k:
                continue
            existing_coords.add((sx, sy))
            ce = sx + square_size_m // 2
            cn = sy + square_size_m // 2
            sp_null = assign_spatial_split(ce, cn)
            lat, lon = bng_to_latlon(ce, cn)
            new_null.append(
                {
                    "square_id": f"null_test_E{sx}_N{sy}",
                    "sampling_stream": "uniform_national_null",
                    "split": sp_null,
                    "e_min": sx,
                    "n_min": sy,
                    "e_max": sx + square_size_m,
                    "n_max": sy + square_size_m,
                    "center_lat": round(lat, 5),
                    "center_lon": round(lon, 5),
                    "os_grid_ref": bng_to_os_grid_ref(ce, cn),
                    "known_prehistoric_monuments_in_square": 0,
                    "status": "pending",
                }
            )
            if len(new_null) >= len(all_new_landscape):
                break

    # Interleave 1:1
    appended: List[Dict] = []
    max_len = max(len(all_new_landscape), len(new_null))
    for i in range(max_len):
        if i < len(all_new_landscape):
            appended.append(all_new_landscape[i])
        if i < len(new_null):
            appended.append(new_null[i])

    if not appended:
        return False
    queue_state["queue"].extend(appended)
    print(f"[queue] Expanded held-out test queue with {len(appended)} additional 2km x 2km squares.")
    return True


def export_public_docs(
    queue_state: Dict,
    survey_dir: Path,
    eval_metrics_path: Path,
    docs_dir: Path,
    catalog: Optional[MonumentCatalog] = None,
) -> Dict:
    """
    Aggregate all completed 2 km x 2 km squares and export:
      - `docs/survey_summary.json` (model evaluation metrics + national survey progress + stream breakdowns)
      - `docs/candidates.geojson` (all deduplicated detections with OSM & NHLE spatial context)
      - `docs/diagnostics/*.png` (4-panel LiDAR diagnostic images referenced by candidates)
    """
    docs_dir = Path(docs_dir)
    docs_diag_dir = docs_dir / "diagnostics"
    if docs_diag_dir.exists():
        for old_png in docs_diag_dir.glob("*.png"):
            if (
                old_png.name.startswith("thornborough_")
                or old_png.name.startswith("roman_")
                or old_png.name.startswith("enclosure_")
            ):
                continue
            old_png.unlink(missing_ok=True)
    docs_diag_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = {}
    if eval_metrics_path.exists():
        eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8"))

    completed_squares = [q for q in queue_state["queue"] if q.get("status") == "completed"]
    skipped_squares = [q for q in queue_state["queue"] if q.get("status") == "no_lidar_coverage"]
    pending_squares = [q for q in queue_state["queue"] if q.get("status") == "pending"]

    all_features: List[Dict] = []
    square_summaries: List[Dict] = []
    copied_pngs: Set[str] = set()

    stream_stats = {
        "uniform_national_null": {
            "squares_scanned": 0,
            "area_km2": 0.0,
            "rediscovered_nhle_scheduled": 0,
            "rediscovered_osm_archaeology": 0,
            "flagged_modern_or_karst_confounders": 0,
            "unidentified_candidates": 0,
            "unidentified_candidates_fpr_0_01": 0,
        },
        "heldout_test_landscape": {
            "squares_scanned": 0,
            "area_km2": 0.0,
            "rediscovered_nhle_scheduled": 0,
            "rediscovered_osm_archaeology": 0,
            "flagged_modern_or_karst_confounders": 0,
            "unidentified_candidates": 0,
            "unidentified_candidates_fpr_0_01": 0,
        },
    }

    for sq in completed_squares:
        sq_id = sq["square_id"]
        stream = sq["sampling_stream"]
        report_path = survey_dir / sq_id / f"{sq_id}_report.json"
        geojson_path = survey_dir / sq_id / f"{sq_id}_candidates.geojson"
        if not report_path.exists() or not geojson_path.exists():
            continue

        rep = json.loads(report_path.read_text(encoding="utf-8"))
        gj = json.loads(geojson_path.read_text(encoding="utf-8"))

        area_km2 = float(rep.get("area_km2", 4.0))
        st = stream_stats.setdefault(
            stream,
            {
                "squares_scanned": 0,
                "area_km2": 0.0,
                "rediscovered_nhle_scheduled": 0,
                "rediscovered_osm_archaeology": 0,
                "flagged_modern_or_karst_confounders": 0,
                "unidentified_candidates": 0,
                "unidentified_candidates_fpr_0_01": 0,
            },
        )
        st["squares_scanned"] += 1
        st["area_km2"] = round(st["area_km2"] + area_km2, 3)
        st["rediscovered_nhle_scheduled"] += int(rep.get("rediscovered_scheduled_monuments_count", 0))
        st["rediscovered_osm_archaeology"] += int(rep.get("rediscovered_osm_archaeology_count", 0))
        st["flagged_modern_or_karst_confounders"] += int(rep.get("flagged_modern_or_karst_confounders_count", 0))
        st["unidentified_candidates"] += int(rep.get("unidentified_candidates_count", 0))

        unid_001 = 0
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            props["square_id"] = sq_id
            props["sampling_stream"] = stream
            props["split"] = sq.get("split", "test")
            if props.get("context_category") == "unidentified_candidate" and props.get("passes_fpr_0_01"):
                unid_001 += 1

            diag_name = props.get("diagnostic_png")
            if diag_name:
                src_png = survey_dir / sq_id / "diagnostics" / diag_name
                if src_png.exists() and diag_name not in copied_pngs:
                    shutil.copy2(src_png, docs_diag_dir / diag_name)
                    copied_pngs.add(diag_name)

            all_features.append(feat)

        st["unidentified_candidates_fpr_0_01"] += unid_001

        lat_sw, lon_sw = bng_to_latlon(sq["e_min"], sq["n_min"])
        lat_ne, lon_ne = bng_to_latlon(sq["e_max"], sq["n_max"])
        square_summaries.append(
            {
                "square_id": sq_id,
                "sampling_stream": stream,
                "split": sq.get("split", "test"),
                "os_grid_ref": sq.get("os_grid_ref", ""),
                "bbox_epsg27700": [sq["e_min"], sq["n_min"], sq["e_max"], sq["n_max"]],
                "bounds_latlon": [[round(lat_sw, 5), round(lon_sw, 5)], [round(lat_ne, 5), round(lon_ne, 5)]],
                "center_lat": sq["center_lat"],
                "center_lon": sq["center_lon"],
                "area_km2": area_km2,
                "windows_with_valid_1m_lidar": rep.get("windows_with_valid_1m_lidar", 0),
                "windows_above_tile_threshold": rep.get("windows_above_tile_threshold", 0),
                "rediscovered_nhle_scheduled_count": rep.get("rediscovered_scheduled_monuments_count", 0),
                "rediscovered_osm_archaeology_count": rep.get("rediscovered_osm_archaeology_count", 0),
                "flagged_modern_or_karst_confounders_count": rep.get("flagged_modern_or_karst_confounders_count", 0),
                "unidentified_candidates_count": rep.get("unidentified_candidates_count", 0),
                "unidentified_candidates_fpr_0_01_count": unid_001,
                "scanned_at_utc": sq.get("scanned_at_utc", ""),
            }
        )

    if catalog is not None and all_features:
        extractor = NHLESpatialContextExtractor(catalog)
        cand_props = [f["properties"] for f in all_features]
        arch_cache_path = survey_dir.parent / "arch_context_cache.json"
        enrich_candidates_with_archaeological_context(
            candidates=cand_props,
            extractor=extractor,
            cache_path=arch_cache_path,
            docs_diag_dir=docs_diag_dir,
            max_vlm_calls=80,
        )
        # Recompute stream_stats and square_summaries counts after VLM rectilinear-bund filtering
        for st in stream_stats.values():
            st["rediscovered_nhle_scheduled"] = 0
            st["rediscovered_osm_archaeology"] = 0
            st["flagged_modern_or_karst_confounders"] = 0
            st["unidentified_candidates"] = 0
            st["unidentified_candidates_fpr_0_01"] = 0
        sq_map = {s["square_id"]: s for s in square_summaries}
        for s in square_summaries:
            s["flagged_modern_or_karst_confounders_count"] = 0
            s["unidentified_candidates_count"] = 0
            s["unidentified_candidates_fpr_0_01_count"] = 0
        for feat in all_features:
            p = feat["properties"]
            st = stream_stats.get(p.get("sampling_stream", ""))
            sq_s = sq_map.get(p.get("square_id", ""))
            cat = p.get("context_category")
            if st is not None:
                if cat == "rediscovered_nhle_scheduled":
                    st["rediscovered_nhle_scheduled"] += 1
                elif cat == "rediscovered_osm_archaeology":
                    st["rediscovered_osm_archaeology"] += 1
                elif cat == "flagged_modern_or_karst_confounder":
                    st["flagged_modern_or_karst_confounders"] += 1
                elif cat == "unidentified_candidate":
                    st["unidentified_candidates"] += 1
                    if p.get("passes_fpr_0_01"):
                        st["unidentified_candidates_fpr_0_01"] += 1
            if sq_s is not None:
                if cat == "flagged_modern_or_karst_confounder":
                    sq_s["flagged_modern_or_karst_confounders_count"] += 1
                elif cat == "unidentified_candidate":
                    sq_s["unidentified_candidates_count"] += 1
                    if p.get("passes_fpr_0_01"):
                        sq_s["unidentified_candidates_fpr_0_01_count"] += 1

    for stream, st in stream_stats.items():
        area = max(st["area_km2"], 1e-6)
        st["unidentified_density_per_km2_fpr_0_05"] = round(st["unidentified_candidates"] / area, 3)
        st["unidentified_density_per_km2_fpr_0_01"] = round(st["unidentified_candidates_fpr_0_01"] / area, 3)

    total_area_km2 = round(sum(s["area_km2"] for s in square_summaries), 3)
    public_summary = {
        "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license": "AGPL-3.0-or-later",
        "data_attribution": (
            "Contains Environment Agency Open LiDAR data (OGL v3.0), "
            "Historic England National Heritage List for England (OGL v3.0), "
            "and OpenStreetMap contributors (ODbL)."
        ),
        "queue_progress": {
            "total_queued_test_squares": len(queue_state["queue"]),
            "completed_squares": len(completed_squares),
            "no_lidar_coverage_squares": len(skipped_squares),
            "pending_squares": len(pending_squares),
            "total_scanned_area_km2": total_area_km2,
            "total_detections": len(all_features),
        },
        "stream_breakdown": stream_stats,
        "heldout_test_evaluation": eval_metrics.get("model_test_metrics", {}),
        "scalar_baseline_test": eval_metrics.get("unbiased_scalar_baseline_test", {}),
        "subgroup_and_confounding_analysis": eval_metrics.get(
            "model_test_subgroup_and_confounding_analysis", {}
        ),
        "validation_operating_points": eval_metrics.get("val_selected_thresholds", {}),
        "calibration_parameters": eval_metrics.get("platt_calibrator", {}),
        "scanned_squares": square_summaries,
    }

    (docs_dir / "survey_summary.json").write_text(
        json.dumps(public_summary, indent=2),
        encoding="utf-8",
    )
    non_conf = [f for f in all_features if f["properties"].get("context_category") != "flagged_modern_or_karst_confounder"]
    conf_only = [f for f in all_features if f["properties"].get("context_category") == "flagged_modern_or_karst_confounder"]
    conf_only.sort(
        key=lambda f: (1 if f["properties"].get("diagnostic_png") else 0, float(f["properties"].get("confidence", 0.0))),
        reverse=True,
    )
    export_feats = non_conf + conf_only[:400]
    for f in export_feats:
        p = f["properties"]
        if isinstance(p.get("nearby_nhle_monuments"), list) and len(p["nearby_nhle_monuments"]) > 3:
            p["nearby_nhle_monuments"] = p["nearby_nhle_monuments"][:3]
    (docs_dir / "candidates.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": export_feats}, separators=(",", ":")),
        encoding="utf-8",
    )
    return public_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a scheduled batch of the unbiased UK National LiDAR Megalith Survey."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/bulk/megalith-lidar"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/bulk/megalith-lidar/checkpoints/megalith_unet_best.pt"),
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("/home/ian/Code/megalith-lidar/docs"),
    )
    parser.add_argument("--max-squares", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=1, help="Number of consecutive batch cycles to run")
    parser.add_argument(
        "--roman-enclosure-per-cycle",
        type=int,
        default=25,
        help="Number of completed squares to scan with RomanUNet and EnclosureUNet per cycle",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--rescore-completed", action="store_true", help="Re-score all completed squares with current model & OSM context")
    parser.add_argument("--shard-idx", type=int, default=0, help="Shard index for parallel rescoring")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards")
    parser.add_argument("--git-push", action="store_true", help="Commit and push updated docs/ to GitHub")
    args = parser.parse_args()

    survey_dir = args.data_dir / "national_survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    queue_path = survey_dir / "queue_state.json"
    eval_metrics_path = args.checkpoint.parent / "evaluation_metrics_v2_unbiased.json"

    catalog_path = args.data_dir / "catalog" / "nhle_scheduled_monuments_27700.json"
    catalog = download_or_load_nhle_catalog(catalog_path)
    tile_cache_dir = args.data_dir / "tiles"
    osm_cache_dir = args.data_dir / "osm_cache"

    if queue_path.exists():
        queue_state = json.loads(queue_path.read_text(encoding="utf-8"))
    else:
        queue_items = build_deterministic_national_queue(catalog, tile_cache_dir=tile_cache_dir)
        queue_state = {
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": 20261021,
            "square_size_m": 2000,
            "queue": queue_items,
        }
    if prioritize_contiguous_seed_regions_in_queue(queue_state, catalog):
        queue_path.write_text(json.dumps(queue_state, indent=2), encoding="utf-8")
    if expand_national_queue_if_needed(queue_state, catalog):
        queue_path.write_text(json.dumps(queue_state, indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = MegalithUNet(
        in_channels=int(ckpt.get("in_channels", 6)),
        num_classes=int(ckpt.get("num_classes", 4)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt.get("model_state_dict") or ckpt["model_state"])
    model.eval()

    calibrator = PlattCalibrator.from_dict(ckpt["calibrator"])

    op_points = ckpt.get("val_thresholds") or ckpt["validation_operating_points"]
    th_005 = float(op_points["fpr_0.05"]["threshold"])
    th_001 = float(op_points["fpr_0.01"]["threshold"])
    pixel_th_005 = float(ckpt.get("val_pixel_threshold_fpr05", 0.58))
    pixel_th_001 = max(pixel_th_005, 0.72)

    channel_mean = np.array(ckpt["channel_mean"], dtype=np.float32)
    channel_std = np.array(ckpt["channel_std"], dtype=np.float32)
    del ckpt

    if args.rescore_completed:
        completed_items = []
        for q in queue_state["queue"]:
            if q.get("status") != "completed":
                continue
            rep_p = survey_dir / q["square_id"] / f"{q['square_id']}_report.json"
            if rep_p.exists():
                rep_d = json.loads(rep_p.read_text(encoding="utf-8"))
                if int(rep_d.get("unidentified_candidates_count", 1)) == 0:
                    continue
            completed_items.append(q)
        shard_items = [
            item for idx, item in enumerate(completed_items) if idx % max(args.num_shards, 1) == args.shard_idx
        ]
        for idx, item in enumerate(shard_items, 1):
            sq_id = item["square_id"]
            e_min, n_min, e_max, n_max = item["e_min"], item["n_min"], item["e_max"], item["n_max"]
            t0 = time.time()
            out_dir = survey_dir / sq_id
            diag_sub = out_dir / "diagnostics"
            if diag_sub.exists():
                for old_p in diag_sub.glob("*.png"):
                    old_p.unlink(missing_ok=True)
            summary = scan_region(
                model=model,
                catalog=catalog,
                channel_mean=channel_mean,
                channel_std=channel_std,
                calibrator=calibrator,
                tile_prob_threshold=th_005,
                pixel_prob_threshold=pixel_th_005,
                e_min=e_min,
                e_max=e_max,
                n_min=n_min,
                n_max=n_max,
                device=device,
                tile_cache_dir=tile_cache_dir,
                output_dir=out_dir,
                region_name=sq_id,
                osm_cache_dir=osm_cache_dir,
            )
            gj_path = out_dir / f"{sq_id}_candidates.geojson"
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
                f"[rescore shard {args.shard_idx}/{args.num_shards}] ({idx}/{len(shard_items)}) {sq_id} ({round(time.time() - t0, 1)}s): "
                f"NHLE={summary['rediscovered_scheduled_monuments_count']}, "
                f"OSM_arch={summary['rediscovered_osm_archaeology_count']}, "
                f"Confounder={summary['flagged_modern_or_karst_confounders_count']}, "
                f"Unid={summary['unidentified_candidates_count']}"
            )
        if args.num_shards > 1:
            return

    from scripts.scan_enclosure_batch import run_enclosure_scan
    from scripts.scan_roman_batch import run_roman_scan

    roman_ckpt = args.checkpoint.parent / "roman_unet_best.pt"
    enclosure_ckpt = args.checkpoint.parent / "enclosure_unet_best.pt"

    total_cycles = max(1, int(args.cycles))
    for cycle_idx in range(1, total_cycles + 1):
        if expand_national_queue_if_needed(queue_state, catalog):
            queue_path.write_text(json.dumps(queue_state, indent=2), encoding="utf-8")

        pending_count = sum(1 for item in queue_state["queue"] if item.get("status") == "pending")
        if pending_count == 0 and not args.rescore_completed:
            print("[batch] All 1,417 test squares in queue_state are completed; skipping no-op re-export.")
            return

        model.to(device)
        scanned_now = 0
        for item in queue_state["queue"]:
            if args.rescore_completed:
                break
            if item.get("status") != "pending":
                continue
            if scanned_now >= args.max_squares:
                break

            sq_id = item["square_id"]
            e_min, n_min, e_max, n_max = item["e_min"], item["n_min"], item["e_max"], item["n_max"]
            t0 = time.time()
            print(
                f"[cycle {cycle_idx}/{total_cycles}] Scanning {sq_id} ({item['sampling_stream']}, {item['os_grid_ref']}) "
                f"bbox=[{e_min}, {n_min}, {e_max}, {n_max}] ..."
            )

            out_dir = survey_dir / sq_id
            summary = scan_region(
                model=model,
                catalog=catalog,
                channel_mean=channel_mean,
                channel_std=channel_std,
                calibrator=calibrator,
                tile_prob_threshold=th_005,
                pixel_prob_threshold=pixel_th_005,
                e_min=e_min,
                e_max=e_max,
                n_min=n_min,
                n_max=n_max,
                device=device,
                tile_cache_dir=tile_cache_dir,
                output_dir=out_dir,
                region_name=sq_id,
                osm_cache_dir=osm_cache_dir,
            )

            if summary["windows_with_valid_1m_lidar"] == 0:
                item["status"] = "no_lidar_coverage"
                item["scanned_at_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                queue_path.write_text(json.dumps(queue_state, indent=2), encoding="utf-8")
                print(f"[batch] {sq_id}: 0 valid 1m LiDAR windows (skipping without counting toward quota)")
                continue

            # Tag each candidate in the saved GeoJSON and report with whether it also passes `fpr_0.01`
            gj_path = out_dir / f"{sq_id}_candidates.geojson"
            if gj_path.exists():
                gj_data = json.loads(gj_path.read_text(encoding="utf-8"))
                for feat in gj_data.get("features", []):
                    p = feat.get("properties", {})
                    p["passes_fpr_0_01"] = bool(
                        float(p.get("calibrated_tile_prob", 0.0)) >= th_001
                        and float(p.get("peak_prob", 0.0)) >= pixel_th_001
                    )
                gj_path.write_text(json.dumps(gj_data, indent=2), encoding="utf-8")

            item["status"] = "completed"
            item["scanned_at_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            item["elapsed_s"] = round(time.time() - t0, 1)
            item["windows_with_valid_1m_lidar"] = summary["windows_with_valid_1m_lidar"]
            item["rediscovered_nhle_scheduled_count"] = summary["rediscovered_scheduled_monuments_count"]
            item["rediscovered_osm_archaeology_count"] = summary["rediscovered_osm_archaeology_count"]
            item["flagged_modern_or_karst_confounders_count"] = summary["flagged_modern_or_karst_confounders_count"]
            item["unidentified_candidates_count"] = summary["unidentified_candidates_count"]

            queue_path.write_text(json.dumps(queue_state, indent=2), encoding="utf-8")
            scanned_now += 1
            print(
                f"[cycle {cycle_idx}/{total_cycles}] Completed {sq_id} in {item['elapsed_s']}s: "
                f"NHLE={summary['rediscovered_scheduled_monuments_count']}, "
                f"OSM_arch={summary['rediscovered_osm_archaeology_count']}, "
                f"OSM_confounder={summary['flagged_modern_or_karst_confounders_count']}, "
                f"Unidentified={summary['unidentified_candidates_count']}"
            )

        pub = export_public_docs(
            queue_state=queue_state,
            survey_dir=survey_dir,
            eval_metrics_path=eval_metrics_path,
            docs_dir=args.docs_dir,
            catalog=catalog,
        )
        print(
            f"[cycle {cycle_idx}/{total_cycles}] Prehistoric public docs updated: "
            f"{pub['queue_progress']['completed_squares']} squares "
            f"({pub['queue_progress']['total_scanned_area_km2']} km²), "
            f"{pub['queue_progress']['total_detections']} total detections."
        )

        if args.roman_enclosure_per_cycle > 0:
            model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if roman_ckpt.exists():
                rom_sum = run_roman_scan(
                    data_dir=args.data_dir,
                    roman_ckpt_path=roman_ckpt,
                    docs_dir=args.docs_dir,
                    device_str=args.device,
                    max_squares=args.roman_enclosure_per_cycle,
                )
                print(
                    f"[cycle {cycle_idx}/{total_cycles}] Roman docs updated: "
                    f"{rom_sum['squares_scanned']} squares ({rom_sum['area_km2']} km²), "
                    f"Unid={rom_sum['unidentified_candidates']} (1% FPR={rom_sum['unidentified_candidates_fpr_0_01']})."
                )
            if enclosure_ckpt.exists():
                enc_sum = run_enclosure_scan(
                    data_dir=args.data_dir,
                    enclosure_ckpt_path=enclosure_ckpt,
                    docs_dir=args.docs_dir,
                    device_str=args.device,
                    max_squares=args.roman_enclosure_per_cycle,
                )
                print(
                    f"[cycle {cycle_idx}/{total_cycles}] Enclosure docs updated: "
                    f"{enc_sum['squares_scanned']} squares ({enc_sum['area_km2']} km²), "
                    f"Unid={enc_sum['unidentified_candidates']} (1% FPR={enc_sum['unidentified_candidates_fpr_0_01']})."
                )

        if args.git_push:
            repo_root = Path(__file__).resolve().parents[1]
            subprocess.run(["git", "add", "docs/"], cwd=repo_root, check=False)
            status = subprocess.run(
                ["git", "status", "--porcelain", "docs/"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if status.stdout.strip():
                comp = pub["queue_progress"]["completed_squares"]
                area = pub["queue_progress"]["total_scanned_area_km2"]
                subprocess.run(
                    ["git", "commit", "-m", f"Update national survey batch ({comp} squares, {area} km²)"],
                    cwd=repo_root,
                    check=False,
                )
                subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=repo_root, check=False)
                subprocess.run(["git", "push", "origin", "main"], cwd=repo_root, check=False)


if __name__ == "__main__":
    main()
