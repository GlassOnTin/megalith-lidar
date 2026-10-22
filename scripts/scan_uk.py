#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from megalith_lidar.catalog import download_or_load_nhle_catalog
from megalith_lidar.dataset import assign_spatial_split
from megalith_lidar.model import MegalithUNet
from megalith_lidar.scanner import scan_region
from megalith_lidar.stats import PlattCalibrator


def pick_unbiased_test_squares(catalog, seed: int = 20261021):
    """
    Select two 2km x 2km (4.0 km²) squares strictly from 20km x 20km blocks assigned to 'test'
    by the SHA-256 block hash (zero manual region cherry-picking):
      1. 'random_null_test_square': A 2km x 2km square drawn at random from a 'test' block containing
         zero known prehistoric monuments (to measure empirical False Discovery Rate per km²).
      2. 'heldout_monument_test_square': The 2km x 2km square inside the hash-assigned 'test' split
         with the highest count of surveyed NHLE prehistoric monuments (for blind recall/localization testing).
    """
    rng = np.random.default_rng(seed)
    test_blocks = sorted(
        {
            (int(r.easting // 20000), int(r.northing // 20000))
            for r in catalog.records
            if assign_spatial_split(r.easting, r.northing) == "test"
        }
    )

    # 1. Pick a random 2km x 2km square in a 'test' block with zero prehistoric monuments
    null_square = None
    for _ in range(500):
        bx, by = test_blocks[int(rng.integers(0, len(test_blocks)))]
        e0 = int(bx * 20000 + rng.integers(0, 9) * 2000)
        n0 = int(by * 20000 + rng.integers(0, 9) * 2000)
        recs = catalog.query_bbox(e0, n0, e0 + 2000, n0 + 2000)
        if not any(r.is_megalithic_or_prehistoric for r in recs) and len(recs) >= 0:
            # Ensure block is inland by checking within 6km of some monument in catalog
            nearby_land = catalog.query_bbox(e0 - 5000, n0 - 5000, e0 + 7000, n0 + 7000)
            if nearby_land:
                null_square = (e0, e0 + 2000, n0, n0 + 2000)
                break

    # 2. Deterministically find the 2km x 2km cell inside the SHA-256 'test' blocks with the most
    # megalithic tombs / stone circles / henges (ties broken by total prehistoric monuments)
    from collections import Counter
    cell_scores = Counter()
    for r in catalog.records:
        if r.is_megalithic_or_prehistoric and assign_spatial_split(r.easting, r.northing) == "test":
            key = (int(r.easting // 2000) * 2000, int(r.northing // 2000) * 2000)
            weight = 3 if r.cls in (1, 2) else 1
            cell_scores[key] += weight

    (me0, mn0), _ = cell_scores.most_common(1)[0]
    monument_square = (me0, me0 + 2000, mn0, mn0 + 2000)
    return null_square, monument_square


def main() -> None:
    parser = argparse.ArgumentParser(description="Unbiased Statistically Calibrated UK LiDAR Search")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/bulk/megalith-lidar"))
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random_null_test_square", "heldout_monument_test_square", "custom"],
        default="heldout_monument_test_square",
    )
    parser.add_argument(
        "--operating-point",
        type=str,
        choices=["fpr_0.01", "fpr_0.05", "fpr_0.10", "max_youden_j"],
        default="fpr_0.05",
    )
    parser.add_argument("--e-min", type=int, default=None)
    parser.add_argument("--e-max", type=int, default=None)
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    catalog = download_or_load_nhle_catalog(data_dir / "catalog" / "nhle_scheduled_monuments_27700.json")
    ckpt_path = data_dir / "checkpoints" / "megalith_unet_best.pt"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MegalithUNet(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ch_mean = np.array(ckpt["channel_mean"], dtype=np.float32)
    ch_std = np.array(ckpt["channel_std"], dtype=np.float32)
    calibrator = PlattCalibrator.from_dict(ckpt["calibrator"])
    val_thresholds = ckpt["val_thresholds"]
    tile_th = float(val_thresholds[args.operating_point]["threshold"])
    pixel_th = float(ckpt.get("val_pixel_threshold_fpr05", 0.50))

    if args.mode == "custom" and args.e_min is not None:
        e_min, e_max, n_min, n_max = args.e_min, args.e_max, args.n_min, args.n_max
        region_name = args.name or f"custom_E{e_min}_N{n_min}_{args.operating_point}"
    else:
        null_sq, mon_sq = pick_unbiased_test_squares(catalog)
        if args.mode == "random_null_test_square":
            e_min, e_max, n_min, n_max = null_sq
            region_name = args.name or f"random_null_test_E{e_min}_N{n_min}_{args.operating_point}"
        else:
            e_min, e_max, n_min, n_max = mon_sq
            region_name = args.name or f"heldout_test_E{e_min}_N{n_min}_{args.operating_point}"

    print(
        f"Scanning region '{region_name}' [E:{e_min}..{e_max}, N:{n_min}..{n_max}] "
        f"at validation-calibrated operating point '{args.operating_point}' "
        f"(tile_prob_th={tile_th:.4f}, pixel_prob_th={pixel_th:.4f})..."
    )

    output_dir = data_dir / "scans_v2_unbiased" / region_name
    summary = scan_region(
        model=model,
        catalog=catalog,
        channel_mean=ch_mean,
        channel_std=ch_std,
        calibrator=calibrator,
        tile_prob_threshold=tile_th,
        pixel_prob_threshold=pixel_th,
        e_min=e_min,
        e_max=e_max,
        n_min=n_min,
        n_max=n_max,
        device=device,
        tile_cache_dir=data_dir / "tiles",
        output_dir=output_dir,
        region_name=region_name,
        stride=args.stride,
    )

    print(
        f"\nScan complete for '{region_name}' ({summary['area_km2']} km²): "
        f"{summary['windows_with_valid_1m_lidar']}/{summary['windows_requested']} valid 1m LiDAR windows "
        f"({summary['windows_above_tile_threshold']} windows passed tile threshold)."
    )
    print(f"  Rediscovered Known Scheduled Monuments: {summary['rediscovered_scheduled_monuments_count']}")
    print(
        f"  Unidentified Candidate Sites          : {summary['unidentified_candidates_count']} "
        f"({summary['unidentified_density_per_km2']} / km²)"
    )
    for i, c in enumerate(summary["unidentified_candidates"][:10], 1):
        print(
            f"  {i:2d}. {c['os_grid_ref']} (E={c['easting']}, N={c['northing']} | {c['lat']:.5f}, {c['lon']:.5f}) "
            f"| seg_score={c['confidence']:.3f} | tile_prob={c['calibrated_tile_prob']:.3f} "
            f"| class={c['predicted_class']} | area={c['area_m2']}m² | relief={c['est_local_relief_m']:+.2f}m"
        )


if __name__ == "__main__":
    main()
