from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

EA_DTM_WCS_URL = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-terrain-model-dtm-1m/wcs"
)
EA_DTM_COVERAGE_ID = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"

EA_DSM_WCS_URL = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-surface-model-last-return-dsm-1m/wcs"
)
EA_DSM_COVERAGE_ID = "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m"


def fetch_wcs_geotiff(
    e_min: int,
    e_max: int,
    n_min: int,
    n_max: int,
    dsm: bool = False,
    timeout: float = 25.0,
    retries: int = 5,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Download a 1m resolution float32 elevation tile from the Environment Agency WCS.
    Returns (array_or_None, status) where status is 'ok', 'invalid_response', or 'network_error'.
    """
    base_url = EA_DSM_WCS_URL if dsm else EA_DTM_WCS_URL
    cov_id = EA_DSM_COVERAGE_ID if dsm else EA_DTM_COVERAGE_ID
    url = (
        f"{base_url}?service=WCS&version=2.0.1&request=GetCoverage"
        f"&coverageId={cov_id}"
        f"&subset=E({int(e_min)},{int(e_max)})"
        f"&subset=N({int(n_min)},{int(n_max)})"
        f"&format=image/tiff"
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "megalith-lidar-uk/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                raw = resp.read()
                if "xml" in ctype.lower() or len(raw) < 512:
                    return None, "invalid_response"
                arr = np.array(Image.open(io.BytesIO(raw)), dtype=np.float32)
                # Replace extreme float32 nodata sentinels (e.g. -3.4e38) with NaN
                bad = ~np.isfinite(arr) | (arr < -100.0) | (arr > 2000.0)
                if np.any(bad):
                    arr = arr.copy()
                    arr[bad] = np.nan
                return arr, "ok"
        except Exception:
            if attempt == retries - 1:
                return None, "network_error"
            time.sleep(0.75 * (attempt + 1))
    return None, "network_error"


def is_valid_elevation_tile(arr: Optional[np.ndarray], min_valid_fraction: float = 0.85) -> bool:
    """Check that an elevation array has at least `min_valid_fraction` finite terrain pixels."""
    if arr is None or arr.ndim != 2 or arr.size == 0:
        return False
    valid = np.isfinite(arr) & (arr > -50.0) & (arr < 1500.0)
    if float(np.mean(valid)) < min_valid_fraction:
        return False
    vals = arr[valid]
    if float(np.std(vals)) < 0.02:
        return False
    return True


def fetch_or_load_paired_tile(
    e_min: int,
    e_max: int,
    n_min: int,
    n_max: int,
    cache_dir: Optional[Path] = None,
    min_valid_fraction: float = 0.85,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fetch or load from disk cache a paired (DTM, DSM) 1m tile for [e_min, e_max] x [n_min, n_max].
    Returns None if the location lacks valid 1m LiDAR coverage.
    Never caches transient network errors as empty tiles.
    """
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        sub = cache_dir / f"E{int(e_min) // 10000:02d}_N{int(n_min) // 10000:02d}"
        cache_file = sub / f"tile_E{int(e_min)}_{int(e_max)}_N{int(n_min)}_{int(n_max)}.npz"
        if cache_file.exists():
            try:
                with np.load(cache_file) as data:
                    if "empty" in data:
                        return None
                    dtm = data["dtm"].astype(np.float32)
                    dsm = data["dsm"].astype(np.float32)
                    if is_valid_elevation_tile(dtm, min_valid_fraction):
                        return dtm, dsm
                    return None
            except Exception:
                pass

    dtm, status = fetch_wcs_geotiff(e_min, e_max, n_min, n_max, dsm=False)
    if status == "network_error":
        # Do NOT poison disk cache on transient network errors
        return None
    if not is_valid_elevation_tile(dtm, min_valid_fraction):
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, empty=np.array([1], dtype=np.uint8))
        return None

    assert dtm is not None
    dsm, _ = fetch_wcs_geotiff(e_min, e_max, n_min, n_max, dsm=True)
    if dsm is None or dsm.shape != dtm.shape or not np.any(np.isfinite(dsm)):
        dsm = dtm.copy()

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            dtm=dtm.astype(np.float32),
            dsm=dsm.astype(np.float32),
        )
    return dtm, dsm
