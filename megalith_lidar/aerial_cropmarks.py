from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from megalith_lidar.catalog import bng_to_latlon

NUM_CROPMARK_CHANNELS = 2
CROPMARK_CHANNEL_NAMES = (
    "band_cropmark_exg_16m",
    "band_soilmark_lum_16m",
)

ARCGIS_WORLD_IMAGERY_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)
ARCGIS_WORLD_IMAGERY_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"
)


def _latlon_to_tile_xy(lat: float, lon: float, z: int) -> Tuple[float, float]:
    n = float(1 << z)
    xt = (lon + 180.0) / 360.0 * n
    lr = math.radians(lat)
    yt = (1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n
    return xt, yt


def fetch_or_load_aerial_tile(
    e_min: float,
    e_max: float,
    n_min: float,
    n_max: float,
    cache_dir: Optional[Path] = None,
    size: int = 256,
    timeout_s: float = 12.0,
    zoom: int = 17,
) -> Optional[np.ndarray]:
    """
    Fetch or load a pixel-aligned EPSG:27700 (British National Grid) RGB aerial orthophoto tile
    of shape `(size, size, 3)` (`uint8`). Uses the high-speed ArcGIS World Imagery CDN tile
    pyramid (`~130 tiles/s`) with exact 4-corner BNG-to-WebMercator bilinear resampling, falling
    back to the `/export` endpoint if needed.
    """
    ie0, ie1, in0, in1 = int(round(e_min)), int(round(e_max)), int(round(n_min)), int(round(n_max))
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"aerial_E{ie0}_N{in0}_E{ie1}_N{in1}_{size}.png"
        if cache_path.exists():
            try:
                im = Image.open(cache_path).convert("RGB")
                if im.size == (size, size):
                    return np.asarray(im, dtype=np.uint8)
            except Exception:
                pass

    headers = {
        "User-Agent": "megalith-lidar-uk-research/1.0 (https://github.com/GlassOnTin/megalith-lidar)",
    }
    try:
        lat_nw, lon_nw = bng_to_latlon(ie0, in1)
        lat_ne, lon_ne = bng_to_latlon(ie1, in1)
        lat_sw, lon_sw = bng_to_latlon(ie0, in0)
        lat_se, lon_se = bng_to_latlon(ie1, in0)

        x_nw, y_nw = _latlon_to_tile_xy(lat_nw, lon_nw, zoom)
        x_ne, y_ne = _latlon_to_tile_xy(lat_ne, lon_ne, zoom)
        x_sw, y_sw = _latlon_to_tile_xy(lat_sw, lon_sw, zoom)
        x_se, y_se = _latlon_to_tile_xy(lat_se, lon_se, zoom)

        tx0 = int(math.floor(min(x_nw, x_sw)))
        tx1 = int(math.floor(max(x_ne, x_se)))
        ty0 = int(math.floor(min(y_nw, y_ne)))
        ty1 = int(math.floor(max(y_sw, y_se)))

        mosaic_w = (tx1 - tx0 + 1) * 256
        mosaic_h = (ty1 - ty0 + 1) * 256
        mosaic = Image.new("RGB", (mosaic_w, mosaic_h))

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                t_url = f"{ARCGIS_WORLD_IMAGERY_TILE_URL}/{zoom}/{ty}/{tx}"
                req = urllib.request.Request(t_url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    im_t = Image.open(io.BytesIO(resp.read())).convert("RGB")
                mosaic.paste(im_t, ((tx - tx0) * 256, (ty - ty0) * 256))

        mos_arr = np.asarray(mosaic, dtype=np.float32)
        v = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
        u = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
        xt_grid = (1.0 - v) * ((1.0 - u) * x_nw + u * x_ne) + v * ((1.0 - u) * x_sw + u * x_se)
        yt_grid = (1.0 - v) * ((1.0 - u) * y_nw + u * y_ne) + v * ((1.0 - u) * y_sw + u * y_se)

        px_x = (xt_grid - tx0) * 256.0 - 0.5
        px_y = (yt_grid - ty0) * 256.0 - 0.5

        out = np.zeros((size, size, 3), dtype=np.uint8)
        for c in range(3):
            out[:, :, c] = np.clip(
                map_coordinates(mos_arr[:, :, c], [px_y, px_x], order=1, mode="nearest"),
                0.0,
                255.0,
            ).astype(np.uint8)
        if cache_path is not None:
            Image.fromarray(out).save(cache_path)
        return out
    except Exception:
        pass

    req_size = min(1024, max(size * 2, 512))
    url = (
        f"{ARCGIS_WORLD_IMAGERY_EXPORT_URL}"
        f"?bbox={ie0},{in0},{ie1},{in1}&bboxSR=27700&imageSR=27700"
        f"&size={req_size},{req_size}&format=png32&f=image"
    )
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_bytes = resp.read()
        im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        if cache_path is not None:
            im.save(cache_path)
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def compute_cropmark_features(
    rgb_u8: np.ndarray,
    dtm: Optional[np.ndarray] = None,
    dsm: Optional[np.ndarray] = None,
    max_bare_ground_ndsm_m: float = 0.50,
) -> np.ndarray:
    """
    Compute 2-channel isotropic scale-space bandpass cropmark & soilmark representation `(2, H, W)`:
      0. band_cropmark_exg_16m : (G_1.2(ExG) - G_16(ExG)) * I(nDSM < 0.5m)
         Isolates 2..32m local vegetation vigor anomalies (+ over buried humic ring-ditches,
         - over parched buried stone walls/banks) while removing whole-field crop color & season.
      1. band_soilmark_lum_16m : (G_1.2(L) - G_16(L)) * I(nDSM < 0.5m)
         Isolates 2..32m local bare-soil/plough-spread albedo anomalies (+ over chalk/gravel
         barrow mounds clipped by ploughing, - over dark organic ditch fills).

    Guarantees:
      - Exact Dihedral (D4) rotational & reflection equivariance
      - Invariant to whole-field illumination/exposure shifts: F(rgb + C) == F(rgb)
      - Gated to bare ground (nDSM < 0.50m) when DTM/DSM are provided so woodland canopy never
        leaks into cropmark channels.
    """
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    r = gaussian_filter(rgb[:, :, 0], sigma=1.2, mode="nearest")
    g = gaussian_filter(rgb[:, :, 1], sigma=1.2, mode="nearest")
    b = gaussian_filter(rgb[:, :, 2], sigma=1.2, mode="nearest")

    denom = r + g + b + 1e-4
    exg = (2.0 * g - r - b) / denom
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    dexg_16m = exg - gaussian_filter(exg, sigma=16.0, mode="nearest")
    dlum_16m = lum - gaussian_filter(lum, sigma=16.0, mode="nearest")

    if dtm is not None and dsm is not None:
        ndsm = np.maximum(0.0, np.asarray(dsm, dtype=np.float32) - np.asarray(dtm, dtype=np.float32))
        bare_gate = gaussian_filter((ndsm < max_bare_ground_ndsm_m).astype(np.float32), sigma=1.5, mode="nearest")
        dexg_16m = dexg_16m * bare_gate
        dlum_16m = dlum_16m * bare_gate

    return np.stack([dexg_16m, dlum_16m], axis=0).astype(np.float32)


def render_cropmark_false_color(
    rgb_u8: np.ndarray,
    dtm: Optional[np.ndarray] = None,
    dsm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render a false-color `(H, W, 3)` `uint8` diagnostic image from the 16m scale-space cropmark channels:
      - Green = +ΔExG_16m (moisture-retaining buried ring-ditch / pit cropmark)
      - Red/Magenta = +ΔL_16m / -ΔExG_16m (chalk/gravel soilmark or parched stone bank)
      - Blue = -ΔL_16m (dark damp organic soilmark)
    """
    feats = compute_cropmark_features(rgb_u8, dtm=dtm, dsm=dsm)
    dexg, dlum = feats[0], feats[1]

    s_exg = max(float(np.percentile(np.abs(dexg), 96)) * 1.4, 0.015)
    s_lum = max(float(np.percentile(np.abs(dlum), 96)) * 1.4, 0.025)

    def _to_u8(arr: np.ndarray, scale: float) -> np.ndarray:
        return np.clip((arr / scale * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)

    c_r = _to_u8(dlum - 0.40 * dexg, s_lum)
    c_g = _to_u8(dexg + 0.25 * dlum, s_exg)
    c_b = _to_u8(-dlum - 0.20 * dexg, s_lum)
    return np.stack([c_r, c_g, c_b], axis=-1)


def compute_candidate_cropmark_metrics(
    rgb_u8: np.ndarray,
    cx_px: float,
    cy_px: float,
    area_m2: float,
    dtm: Optional[np.ndarray] = None,
    dsm: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Measure concentric ring-ditch cropmark (`+ΔExG` annulus vs core) and central mound soilmark
    albedo contrast (`ΔL`) around a candidate centroid `(cx_px, cy_px)`.
    """
    feats = compute_cropmark_features(rgb_u8, dtm=dtm, dsm=dsm)
    dexg, dlum = feats[0], feats[1]
    h, w = dexg.shape

    r_eq = max(4.0, math.sqrt(max(area_m2, 16.0) / math.pi))
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - cx_px, yy - cy_px)

    core_mask = dist <= (0.65 * r_eq)
    ring_mask = (dist > (0.70 * r_eq)) & (dist <= (1.35 * r_eq))

    core_exg = float(np.mean(dexg[core_mask])) if np.any(core_mask) else 0.0
    ring_exg = float(np.mean(dexg[ring_mask])) if np.any(ring_mask) else 0.0
    core_lum = float(np.mean(dlum[core_mask])) if np.any(core_mask) else 0.0
    ring_lum = float(np.mean(dlum[ring_mask])) if np.any(ring_mask) else 0.0

    return {
        "cropmark_ring_exg_diff": round(ring_exg - core_exg, 4),
        "cropmark_ring_exg_mean": round(ring_exg, 4),
        "soilmark_core_lum_diff": round(core_lum - ring_lum, 4),
    }
