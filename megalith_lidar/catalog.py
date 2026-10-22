from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

CLASS_BACKGROUND = 0
CLASS_MEGALITHIC_TOMB_LONG_BARROW = 1
CLASS_CIRCLE_HENGE_STONE = 2
CLASS_BARROW_CAIRN = 3
CLASS_OTHER_MONUMENT = 4

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_MEGALITHIC_TOMB_LONG_BARROW: "megalithic_tomb_long_barrow",
    CLASS_CIRCLE_HENGE_STONE: "stone_circle_henge_alignment",
    CLASS_BARROW_CAIRN: "round_barrow_cairn",
    CLASS_OTHER_MONUMENT: "other_scheduled_monument",
}

_TOMB_PAT = re.compile(
    r"\b("
    r"long\s+barrow|long\s+cairn|bank\s+barrow|chambered\s+tomb|chambered\s+barrow|"
    r"chambered\s+cairn|chambered\s+long|megalithic\s+tomb|megalithic\s+chamber|"
    r"dolmen|portal\s+dolmen|quoit|passage\s+grave|gallery\s+grave|cromlech|"
    r"medway\s+megalith|sarsen\s+chamber|entrance\s+grave|scillonian\s+entrance\s+grave"
    r")s?\b",
    re.IGNORECASE,
)

_CIRCLE_HENGE_STONE_PAT = re.compile(
    r"\b("
    r"stone\s+circle|timber\s+circle|henge|hengiform|cursus|causewayed\s+enclosure|"
    r"standing\s+stone|menhir|stone\s+row|stone\s+alignment|stone\s+avenue|"
    r"ceremonial\s+avenue|monolith|orthostat|sarsen\s+stone|four\s+poster\s+stone"
    r")s?\b",
    re.IGNORECASE,
)

_BARROW_CAIRN_PAT = re.compile(
    r"\b("
    r"bowl\s+barrow|bell\s+barrow|disc\s+barrow|saucer\s+barrow|pond\s+barrow|"
    r"round\s+barrow|barrow\s+cemetery|barrow|round\s+cairn|ring\s+cairn|tor\s+cairn|"
    r"platform\s+cairn|kerbed\s+cairn|clearance\s+cairn|cairnfield|cairn\s+circle|"
    r"cist|prehistoric\s+cairn|bronze\s+age\s+barrow|tumulus|tumuli"
    r")s?\b",
    re.IGNORECASE,
)

_EXCLUDE_FALSE_PAT = re.compile(
    r"\b("
    r"wayside\s+cross|churchyard\s+cross|market\s+cross|preaching\s+cross|"
    r"boundary\s+stone|milestone|parish\s+boundary|guide\s+post|pillbox|"
    r"anti-aircraft|World\s+War|pumping\s+engine|blast\s+furnace|colliery"
    r")\b",
    re.IGNORECASE,
)

_TRANSFORMER_27700_TO_4326 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

_OS_GRID_LETTERS = [
    ["SV", "SW", "SX", "SY", "SZ", "TV", "TW"],
    ["SQ", "SR", "SS", "ST", "SU", "TQ", "TR"],
    ["SL", "SM", "SN", "SO", "SP", "TL", "TM"],
    ["SF", "SG", "SH", "SJ", "SK", "TF", "TG"],
    ["SA", "SB", "SC", "SD", "SE", "TA", "TB"],
    ["NV", "NW", "NX", "NY", "NZ", "OV", "OW"],
    ["NQ", "NR", "NS", "NT", "NU", "OQ", "OR"],
    ["NL", "NM", "NN", "NO", "NP", "OL", "OM"],
    ["NF", "NG", "NH", "NJ", "NK", "OF", "OG"],
    ["NA", "NB", "NC", "ND", "NE", "OA", "OB"],
    ["HV", "HW", "HX", "HY", "HZ", "JV", "JW"],
    ["HQ", "HR", "HS", "HT", "HU", "JQ", "JR"],
    ["HL", "HM", "HN", "HO", "HP", "JL", "JM"],
]


def classify_monument_name(name: str) -> int:
    """
    Classify a Scheduled Monument title into prehistoric monument morphological classes
    using only generic archaeological type terms (zero site-specific proper noun lists).
    """
    if not name:
        return CLASS_OTHER_MONUMENT
    if _EXCLUDE_FALSE_PAT.search(name):
        return CLASS_OTHER_MONUMENT
    if _TOMB_PAT.search(name):
        return CLASS_MEGALITHIC_TOMB_LONG_BARROW
    if _CIRCLE_HENGE_STONE_PAT.search(name):
        return CLASS_CIRCLE_HENGE_STONE
    if _BARROW_CAIRN_PAT.search(name):
        return CLASS_BARROW_CAIRN
    return CLASS_OTHER_MONUMENT


def bng_to_latlon(easting: float, northing: float) -> Tuple[float, float]:
    """Convert British National Grid (EPSG:27700) Easting/Northing to WGS84 (lat, lon)."""
    lon, lat = _TRANSFORMER_27700_TO_4326.transform(easting, northing)
    return float(lat), float(lon)


def bng_to_os_grid_ref(easting: float, northing: float, digits: int = 10) -> str:
    """Convert British National Grid (EPSG:27700) Easting/Northing to OS Grid Reference."""
    e = int(round(easting))
    n = int(round(northing))
    if e < 0 or n < 0 or e >= 700000 or n >= 1300000:
        return f"E{e} N{n}"
    col = e // 100000
    row = n // 100000
    prefix = _OS_GRID_LETTERS[row][col]
    e_rem = e % 100000
    n_rem = n % 100000
    half = digits // 2
    scale = 10 ** (5 - half)
    return f"{prefix} {e_rem // scale:0{half}d} {n_rem // scale:0{half}d}"


@dataclass
class MonumentRecord:
    list_entry: int
    name: str
    easting: float
    northing: float
    area_ha: float
    ngr: str
    cls: int
    rings: List[List[List[float]]] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.rings:
            xs = [pt[0] for ring in self.rings for pt in ring]
            ys = [pt[1] for ring in self.rings for pt in ring]
            self.bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            r = max(5.0, math.sqrt(max(self.area_ha, 0.005) * 10000.0 / math.pi))
            self.bbox = (
                self.easting - r,
                self.northing - r,
                self.easting + r,
                self.northing + r,
            )

    @property
    def is_megalithic_or_prehistoric(self) -> bool:
        return self.cls in (
            CLASS_MEGALITHIC_TOMB_LONG_BARROW,
            CLASS_CIRCLE_HENGE_STONE,
            CLASS_BARROW_CAIRN,
        )


def _point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def distance_to_monument(x: float, y: float, rec: MonumentRecord) -> float:
    """Compute distance in metres from point (x, y) in EPSG:27700 to a MonumentRecord (0.0 if inside)."""
    min_x, min_y, max_x, max_y = rec.bbox
    # Fast bbox lower bound check
    dx_box = max(min_x - x, 0.0, x - max_x)
    dy_box = max(min_y - y, 0.0, y - max_y)
    if not rec.rings:
        return math.hypot(dx_box, dy_box)

    for ring in rec.rings:
        if len(ring) >= 3 and _point_in_ring(x, y, ring):
            return 0.0

    best = math.hypot(x - rec.easting, y - rec.northing)
    for ring in rec.rings:
        for i in range(len(ring) - 1):
            d = _point_to_segment_dist(x, y, ring[i][0], ring[i][1], ring[i + 1][0], ring[i + 1][1])
            if d < best:
                best = d
    return best


class MonumentCatalog:
    """Spatial index over Scheduled Monuments in EPSG:27700 with fast rasterization and lookup."""

    def __init__(self, records: Iterable[MonumentRecord], cell_size_m: float = 2000.0) -> None:
        self.records: List[MonumentRecord] = list(records)
        self.cell_size_m = cell_size_m
        self.grid: Dict[Tuple[int, int], List[MonumentRecord]] = defaultdict(list)
        for rec in self.records:
            min_x, min_y, max_x, max_y = rec.bbox
            gx0 = int(math.floor(min_x / cell_size_m))
            gx1 = int(math.floor(max_x / cell_size_m))
            gy0 = int(math.floor(min_y / cell_size_m))
            gy1 = int(math.floor(max_y / cell_size_m))
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    self.grid[(gx, gy)].append(rec)

    def query_bbox(self, e_min: float, n_min: float, e_max: float, n_max: float) -> List[MonumentRecord]:
        gx0 = int(math.floor(e_min / self.cell_size_m))
        gx1 = int(math.floor(e_max / self.cell_size_m))
        gy0 = int(math.floor(n_min / self.cell_size_m))
        gy1 = int(math.floor(n_max / self.cell_size_m))
        seen = set()
        out = []
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                for rec in self.grid.get((gx, gy), ()):
                    if id(rec) in seen:
                        continue
                    bx0, by0, bx1, by1 = rec.bbox
                    if bx1 >= e_min and bx0 <= e_max and by1 >= n_min and by0 <= n_max:
                        seen.add(id(rec))
                        out.append(rec)
        return out

    def find_nearby(
        self, easting: float, northing: float, max_distance_m: float = 50.0
    ) -> List[Tuple[MonumentRecord, float]]:
        candidates = self.query_bbox(
            easting - max_distance_m,
            northing - max_distance_m,
            easting + max_distance_m,
            northing + max_distance_m,
        )
        matches = []
        for rec in candidates:
            d = distance_to_monument(easting, northing, rec)
            if d <= max_distance_m:
                matches.append((rec, d))
        matches.sort(key=lambda pair: pair[1])
        return matches

    def rasterize_tile(
        self,
        e_min: float,
        e_max: float,
        n_min: float,
        n_max: float,
        size: int = 256,
    ) -> np.ndarray:
        """
        Rasterize all prehistoric/megalithic monument polygons intersecting [e_min, e_max] x [n_min, n_max]
        into a (size, size) uint8 class mask using their exact surveyed NHLE polygon geometry.
        """
        mask_im = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask_im)
        scale_x = size / (e_max - e_min)
        scale_y = size / (n_max - n_min)

        intersecting = self.query_bbox(e_min, n_min, e_max, n_max)
        intersecting.sort(key=lambda r: -r.cls)
        for rec in intersecting:
            if not rec.is_megalithic_or_prehistoric:
                continue
            cls_val = int(rec.cls)
            for ring in rec.rings:
                if len(ring) >= 3:
                    pts = [((pt[0] - e_min) * scale_x, (n_max - pt[1]) * scale_y) for pt in ring]
                    draw.polygon(pts, fill=cls_val, outline=cls_val)
        return np.array(mask_im, dtype=np.uint8)


NHLE_LAYER6_QUERY_URL = (
    "https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services/"
    "National_Heritage_List_for_England_NHLE_v02_VIEW/FeatureServer/6/query"
)


def _fetch_nhle_chunk(offset: int, batch_size: int = 2000) -> List[dict]:
    params = urllib.parse.urlencode(
        {
            "where": "1=1",
            "outFields": "OBJECTID,ListEntry,Name,Easting,Northing,area_ha,NGR",
            "returnGeometry": "true",
            "maxAllowableOffset": "1",
            "geometryPrecision": "1",
            "outSR": "27700",
            "orderByFields": "OBJECTID ASC",
            "resultOffset": str(offset),
            "resultRecordCount": str(batch_size),
            "f": "json",
        }
    )
    url = f"{NHLE_LAYER6_QUERY_URL}?{params}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "megalith-lidar-uk/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("features", [])
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.0 * (attempt + 1))
    return []


def download_or_load_nhle_catalog(cache_path: Path) -> MonumentCatalog:
    """Download all ~20,003 Historic England Scheduled Monuments with 1m polygons or load from cache."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        raw_list = json.loads(cache_path.read_text(encoding="utf-8"))
        records = [
            MonumentRecord(
                list_entry=int(r["list_entry"]),
                name=r["name"],
                easting=float(r["easting"]),
                northing=float(r["northing"]),
                area_ha=float(r.get("area_ha") or 0.01),
                ngr=r.get("ngr") or "",
                cls=classify_monument_name(r["name"]),
                rings=r.get("rings") or [],
            )
            for r in raw_list
        ]
        return MonumentCatalog(records)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Query total count first
    count_params = urllib.parse.urlencode({"where": "1=1", "returnCountOnly": "true", "f": "json"})
    req = urllib.request.Request(f"{NHLE_LAYER6_QUERY_URL}?{count_params}", headers={"User-Agent": "megalith-lidar-uk/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        total = int(json.loads(resp.read().decode("utf-8"))["count"])

    batch_size = 2000
    offsets = list(range(0, total, batch_size))
    features: List[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(_fetch_nhle_chunk, off, batch_size): off for off in offsets}
        for fut in as_completed(future_map):
            features.extend(fut.result())

    serialized = []
    records = []
    for feat in features:
        attr = feat.get("attributes") or {}
        geom = feat.get("geometry") or {}
        easting = attr.get("Easting")
        northing = attr.get("Northing")
        if easting is None or northing is None:
            continue
        name = (attr.get("Name") or "").strip()
        rings = geom.get("rings") or []
        item = {
            "list_entry": int(attr.get("ListEntry") or 0),
            "name": name,
            "easting": float(easting),
            "northing": float(northing),
            "area_ha": float(attr.get("area_ha") or 0.01),
            "ngr": (attr.get("NGR") or "").strip(),
            "rings": rings,
        }
        serialized.append(item)
        records.append(
            MonumentRecord(
                list_entry=item["list_entry"],
                name=item["name"],
                easting=item["easting"],
                northing=item["northing"],
                area_ha=item["area_ha"],
                ngr=item["ngr"],
                cls=classify_monument_name(name),
                rings=rings,
            )
        )

    cache_path.write_text(json.dumps(serialized), encoding="utf-8")
    return MonumentCatalog(records)
