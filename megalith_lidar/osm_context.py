from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pyproj import Transformer

from megalith_lidar.catalog import bng_to_latlon

_TRANSFORMER_4326_TO_27700 = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

OSM_ARCHAEOLOGY_HISTORIC_VALUES = frozenset(
    {
        "megalith",
        "archaeological_site",
        "tumulus",
        "barrow",
        "stone",
    }
)

OSM_CONFOUNDER_MAN_MADE_VALUES = frozenset(
    {
        "embankment",
        "dyke",
        "spoil_heap",
        "reservoir_covered",
        "bunker_silo",
        "adit",
        "mineshaft",
        "water_works",
        "wastewater_plant",
        "storage_tank",
        "works",
        "mast",
        "tower",
        "bridge",
        "pier",
        "breakwater",
        "groyne",
    }
)

OSM_CONFOUNDER_LANDUSE_VALUES = frozenset(
    {
        "quarry",
        "landfill",
        "reservoir",
        "basin",
        "industrial",
        "military",
        "recreation_ground",
        "commercial",
        "retail",
        "railway",
        "garages",
        "construction",
        "brownfield",
    }
)

OSM_CONFOUNDER_LEISURE_VALUES = frozenset(
    {
        "golf_course",
        "park",
        "garden",
        "pitch",
        "stadium",
        "track",
        "horse_riding",
        "marina",
        "playground",
    }
)

OSM_CONFOUNDER_GOLF_VALUES = frozenset(
    {
        "bunker",
        "green",
        "tee",
        "fairway",
        "rough",
        "driving_range",
    }
)

OSM_CONFOUNDER_NATURAL_VALUES = frozenset(
    {
        "sinkhole",
        "cliff",
        "scree",
        "bare_rock",
        "shingle",
        "beach",
        "dune",
    }
)

OSM_CONFOUNDER_WATERWAY_VALUES = frozenset(
    {
        "dam",
        "weir",
        "lock_gate",
        "canal",
        "ditch",
        "drain",
    }
)


def classify_osm_tags(tags: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Classify an OpenStreetMap tag dictionary into either:
      - ("osm_archaeology", subtype_description)
      - ("osm_confounder", subtype_description)
      - None (neutral background feature such as footpath or parish boundary)
    """
    if not tags:
        return None

    building = tags.get("building")
    historic = tags.get("historic", "")
    man_made = tags.get("man_made", "")
    landuse = tags.get("landuse", "")
    leisure = tags.get("leisure", "")
    golf = tags.get("golf", "")
    natural = tags.get("natural", "")
    waterway = tags.get("waterway", "")
    railway = tags.get("railway", "")

    # Standing buildings (including 18th-century garden temples/follies) are confounders, not earthworks
    if building and building != "no":
        if historic:
            return ("osm_confounder", f"building={building} (historic={historic})")
        return ("osm_confounder", f"building={building}")

    # Prehistoric / earthwork archaeology recorded in OSM
    if (
        historic in OSM_ARCHAEOLOGY_HISTORIC_VALUES
        or man_made == "cairn"
        or "megalith_type" in tags
        or "archaeological_site" in tags
        or tags.get("site_type") in {"megalith", "tumulus", "stone_circle", "cairn", "barrow", "henge"}
    ):
        label_parts = []
        if historic:
            label_parts.append(f"historic={historic}")
        if man_made == "cairn":
            label_parts.append("man_made=cairn")
        if "megalith_type" in tags:
            label_parts.append(f"megalith_type={tags['megalith_type']}")
        if "archaeological_site" in tags:
            label_parts.append(f"archaeological_site={tags['archaeological_site']}")
        if "site_type" in tags:
            label_parts.append(f"site_type={tags['site_type']}")
        return ("osm_archaeology", ", ".join(label_parts) or "osm_archaeology")

    # Modern anthropogenic or karst/geomorphological confounders
    if man_made in OSM_CONFOUNDER_MAN_MADE_VALUES:
        return ("osm_confounder", f"man_made={man_made}")
    if landuse in OSM_CONFOUNDER_LANDUSE_VALUES:
        return ("osm_confounder", f"landuse={landuse}")
    if golf in OSM_CONFOUNDER_GOLF_VALUES:
        return ("osm_confounder", f"golf={golf}")
    if leisure in OSM_CONFOUNDER_LEISURE_VALUES:
        return ("osm_confounder", f"leisure={leisure}")
    if natural in OSM_CONFOUNDER_NATURAL_VALUES:
        return ("osm_confounder", f"natural={natural}")
    if waterway in OSM_CONFOUNDER_WATERWAY_VALUES:
        return ("osm_confounder", f"waterway={waterway}")
    if railway in {"rail", "abandoned", "disused", "embankment", "cutting", "tram", "light_rail"}:
        return ("osm_confounder", f"railway={railway}")

    return None


def _point_in_polygon(e: float, n: float, ring: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon check in EPSG:27700 metres."""
    inside = False
    num = len(ring)
    if num < 3:
        return False
    j = num - 1
    for i in range(num):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > n) != (yj > n)) and (e < (xj - xi) * (n - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_segment_distance_m(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _point_to_geometry_distance_m(
    e: float,
    n: float,
    coords: Sequence[Tuple[float, float]],
    is_closed_area: bool,
) -> float:
    if not coords:
        return float("inf")
    if len(coords) == 1:
        return math.hypot(e - coords[0][0], n - coords[0][1])
    if is_closed_area and _point_in_polygon(e, n, coords):
        return 0.0
    min_d = float("inf")
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        d = _point_to_segment_distance_m(e, n, x1, y1, x2, y2)
        if d < min_d:
            min_d = d
    return min_d


@dataclass
class OSMParsedFeature:
    osm_id: str
    category: str  # "osm_archaeology" or "osm_confounder"
    subtype: str
    name: Optional[str]
    wikidata: Optional[str]
    coords_27700: List[Tuple[float, float]]
    is_closed_area: bool
    bbox_27700: Tuple[float, float, float, float]


class OSMSpatialIndex:
    """
    Spatial cross-check index built from the official OpenStreetMap API v0.6 bbox endpoint
    (`https://api.openstreetmap.org/api/0.6/map.json`).
    """

    def __init__(self, features: List[OSMParsedFeature]) -> None:
        self.features = features

    @classmethod
    def from_osm_json(cls, payload: Dict) -> "OSMSpatialIndex":
        elements = payload.get("elements", [])
        nodes_27700: Dict[int, Tuple[float, float]] = {}
        raw_nodes = []
        raw_ways: Dict[int, Dict] = {}
        raw_relations = []

        for el in elements:
            etype = el.get("type")
            if etype == "node" and "lat" in el and "lon" in el:
                raw_nodes.append(el)
            elif etype == "way":
                raw_ways[el["id"]] = el
            elif etype == "relation":
                raw_relations.append(el)

        if raw_nodes:
            lons = [float(el["lon"]) for el in raw_nodes]
            lats = [float(el["lat"]) for el in raw_nodes]
            es, ns = _TRANSFORMER_4326_TO_27700.transform(lons, lats)
            for el, e_val, n_val in zip(raw_nodes, es, ns):
                nodes_27700[int(el["id"])] = (float(e_val), float(n_val))

        parsed: List[OSMParsedFeature] = []

        # 1. Tagged nodes
        for el in raw_nodes:
            tags = el.get("tags")
            if not tags:
                continue
            classified = classify_osm_tags(tags)
            if classified is None:
                continue
            cat, subtype = classified
            pt = nodes_27700.get(int(el["id"]))
            if pt is None:
                continue
            e_val, n_val = pt
            parsed.append(
                OSMParsedFeature(
                    osm_id=f"node/{el['id']}",
                    category=cat,
                    subtype=subtype,
                    name=tags.get("name"),
                    wikidata=tags.get("wikidata"),
                    coords_27700=[pt],
                    is_closed_area=False,
                    bbox_27700=(e_val, n_val, e_val, n_val),
                )
            )

        # 2. Tagged ways
        for wid, el in raw_ways.items():
            tags = el.get("tags")
            if not tags:
                continue
            classified = classify_osm_tags(tags)
            if classified is None:
                continue
            cat, subtype = classified
            nd_ids = el.get("nodes", [])
            coords = [nodes_27700[nid] for nid in nd_ids if nid in nodes_27700]
            if not coords:
                continue
            is_closed = len(nd_ids) >= 4 and nd_ids[0] == nd_ids[-1]
            # Linear embankments/cliff lines are not filled area polygons unless tagged area=yes or landuse/leisure/building
            has_area_tag = (
                tags.get("area") == "yes"
                or "landuse" in tags
                or "leisure" in tags
                or "building" in tags
                or "golf" in tags
                or tags.get("natural") in {"sinkhole", "bare_rock", "scree", "shingle", "beach", "dune"}
                or cat == "osm_archaeology"
            )
            is_area = is_closed and has_area_tag
            es_w = [c[0] for c in coords]
            ns_w = [c[1] for c in coords]
            parsed.append(
                OSMParsedFeature(
                    osm_id=f"way/{wid}",
                    category=cat,
                    subtype=subtype,
                    name=tags.get("name"),
                    wikidata=tags.get("wikidata"),
                    coords_27700=coords,
                    is_closed_area=is_area,
                    bbox_27700=(min(es_w), min(ns_w), max(es_w), max(ns_w)),
                )
            )

        # 3. Tagged multipolygon/site relations whose member ways carry geometry
        for rel in raw_relations:
            tags = rel.get("tags")
            if not tags:
                continue
            classified = classify_osm_tags(tags)
            if classified is None:
                continue
            cat, subtype = classified
            for mem in rel.get("members", []):
                if mem.get("type") == "way" and mem.get("role", "outer") in {"outer", ""}:
                    way_el = raw_ways.get(mem.get("ref"))
                    if not way_el:
                        continue
                    nd_ids = way_el.get("nodes", [])
                    coords = [nodes_27700[nid] for nid in nd_ids if nid in nodes_27700]
                    if not coords:
                        continue
                    is_closed = len(nd_ids) >= 4 and nd_ids[0] == nd_ids[-1]
                    es_w = [c[0] for c in coords]
                    ns_w = [c[1] for c in coords]
                    parsed.append(
                        OSMParsedFeature(
                            osm_id=f"relation/{rel['id']}(way/{way_el['id']})",
                            category=cat,
                            subtype=subtype,
                            name=tags.get("name"),
                            wikidata=tags.get("wikidata"),
                            coords_27700=coords,
                            is_closed_area=is_closed,
                            bbox_27700=(min(es_w), min(ns_w), max(es_w), max(ns_w)),
                        )
                    )

        return cls(parsed)

    def query_point(
        self,
        easting: float,
        northing: float,
        archaeology_radius_m: float = 45.0,
        confounder_radius_m: float = 40.0,
    ) -> Dict[str, List[Dict]]:
        """
        Return all OSM recorded archaeological sites within `archaeology_radius_m` and
        all OSM modern/karst confounders within `confounder_radius_m` (or containing polygon).
        """
        arch_matches: List[Dict] = []
        conf_matches: List[Dict] = []

        for feat in self.features:
            max_r = archaeology_radius_m if feat.category == "osm_archaeology" else confounder_radius_m
            b_emin, b_nmin, b_emax, b_nmax = feat.bbox_27700
            if (
                easting < b_emin - max_r
                or easting > b_emax + max_r
                or northing < b_nmin - max_r
                or northing > b_nmax + max_r
            ):
                continue

            dist_m = _point_to_geometry_distance_m(
                easting, northing, feat.coords_27700, feat.is_closed_area
            )
            if dist_m <= max_r:
                entry = {
                    "osm_id": feat.osm_id,
                    "subtype": feat.subtype,
                    "name": feat.name,
                    "wikidata": feat.wikidata,
                    "distance_m": round(float(dist_m), 1),
                    "inside_polygon": bool(feat.is_closed_area and dist_m == 0.0),
                }
                if feat.category == "osm_archaeology":
                    arch_matches.append(entry)
                else:
                    conf_matches.append(entry)

        arch_matches.sort(key=lambda m: m["distance_m"])
        conf_matches.sort(key=lambda m: m["distance_m"])
        return {
            "osm_archaeology_matches": arch_matches,
            "osm_confounder_matches": conf_matches,
        }


def fetch_or_load_osm_context(
    e_min: int,
    n_min: int,
    e_max: int,
    n_max: int,
    osm_cache_dir: Path,
    margin_m: int = 80,
    timeout_s: float = 20.0,
) -> Optional[OSMSpatialIndex]:
    """
    Fetch or load the OpenStreetMap v0.6 map.json bounding box covering `[e_min, n_min, e_max, n_max]`
    plus `margin_m` padding so boundary candidates have full spatial context.
    """
    osm_cache_dir = Path(osm_cache_dir)
    osm_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = osm_cache_dir / f"osm_E{e_min}_N{n_min}_E{e_max}_N{n_max}.json"

    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return OSMSpatialIndex.from_osm_json(payload)
        except Exception:
            pass

    lat_sw, lon_sw = bng_to_latlon(e_min - margin_m, n_min - margin_m)
    lat_ne, lon_ne = bng_to_latlon(e_max + margin_m, n_max + margin_m)
    lat_nw, lon_nw = bng_to_latlon(e_min - margin_m, n_max + margin_m)
    lat_se, lon_se = bng_to_latlon(e_max + margin_m, n_min - margin_m)

    west = min(lon_sw, lon_nw)
    east = max(lon_ne, lon_se)
    south = min(lat_sw, lat_se)
    north = max(lat_ne, lat_nw)

    url = f"https://api.openstreetmap.org/api/0.6/map.json?bbox={west:.5f},{south:.5f},{east:.5f},{north:.5f}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "megalith-lidar-uk-research/1.0 (https://github.com/GlassOnTin/megalith-lidar)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_bytes = resp.read()
        payload = json.loads(raw_bytes.decode("utf-8"))
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
        return OSMSpatialIndex.from_osm_json(payload)
    except Exception:
        return None


def annotate_candidates_with_spatial_context(
    candidates: List[Dict],
    osm_index: Optional[OSMSpatialIndex],
) -> List[Dict]:
    """
    Annotate each candidate detection with OpenStreetMap archaeological and confounder context
    WITHOUT altering `confidence`, `peak_prob`, `mean_prob`, or `calibrated_tile_prob`.
    """
    for cand in candidates:
        osm_arch = []
        osm_conf = []
        if osm_index is not None:
            res = osm_index.query_point(cand["easting"], cand["northing"])
            osm_arch = res["osm_archaeology_matches"]
            osm_conf = res["osm_confounder_matches"]

        cand["osm_archaeology_matches"] = osm_arch
        cand["osm_confounder_matches"] = osm_conf

        if cand.get("is_known_monument"):
            cand["context_category"] = "rediscovered_nhle_scheduled"
        elif len(osm_arch) > 0:
            cand["context_category"] = "rediscovered_osm_archaeology"
        elif len(osm_conf) > 0:
            cand["context_category"] = "flagged_modern_or_karst_confounder"
        else:
            cand["context_category"] = "unidentified_candidate"

    return candidates
