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
        "roman_road",
        "roman_villa",
        "fort",
        "camp",
        "castle",
        "earthworks",
    }
)

OSM_CONFOUNDER_MAN_MADE_VALUES = frozenset(
    {
        "embankment",
        "dyke",
        "spoil_heap",
        "reservoir_covered",
        "bunker_silo",
        "silage_clamp",
        "manure_heap",
        "adit",
        "mineshaft",
        "water_works",
        "wastewater_plant",
        "storage_tank",
        "silo",
        "works",
        "mast",
        "tower",
        "bridge",
        "pier",
        "breakwater",
        "groyne",
        "pipeline",
        "cutline",
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
        "farmyard",
        "allotments",
        "orchard",
        "cemetery",
        "churchyard",
        "forest",
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
        "wood",
        "tree",
        "tree_row",
        "water",
        "wetland",
    }
)

OSM_CONFOUNDER_WATERWAY_VALUES = frozenset(
    {
        "dam",
        "weir",
        "lock_gate",
        "canal",
        "river",
        "stream",
        "ditch",
        "drain",
    }
)

OSM_CONFOUNDER_HIGHWAY_VALUES = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "track",
        "bridleway",
        "byway",
        "road",
    }
)

OSM_CONFOUNDER_POWER_VALUES = frozenset(
    {
        "pole",
        "tower",
        "line",
        "minor_line",
        "substation",
        "plant",
        "generator",
    }
)

OSM_CONFOUNDER_BARRIER_VALUES = frozenset(
    {
        "hedge",
        "fence",
        "wall",
        "ditch",
        "retaining_wall",
        "embankment",
        "city_wall",
    }
)


def classify_osm_tags_with_radius(tags: Dict[str, str]) -> Optional[Tuple[str, str, float]]:
    """
    Classify an OpenStreetMap tag dictionary into:
      - ("osm_archaeology", subtype_description, match_radius_m)
      - ("osm_confounder", subtype_description, match_radius_m)
      - None
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
    highway = tags.get("highway", "")
    power = tags.get("power", "")
    barrier = tags.get("barrier", "")
    military = tags.get("military", "")

    # Standing buildings (including 18th-century garden temples/follies) are confounders, not earthworks
    if building and building != "no":
        if historic:
            return ("osm_confounder", f"building={building} (historic={historic})", 35.0)
        return ("osm_confounder", f"building={building}", 35.0)

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
        return ("osm_archaeology", ", ".join(label_parts) or "osm_archaeology", 45.0)

    # Modern anthropogenic, agricultural, utility, forestry, or karst confounders
    if military and military != "no":
        return ("osm_confounder", f"military={military}", 35.0)
    if man_made in OSM_CONFOUNDER_MAN_MADE_VALUES:
        return ("osm_confounder", f"man_made={man_made}", 35.0)
    if landuse in OSM_CONFOUNDER_LANDUSE_VALUES:
        rad = 24.0 if landuse == "forest" else 30.0
        return ("osm_confounder", f"landuse={landuse}", rad)
    if golf in OSM_CONFOUNDER_GOLF_VALUES:
        return ("osm_confounder", f"golf={golf}", 35.0)
    if leisure in OSM_CONFOUNDER_LEISURE_VALUES:
        return ("osm_confounder", f"leisure={leisure}", 35.0)
    if natural in OSM_CONFOUNDER_NATURAL_VALUES:
        rad = 24.0 if natural in {"wood", "tree", "tree_row"} else 30.0
        return ("osm_confounder", f"natural={natural}", rad)
    if waterway in OSM_CONFOUNDER_WATERWAY_VALUES:
        rad = 18.0 if waterway in {"stream", "ditch", "drain"} else 28.0
        return ("osm_confounder", f"waterway={waterway}", rad)
    if railway in {"rail", "abandoned", "disused", "embankment", "cutting", "tram", "light_rail"}:
        return ("osm_confounder", f"railway={railway}", 30.0)
    if power in OSM_CONFOUNDER_POWER_VALUES:
        return ("osm_confounder", f"power={power}", 18.0)
    if highway in OSM_CONFOUNDER_HIGHWAY_VALUES:
        return ("osm_confounder", f"highway={highway}", 35.0)
    if barrier in OSM_CONFOUNDER_BARRIER_VALUES:
        return ("osm_confounder", f"barrier={barrier}", 15.0)

    return None


def classify_osm_tags(tags: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Backwards-compatible 2-tuple wrapper around `classify_osm_tags_with_radius`."""
    res = classify_osm_tags_with_radius(tags)
    if res is None:
        return None
    return (res[0], res[1])


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
    match_radius_m: float = 35.0


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
            classified = classify_osm_tags_with_radius(tags)
            if classified is None:
                continue
            cat, subtype, rad = classified
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
                    match_radius_m=rad,
                )
            )

        # 2. Tagged ways
        for wid, el in raw_ways.items():
            tags = el.get("tags")
            if not tags:
                continue
            classified = classify_osm_tags_with_radius(tags)
            if classified is None:
                continue
            cat, subtype, rad = classified
            nd_ids = el.get("nodes", [])
            coords = [nodes_27700[nid] for nid in nd_ids if nid in nodes_27700]
            if not coords:
                continue
            is_closed = len(nd_ids) >= 4 and nd_ids[0] == nd_ids[-1]
            has_area_tag = (
                tags.get("area") == "yes"
                or "landuse" in tags
                or "leisure" in tags
                or "building" in tags
                or "golf" in tags
                or "military" in tags
                or tags.get("natural") in {"sinkhole", "bare_rock", "scree", "shingle", "beach", "dune", "wood", "water", "wetland"}
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
                    match_radius_m=rad,
                )
            )

        # 3. Tagged multipolygon/site relations whose member ways carry geometry
        for rel in raw_relations:
            tags = rel.get("tags")
            if not tags:
                continue
            classified = classify_osm_tags_with_radius(tags)
            if classified is None:
                continue
            cat, subtype, rad = classified
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
                            match_radius_m=rad,
                        )
                    )

        return cls(parsed)

    def query_point(
        self,
        easting: float,
        northing: float,
        archaeology_radius_m: float = 45.0,
        confounder_radius_m: Optional[float] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Return all OSM recorded archaeological sites within `archaeology_radius_m` and
        all OSM confounders within their feature-specific `match_radius_m` (or containing polygon).
        """
        arch_matches: List[Dict] = []
        conf_matches: List[Dict] = []

        for feat in self.features:
            if feat.category == "osm_archaeology":
                max_r = archaeology_radius_m
            else:
                max_r = confounder_radius_m if confounder_radius_m is not None else feat.match_radius_m

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
    max_canopy_ndsm_m: float = 2.8,
    min_positive_relief_m: float = 0.25,
    target_domain: str = "prehistoric",
) -> List[Dict]:
    """
    Annotate each candidate detection with OpenStreetMap archaeological and confounder context
    plus intrinsic 1m LiDAR physical confounder checks (tree canopy / standing structure nDSM,
    sunken marl/chalk pits, or LiDAR void/TIN facets) WITHOUT altering `confidence`,
    `peak_prob`, `mean_prob`, or `calibrated_tile_prob`.
    """
    is_roman = target_domain == "roman"
    for cand in candidates:
        osm_arch = []
        osm_conf = []
        if osm_index is not None:
            res = osm_index.query_point(cand["easting"], cand["northing"])
            osm_arch = res["osm_archaeology_matches"]
            raw_conf = list(res["osm_confounder_matches"])
            if is_roman:
                # 14-21% of scheduled Roman villas, forts, and Roman road aggers sit inside woodland
                # (landuse=forest / natural=wood) where 1m bare-earth LiDAR sees through the canopy.
                # Keep modern roads, railways, buildings, quarries, reservoirs, and golf courses as
                # confounders, while allowing woodland-covered Roman earthworks through.
                osm_conf = [
                    m
                    for m in raw_conf
                    if m.get("subtype") not in {"landuse=forest", "natural=wood", "natural=tree", "natural=tree_row"}
                ]
            else:
                osm_conf = raw_conf

        # Intrinsic 1m LiDAR physical confounder checks (when outside known monuments)
        ndsm_m = float(cand.get("est_surface_prominence_m", 0.0))
        rel_m = float(cand.get("est_local_relief_m", 0.5))
        max_pos_rel_m = float(cand.get("max_positive_relief_m", max(rel_m, 0.0)))
        centroid_rel_m = float(cand.get("centroid_local_relief_m", rel_m))
        patch_min_rel_m = float(cand.get("patch_min_relief_m", 0.0))
        slope_m = float(cand.get("mean_slope_tangent", 0.08))
        has_tin_facet = bool(cand.get("has_lidar_tin_facet", False))

        if not is_roman and ndsm_m >= max_canopy_ndsm_m:
            osm_conf.append(
                {
                    "osm_id": "lidar/ndsm",
                    "subtype": f"canopy_or_structure (nDSM={ndsm_m:.1f}m)",
                    "name": "Tall tree canopy or standing structure",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )
        elif not is_roman and (ndsm_m >= 0.75 or (ndsm_m >= 0.50 and slope_m >= 0.075)):
            osm_conf.append(
                {
                    "osm_id": "lidar/clamp_or_scrub",
                    "subtype": f"surface_vegetation_or_clamp (nDSM={ndsm_m:.2f}m, slope={slope_m:.2f})",
                    "name": "Surface vegetation scrub, felled timber, or agricultural clamp",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )
        elif rel_m >= 2.20 or (slope_m >= 0.165 and rel_m >= 1.65):
            osm_conf.append(
                {
                    "osm_id": "lidar/reservoir_or_bund",
                    "subtype": f"covered_reservoir_or_steep_bund (relief={rel_m:+.2f}m, slope={slope_m:.2f})",
                    "name": "Covered reservoir, steep embankment, or modern spoil bund",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )
        elif not is_roman and (rel_m >= 2.00 or (slope_m >= 0.145 and rel_m >= 1.50)):
            osm_conf.append(
                {
                    "osm_id": "lidar/reservoir_or_bund",
                    "subtype": f"covered_reservoir_or_steep_bund (relief={rel_m:+.2f}m, slope={slope_m:.2f})",
                    "name": "Covered reservoir, steep embankment, or modern spoil bund",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )
        elif not is_roman and (patch_min_rel_m <= -0.35 and centroid_rel_m < 0.25):
            osm_conf.append(
                {
                    "osm_id": "lidar/cut_fill_bund",
                    "subtype": f"cut_and_fill_u_bund_or_pit (bank={rel_m:+.2f}m, pit={patch_min_rel_m:+.2f}m, center={centroid_rel_m:+.2f}m)",
                    "name": "Cut-and-fill U-bund, military gun-pit, or quarry scoop embankment",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )
        if rel_m < -0.15 or max_pos_rel_m < (0.20 if is_roman else min_positive_relief_m):
            osm_conf.append(
                {
                    "osm_id": "lidar/pit",
                    "subtype": f"sunken_pit_or_pond (relief={rel_m:+.2f}m)",
                    "name": "Negative-relief pit, pond, or marl scoop",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )
        if has_tin_facet:
            osm_conf.append(
                {
                    "osm_id": "lidar/void",
                    "subtype": "lidar_coverage_edge_or_tin_facet",
                    "name": "LiDAR coverage boundary or interpolated TIN facet",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )

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
