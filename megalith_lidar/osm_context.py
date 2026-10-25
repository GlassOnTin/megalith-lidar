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
        "sand",
        "coastline",
        "mud",
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
        if natural in {"wood", "tree", "tree_row"}:
            rad = 24.0
        elif natural in {"dune", "sand", "beach", "shingle", "coastline", "mud"}:
            rad = 45.0
        else:
            rad = 30.0
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
                or tags.get("natural") in {"sinkhole", "bare_rock", "scree", "shingle", "beach", "dune", "sand", "mud", "wood", "water", "wetland"}
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

    def query_environmental_context(
        self,
        easting: float,
        northing: float,
        radius_m: float = 400.0,
        max_items: int = 8,
    ) -> List[Dict]:
        """
        Return nearby geomorphic, hydrological, extractive, military, or recreational
        context features within `radius_m` (up to `max_items` deduplicated features) so
        the academic context synthesis can evaluate landscape-level false positives
        (e.g., meandering rivers, coastal sand dunes, quarries, airfields, golf courses).
        """
        env_matches: List[Dict] = []
        skip_distant_minor = {
            "barrier=fence",
            "barrier=hedge",
            "barrier=wall",
            "barrier=ditch",
            "power=pole",
            "power=minor_line",
            "highway=service",
            "highway=residential",
            "highway=unclassified",
            "highway=track",
            "highway=bridleway",
            "highway=byway",
            "building=yes",
            "building=house",
            "building=detached",
            "building=shed",
            "building=garage",
            "natural=tree",
            "natural=tree_row",
        }
        for feat in self.features:
            if feat.category == "osm_archaeology":
                continue
            b_emin, b_nmin, b_emax, b_nmax = feat.bbox_27700
            if (
                easting < b_emin - radius_m
                or easting > b_emax + radius_m
                or northing < b_nmin - radius_m
                or northing > b_nmax + radius_m
            ):
                continue
            dist_m = _point_to_geometry_distance_m(
                easting, northing, feat.coords_27700, feat.is_closed_area
            )
            if dist_m > radius_m:
                continue
            if dist_m > 45.0 and feat.subtype in skip_distant_minor and not feat.name:
                continue
            env_matches.append(
                {
                    "osm_id": feat.osm_id,
                    "subtype": feat.subtype,
                    "name": feat.name,
                    "distance_m": round(float(dist_m), 1),
                    "inside_polygon": bool(feat.is_closed_area and dist_m == 0.0),
                }
            )
        env_matches.sort(key=lambda m: (m["distance_m"], 0 if m.get("name") else 1))
        deduped: List[Dict] = []
        seen_keys = set()
        for m in env_matches:
            key = (m["subtype"], m.get("name") or "")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(m)
            if len(deduped) >= max_items:
                break
        return deduped


def is_active_holocene_coastal_dune_belt(easting: float, northing: float) -> Optional[str]:
    """
    Return the named Holocene mobile coastal sand-dune system if `(easting, northing)` lies
    inside a rapidly shifting aeolian sand-dune / beach-barrier belt where unburied Bronze Age
    or Roman earthwork morphology cannot survive on the active surface.
    """
    # Sefton Coast mobile dune belt: Crosby -> Formby Point -> Freshfield -> Ainsdale -> Birkdale
    if 325000 <= easting <= 331800 and 403500 <= northing <= 420500:
        return "Sefton Coast (Formby / Ainsdale / Birkdale) Holocene mobile aeolian sand-dune belt"
    # North Devon Braunton Burrows & Saunton Sands mobile dune system
    if 243500 <= easting <= 247200 and 131500 <= northing <= 137500:
        return "Braunton Burrows (North Devon) Holocene mobile coastal sand-dune system"
    # Studland Peninsula mobile dune spit (Dorset)
    if 402500 <= easting <= 404500 and 82500 <= northing <= 86500:
        return "Studland Heath / South Haven Holocene coastal dune spit"
    # Holy Island / Ross Links / Bamburgh coastal dunes (Northumberland)
    if 408000 <= easting <= 415500 and 634000 <= northing <= 643500:
        return "Lindisfarne / Ross Links / Bamburgh Holocene coastal sand-dune system"
    return None


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
    is_roman = target_domain in ("roman", "enclosure")
    is_enclosure = target_domain == "enclosure"
    for cand in candidates:
        osm_arch = []
        osm_conf = []
        env_feats = []
        e_pt = float(cand["easting"])
        n_pt = float(cand["northing"])
        if osm_index is not None:
            res = osm_index.query_point(e_pt, n_pt)
            osm_arch = res["osm_archaeology_matches"]
            raw_conf = list(res["osm_confounder_matches"])
            env_feats = osm_index.query_environmental_context(e_pt, n_pt, radius_m=400.0)
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
        cand["nearby_environmental_features"] = env_feats

        dune_belt = is_active_holocene_coastal_dune_belt(e_pt, n_pt)
        near_coastal_dune_osm = next(
            (
                m
                for m in env_feats
                if m.get("subtype") in {"natural=dune", "natural=sand", "natural=beach", "natural=shingle", "natural=coastline"}
                and float(m.get("distance_m", 999.0)) <= 180.0
            ),
            None,
        )
        if dune_belt or near_coastal_dune_osm:
            dune_desc = dune_belt or (
                f"Coastal dune/beach margin ({near_coastal_dune_osm['subtype']} at {near_coastal_dune_osm['distance_m']:.0f}m)"
                if near_coastal_dune_osm
                else "Holocene coastal sand-dune system"
            )
            osm_conf.append(
                {
                    "osm_id": "geomorph/coastal_sand_dunes",
                    "subtype": f"holocene_coastal_sand_dune_or_blowout ({dune_desc})",
                    "name": f"Active Holocene coastal sand-dune blowout, parabolic dune ridge, or beach slack ({dune_desc})",
                    "wikidata": None,
                    "distance_m": float(near_coastal_dune_osm["distance_m"]) if near_coastal_dune_osm else 0.0,
                    "inside_polygon": dune_belt is not None,
                }
            )

        # Intrinsic 1m LiDAR physical confounder checks (when outside known monuments)
        ndsm_m = float(cand.get("est_surface_prominence_m", 0.0))
        rel_m = float(cand.get("est_local_relief_m", 0.5))
        max_pos_rel_m = float(cand.get("max_positive_relief_m", max(rel_m, 0.0)))
        centroid_rel_m = float(cand.get("centroid_local_relief_m", rel_m))
        patch_min_rel_m = float(cand.get("patch_min_relief_m", 0.0))
        slope_m = float(cand.get("mean_slope_tangent", 0.08))
        area_m2 = float(cand.get("area_m2", 400.0))
        elong_m = float(cand.get("elongation", 1.2))
        ridge_span_m = float(cand.get("ridge_continuation_span_m", 0.0))
        ridge_elong = float(cand.get("ridge_continuation_elongation", 1.0))
        channel_span_m = float(cand.get("adjacent_channel_span_m", 0.0))
        has_tin_facet = bool(cand.get("has_lidar_tin_facet", False))

        if is_enclosure:
            if area_m2 > 8500.0 or area_m2 < 300.0:
                osm_conf.append(
                    {
                        "osm_id": "lidar/whole_tile_blob",
                        "subtype": f"oversized_floodplain_or_moorland_blob (area={area_m2:.0f}m2)",
                        "name": "Oversized floodplain terrace or whole-tile texture blob",
                        "wikidata": None,
                        "distance_m": 0.0,
                        "inside_polygon": True,
                    }
                )
            elif (
                slope_m > 0.115
                or max_pos_rel_m > 1.90
                or abs(rel_m) > 1.90
                or patch_min_rel_m <= -0.60
                or elong_m > 1.55
            ):
                osm_conf.append(
                    {
                        "osm_id": "lidar/scarp_or_meander",
                        "subtype": f"river_meander_or_lynchet_scarp (relief={rel_m:+.2f}m, slope={slope_m:.3f}, elong={elong_m:.2f})",
                        "name": "River meander bank, quarry scarp, or linear field lynchet",
                        "wikidata": None,
                        "distance_m": 0.0,
                        "inside_polygon": True,
                    }
                )
            elif ndsm_m > 0.45:
                osm_conf.append(
                    {
                        "osm_id": "lidar/ndsm",
                        "subtype": f"canopy_or_structure (nDSM={ndsm_m:.2f}m)",
                        "name": "Tree canopy or standing structure",
                        "wikidata": None,
                        "distance_m": 0.0,
                        "inside_polygon": True,
                    }
                )
        elif not is_roman and ndsm_m >= max_canopy_ndsm_m:
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
        elif not is_enclosure and (rel_m >= 2.20 or (slope_m >= 0.165 and rel_m >= 1.65)):
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
        elif (
            not is_enclosure
            and area_m2 <= 14000.0
            and (
                (centroid_rel_m <= 0.12 and patch_min_rel_m <= -0.22 and (max_pos_rel_m - centroid_rel_m) >= 0.70)
                or cand.get("vlm_shape_verdict") == "OBVIOUS_MODERN_CONFOUNDER"
                or (
                    cand.get("vlm_shape_verdict") == "RECTILINEAR_OR_U_BUND"
                    and centroid_rel_m <= 0.25
                    and ((max_pos_rel_m - centroid_rel_m) >= 0.50 or patch_min_rel_m <= -0.25)
                )
                or (
                    max_pos_rel_m >= 0.80
                    and centroid_rel_m <= 0.35
                    and (max_pos_rel_m - centroid_rel_m) >= 0.65
                    and (
                        float(cand.get("soilmark_core_lum_diff", 0.0)) <= -0.038
                        or (ndsm_m >= 0.85 and float(cand.get("cropmark_ring_exg_diff", 0.0)) >= 0.04)
                        or (
                            str(cand.get("arch_context_source") or "") in {"qwen3.8_vlm_vision", "qwen3.8_llm"}
                            and any(
                                phrase in str(cand.get("arch_context_note") or "").lower()
                                for phrase in (
                                    "is a false positive",
                                    "likely a modern",
                                    "likely represents a modern",
                                    "confounding the model",
                                    "modern agricultural",
                                    "modern field boundary",
                                    "building platform",
                                    "modern utility",
                                    "modern pond",
                                    "farm pond",
                                    "slurry",
                                    "gravel pit",
                                )
                            )
                        )
                        or (
                            float(cand.get("pond_unet_tile_prob", 0.0)) >= 0.74
                            and float(cand.get("pond_unet_local_peak", 0.0)) >= 0.45
                        )
                    )
                )
                or (
                    ndsm_m < 1.5
                    and patch_min_rel_m <= -0.25
                    and float(cand.get("soilmark_core_lum_diff", 0.0)) <= -0.045
                    and float(cand.get("cropmark_ring_exg_diff", 0.0)) <= -0.045
                )
                or (
                    float(cand.get("pond_unet_tile_prob", 0.0)) >= 0.76
                    and float(cand.get("pond_unet_local_peak", 0.0)) >= 0.55
                )
                or (
                    float(cand.get("pond_unet_local_peak", 0.0)) >= 0.55
                    and (
                        (
                            float(cand.get("cropmark_ring_exg_diff", 0.0)) <= -0.020
                            and float(cand.get("soilmark_core_lum_diff", 0.0)) <= -0.018
                        )
                        or (
                            float(cand.get("adjacent_channel_span_m", 0.0)) >= 55.0
                            and float(cand.get("cropmark_ring_exg_diff", 0.0)) <= -0.020
                        )
                    )
                )
                or (
                    float(cand.get("pond_unet_local_peak", 0.0)) >= 0.75
                    and patch_min_rel_m <= -0.30
                )
                or (
                    float(cand.get("ring_ditch4_m", 0.0)) <= -0.045
                    and patch_min_rel_m <= -0.35
                    and slope_m >= 0.12
                    and float(cand.get("cropmark_ring_exg_diff", 0.0)) <= 0.0
                )
            )
        ):
            osm_conf.append(
                {
                    "osm_id": "lidar/slurry_lagoon_or_bund",
                    "subtype": (
                        f"embanked_slurry_lagoon_or_farm_reservoir "
                        f"(bank={max_pos_rel_m:+.2f}m, basin={centroid_rel_m:+.2f}m, area={area_m2:.0f}m2)"
                    ),
                    "name": "Embanked pond, slurry lagoon, gravel pit, or levelled court bund",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                }
            )
        elif not is_roman and (
            rel_m >= 1.85
            or (slope_m >= 0.145 and rel_m >= 1.50)
            or (max_pos_rel_m >= 0.95 and slope_m >= 0.105 and ndsm_m >= 0.20)
        ):
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
        elif not is_roman and (
            area_m2 <= 950.0
            and elong_m < 2.2
            and ridge_span_m >= 95.0
            and ridge_elong >= 4.0
            and channel_span_m >= 85.0
        ):
            osm_conf.append(
                {
                    "osm_id": "lidar/meander_scroll_bar",
                    "subtype": (
                        f"fluvial_meander_or_palaeochannel_scroll_bar "
                        f"(ridge={ridge_span_m:.0f}m, elong={ridge_elong:.1f}x, channel={channel_span_m:.0f}m)"
                    ),
                    "name": "Fluvial meander scroll-bar, levee ridge, or palaeochannel bank",
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
