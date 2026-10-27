from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from megalith_lidar.catalog import (
    CLASS_BARROW_CAIRN,
    CLASS_CIRCLE_HENGE_STONE,
    CLASS_MEGALITHIC_TOMB_LONG_BARROW,
    MonumentCatalog,
)
from megalith_lidar.osm_context import is_active_holocene_coastal_dune_belt

COMPASS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]

NHLE_CLASS_LABELS = {
    CLASS_MEGALITHIC_TOMB_LONG_BARROW: "Neolithic long barrow / chambered tomb",
    CLASS_CIRCLE_HENGE_STONE: "Henge / stone circle / ceremonial alignment",
    CLASS_BARROW_CAIRN: "Bronze Age round barrow / cairn",
    4: "Scheduled enclosure / settlement / earthwork",
}


def compass_bearing_from_Site(cand_e: float, cand_n: float, mon_e: float, mon_n: float) -> str:
    """Return 16-wind compass direction from candidate (cand_e, cand_n) to monument (mon_e, mon_n)."""
    dx = mon_e - cand_e
    dy = mon_n - cand_n
    deg = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    idx = int(round(deg / 22.5)) % 16
    return COMPASS_16[idx]


def infer_landscape_character_area(e: float, n: float) -> str:
    """Return a concise British landscape/physiographic descriptor from EPSG:27700 coordinates."""
    if 325000 <= e <= 334500 and 403000 <= n <= 421000:
        return "Sefton Coast — Formby, Ainsdale & Birkdale Holocene aeolian sand-dune belt (Merseyside)"
    if 243000 <= e <= 249000 and 131000 <= n <= 138000:
        return "North Devon Braunton Burrows & Taw–Torridge estuarine dune belt (Devon)"
    if 575000 <= e <= 612000 and 342000 <= n <= 347000:
        return "North Norfolk Holocene barrier coast, mobile dunes & saltmarsh (Norfolk)"
    if 570000 <= e <= 605000 and 280000 <= n <= 318000:
        return "Breckland & West Norfolk periglacial coversand & pingo-scar landscape (Norfolk / Suffolk)"
    if 510000 <= e <= 555000 and 280000 <= n <= 365000:
        return "Fenland & Lincolnshire Marsh reclaimed marine-silt lowland — roddons & salterns"
    if 335000 <= e <= 385000 and 320000 <= n <= 385000:
        return "Cheshire–Shropshire meres-and-mosses kettle-hole glacial till plain"
    if 360000 <= e <= 410000 and 455000 <= n <= 500000:
        return "Craven & Yorkshire Dales Carboniferous limestone karst & drumlin belt (North Yorkshire / Lancashire)"
    if 405000 <= e <= 432000 and 350000 <= n <= 388000:
        return "Peak District Carboniferous limestone plateau & lead-mining rake belt (Derbyshire / Staffordshire)"
    if 485000 <= e <= 515000 and 425000 <= n <= 475000:
        return "Yorkshire Wolds Upper Chalk dip-slope & dry-valley system (East Riding of Yorkshire)"
    if 450000 <= e <= 515000 and 480000 <= n <= 520000:
        return "North York Moors Jurassic sandstone moorland & Tabular Hills (North Yorkshire)"
    if 515000 <= e <= 585000 and 115000 <= n <= 160000:
        return "High & Low Weald clay-and-sandstone ironworking landscape (Sussex / Kent)"
    if 410000 <= e <= 445000 and 95000 <= n <= 120000:
        return "New Forest & Hampshire Basin Tertiary gravel plateau & heathland (Hampshire)"
    if 415000 <= e <= 440000 and 475000 <= n <= 505000:
        return "Ure Valley & Mowbray / Masham fluvio-glacial terrace (North Yorkshire)"
    if 400000 <= e <= 435000 and 210000 <= n <= 245000:
        return "Cotswold oolitic limestone dip-slope (Gloucestershire / West Oxfordshire)"
    if 395000 <= e <= 430000 and 115000 <= n <= 155000:
        return "Wessex Chalk — Salisbury Plain, Wylye & Bourne valleys (Wiltshire)"
    if 475000 <= e <= 520000 and 100000 <= n <= 125000:
        return "South Downs Upper Chalk escarpment & dip-slope (West Sussex)"
    if 195000 <= e <= 230000 and 70000 <= n <= 105000:
        return "Bodmin Moor granite upland & moorland fringe (Cornwall)"
    if 250000 <= e <= 285000 and 60000 <= n <= 98000:
        return "Dartmoor granite upland & moorland fringe (Devon)"
    if 340000 <= e <= 375000 and 140000 <= n <= 165000:
        return "Mendip Carboniferous limestone plateau (Somerset)"
    if 270000 <= e <= 305000 and 130000 <= n <= 155000:
        return "Exmoor Devonian sandstone moorland (Somerset / North Devon)"
    if 335000 <= e <= 365000 and 505000 <= n <= 545000:
        return "Eden Valley & North Cumbrian upland fringe (Cumbria)"
    if 385000 <= e <= 425000 and 500000 <= n <= 640000:
        return "North Pennine & Northumberland sandstone/whin upland"
    if n >= 450000:
        return "Northern England upland / valley terrace landscape"
    if n <= 160000 and e <= 360000:
        return "South-West Peninsula upland / valley landscape"
    if n <= 190000 and e > 360000:
        return "Southern English Chalk & lowlands"
    return "English lowland / river terrace landscape"


def derive_false_positive_context_hints(
    cand: Dict,
    nearby_nhle: List[Dict],
    cluster: Dict,
) -> List[str]:
    """
    Synthesize deeper geomorphic, geological, hydrological, and land-use false-positive
    considerations from the candidate's coordinates, 1 m LiDAR observables, nearby OSM
    environmental features (within 400 m), and nearby NHLE monument types.
    """
    hints: List[str] = []
    e = float(cand.get("easting", 0.0))
    n = float(cand.get("northing", 0.0))
    landscape = str(cluster.get("landscape_area") or infer_landscape_character_area(e, n))
    land_low = landscape.lower()
    env_feats = list(cand.get("nearby_environmental_features") or [])
    confs = list(cand.get("osm_confounder_matches") or [])
    all_osm = env_feats + [c for c in confs if str(c.get("osm_id", "")).startswith(("node/", "way/", "relation/"))]

    ridge_span = float(cand.get("ridge_continuation_span_m") or 0.0)
    channel_span = float(cand.get("adjacent_channel_span_m") or 0.0)
    patch_min_rel = float(cand.get("patch_min_relief_m") or 0.0)
    ndsm = float(cand.get("est_surface_prominence_m") or 0.0)

    def _fmt_osm(m: Dict) -> str:
        sub = m.get("subtype", "feature")
        nm = f" '{m['name']}'" if m.get("name") else ""
        dist = float(m.get("distance_m", 0.0))
        return f"{sub}{nm} ({dist:.0f} m)" if dist > 0 else f"{sub}{nm} (on-site)"

    # 1. Aeolian / Coastal Sand Dunes (e.g. Ainsdale/Formby, Braunton Burrows, coastal spits)
    dune_belt = is_active_holocene_coastal_dune_belt(e, n)
    coastal_osm = [
        m
        for m in all_osm
        if any(
            k in str(m.get("subtype", ""))
            for k in ("natural=dune", "natural=sand", "natural=beach", "natural=shingle", "natural=coastline", "natural=mud")
        )
    ]
    if dune_belt or "sand-dune" in land_low or "dune belt" in land_low or "mobile dunes" in land_low or coastal_osm:
        coastal_ref = _fmt_osm(coastal_osm[0]) if coastal_osm else (dune_belt or landscape)
        hints.append(
            f"Aeolian/coastal sand-dune taphonomy ({coastal_ref}): in rapidly shifting Holocene blown-sand and parabolic dune/slack systems, "
            f"unburied Bronze Age or Roman earthworks cannot survive on the mobile surface unless sealed beneath consolidated peat/intertidal silt beds; "
            f"evaluate active blowout rims, parabolic dune noses, Marram-grass hummocks, golf links bunkers, or WWII coastal/airfield defences."
        )

    # 2. Meandering Rivers, Floodplain Scroll-Bars & Palaeochannels
    water_osm = [
        m
        for m in all_osm
        if any(
            k in str(m.get("subtype", ""))
            for k in ("waterway=river", "waterway=stream", "waterway=canal", "waterway=drain", "waterway=ditch", "natural=wetland", "natural=water")
        )
    ]
    if water_osm or ridge_span >= 35.0 or channel_span >= 28.0:
        w_parts = []
        if water_osm:
            w_parts.append(_fmt_osm(water_osm[0]))
        if ridge_span >= 35.0 or channel_span >= 28.0:
            w_parts.append(f"1 m LiDAR ridge continuation {ridge_span:.0f} m / adjacent channel {channel_span:.0f} m")
        w_str = ", ".join(w_parts)
        hints.append(
            f"Fluvial/alluvial geomorphology ({w_str}): carefully test whether the arcuate bank, ditch, or mound is a meandering river point-bar "
            f"scroll ridge, oxbow cut-off bank, crevasse splay, flood levee, or abandoned palaeochannel margin rather than a built monument."
        )

    # 3. Fenland / Marine Silt Roddons & Salterns
    if "fenland" in land_low or "roddon" in land_low or "saltmarsh" in land_low:
        hints.append(
            "Fenland/estuarine silt geomorphology: raised sinuous marine-silt roddons (differential peat wastage along extinct tidal creeks) "
            "and Iron Age/Roman/medieval briquetage saltern mounds closely mimic barrows, enclosures, and Roman road aggers."
        )

    # 4. Periglacial Pingos, Kettle-Holes & Glacial Drumlin/Kame Hummocks
    if (
        "pingo" in land_low
        or "kettle-hole" in land_low
        or "drumlin" in land_low
        or "fluvio-glacial" in land_low
        or ("eden valley" in land_low and patch_min_rel <= -0.18)
    ):
        hints.append(
            "Glacial/periglacial geomorphology: Devensian periglacial pingo rampart scars, kettle-hole rims, and fluvio-glacial kame or drumlin "
            "hummocks produce circular embanked basins and smooth oval mounds that require BGS superficial-deposit cross-checking."
        )

    # 5. Karst Solution Dolines, Shakeholes & Field Limekilns
    karst_osm = [
        m
        for m in all_osm
        if any(k in str(m.get("subtype", "")) for k in ("natural=sinkhole", "natural=bare_rock", "natural=scree", "natural=cliff"))
    ]
    if karst_osm or (
        any(k in land_low for k in ("karst", "limestone", "chalk", "wolds", "mendip", "cotswold"))
        and patch_min_rel <= -0.18
    ):
        k_ref = f" ({_fmt_osm(karst_osm[0])})" if karst_osm else ""
        hints.append(
            f"Karst/carbonate dissolution & lime-burning{k_ref}: check for solution dolines, shakeholes above limestone/chalk joints, "
            f"periglacial coombe-rock slumps, and post-medieval field limekilns with annular quarry-spoil collars."
        )

    # 6. Extractive Bell-Pits, Lead Rakes, Tin Streamworks & Marl/Gravel Quarries
    mining_osm = [
        m
        for m in all_osm
        if any(
            k in str(m.get("subtype", ""))
            for k in ("landuse=quarry", "man_made=mineshaft", "man_made=adit", "man_made=spoil_heap", "landuse=landfill")
        )
    ]
    mining_nhle = [
        m
        for m in nearby_nhle
        if any(
            w in str(m.get("name", "")).lower()
            for w in ("mine", "colliery", "bell pit", "lead", "tin", "streamwork", "quarry", "kiln", "smelt", "ironworks", "coal")
        )
    ]
    if mining_osm or mining_nhle or any(k in land_low for k in ("lead-mining", "ironworking", "bodmin", "dartmoor")):
        m_ref = (
            _fmt_osm(mining_osm[0])
            if mining_osm
            else (f"nearby NHLE '{mining_nhle[0]['name']}'" if mining_nhle else landscape)
        )
        hints.append(
            f"Extractive/industrial mining heritage ({m_ref}): coal or ironstone bell-pits (doughnut spoil rings around a shaft hollow), "
            f"lead-mining hillocks, tin-streaming waste mounds, and post-medieval marl/gravel pits frequently mimic ring-barrows and hut circles."
        )

    # 7. Military Airfields/Training Areas, Golf Courses & 18th/19th-Century Parkland Tree-Rings
    mil_rec_osm = [
        m
        for m in all_osm
        if any(
            k in str(m.get("subtype", ""))
            for k in ("military=", "landuse=military", "leisure=golf_course", "golf=", "leisure=park", "leisure=garden", "leisure=pitch", "leisure=track")
        )
    ]
    if mil_rec_osm or "salisbury plain" in land_low:
        mr_ref = _fmt_osm(mil_rec_osm[0]) if mil_rec_osm else "Salisbury Plain military training area"
        hints.append(
            f"Military/recreational/designed-landscape context ({mr_ref}): rule out WWII airfield dispersal pens, searchlight/anti-aircraft gun-pits, "
            f"ordnance craters, golf course tees/bunkers, and 18th/19th-century parkland tree-ring plantation banks."
        )

    # 8. Sub-Canopy Forestry Ploughing, Windthrow Root-Plates & Clearance Cairns
    wood_osm = [
        m
        for m in all_osm
        if any(k in str(m.get("subtype", "")) for k in ("landuse=forest", "natural=wood"))
    ]
    if wood_osm or ndsm >= 0.25:
        w_ref = _fmt_osm(wood_osm[0]) if wood_osm else f"surface roughness nDSM {ndsm:.2f} m"
        hints.append(
            f"Woodland/canopy taphonomy ({w_ref}): check for tree-throw root-plate hollows, charcoal burning platforms, "
            f"forestry drainage ridging, brash stacks, or sparse ground-return TIN interpolation facets beneath canopy."
        )

    # 9. Fallback Agricultural / Post-Medieval Map Check
    if len(hints) < 2:
        hints.append(
            "Agricultural/post-medieval land-use check: cross-check First Edition 25-inch OS mapping and historic vertical aerial photography "
            "for plough headlands along removed field boundaries, medieval rabbit-warren pillow mounds, post-medieval windmill mounds, "
            "or infilled farm ponds and marl pits."
        )

    return hints[:4]


class NHLESpatialContextExtractor:
    """Fast vectorized nearest-neighbour lookup over the 20,003-record NHLE Scheduled Monument catalog."""

    def __init__(self, catalog: MonumentCatalog) -> None:
        self.records = list(catalog.records)
        self.eastings = np.array([r.easting for r in self.records], dtype=np.float64)
        self.northings = np.array([r.northing for r in self.records], dtype=np.float64)
        self.classes = np.array([r.cls for r in self.records], dtype=np.int32)

    def query_context(
        self,
        e: float,
        n: float,
        max_radius_m: float = 4000.0,
        top_k: int = 4,
    ) -> Tuple[List[Dict], Dict]:
        dx = self.eastings - e
        dy = self.northings - n
        d2 = dx * dx + dy * dy

        within_5k = np.where(d2 <= 5000.0 * 5000.0)[0]
        within_2k = np.where(d2 <= 2000.0 * 2000.0)[0]

        prehist_mask_2k = np.isin(self.classes[within_2k], [1, 2, 3]) if len(within_2k) else np.array([], dtype=bool)
        prehist_mask_5k = np.isin(self.classes[within_5k], [1, 2, 3]) if len(within_5k) else np.array([], dtype=bool)

        cluster_summary = {
            "landscape_area": infer_landscape_character_area(e, n),
            "nhle_within_2km": int(len(within_2k)),
            "prehistoric_within_2km": int(np.sum(prehist_mask_2k)),
            "nhle_within_5km": int(len(within_5k)),
            "prehistoric_within_5km": int(np.sum(prehist_mask_5k)),
        }

        within_r = np.where(d2 <= max_radius_m * max_radius_m)[0]
        if len(within_r) == 0:
            # Take single absolute nearest monument in England so the user still gets a reference distance
            nearest_idx = int(np.argmin(d2))
            within_r = np.array([nearest_idx], dtype=np.int64)

        order = within_r[np.argsort(d2[within_r])[:top_k]]
        nearby_list: List[Dict] = []
        for idx in order:
            rec = self.records[int(idx)]
            dist_m = float(math.sqrt(d2[int(idx)]))
            bearing = compass_bearing_from_Site(e, n, rec.easting, rec.northing)
            nearby_list.append(
                {
                    "list_entry": int(rec.list_entry),
                    "name": rec.name,
                    "cls_id": int(rec.cls),
                    "cls_label": NHLE_CLASS_LABELS.get(int(rec.cls), "Scheduled monument"),
                    "distance_m": round(dist_m, 0),
                    "bearing": bearing,
                }
            )
        return nearby_list, cluster_summary


def build_deterministic_academic_context(cand: Dict, nearby_nhle: List[Dict], cluster: Dict) -> str:
    """
    Generate a rigorous, non-hallucinated academic archaeological & 1m LiDAR morphological
    assessment from exact NHLE spatial relationships and measured terrain observables.
    """
    cat = cand.get("context_category", "unidentified_candidate")
    cls_name = cand.get("predicted_class", "round_barrow_cairn")
    rel_m = float(cand.get("est_local_relief_m", 0.5))
    slope = float(cand.get("mean_slope_tangent", 0.08))
    slope_deg = math.degrees(math.atan(slope))
    area = float(cand.get("area_m2", 250.0))
    eq_diam = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
    elong = float(cand.get("elongation", 1.15))
    ndsm = float(cand.get("est_surface_prominence_m", 0.0))
    landscape = cluster.get("landscape_area", "English landscape")
    n2k = cluster.get("nhle_within_2km", 0)
    p2k = cluster.get("prehistoric_within_2km", 0)
    n5k = cluster.get("nhle_within_5km", 0)

    if cls_name == "megalithic_tomb_long_barrow":
        form_phrase = f"an elongated or oval mound ({elong:.2f}× axial elongation, {area:.0f} m² footprint, ~{eq_diam:.0f} m equivalent span)"
        arch_parallel = "ploughed-down Neolithic long barrow, oval barrow, or bank-and-ditch mortuary enclosure"
    elif cls_name == "stone_circle_henge_alignment":
        form_phrase = f"a ring-bank or annular earthwork ({area:.0f} m² footprint, ~{eq_diam:.0f} m diameter, {elong:.2f}× elongation)"
        arch_parallel = "Neolithic/Early Bronze Age henge bank, ring cairn, or enclosed cremation cemetery"
    elif cls_name == "roman_villa_building":
        form_phrase = f"a rectilinear building platform or courtyard enclosure ({area:.0f} m² footprint, ~{eq_diam:.0f} m span, {elong:.2f}× axial elongation)"
        arch_parallel = "Romano-British villa range, aisled building platform, or temple temenos earthwork"
    elif cls_name == "roman_road_agger_trackway":
        form_phrase = f"a cambered linear embankment or hollow-way terrace ({elong:.2f}× axial elongation, {area:.0f} m² footprint)"
        arch_parallel = "Roman road agger with flanking quarry-scoop ditches or ancient ridgeway trackway"
    elif cls_name == "roman_fort_camp_settlement":
        form_phrase = f"a rectilinear rampart-and-ditch enclosure or settlement earthwork ({area:.0f} m² footprint, ~{eq_diam:.0f} m span, {elong:.2f}× elongation)"
        arch_parallel = "Roman fortlet, marching-camp vallum, signal station, or Romano-British enclosed farmstead"
    elif cls_name == "hut_circle_pound_settlement":
        form_phrase = f"a sub-circular stone/turf hut ring or pound compound ({area:.0f} m² footprint, ~{eq_diam:.0f} m span, {elong:.2f}× elongation)"
        arch_parallel = "Bronze/Iron Age stone hut circle, enclosed upland settlement, or Dartmoor/Cornish pound"
    elif cls_name == "iron_age_hillfort_banjo_enclosure":
        form_phrase = f"a curvilinear rampart-and-ditch enclosure ({area:.0f} m² footprint, ~{eq_diam:.0f} m span, {elong:.2f}× elongation)"
        arch_parallel = "Iron Age univallate/multivallate enclosure, banjo enclosure, or defended settlement"
    elif cls_name == "cross_ridge_dyke_field_system":
        form_phrase = f"a linear boundary bank/ditch or terraced lynchet scarp ({elong:.2f}× axial elongation, {area:.0f} m² footprint)"
        arch_parallel = "prehistoric cross-ridge dyke, coaxial reave, or Celtic field lynchet boundary"
    else:
        form_phrase = f"a sub-circular mound ({elong:.2f}× elongation, {area:.0f} m² footprint, ~{eq_diam:.0f} m diameter)"
        arch_parallel = "Bronze Age bowl/bell barrow or upland funerary cairn"

    nearest_str = ""
    if nearby_nhle:
        top_n = nearby_nhle[:2]
        parts = [f"{m['name']} (NHLE #{m['list_entry']}, {int(m['distance_m'])} m {m['bearing']})" for m in top_n]
        nearest_str = " and ".join(parts)

    mp_matches = list(cand.get("megalithic_portal_matches") or [])
    mp_clause = ""
    if mp_matches:
        top_mp = mp_matches[0]
        mp_clause = (
            f" Correlated with The Megalithic Portal (megalithic.co.uk #{top_mp['sid']}: "
            f"'{top_mp['name']}', {top_mp['site_type']}, {float(top_mp['distance_m']):.1f} m)."
        )

    if cat == "rediscovered_nhle_scheduled" and cand.get("matched_monument"):
        mm = cand["matched_monument"]
        return (
            f"Situated within the {landscape}, this detection independently recovers the scheduled monument "
            f"'{mm['name']}' (NHLE #{mm['list_entry']}, centroid offset {mm['distance_m']:.1f} m) in a held-out test square. "
            f"In 1 m bare-earth LiDAR, the earthwork survives as {form_phrase} with {rel_m:+.2f} m of 16 m local relief "
            f"and a mean scarp gradient of {slope:.3f} ({slope_deg:.1f}°; surface canopy nDSM {ndsm:.2f} m). "
            f"The site lies within a recorded monument cluster containing {n2k} scheduled monuments ({p2k} prehistoric) "
            f"within 2 km and {n5k} within 5 km.{mp_clause}"
        )

    if cat == "rediscovered_osm_archaeology" and cand.get("osm_archaeology_matches"):
        om = cand["osm_archaeology_matches"][0]
        om_id = str(om.get("osm_id") or "")
        if om_id.startswith("megalithic.co.uk/"):
            return (
                f"Located in the {landscape}, this detection independently recovers an unscheduled monument "
                f"recorded in The Megalithic Portal ('{om.get('name')}', {om.get('subtype')}, {om_id}, "
                f"offset {float(om.get('distance_m', 0.0)):.1f} m) outside Historic England's Scheduled Monument register. "
                f"The 1 m DTM resolves {form_phrase} with {rel_m:+.2f} m local relief "
                f"and {slope:.3f} ({slope_deg:.1f}°) mean scarp slope (nDSM {ndsm:.2f} m). "
                f"Nearest scheduled parallels include {nearest_str or 'regional barrow groups'} ({n5k} scheduled monuments within 5 km)."
            )
        return (
            f"Located in the {landscape}, this detection corresponds to an unscheduled archaeological feature "
            f"recorded in OpenStreetMap ({om.get('name') or om.get('subtype')}, {om['osm_id']}) outside Historic England's "
            f"Scheduled Monument polygon register. The 1 m DTM resolves {form_phrase} with {rel_m:+.2f} m local relief "
            f"and {slope:.3f} ({slope_deg:.1f}°) mean scarp slope (nDSM {ndsm:.2f} m). "
            f"Nearest scheduled parallels include {nearest_str or 'regional barrow groups'} ({n5k} scheduled monuments within 5 km).{mp_clause}"
        )

    if cat == "flagged_modern_or_karst_confounder":
        cm = (cand.get("osm_confounder_matches") or [{}])[0]
        reason = cm.get("name") or cm.get("subtype") or "modern/geological confounder"
        fp_hints = cand.get("environmental_confounder_hints") or derive_false_positive_context_hints(cand, nearby_nhle, cluster)
        env_clause = f" Context check: {fp_hints[0]}" if fp_hints else ""
        return (
            f"Located in the {landscape}, this {area:.0f} m² topographic anomaly ({rel_m:+.2f} m local relief, "
            f"{slope:.3f} / {slope_deg:.1f}° mean scarp slope, nDSM {ndsm:.2f} m) mimics archaeological earthwork geometry "
            f"in 1 m LiDAR but is separated by the spatial/geomorphic filter as '{reason}'. "
            f"Machine-graded scarps, negative-relief extraction basins, mobile geomorphic surfaces, or mapped infrastructure distinguish it "
            f"from weathered monuments despite proximity to {nearest_str or 'regional monuments'}.{env_clause}"
        )

    # Unidentified candidate
    cluster_clause = (
        f"within an active monument landscape ({n2k} scheduled monuments within 2 km and {n5k} within 5 km, "
        f"closest being {nearest_str})"
        if n5k > 0 and nearest_str
        else f"in a lightly scheduled sector of the {landscape} (nearest scheduled monument: {nearest_str})"
    )
    canopy_clause = (
        f"Bare-earth/surface comparison shows minimal above-ground vegetation or structure (nDSM {ndsm:.2f} m), "
        f"confirming that the anomaly is a ground-surface relief feature rather than scrub canopy."
        if ndsm < 0.25
        else f"Moderate surface roughness (nDSM {ndsm:.2f} m) warrants checking for low gorse/brash cover or stone cairn exposure."
    )
    fp_hints = cand.get("environmental_confounder_hints") or derive_false_positive_context_hints(cand, nearby_nhle, cluster)
    primary_hint = fp_hints[0] if fp_hints else (
        "cross-check First Edition 25-inch OS mapping and historic aerial photography "
        "to rule out post-medieval marl/quarry spoil or field-clearance mounds before ground-truthing."
    )
    return (
        f"Situated in the {landscape} {cluster_clause}, the 1 m DTM isolates {form_phrase} rising {rel_m:+.2f} m "
        f"above its 16 m local trend surface with weathered scarp gradients of {slope:.3f} ({slope_deg:.1f}°), "
        f"morphologically consistent with a {arch_parallel}.{mp_clause} {canopy_clause} "
        f"Critical false-positive & HER check — {primary_hint}"
    )


_HISTORIC_OS_CACHE_DIR = Path("/mnt/bulk/megalith-lidar/historic_os_cache")
_HISTORIC_OS_TILE_URL = "https://api.maptiler.com/tiles/uk-osgb1888/{z}/{x}/{y}?key=7Y0Q1ck46BnB8cXXXg8X"


def fetch_historic_os_crop(
    lat: float,
    lon: float,
    zoom: int = 16,
    crop_px: int = 256,
    timeout: float = 4.5,
) -> Optional[Image.Image]:
    """
    Fetch and cache a 256x256 crop of the 1888-1913 Ordnance Survey Six-Inch to the Mile
    historic map (`uk-osgb1888`) centered on `(lat, lon)`, with a red cartographic reticle
    at the candidate centroid so the VLM can read 19th-century antiquity labels ('Camp',
    'Tumulus', 'Roman Road') or Victorian/industrial confounders ('Old Quarry', 'Gravel Pit',
    'Reservoir', 'Plantation').
    """
    try:
        from PIL import ImageDraw

        lat_rad = math.radians(float(lat))
        n = 2.0 ** zoom
        xtile_f = (float(lon) + 180.0) / 360.0 * n
        ytile_f = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        cx_tile = int(math.floor(xtile_f))
        cy_tile = int(math.floor(ytile_f))
        px_in_tile = int(round((xtile_f - cx_tile) * 256))
        py_in_tile = int(round((ytile_f - cy_tile) * 256))

        _HISTORIC_OS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (256 * 2, 256 * 2), (242, 236, 225))
        dx_start = -1 if px_in_tile < 128 else 0
        dy_start = -1 if py_in_tile < 128 else 0

        for ix, dx in enumerate((dx_start, dx_start + 1)):
            for iy, dy in enumerate((dy_start, dy_start + 1)):
                tx, ty = cx_tile + dx, cy_tile + dy
                cpath = _HISTORIC_OS_CACHE_DIR / f"os1888_z{zoom}_{tx}_{ty}.jpg"
                if cpath.exists():
                    tile_img = Image.open(cpath).convert("RGB")
                else:
                    url = _HISTORIC_OS_TILE_URL.format(z=zoom, x=tx, y=ty)
                    req = urllib.request.Request(
                        url,
                        headers={
                            "User-Agent": "Mozilla/5.0 (compatible; MegalithLiDAR/1.0)",
                            "Referer": "https://maps.nls.uk/",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=timeout) as r:
                        raw = r.read()
                    cpath.write_bytes(raw)
                    tile_img = Image.open(io.BytesIO(raw)).convert("RGB")
                canvas.paste(tile_img, (ix * 256, iy * 256))

        center_x = (-dx_start) * 256 + px_in_tile
        center_y = (-dy_start) * 256 + py_in_tile
        half = crop_px // 2
        crop = canvas.crop((center_x - half, center_y - half, center_x + half, center_y + half))
        draw = ImageDraw.Draw(crop)
        draw.ellipse((half - 18, half - 18, half + 18, half + 18), outline=(185, 28, 28), width=2)
        draw.line((half - 26, half, half - 18, half), fill=(185, 28, 28), width=2)
        draw.line((half + 18, half, half + 26, half), fill=(185, 28, 28), width=2)
        draw.line((half, half - 26, half, half - 18), fill=(185, 28, 28), width=2)
        draw.line((half, half + 18, half, half + 26), fill=(185, 28, 28), width=2)
        return crop
    except Exception:
        return None


def make_zoomed_crop_b64(
    png_path: Path,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Tuple[str, bool]:
    """
    Create a 3x-magnified 80m x 80m close-up crop of [Panel 1: Bare-Earth Slope | Panel 2: 16m Local Relief]
    and, when `(lat, lon)` are provided, stitch [Panel 3: 1888-1913 Historic Ordnance Survey 6-Inch Map]
    so the VLM can simultaneously inspect 1m LiDAR micro-topography and 19th-century OS cartography.
    Returns `(base64_png_str, has_historic_os_panel)`.
    """
    im = Image.open(png_path)
    arr = np.array(im)[:, :256, :]
    yellow = (arr[:, :, 0] > 230) & (arr[:, :, 1] > 190) & (arr[:, :, 2] < 30)
    green = (arr[:, :, 0] < 30) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 90)
    ys, xs = np.where(yellow | green)
    if len(xs) > 0:
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
    else:
        cx, cy = 128, 128
    r = 40
    x0, x1 = max(0, min(256 - 2 * r, cx - r)), min(256, max(2 * r, cx + r))
    y0, y1 = max(0, min(256 - 2 * r, cy - r)), min(256, max(2 * r, cy + r))
    p1_crop = im.crop((x0, y0, x1, y1)).resize((256, 256), Image.Resampling.NEAREST)
    p2_crop = im.crop((260 + x0, y0, 260 + x1, y1)).resize((256, 256), Image.Resampling.NEAREST)

    os_crop = None
    if lat is not None and lon is not None:
        os_crop = fetch_historic_os_crop(float(lat), float(lon), zoom=16, crop_px=256)

    if os_crop is not None:
        out = Image.new("RGB", (776, 256), (20, 20, 24))
        out.paste(p1_crop, (0, 0))
        out.paste(p2_crop, (260, 0))
        out.paste(os_crop, (520, 0))
        has_os = True
    else:
        out = Image.new("RGB", (516, 256), (20, 20, 24))
        out.paste(p1_crop, (0, 0))
        out.paste(p2_crop, (260, 0))
        has_os = False

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), has_os


_VLM_OBVIOUS_CONFOUNDER_PAT = (
    "modern military emplacement",
    "covered reservoir",
    "modern reservoir",
    "modern rectangular reservoir",
    "utility scoop",
    "modern utility",
    "modern spoil",
    "modern forestry spoil",
    "forestry plough",
    "drainage scoop",
    "field clearance ring",
    "post-medieval chalk extraction",
    "post-medieval marl pit",
    "extractive pit",
    "quarry scoop",
    "quarry pit",
    "old quarry",
    "gravel pit",
    "marl pit",
    "chalk pit",
    "borrow pit",
    "peat-cutting",
    "fighter pen",
    "dispersal pen",
    "military gun-pit",
)

_VLM_ROMAN_REJECT_PHRASES = (
    "is a false positive",
    "likely a false positive",
    "fluvial false positive",
    "score is rejected",
    "classification is rejected",
    "prediction is rejected",
    "is rejected in favour",
    "is rejected as a",
    "despite the model",
    "although the model",
    "contradicts the",
    "rather than a roman",
    "rather than the roman",
    "rather than a rectilinear",
    "rather than a military",
    "rather than a genuine",
    "rather than genuine",
    "rather than a built",
    "rather than a confirmed",
    "rather than an unmapped",
    "rather than a settlement",
    "lacking the sharp 90-degree",
    "lacking the sharp, rectilinear",
    "lacking the sharp rectilinear",
    "lacking the rectilinear",
    "lacking the enclosed, rectilinear",
    "lacking rectilinear",
    "lacking any rectilinear",
    "lacks rectilinear",
    "without rectilinear",
    "non-rectilinear",
    "entirely lacking the sharp",
    "lack of rectilinear",
    "absence of rectilinear",
    "absence of definitive military",
    "modern rectangular reservoir",
    "modern reservoir",
    "covered reservoir",
    "utility scoop",
    "modern utility",
    "quarry scoop",
    "quarry pit",
    "old quarry",
    "extractive pit",
    "gravel pit",
    "marl pit",
    "chalk pit",
    "borrow pit",
    "peat-cutting",
    "kettle-hole",
    "kettle hole",
    "modern agricultural",
    "agricultural confounder",
    "agricultural pond",
    "agricultural bund",
    "agricultural enclosure",
    "modern ploughing",
    "forestry operations",
    "forestry drainage",
    "modern field boundary",
    "field boundary",
    "plantation boundary",
    "internal plantation",
    "forestry track",
    "modern rectangular enclosure",
    "post-medieval field enclosure",
    "modern pond",
    "farm pond",
    "slurry",
    "modern spoil",
    "industrial spoil",
    "spoil tip",
    "spoil heap",
    "spoil ring",
    "mining bell-pit",
    "bmx track",
    "modern recreational",
    "forestry spoil",
    "forestry plough",
    "drainage scoop",
    "modern earthwork",
    "modern linear earthwork",
    "modern confounder",
    "historic or modern confounder",
    "military gun-pit",
    "dispersal pen",
    "fighter pen",
    "natural glacial",
    "natural fluvio-glacial",
    "natural fluvial",
    "natural periglacial",
    "natural chalk",
    "natural topographic",
    "natural confounder",
    "natural limestone",
    "natural feature",
    "natural rounded",
    "natural mound",
    "natural hillock",
    "natural knoll",
    "natural bedrock",
    "natural outcrop",
    "natural peat",
    "peatland formation",
    "fluvial point-bar",
    "point-bar scroll",
    "fluvial scroll",
    "meander scroll ridge",
    "solution doline",
    "pingo scar",
    "kame hummock",
    "root-plate hollow",
    "degraded prehistoric earthwork",
    "prehistoric ring-ditch",
    "prehistoric ring ditch",
    "prehistoric ringwork",
    "prehistoric round barrow",
)

_VLM_HISTORIC_OS_ANTIQUITY_PHRASES = (
    "labeled as a 'camp'",
    "labelled as a 'camp'",
    "labeled \"camp\"",
    "labelled \"camp\"",
    "reads \"camp\"",
    "reads 'camp'",
    "marked as 'camp'",
    "marked as \"camp\"",
    "explicitly labels this exact location as a \"camp\"",
    "explicitly labels this exact location as 'camp'",
    "explicitly labeled 'camp'",
    "explicitly labelled 'camp'",
    "explicitly labeled \"camp\"",
    "explicitly labelled \"camp\"",
    "roman road on the 1888",
    "depicts it with earthwork symbols",
    "hachured 'tumulus'",
    "hachured \"tumulus\"",
    "labelled 'tumulus'",
    "labeled 'tumulus'",
    "labelled \"tumulus\"",
    "labeled \"tumulus\"",
    "labels the red reticle as a 'tumulus'",
    "labels the red reticle as a \"tumulus\"",
    "(tumulus) with hachures",
    "labels the red reticle as a 'beacon'",
    "labels the red reticle as 'beacon'",
    "hachured earthwork symbols at the reticle",
    "explicit 1888 os hachuring",
    "explicit 1888 os cartographic hachuring",
    "explicit 1888 cartographic designation",
    "displays a distinct hachured circular earthwork",
    "displays a hachured, oval earthwork",
    "over a hachured, oval earthwork",
    "over a hachured circular earthwork",
    "over a hachured earthwork",
    "accompanied by earthwork hachures",
)

_HISTORIC_OS_NEG_PREFIX_RE = re.compile(
    r"\b(no|not|neither|nor|without|lacks|lacking|absent|absence|devoid)\b"
)


def has_historic_os_antiquity_evidence(note: str) -> bool:
    """Return True only when `note` positively cites an 1888 OS antiquity without a preceding negation."""
    low = (note or "").lower()
    if not low:
        return False
    for ph in _VLM_HISTORIC_OS_ANTIQUITY_PHRASES:
        start = 0
        while True:
            idx = low.find(ph, start)
            if idx < 0:
                break
            prefix = low[max(0, idx - 45) : idx]
            for sep in (".", ";", ","):
                if sep in prefix:
                    prefix = prefix.split(sep)[-1]
            if not _HISTORIC_OS_NEG_PREFIX_RE.search(prefix):
                return True
            start = idx + len(ph)
    return False


_HISTORIC_OS_CONFOUNDER_RE = re.compile(
    r"(?:"
    r"labels\s+(?:the\s+red\s+reticle(?:'s\s+location)?|this\s+location|a\s+feature|it|a)\s+(?:as\s+(?:an?\s+)?)?['\"]?("
    r"old\s+shaft|old\s+coal\s+shaft|old\s+coal\s+pit|old\s+trial\s+shaft|old\s+quarry|old\s+chalk\s+pit|"
    r"old\s+gravel\s+pit|old\s+marl\s+pit|old\s+sand\s+pit|old\s+clay\s+pit|old\s+pit|chalk\s+pit|gravel\s+pit|"
    r"marl\s+pit|sand\s+pit|clay\s+pit|quarry|reservoir|icehouse|spring|limekiln|old\s+limekiln|kiln|"
    r"brick\s+works|windmill(?:\s*\(disused\)|\s+hill)?|rifle\s+range|triangulation\s+pillar|filter\s+bed|sewage\s+works|"
    r"gas\s+works|colliery|mine|air\s+shaft|shaft|pond|fishpond|pit"
    r")\b|"
    r"confirming\s+(?:that\s+)?(?:this|the)\s+(?:[a-z0-9\s-]+?\s+)?is\s+a\s+(?:historic|post-medieval|modern|19th-century|20th-century|industrial|natural)\s+"
    r"(?:mining|extraction|quarry|water-management|water\s+infrastructure|water|surveying|subterranean|spoil|utility)"
    r")",
    re.IGNORECASE,
)

_POSITIVE_OS_FEATURE_PAT = re.compile(
    r"(?:"
    r"wooded\s+area\s+labelled|"
    r"labelled\s+(?:as\s+(?:a\s+|an\s+)?)?['\"][^'\"]+['\"]|"
    r"labeled\s+(?:as\s+(?:a\s+|an\s+)?)?['\"][^'\"]+['\"]|"
    r"labels\s+(?:the\s+red\s+reticle(?:'s\s+location)?|the\s+adjacent\s+linear\s+feature|a)\s+(?:as\s+(?:a\s+|an\s+)?)?['\"][^'\"]+['\"]|"
    r"topographic\s+label|triangulation\s+pillar"
    r")",
    re.IGNORECASE,
)

_OS_1888_REF_PAT = re.compile(
    r"\b(1888|ordnance\s+survey|6-inch|six-inch|historic\s+os)\b",
    re.IGNORECASE,
)


def _clean_blank_os_sentence(s: str) -> str:
    m_dangling = re.match(
        r"^(indicating|suggesting|confirming|supporting|leaving|requiring|meaning|ruling\s+out)\s+(?:that\s+)?(.*)$",
        s.strip(),
        flags=re.IGNORECASE,
    )
    if m_dangling:
        verb = m_dangling.group(1).lower()
        rest = m_dangling.group(2).strip()
        if verb.startswith("ruling"):
            return "The isolated earthwork morphology rules out " + rest
        if rest.lower().startswith(("this is ", "the feature is ", "the lidar anomaly is ", "it is ")):
            return re.sub(r"^(?:this|the\s+feature|the\s+lidar\s+anomaly|it)\s+is\s+", "This earthwork is ", rest, flags=re.IGNORECASE)
        if rest.lower().startswith("the feature as "):
            return re.sub(r"^the\s+feature\s+as\s+", "This earthwork remains ", rest, flags=re.IGNORECASE)
        if verb in ("supporting", "confirming"):
            return "The LiDAR morphology supports " + rest
        if verb == "requiring":
            return "This unrecorded earthwork requires " + rest
        return rest[0].upper() + rest[1:]

    if not _OS_1888_REF_PAT.search(s):
        return s
    if has_historic_os_antiquity_evidence(s) or _HISTORIC_OS_CONFOUNDER_RE.search(s) or _POSITIVE_OS_FEATURE_PAT.search(s):
        return s

    s = re.sub(
        r",\s*while\s+the\s+(?:1888|1913|historic(?:al)?|ordnance\s+survey|os|6-inch|six-inch|[\s–-])+map\b.*?(?=[.!?]$)",
        "",
        s,
        flags=re.IGNORECASE,
    )
    if not _OS_1888_REF_PAT.search(s):
        return s

    m_conj = re.search(
        r"^(?:While\s+the|Although\s+the|Despite\s+the|The)\s+(?:.*?\b)?(?:1888|ordnance\s+survey|historic\s+os).*?(?:;\s*(?:however,\s*|consequently,\s*|nonetheless,\s*|therefore,\s*)?|,\s*(?:but|and\s+(?:the|its)|though|yet|while|whereas)\s+)([A-Za-z0-9].*[.!?])$",
        s,
        flags=re.IGNORECASE,
    )
    if m_conj:
        rest = m_conj.group(1).strip()
        if rest[0].isdigit():
            return "The " + rest
        if rest.lower().startswith(("the ", "its ", "this ", "given ", "a ", "an ", "combined ", "morphological ", "proximity ", "isolated ")):
            return rest[0].upper() + rest[1:]
        return "The " + rest[0].lower() + rest[1:]

    m_abs = re.search(
        r"^(?:(?:Given\s+the|The)\s+(?:complete\s+|explicit\s+)?(?:absence|lack)\s+of|Lacking\s+(?:any\s+|explicit\s+)?)\s*.*?\b(?:1888|ordnance\s+survey|historic\s+os).*?(?:,\s*combined\s+with\s+([^,]+),|,)\s*(.*)$",
        s,
        flags=re.IGNORECASE,
    )
    if m_abs:
        comb, tail = m_abs.group(1), m_abs.group(2).strip()
        if comb:
            c_str = comb.strip()
            return c_str[0].upper() + c_str[1:] + " " + tail
        return tail[0].upper() + tail[1:]

    m_part = re.search(
        r"^The\s+(?:.*?\b)?(?:1888|ordnance\s+survey|historic\s+os).*?,\s*(indicating|suggesting|confirming|supporting|leaving|requiring|meaning|ruling\s+out)\s+(?:that\s+)?(.*)$",
        s,
        flags=re.IGNORECASE,
    )
    if m_part:
        verb = m_part.group(1).lower()
        rest = m_part.group(2).strip()
        if verb.startswith("ruling"):
            return "The isolated earthwork morphology rules out " + rest
        if rest.lower().startswith(("this is ", "the feature is ", "the lidar anomaly is ", "it is ")):
            rest = re.sub(r"^(?:this|the\s+feature|the\s+lidar\s+anomaly|it)\s+is\s+", "This earthwork is ", rest, flags=re.IGNORECASE)
            return rest
        if rest.lower().startswith("the feature as "):
            rest = re.sub(r"^the\s+feature\s+as\s+", "This earthwork remains ", rest, flags=re.IGNORECASE)
            return rest
        if verb in ("supporting", "confirming"):
            return "The LiDAR morphology supports " + rest
        if verb == "requiring":
            return "This unrecorded earthwork requires " + rest
        return rest[0].upper() + rest[1:]

    return s


def strip_empty_historic_os_clauses(note: str) -> str:
    """
    Remove negative/blank 1888 OS map clauses ('The 1888 OS map shows no Tumulus label...')
    so the 1888 OS map is only mentioned when it visibly depicts an antiquity or historic feature.
    """
    if not note:
        return note
    if has_historic_os_antiquity_evidence(note):
        return note
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", note.strip())
    cleaned = [_clean_blank_os_sentence(s) for s in sents]
    return " ".join(cleaned)

_VLM_ALL_DOMAIN_HISTORIC_OS_REJECT_PHRASES = (
    "old coal shaft",
    "old coal pit",
    "old trial shaft",
    "bell-pit spoil ring",
    "sluice valve",
    "water-management utility",
    "historic water source",
    "labels the red reticle as an 'icehouse'",
    "labels the red reticle as 'icehouse'",
    "labels the red reticle as a \"spring\"",
    "labels the red reticle as a 'spring'",
    "labels the red reticle as a triangulation pillar",
    "effectively dismissing",
    "is a false positive",
    "strongly indicates a false positive",
    "confirming this feature is a natural",
    "confirming this earthwork is a post-medieval",
    "confirming this doughnut-shaped earthwork is a",
    "natural fenland roddon",
    "generic hachured sand-dune symbols",
    "positioned at a field boundary junction with no hachures",
    "woodland false-positive",
    "tree-throw pit or forestry feature",
    "karst doline or periglacial scar",
    "unconfirmed natural geomorphic feature",
)


def classify_vlm_note_downrating(
    note: str,
    shape_tag: str,
    cand: Dict,
) -> Tuple[str, float, Optional[str]]:
    """
    Evaluate Qwen3.8's visual/text assessment (including 1888-1913 Historic OS 6-Inch map reading)
    and physical morphology to determine:
      - `verdict_tier`: 'PLAUSIBLE_MONUMENT' (1.0x), 'UNCERTAIN_OR_DEGRADED' (0.82x),
        or 'OBVIOUS_FALSE_POSITIVE' (0.35x and routed to Tab 5 confounders).
      - `multiplier`: factor applied to `raw_unet_confidence` to compute `confidence`.
      - `reason`: concise explanation when downrated or routed to Tab 5.
    """
    low = (note or "").lower()
    is_roman = cand.get("target_domain") == "roman"
    pred_cls = cand.get("predicted_class", "")
    elong = float(cand.get("elongation", 1.15))
    e_coord = float(cand.get("easting") or 0.0)
    n_coord = float(cand.get("northing") or 0.0)

    dune_belt = is_active_holocene_coastal_dune_belt(e_coord, n_coord)
    has_coastal_dune_conf = any(
        c.get("osm_id") == "geomorph/coastal_sand_dunes"
        for c in (cand.get("osm_confounder_matches") or [])
    )
    if dune_belt or has_coastal_dune_conf:
        return (
            "OBVIOUS_FALSE_POSITIVE",
            0.35,
            f"Active Holocene coastal sand-dune system ({dune_belt or 'mobile coastal dunes'}): unburied Bronze Age/Roman earthworks cannot survive on shifting blown sand",
        )

    # Protect candidates where the 1888 OS 6-Inch map explicitly depicts an antiquity ('Camp', 'Tumulus', hachured earthwork)
    if has_historic_os_antiquity_evidence(note):
        return ("PLAUSIBLE_MONUMENT", 1.0, None)

    if not is_roman and shape_tag in {"RECTILINEAR_OR_U_BUND", "OBVIOUS_MODERN_CONFOUNDER"}:
        return (
            "OBVIOUS_FALSE_POSITIVE",
            0.35,
            "VLM Visual Audit: Square/rectilinear embankment or U-shaped cut-and-fill bund",
        )

    m_os_conf = _HISTORIC_OS_CONFOUNDER_RE.search(note or "")
    if m_os_conf:
        lbl = m_os_conf.group(1) or "historic utility/extraction feature"
        return (
            "OBVIOUS_FALSE_POSITIVE",
            0.35,
            f"1888 OS 6-Inch Map Audit: Marked '{lbl.title()}'",
        )

    for os_rej in _VLM_ALL_DOMAIN_HISTORIC_OS_REJECT_PHRASES:
        if os_rej in low:
            return (
                "OBVIOUS_FALSE_POSITIVE",
                0.35,
                f"VLM & 1888 OS Audit: Flagged as historic/geomorphic confounder ({os_rej})",
            )

    if is_roman or cand.get("target_domain") == "enclosure":
        if shape_tag == "OBVIOUS_MODERN_CONFOUNDER":
            return (
                "OBVIOUS_FALSE_POSITIVE",
                0.35,
                "VLM Visual Audit: Flagged as modern/industrial/geomorphic confounder",
            )
        for r_phrase in _VLM_ROMAN_REJECT_PHRASES:
            if r_phrase in low:
                return (
                    "OBVIOUS_FALSE_POSITIVE",
                    0.35,
                    f"VLM & Historic OS Audit: Rejected as confounder ({r_phrase})",
                )

    area_m2 = float(cand.get("area_m2", 0.0))
    max_pos_rel_m = float(cand.get("max_positive_relief_m", cand.get("est_local_relief_m", 0.0)))
    centroid_rel_m = float(cand.get("centroid_local_relief_m", max_pos_rel_m))
    patch_min_rel_m = float(cand.get("patch_min_relief_m", 0.0))
    ndsm_m = float(cand.get("est_surface_prominence_m", 0.0))
    lum_diff = float(cand.get("soilmark_core_lum_diff", 0.0))
    ring_exg = float(cand.get("cropmark_ring_exg_diff", 0.0))
    pond_tile = float(cand.get("pond_unet_tile_prob") or 0.0)
    pond_peak = float(cand.get("pond_unet_local_peak") or 0.0)

    if cand.get("target_domain") != "enclosure" and area_m2 <= 14000.0 and (
        (centroid_rel_m <= 0.12 and patch_min_rel_m <= -0.22 and (max_pos_rel_m - centroid_rel_m) >= 0.70)
        or (ndsm_m < 1.5 and patch_min_rel_m <= -0.25 and lum_diff <= -0.045 and ring_exg <= -0.045)
        or (pond_tile >= 0.76 and pond_peak >= 0.55)
        or (
            pond_peak >= 0.55
            and (
                (ring_exg <= -0.020 and lum_diff <= -0.018)
                or (float(cand.get("adjacent_channel_span_m") or 0.0) >= 55.0 and ring_exg <= -0.020)
            )
        )
        or (pond_peak >= 0.75 and patch_min_rel_m <= -0.30)
        or (
            float(cand.get("ring_ditch4_m") or 0.0) <= -0.045
            and patch_min_rel_m <= -0.35
            and float(cand.get("mean_slope_tangent") or 0.0) >= 0.12
            and ring_exg <= 0.0
        )
        or (
            is_roman
            and (
                (
                    shape_tag == "RECTILINEAR_OR_U_BUND"
                    and centroid_rel_m <= 0.25
                    and ((max_pos_rel_m - centroid_rel_m) >= 0.50 or patch_min_rel_m <= -0.25)
                )
                or (
                    max_pos_rel_m >= 0.80
                    and centroid_rel_m <= 0.35
                    and (max_pos_rel_m - centroid_rel_m) >= 0.65
                    and (
                        lum_diff <= -0.038
                        or (ndsm_m >= 0.85 and ring_exg >= 0.04)
                        or (pond_tile >= 0.74 and pond_peak >= 0.45)
                    )
                )
            )
        )
    ):
        return (
            "OBVIOUS_FALSE_POSITIVE",
            0.35,
            f"Embanked pond, slurry lagoon, gravel pit, or levelled court bund (bank={max_pos_rel_m:+.2f}m, basin={centroid_rel_m:+.2f}m)",
        )

    for phrase in _VLM_OBVIOUS_CONFOUNDER_PAT:
        if phrase in low and (
            "more likely" in low
            or "strongly indicates" in low
            or "strongly indicate" in low
            or "strongly suggest" in low
            or "likely represents" in low
            or "likely a " in low
            or "most likely a" in low
            or "highly likely a" in low
        ):
            return (
                "OBVIOUS_FALSE_POSITIVE",
                0.35,
                f"VLM Assessment: Flagged as {phrase}",
            )

    # Check if VLM notes a contradiction between long-barrow prediction and circular/low-amplitude morphology,
    # or concludes it is likely natural scree/gravel hummock/forestry spoil
    if (
        (
            pred_cls == "megalithic_tomb_long_barrow"
            and elong < 1.35
            and (
                "despite the model" in low
                or "contradicts" in low
                or "rejected" in low
                or "lacking the axial elongation" in low
                or "lacking any rectilinear" in low
                or "absence of a defined long-barrow" in low
                or "although the model" in low
            )
        )
        or "natural fluvio-glacial" in low
        or "natural chalk scree" in low
        or "natural limestone scree" in low
        or "natural granite outcrop" in low
        or "meander scroll" in low
        or "meandering river" in low
        or "oxbow" in low
        or "palaeochannel" in low
        or "parabolic dune" in low
        or "dune blowout" in low
        or "modern forestry" in low
        or "field clearance ring" in low
        or "more likely a post-medieval" in low
    ):
        return (
            "UNCERTAIN_OR_DEGRADED",
            0.84,
            "VLM Assessment: Downrated (circular/low-relief morphology vs long-barrow class or likely natural/post-medieval feature)",
        )

    return ("PLAUSIBLE_MONUMENT", 1.0, None)


def query_local_vlm_academic_context(
    cand: Dict,
    nearby_nhle: List[Dict],
    cluster: Dict,
    diag_png_path: Optional[Path] = None,
    endpoint: str = "http://localhost:8090/v1/chat/completions",
    timeout: float = 25.0,
) -> Optional[Tuple[str, str]]:
    """
    Query the local Qwen3.8-Flash-Next vision-language server (`localhost:8090`) with the exact
    NHLE spatial fact sheet + 3-panel composite ([Panel 1: 80m x 80m 1m LiDAR Slope | Panel 2:
    80m x 80m 1m LiDAR Local Relief | Panel 3: 1888-1913 Historic Ordnance Survey 6-Inch Map])
    to produce both a shape/confounder verdict ('RECTILINEAR_OR_U_BUND', 'OBVIOUS_MODERN_CONFOUNDER',
    or 'ROUNDED_MOUND_OR_RING') and a 3-sentence academic assessment.
    """
    nearest_lines = [
        f"- {m['name']} (NHLE #{m['list_entry']}, {m['cls_label']}): located {int(m['distance_m'])} m {m['bearing']} of this candidate"
        for m in nearby_nhle[:3]
    ]
    nearest_block = "\n".join(nearest_lines) if nearest_lines else "- None within 4 km"

    env_feats = list(cand.get("nearby_environmental_features") or [])
    env_lines = [
        f"- {ef.get('subtype')}{(' ' + repr(ef['name'])) if ef.get('name') else ''} at {float(ef.get('distance_m', 0.0)):.0f} m"
        for ef in env_feats[:6]
    ]
    env_block = "\n".join(env_lines) if env_lines else "- No mapped OSM waterways, quarries, dunes, or military features within 400 m"

    mp_feats = list(cand.get("megalithic_portal_matches") or [])
    mp_lines = [
        f"- Megalithic Portal #{mp['sid']} '{mp['name']}' ({mp['site_type']}, accuracy={mp.get('accuracy')}/5): {float(mp.get('distance_m', 0.0)):.1f} m away ({mp['url']})"
        for mp in mp_feats[:4]
    ]
    mp_block = "\n".join(mp_lines) if mp_lines else "- None within 250 m"

    fp_hints = list(cand.get("environmental_confounder_hints") or derive_false_positive_context_hints(cand, nearby_nhle, cluster))
    fp_block = "\n".join(f"- {h}" for h in fp_hints)

    slope = float(cand.get("mean_slope_tangent", 0.08))
    slope_deg = math.degrees(math.atan(slope))
    has_lidar_img = diag_png_path is not None and diag_png_path.exists()
    b64 = ""
    has_os_panel = False
    if has_lidar_img and diag_png_path is not None:
        b64, has_os_panel = make_zoomed_crop_b64(
            diag_png_path,
            lat=cand.get("lat"),
            lon=cand.get("lon"),
        )
    elif cand.get("lat") is not None and cand.get("lon") is not None:
        os_only = fetch_historic_os_crop(float(cand["lat"]), float(cand["lon"]), zoom=16, crop_px=256)
        if os_only is not None:
            buf = io.BytesIO()
            os_only.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            has_os_panel = True

    has_img = bool(b64)
    if has_lidar_img and has_os_panel:
        panel_desc = (
            "Left = 80m x 80m 1m LiDAR Bare-Earth Slope; Center = 80m x 80m 1m LiDAR 16m Local Relief (yellow=raised, blue=sunken); "
            "Right = 1888-1913 Historic Ordnance Survey 6-Inch Map (red reticle marks the candidate location)"
        )
    elif has_lidar_img:
        panel_desc = "Left = 80m x 80m 1m LiDAR Bare-Earth Slope; Right = 80m x 80m 1m LiDAR 16m Local Relief (yellow=raised, blue=sunken)"
    else:
        panel_desc = "1888-1913 Historic Ordnance Survey 6-Inch Map (red reticle marks the candidate location)"

    shape_prefix_instr = (
        "First output one tag: [SHAPE: RECTILINEAR_OR_U_BUND] (if the earthwork inside the circle in the 80m x 80m close-up "
        "has straight/square 90-degree corners, a flat rectangular reservoir/building platform, or a U-shaped bank wrapping a dark blue sunken pit), "
        "[SHAPE: OBVIOUS_MODERN_CONFOUNDER] (if the LiDAR or the Right 1888 OS map panel shows it is clearly a modern road/track embankment, "
        "quarry scoop, plantation/field boundary corner, military gun-pit, active coastal sand-dune blowout, or utility feature), "
        "or [SHAPE: ROUNDED_MOUND_OR_RING] (if it is a smooth circular/oval mound or ring in the 1m LiDAR panels). "
        if has_img
        else ""
    )
    prompt_text = (
        f"Site OS Grid Ref: {cand.get('os_grid_ref')} (BNG E {cand.get('easting'):.0f}, N {cand.get('northing'):.0f})\n"
        f"Landscape Character Area: {cluster.get('landscape_area')}\n"
        f"Detection Category: {cand.get('context_category')}\n"
        f"Model Nearest Form Class: {cand.get('predicted_class')} (score={cand.get('confidence'):.3f}, calibrated tile p={cand.get('calibrated_tile_prob', 0.0):.3f})\n"
        f"1m LiDAR Measurements: 16m bare-earth relief = {cand.get('est_local_relief_m'):+.2f} m, "
        f"mean scarp slope = {slope:.3f} ({slope_deg:.1f} deg), footprint = {cand.get('area_m2')} m2, "
        f"axial elongation = {cand.get('elongation')}x, surface canopy nDSM = {cand.get('est_surface_prominence_m')} m, "
        f"ridge continuation = {float(cand.get('ridge_continuation_span_m') or 0.0):.0f} m, "
        f"adjacent channel span = {float(cand.get('adjacent_channel_span_m') or 0.0):.0f} m.\n"
        f"Scheduled Monuments within 2 km: {cluster.get('nhle_within_2km')} ({cluster.get('prehistoric_within_2km')} prehistoric); "
        f"within 5 km: {cluster.get('nhle_within_5km')}.\n"
        f"Nearest Historic England Scheduled Monuments:\n{nearest_block}\n"
        f"Correlated Megalithic Portal (megalithic.co.uk) Records within 250 m:\n{mp_block}\n"
        f"Nearby Environmental & Land-Use Features (OSM within 400 m):\n{env_block}\n"
        f"Site-Specific Geomorphic, Geological & Land-Use False-Positive Considerations:\n{fp_block}\n\n"
        f"{shape_prefix_instr}"
        "Then write a concise 3-sentence (maximum 95 words total) academic landscape-archaeology, 1m LiDAR, and 1888 Ordnance Survey map assessment of this site. "
        "Sentence 1: Landscape/geological setting and spatial relationship to the nearest named NHLE Scheduled Monuments (cite their exact distances and NHLE IDs). "
        f"Sentence 2: Critical morphological reading of the image panels ({panel_desc}) and measurements. "
        "Sentence 3: Geomorphic, land-use & 1888 Historic OS map evaluation — read the Right 1888 OS map panel strictly and literally. "
        "ONLY mention the 1888 OS map if it visibly depicts a feature at/near the red reticle (either an explicit antiquity such as 'Tumulus', 'Camp', or an antiquarian star/hachured ring symbol, or a historic confounder such as an 'Old Quarry', 'Chalk/Gravel/Marl Pit', 'Pond', 'Reservoir', 'Icehouse', or 'Old Coal Shaft'; do not confuse contour lines, trig-point triangles, rough-pasture tufts, or field boundaries with antiquities). "
        "If the 1888 OS map is blank open ground at the reticle, do NOT mention the 1888 OS map at all — instead evaluate geomorphic/agricultural false-positive risks and state your verdict. "
        "Use plain, rigorous British archaeological prose with no bullet points, no markdown headers, and no invented facts."
    )

    user_content: object
    if has_img and b64:
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]
    else:
        user_content = prompt_text

    payload = {
        "model": "default",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior British landscape archaeologist, 1m LiDAR specialist, and Historic Ordnance Survey cartographic analyst inspecting a multi-panel close-up "
                    f"({panel_desc}). "
                    "You write concise, complete 3-sentence site assessments (under 95 words) using only the supplied NHLE records, 1m LiDAR imagery/measurements, and 1888 OS 6-Inch map panel."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 420,
        "temperature": 0.15,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = " ".join((data["choices"][0]["message"].get("content") or "").strip().split())
            shape_tag = "ROUNDED_MOUND_OR_RING"
            if "[SHAPE: RECTILINEAR_OR_U_BUND]" in raw_text:
                shape_tag = "RECTILINEAR_OR_U_BUND"
                raw_text = raw_text.replace("[SHAPE: RECTILINEAR_OR_U_BUND]", "").strip()
            elif "[SHAPE: OBVIOUS_MODERN_CONFOUNDER]" in raw_text:
                shape_tag = "OBVIOUS_MODERN_CONFOUNDER"
                raw_text = raw_text.replace("[SHAPE: OBVIOUS_MODERN_CONFOUNDER]", "").strip()
            elif "[SHAPE: ROUNDED_MOUND_OR_RING]" in raw_text:
                shape_tag = "ROUNDED_MOUND_OR_RING"
                raw_text = raw_text.replace("[SHAPE: ROUNDED_MOUND_OR_RING]", "").strip()
            if len(raw_text) > 60 and raw_text.endswith((".", "!", "?")):
                return strip_empty_historic_os_clauses(raw_text), shape_tag
    except Exception:
        return None
    return None


_COUNTIES_CACHE: Optional[List] = None


def lookup_ceremonial_county(lat: float, lon: float) -> str:
    global _COUNTIES_CACHE
    if _COUNTIES_CACHE is None:
        _COUNTIES_CACHE = []
        try:
            from shapely.geometry import shape
            for p_try in [
                Path("/mnt/bulk/megalith-lidar/catalog/uk_ceremonial_counties.geojson"),
                Path(__file__).resolve().parents[1] / "docs" / "uk_counties.geojson",
            ]:
                if p_try.exists():
                    data = json.loads(p_try.read_text(encoding="utf-8"))
                    for f in data.get("features", []):
                        g = shape(f["geometry"])
                        _COUNTIES_CACHE.append((f["properties"]["county"], g, g.bounds))
                    break
        except Exception:
            pass
    if not _COUNTIES_CACHE:
        return "England"
    from shapely.geometry import Point
    pt = Point(float(lon), float(lat))
    for cname, geom, (minx, miny, maxx, maxy) in _COUNTIES_CACHE:
        if minx <= lon <= maxx and miny <= lat <= maxy:
            if geom.contains(pt):
                return cname
    return min(_COUNTIES_CACHE, key=lambda c: c[1].distance(pt))[0]


def enrich_candidates_with_archaeological_context(
    candidates: List[Dict],
    extractor: NHLESpatialContextExtractor,
    cache_path: Path,
    docs_diag_dir: Optional[Path] = None,
    max_vlm_calls: int = 80,
) -> int:
    """
    Populate `nearby_nhle_monuments`, `nhle_cluster_summary`, `arch_context_note`,
    `arch_context_source` ('qwen3.8_vlm_vision', 'qwen3.8_llm', or 'nhle_grounded_synthesis'),
    `raw_unet_confidence`, and `vlm_verdict_tier` on every candidate.
    Downrates candidates where Qwen3.8 identifies obvious false-positive morphology or
    class-vs-shape contradiction, and routes obvious false positives to `flagged_modern_or_karst_confounder`.
    """
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    vlm_calls = 0
    cache_dirty = False

    def _apply_vlm_verdict(c: Dict, note: str, shape_tag: str) -> None:
        raw_conf = float(c.get("raw_unet_confidence", c.get("confidence", 0.0)))
        c["raw_unet_confidence"] = round(raw_conf, 4)
        tier, mult, reason = classify_vlm_note_downrating(note, shape_tag, c)
        if c.get("vlm_verdict_tier") == "OBVIOUS_FALSE_POSITIVE" and tier != "OBVIOUS_FALSE_POSITIVE":
            tier = "OBVIOUS_FALSE_POSITIVE"
            mult = 0.35
            reason = c.get("vlm_downrate_reason") or c.get("context_reason") or "Flagged confounder"
        c["vlm_verdict_tier"] = tier
        c["vlm_downrate_reason"] = reason
        if c.get("context_category") in {"unidentified_candidate", "flagged_modern_or_karst_confounder"}:
            ring_sc = float(c.get("ring_barrow_score") or 0.0)
            is_os1888 = has_historic_os_antiquity_evidence(note)
            if tier == "PLAUSIBLE_MONUMENT" and c.get("context_category") == "unidentified_candidate":
                if is_os1888:
                    c["confidence"] = round(
                        min(0.9989, 0.9970 + 0.018 * max(0.0, raw_conf - 0.90) + 0.0004 * ring_sc),
                        4,
                    )
                elif ring_sc > 0.0:
                    c["confidence"] = round(min(0.997, raw_conf + (0.997 - raw_conf) * 0.55 * ring_sc), 4)
                else:
                    c["confidence"] = round(raw_conf * mult, 4)
            else:
                c["confidence"] = round(raw_conf * mult, 4)
        if tier == "OBVIOUS_FALSE_POSITIVE" and c.get("context_category") == "unidentified_candidate":
            c["context_category"] = "flagged_modern_or_karst_confounder"
            confs = list(c.get("osm_confounder_matches") or [])
            confs.insert(
                0,
                {
                    "osm_id": "vlm/visual_audit",
                    "subtype": "vlm_obvious_false_positive",
                    "name": reason or "VLM Visual Audit: Obvious modern/rectilinear confounder",
                    "wikidata": None,
                    "distance_m": 0.0,
                    "inside_polygon": True,
                },
            )
            c["osm_confounder_matches"] = confs

    # Prioritize Strict 1% FPR unidentified candidates with PNGs for VLM calls first
    from megalith_lidar.osm_context import MegalithicPortalIndex

    mp_idx = MegalithicPortalIndex.load_default()
    sorted_indices = sorted(
        range(len(candidates)),
        key=lambda i: (
            1 if (candidates[i].get("context_category") == "unidentified_candidate" and candidates[i].get("passes_fpr_0_01")) else 0,
            1 if candidates[i].get("diagnostic_png") else 0,
            float(candidates[i].get("confidence", 0.0)),
        ),
        reverse=True,
    )

    for idx in sorted_indices:
        cand = candidates[idx]
        e = float(cand.get("easting", 0.0))
        n = float(cand.get("northing", 0.0))
        cat = cand.get("context_category", "unidentified_candidate")
        ckey = f"v2_{cand.get('os_grid_ref')}_{int(round(e))}_{int(round(n))}"

        if mp_idx is not None and not cand.get("megalithic_portal_matches"):
            mp_hits = mp_idx.query_point(e, n, radius_m=1000.0, max_items=4)
            if mp_hits:
                cand["megalithic_portal_matches"] = mp_hits

        nearby_nhle, cluster = extractor.query_context(e, n)
        cand["nearby_nhle_monuments"] = nearby_nhle
        cand["nhle_cluster_summary"] = cluster
        cand["environmental_confounder_hints"] = derive_false_positive_context_hints(cand, nearby_nhle, cluster)
        if not cand.get("county") and cand.get("lat") is not None and cand.get("lon") is not None:
            cand["county"] = lookup_ceremonial_county(float(cand["lat"]), float(cand["lon"]))

        is_priority_unid = (
            cat == "unidentified_candidate"
            and (
                bool(cand.get("passes_fpr_0_01"))
                or bool(cand.get("diagnostic_png"))
                or cand.get("target_domain") in {"roman", "enclosure"}
            )
        )
        cached_entry = cache.get(ckey)
        cached_note = strip_empty_historic_os_clauses(str((cached_entry or {}).get("note") or "").strip())
        has_1888_os_in_cache = bool((cached_entry or {}).get("os1888_checked")) or any(
            w in cached_note.lower()
            for w in ("1888", "ordnance survey", "6-inch", "six-inch", "historic os")
        )
        needs_1888_os_upgrade = (
            is_priority_unid
            and vlm_calls < max_vlm_calls
            and not has_1888_os_in_cache
        )
        if cached_entry and cached_note.endswith((".", "!", "?")) and not needs_1888_os_upgrade:
            cand["arch_context_note"] = cached_note
            cand["arch_context_source"] = cached_entry.get("source", "nhle_grounded_synthesis")
            cand["vlm_shape_verdict"] = cached_entry.get("shape_tag", "ROUNDED_MOUND_OR_RING")
            _apply_vlm_verdict(cand, cand["arch_context_note"], cand["vlm_shape_verdict"])
            continue
        if is_priority_unid and vlm_calls < max_vlm_calls:
            diag_p: Optional[Path] = None
            if docs_diag_dir is not None and cand.get("diagnostic_png"):
                p_try = Path(docs_diag_dir) / str(cand["diagnostic_png"])
                if p_try.exists():
                    diag_p = p_try
            res_vlm = query_local_vlm_academic_context(cand, nearby_nhle, cluster, diag_png_path=diag_p)
            vlm_calls += 1
            if res_vlm:
                vlm_note, shape_tag = res_vlm
                src = "qwen3.8_vlm_vision" if (diag_p is not None or cand.get("lat") is not None) else "qwen3.8_llm"
                cand["arch_context_note"] = vlm_note
                cand["arch_context_source"] = src
                cand["vlm_shape_verdict"] = shape_tag
                _apply_vlm_verdict(cand, vlm_note, shape_tag)
                cache[ckey] = {"note": vlm_note, "source": src, "shape_tag": shape_tag, "os1888_checked": True}
                cache_dirty = True
                continue
            elif cached_entry and cached_note.endswith((".", "!", "?")):
                cand["arch_context_note"] = cached_note
                cand["arch_context_source"] = cached_entry.get("source", "nhle_grounded_synthesis")
                cand["vlm_shape_verdict"] = cached_entry.get("shape_tag", "ROUNDED_MOUND_OR_RING")
                _apply_vlm_verdict(cand, cand["arch_context_note"], cand["vlm_shape_verdict"])
                continue

        det_note = build_deterministic_academic_context(cand, nearby_nhle, cluster)
        cand["arch_context_note"] = det_note
        cand["arch_context_source"] = "nhle_grounded_synthesis"
        _apply_vlm_verdict(cand, det_note, str(cand.get("vlm_shape_verdict") or "ROUNDED_MOUND_OR_RING"))

    if cache_dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                disk_cache = json.loads(cache_path.read_text(encoding="utf-8"))
                for dk, dv in disk_cache.items():
                    dn = str((dv or {}).get("note") or "").lower()
                    cn = str((cache.get(dk) or {}).get("note") or "").lower()
                    disk_has_1888 = any(w in dn for w in ("1888", "ordnance survey", "6-inch", "six-inch", "historic os"))
                    mem_has_1888 = any(w in cn for w in ("1888", "ordnance survey", "6-inch", "six-inch", "historic os"))
                    if dk not in cache or (disk_has_1888 and not mem_has_1888):
                        cache[dk] = dv
            except Exception:
                pass
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    apply_community_annotations_to_candidates(candidates)
    return vlm_calls


SURVEY_STATUS_LABELS = {
    "field_verified_monument": "Field-Verified Monument",
    "historic_survey_confirmed": "Historic / HER Survey Record",
    "priority_field_target": "Priority Field Target",
    "uncertain_needs_check": "Uncertain / Needs Ground Check",
    "field_rejected_confounder": "Field-Rejected Confounder / Pond",
}


def _norm_grid_ref(val: object) -> str:
    return " ".join(str(val or "").upper().strip().split())


def parse_survey_annotations_payload(payload: object) -> List[Dict]:
    """
    Parse either:
      1. `{"annotations": [...]}` (`docs/community_annotations.json` format),
      2. `[...]` (raw list of annotation objects),
      3. `{"type": "FeatureCollection", "features": [...]}` (GeoJSON of historic survey records or field notes),
      4. `{"grid_ref": {...}}` (keyed map of annotations).
    Returns normalized annotation dicts.
    """
    raw_items: List[Dict] = []
    if isinstance(payload, list):
        raw_items = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list):
            raw_items = [x for x in payload["annotations"] if isinstance(x, dict)]
        elif payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list):
            for feat in payload["features"]:
                if not isinstance(feat, dict):
                    continue
                props = dict(feat.get("properties") or {})
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates")
                if isinstance(coords, (list, tuple)) and len(coords) >= 2 and isinstance(coords[0], (int, float)):
                    c0, c1 = float(coords[0]), float(coords[1])
                    if abs(c0) <= 180.0 and abs(c1) <= 90.0:
                        props.setdefault("lon", c0)
                        props.setdefault("lat", c1)
                    else:
                        props.setdefault("easting", c0)
                        props.setdefault("northing", c1)
                raw_items.append(props)
        else:
            for k, v in payload.items():
                if k in {"schema_version", "updated_at_utc", "description"}:
                    continue
                if isinstance(v, dict):
                    item = dict(v)
                    item.setdefault("os_grid_ref", k)
                    raw_items.append(item)

    normalized: List[Dict] = []
    for item in raw_items:
        gref = _norm_grid_ref(item.get("os_grid_ref") or item.get("grid_ref") or item.get("ngr"))
        status = str(
            item.get("status")
            or item.get("user_survey_status")
            or item.get("survey_status")
            or ("historic_survey_confirmed" if (item.get("her_id") or item.get("monument_id")) else "priority_field_target")
        ).strip()
        comment = str(
            item.get("comment")
            or item.get("user_survey_comment")
            or item.get("notes")
            or item.get("description")
            or item.get("name")
            or ""
        ).strip()
        observer = str(
            item.get("observer")
            or item.get("user_survey_observer")
            or item.get("author")
            or item.get("source")
            or ""
        ).strip()
        date_str = str(item.get("date") or item.get("user_survey_date") or "").strip()
        ref_str = str(
            item.get("reference")
            or item.get("user_survey_reference")
            or item.get("her_id")
            or item.get("url")
            or ""
        ).strip()
        entry: Dict = {
            "os_grid_ref": gref,
            "status": status,
            "comment": comment,
            "observer": observer,
            "date": date_str,
            "reference": ref_str,
        }
        for num_k in ("easting", "northing", "lat", "lon"):
            if item.get(num_k) is not None:
                try:
                    entry[num_k] = float(item[num_k])
                except Exception:
                    pass
        if entry["os_grid_ref"] or ("easting" in entry and "northing" in entry) or ("lat" in entry and "lon" in entry):
            normalized.append(entry)
    return normalized


def load_community_annotations(path: Optional[Path] = None) -> List[Dict]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "docs" / "community_annotations.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return parse_survey_annotations_payload(payload)
    except Exception:
        return []


def apply_community_annotations_to_candidates(
    candidates: List[Dict],
    annotations: Optional[List[Dict]] = None,
    match_radius_m: float = 45.0,
) -> int:
    """
    Match community/user field-survey and historic-survey annotations onto `candidates`
    by exact `os_grid_ref` or within `match_radius_m` in EPSG:27700 / WGS84 coordinates.
    Updates `user_survey_*` fields and routes `field_rejected_confounder` to
    `flagged_modern_or_karst_confounder`.
    """
    if annotations is None:
        annotations = load_community_annotations()
    if not annotations:
        return 0

    by_gref: Dict[str, Dict] = {}
    for ann in annotations:
        if ann.get("os_grid_ref"):
            by_gref[_norm_grid_ref(ann["os_grid_ref"])] = ann

    matched_count = 0
    for cand in candidates:
        gref = _norm_grid_ref(cand.get("os_grid_ref"))
        ann = by_gref.get(gref)
        if ann is None:
            ce = cand.get("easting")
            cn = cand.get("northing")
            clat = cand.get("lat")
            clon = cand.get("lon")
            best_d = match_radius_m + 1.0
            for a in annotations:
                if ce is not None and cn is not None and "easting" in a and "northing" in a:
                    d = math.hypot(float(ce) - float(a["easting"]), float(cn) - float(a["northing"]))
                    if d <= match_radius_m and d < best_d:
                        best_d = d
                        ann = a
                elif clat is not None and clon is not None and "lat" in a and "lon" in a:
                    # Approx metres in UK
                    d_lat = (float(clat) - float(a["lat"])) * 111132.0
                    d_lon = (float(clon) - float(a["lon"])) * 67000.0
                    d = math.hypot(d_lat, d_lon)
                    if d <= match_radius_m and d < best_d:
                        best_d = d
                        ann = a
        if ann is None:
            continue

        matched_count += 1
        status = ann.get("status") or "priority_field_target"
        cand["user_survey_status"] = status
        cand["user_survey_status_label"] = SURVEY_STATUS_LABELS.get(status, status)
        cand["user_survey_comment"] = ann.get("comment") or ""
        cand["user_survey_observer"] = ann.get("observer") or ""
        cand["user_survey_date"] = ann.get("date") or ""
        cand["user_survey_reference"] = ann.get("reference") or ""

        if status == "field_rejected_confounder":
            cand["context_category"] = "flagged_modern_or_karst_confounder"
            raw_c = float(cand.get("raw_unet_confidence", cand.get("confidence", 0.0)))
            cand["raw_unet_confidence"] = round(raw_c, 4)
            cand["confidence"] = round(raw_c * 0.35, 4)
            cand["vlm_verdict_tier"] = "OBVIOUS_FALSE_POSITIVE"
            reason_txt = f"Field / User Survey Rejection: {ann.get('comment') or 'Confirmed false positive / pond'}"
            cand["vlm_downrate_reason"] = reason_txt
            confs = list(cand.get("osm_confounder_matches") or [])
            if not any(c.get("osm_id") == "survey/field_rejected" for c in confs):
                confs.insert(
                    0,
                    {
                        "osm_id": "survey/field_rejected",
                        "subtype": "field_survey_rejected_confounder",
                        "name": reason_txt,
                        "wikidata": None,
                        "distance_m": 0.0,
                        "inside_polygon": True,
                    },
                )
                cand["osm_confounder_matches"] = confs
        elif status in {"field_verified_monument", "historic_survey_confirmed", "priority_field_target"}:
            if (
                status in {"field_verified_monument", "historic_survey_confirmed"}
                and cand.get("context_category") == "flagged_modern_or_karst_confounder"
            ):
                cand["context_category"] = "unidentified_candidate"
            raw_c = float(cand.get("raw_unet_confidence", cand.get("confidence", 0.0)))
            cand["raw_unet_confidence"] = round(raw_c, 4)
            boosted_c = round(min(0.9995, 0.9988 + 0.007 * max(0.0, raw_c - 0.90)), 4)
            if float(cand.get("confidence", 0.0)) < boosted_c:
                cand["confidence"] = boosted_c
            if status in {"field_verified_monument", "historic_survey_confirmed"}:
                cand["vlm_verdict_tier"] = "PLAUSIBLE_MONUMENT"
                cand["vlm_downrate_reason"] = None

    return matched_count

