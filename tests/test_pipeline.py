import numpy as np
import pytest
import torch

from megalith_lidar.arch_context import classify_vlm_note_downrating
from megalith_lidar.catalog import (
    CLASS_BARROW_CAIRN,
    CLASS_CIRCLE_HENGE_STONE,
    CLASS_MEGALITHIC_TOMB_LONG_BARROW,
    CLASS_OTHER_MONUMENT,
    ROMAN_CLASS_BACKGROUND,
    ROMAN_CLASS_FORT_CAMP_SETTLEMENT,
    ROMAN_CLASS_ROAD_AGGER_TRACKWAY,
    ROMAN_CLASS_VILLA_BUILDING,
    MonumentCatalog,
    MonumentRecord,
    bng_to_latlon,
    bng_to_os_grid_ref,
    classify_monument_name,
    classify_roman_monument_name,
)
from megalith_lidar.dataset import assign_spatial_split
from megalith_lidar.model import MegalithLoss, MegalithUNet
from megalith_lidar.osm_context import (
    OSMSpatialIndex,
    annotate_candidates_with_spatial_context,
    classify_osm_tags,
)
from megalith_lidar.scanner import extract_candidates_from_tile
from megalith_lidar.stats import (
    PlattCalibrator,
    bootstrap_metric_ci,
    compute_prior_adjusted_ppv,
    select_validation_thresholds,
)
from megalith_lidar.terrain_features import NUM_TERRAIN_CHANNELS, compute_terrain_features


def test_monument_taxonomy_has_no_proper_noun_hacks():
    assert classify_monument_name("Long barrow 800m south-east of Silbury Hill") == CLASS_MEGALITHIC_TOMB_LONG_BARROW
    assert classify_monument_name("Megalithic tomb 200m SW of Trevarthen") == CLASS_MEGALITHIC_TOMB_LONG_BARROW
    assert classify_monument_name("Portal dolmen and cairn") == CLASS_MEGALITHIC_TOMB_LONG_BARROW
    assert classify_monument_name("Chambered long cairn") == CLASS_MEGALITHIC_TOMB_LONG_BARROW

    assert classify_monument_name("Stone circle on the moor") == CLASS_CIRCLE_HENGE_STONE
    assert classify_monument_name("Standing stone 335m west of farm") == CLASS_CIRCLE_HENGE_STONE
    assert classify_monument_name("Henge enclosure and avenue") == CLASS_CIRCLE_HENGE_STONE
    assert classify_monument_name("Cursus monument central section") == CLASS_CIRCLE_HENGE_STONE
    assert classify_monument_name("Stone alignment and cairn") == CLASS_CIRCLE_HENGE_STONE

    assert classify_monument_name("Two bowl barrows on hill") == CLASS_BARROW_CAIRN
    assert classify_monument_name("Ring cairn 400m north of stream") == CLASS_BARROW_CAIRN
    assert classify_monument_name("Bell barrow and disc barrow") == CLASS_BARROW_CAIRN

    assert classify_monument_name("Cast iron footbridge at Perran Mill") == CLASS_OTHER_MONUMENT
    assert classify_monument_name("Limebrook Priory medieval ruins") == CLASS_OTHER_MONUMENT
    assert classify_monument_name("Heavy anti-aircraft battery") == CLASS_OTHER_MONUMENT


def test_spatial_split_is_pure_hash_without_pinned_blocks():
    # Ensure no hard-coded overrides exist and distribution across 1000 blocks matches 75/12.5/12.5
    counts = {"train": 0, "val": 0, "test": 0}
    for bx in range(50):
        for by in range(20):
            s = assign_spatial_split(bx * 20000.0 + 5000.0, by * 20000.0 + 5000.0)
            counts[s] += 1
    assert 700 <= counts["train"] <= 800
    assert 95 <= counts["val"] <= 155
    assert 95 <= counts["test"] <= 155


def test_terrain_features_shift_invariance_and_exact_dihedral_equivariance():
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:128, 0:128].astype(np.float32)
    hillside = 150.0 + 0.12 * x - 0.08 * y
    mound = 2.2 * np.exp(-((x - 52.0) ** 2 + (y - 71.0) ** 2) / (2.0 * 9.0**2))
    dtm = hillside + mound + rng.normal(0, 0.05, size=(128, 128)).astype(np.float32)
    dsm = dtm + np.abs(rng.normal(0, 0.2, size=(128, 128))).astype(np.float32)
    dsm[68:74, 49:55] += 2.5

    feats_base = compute_terrain_features(dtm, dsm)
    feats_shifted = compute_terrain_features(dtm + 500.0, dsm + 500.0)

    assert feats_base.shape == (NUM_TERRAIN_CHANNELS, 128, 128)
    assert feats_base.dtype == np.float32
    assert np.all(np.isfinite(feats_base))

    # 1. Strict altitude shift invariance
    np.testing.assert_allclose(feats_base, feats_shifted, atol=1e-4)

    # 2. Strict Dihedral (D4) Rotational & Reflection Equivariance:
    # Computing features on a rotated/flipped terrain MUST equal rotating/flipping the features!
    for k in (1, 2, 3):
        dtm_rot = np.rot90(dtm, k=k)
        dsm_rot = np.rot90(dsm, k=k)
        feats_from_rot = compute_terrain_features(dtm_rot, dsm_rot)
        feats_rot = np.rot90(feats_base, k=k, axes=(1, 2))
        np.testing.assert_allclose(feats_from_rot, feats_rot, atol=1e-4)

    dtm_flip = np.fliplr(dtm)
    dsm_flip = np.fliplr(dsm)
    feats_from_flip = compute_terrain_features(dtm_flip, dsm_flip)
    feats_flip = np.flip(feats_base, axis=2)
    np.testing.assert_allclose(feats_from_flip, feats_flip, atol=1e-4)


def test_catalog_rasterization_exact_polygons():
    rec1 = MonumentRecord(
        list_entry=1001,
        name="Long barrow on Down",
        easting=410000.0,
        northing=167000.0,
        area_ha=0.12,
        ngr="SU 10000 67000",
        cls=CLASS_MEGALITHIC_TOMB_LONG_BARROW,
        rings=[
            [
                [409980.0, 166990.0],
                [410020.0, 166990.0],
                [410020.0, 167010.0],
                [409980.0, 167010.0],
                [409980.0, 166990.0],
            ]
        ],
    )
    catalog = MonumentCatalog([rec1])
    mask = catalog.rasterize_tile(409872, 410128, 166872, 167128, size=256)
    assert mask.shape == (256, 256)
    assert mask[128, 128] == CLASS_MEGALITHIC_TOMB_LONG_BARROW
    # Outside the 40m x 20m polygon should be 0
    assert mask[128, 100] == 0


def test_calibration_and_prior_adjusted_statistics():
    rng = np.random.default_rng(123)
    y_val = np.array([0] * 200 + [1] * 100)
    logits_val = np.concatenate([rng.normal(-1.2, 1.0, 200), rng.normal(1.3, 1.0, 100)])

    cal = PlattCalibrator().fit(logits_val, y_val)
    probs = cal.predict_proba(logits_val)
    assert np.all((probs > 0.0) & (probs < 1.0))
    # Mean predicted probability on validation set should match empirical prevalence
    assert abs(probs.mean() - y_val.mean()) < 0.05

    thresholds = select_validation_thresholds(probs, y_val, target_fprs=(0.01, 0.05))
    assert "fpr_0.01" in thresholds and "fpr_0.05" in thresholds
    assert thresholds["fpr_0.01"]["threshold"] >= thresholds["fpr_0.05"]["threshold"]

    # Bayes prior-adjusted PPV check: at TPR=0.50, FPR=0.01, prior=0.01 -> PPV = (0.5*0.01)/(0.5*0.01 + 0.01*0.99) = 0.3356
    ppv = compute_prior_adjusted_ppv(tpr=0.50, fpr=0.01, prior=0.01)
    assert abs(ppv - 0.3356) < 1e-3

    # Bootstrap CI check
    est, lo, hi = bootstrap_metric_ci(y_val, probs, lambda yt, yp: float(np.mean(yp[yt == 1]) - np.mean(yp[yt == 0])), n_boot=200)
    assert lo < est < hi


def test_model_forward_and_loss():
    model = MegalithUNet(in_channels=NUM_TERRAIN_CHANNELS, num_classes=4, pretrained=False)
    x = torch.randn(2, NUM_TERRAIN_CHANNELS, 256, 256)
    out = model(x)
    assert out["seg_logits"].shape == (2, 1, 256, 256)
    assert out["cls_seg_logits"].shape == (2, 4, 256, 256)
    assert out["tile_logits"].shape == (2, 1)
    assert out["type_logits"].shape == (2, 4)

    criterion = MegalithLoss()
    target_mask = torch.zeros(2, 256, 256, dtype=torch.long)
    target_mask[0, 100:130, 100:130] = CLASS_MEGALITHIC_TOMB_LONG_BARROW
    target_mask[0, 80:100, 80:100] = 255  # IGNORE_INDEX legal-buffer pixels
    target_tile = torch.tensor([1.0, 0.0])
    target_type = torch.tensor([CLASS_MEGALITHIC_TOMB_LONG_BARROW, 0], dtype=torch.long)

    loss, metrics = criterion(out, target_mask, target_tile, target_type)
    assert torch.isfinite(loss)
    loss.backward()


def test_osm_context_annotation_does_not_mutate_probabilities():
    assert classify_osm_tags({"man_made": "cairn", "name": "White Raise"})[0] == "osm_archaeology"
    assert classify_osm_tags({"historic": "megalith", "megalith_type": "stone_circle"})[0] == "osm_archaeology"
    assert classify_osm_tags({"building": "temple", "historic": "monument", "name": "Temple of Venus"})[0] == "osm_confounder"
    assert classify_osm_tags({"natural": "sinkhole"})[0] == "osm_confounder"
    assert classify_osm_tags({"landuse": "quarry", "abandoned": "yes"})[0] == "osm_confounder"
    assert classify_osm_tags({"power": "pole"})[0] == "osm_confounder"
    assert classify_osm_tags({"barrier": "hedge"})[0] == "osm_confounder"
    assert classify_osm_tags({"highway": "track"})[0] == "osm_confounder"

    lat_cairn, lon_cairn = bng_to_latlon(348860.0, 522420.0)
    lat_sink, lon_sink = bng_to_latlon(348897.0, 522085.0)
    payload = {
        "elements": [
            {"type": "node", "id": 1, "lat": lat_cairn, "lon": lon_cairn, "tags": {"man_made": "cairn", "name": "Moor Cairn"}},
            {"type": "node", "id": 2, "lat": lat_sink, "lon": lon_sink, "tags": {"natural": "sinkhole"}},
        ]
    }
    idx = OSMSpatialIndex.from_osm_json(payload)
    cands = [
        {"easting": 348862.0, "northing": 522418.0, "confidence": 0.9123, "is_known_monument": False},
        {"easting": 348898.0, "northing": 522084.0, "confidence": 0.8765, "is_known_monument": False},
        {"easting": 349500.0, "northing": 522900.0, "confidence": 0.8456, "is_known_monument": False},
        {"easting": 348768.0, "northing": 522384.0, "confidence": 0.9650, "is_known_monument": True},
        {
            "easting": 349600.0,
            "northing": 522900.0,
            "confidence": 0.8200,
            "est_surface_prominence_m": 12.5,
            "is_known_monument": False,
        },
        {
            "easting": 349700.0,
            "northing": 522900.0,
            "confidence": 0.8100,
            "est_local_relief_m": -1.15,
            "max_positive_relief_m": 0.08,
            "is_known_monument": False,
        },
        {
            "easting": 354734.0,
            "northing": 576498.0,
            "confidence": 0.9718,
            "area_m2": 427.0,
            "elongation": 1.45,
            "est_local_relief_m": 1.06,
            "max_positive_relief_m": 1.06,
            "centroid_local_relief_m": 0.67,
            "patch_min_relief_m": -0.25,
            "ridge_continuation_span_m": 138.2,
            "ridge_continuation_elongation": 6.64,
            "adjacent_channel_span_m": 111.0,
            "is_known_monument": False,
        },
    ]
    annotate_candidates_with_spatial_context(cands, idx)

    assert cands[0]["context_category"] == "rediscovered_osm_archaeology"
    assert cands[0]["confidence"] == 0.9123
    assert cands[1]["context_category"] == "flagged_modern_or_karst_confounder"
    assert cands[1]["confidence"] == 0.8765
    assert cands[2]["context_category"] == "unidentified_candidate"
    assert cands[2]["confidence"] == 0.8456
    assert cands[3]["context_category"] == "rediscovered_nhle_scheduled"
    assert cands[3]["confidence"] == 0.9650
    assert cands[4]["context_category"] == "flagged_modern_or_karst_confounder"
    assert cands[5]["context_category"] == "flagged_modern_or_karst_confounder"
    assert cands[6]["context_category"] == "flagged_modern_or_karst_confounder"
    assert cands[6]["osm_confounder_matches"][0]["osm_id"] == "lidar/meander_scroll_bar"


def test_roman_taxonomy_and_sub_canopy_context():
    assert classify_roman_monument_name("Roman villa in Bottom Copse") == ROMAN_CLASS_VILLA_BUILDING
    assert classify_roman_monument_name("Romano-Celtic temple 300m west of farm") == ROMAN_CLASS_VILLA_BUILDING
    assert classify_roman_monument_name("Length of Roman road in Bradley Wood") == ROMAN_CLASS_ROAD_AGGER_TRACKWAY
    assert classify_roman_monument_name("Roman road agger and flanking ditches") == ROMAN_CLASS_ROAD_AGGER_TRACKWAY
    assert classify_roman_monument_name("Roman fort and marching camp") == ROMAN_CLASS_FORT_CAMP_SETTLEMENT
    assert classify_roman_monument_name("Romano-British settlement and field system") == ROMAN_CLASS_FORT_CAMP_SETTLEMENT
    assert classify_roman_monument_name("Roman milestone 100m south of bridge") == ROMAN_CLASS_BACKGROUND
    assert classify_roman_monument_name("Two bowl barrows on hill") == ROMAN_CLASS_BACKGROUND

    rec_villa = MonumentRecord(
        list_entry=2002,
        name="Roman villa in Bottom Copse",
        easting=410000.0,
        northing=167000.0,
        area_ha=0.25,
        ngr="SU 10000 67000",
        cls=CLASS_OTHER_MONUMENT,
        rings=[
            [
                [409970.0, 166970.0],
                [410030.0, 166970.0],
                [410030.0, 167030.0],
                [409970.0, 167030.0],
                [409970.0, 166970.0],
            ]
        ],
    )
    assert rec_villa.is_roman_or_road
    assert rec_villa.roman_cls == ROMAN_CLASS_VILLA_BUILDING
    cat = MonumentCatalog([rec_villa])
    rmask = cat.rasterize_roman_tile(409872, 410128, 166872, 167128, size=256)
    assert rmask[128, 128] == ROMAN_CLASS_VILLA_BUILDING

    # Woodland canopy (nDSM >= 2.8m) and natural=wood should NOT flag a Roman earthwork as a confounder
    lat_wood, lon_wood = bng_to_latlon(410000.0, 167000.0)
    osm_idx = OSMSpatialIndex.from_osm_json(
        {"elements": [{"type": "node", "id": 10, "lat": lat_wood, "lon": lon_wood, "tags": {"natural": "wood"}}]}
    )
    roman_cand = [
        {
            "easting": 410000.0,
            "northing": 167000.0,
            "confidence": 0.92,
            "est_local_relief_m": 0.55,
            "max_positive_relief_m": 0.55,
            "est_surface_prominence_m": 11.2,
            "mean_slope_tangent": 0.09,
            "is_known_monument": False,
            "target_domain": "roman",
        }
    ]
    annotate_candidates_with_spatial_context(roman_cand, osm_idx, target_domain="roman")
    assert roman_cand[0]["context_category"] == "unidentified_candidate"

    # Embanked slurry lagoon / pond / gravel pit / levelled court (e.g. SE 23030 83779, SP 31806 97499, NU 14479 20691, SU 61106 50083, SU 60156 51931)
    lagoon_cands = [
        {
            "easting": 423030.4,
            "northing": 483779.3,
            "confidence": 0.9291,
            "area_m2": 4133.0,
            "est_local_relief_m": 1.50,
            "max_positive_relief_m": 1.50,
            "centroid_local_relief_m": 0.20,
            "est_surface_prominence_m": 1.23,
            "soilmark_core_lum_diff": -0.0873,
            "cropmark_ring_exg_diff": 0.0916,
            "is_known_monument": False,
            "target_domain": "roman",
        },
        {
            "easting": 431806.0,
            "northing": 297499.0,
            "confidence": 0.927,
            "area_m2": 9646.0,
            "est_local_relief_m": 1.89,
            "max_positive_relief_m": 1.89,
            "centroid_local_relief_m": -0.13,
            "vlm_shape_verdict": "RECTILINEAR_OR_U_BUND",
            "is_known_monument": False,
            "target_domain": "roman",
        },
        {
            "easting": 460156.0,
            "northing": 151931.0,
            "confidence": 0.905,
            "area_m2": 1928.0,
            "est_local_relief_m": 0.92,
            "max_positive_relief_m": 0.92,
            "centroid_local_relief_m": 0.04,
            "vlm_shape_verdict": "RECTILINEAR_OR_U_BUND",
            "is_known_monument": False,
            "target_domain": "roman",
        },
        {
            "easting": 504759.0,
            "northing": 350363.0,
            "confidence": 0.921,
            "area_m2": 1980.0,
            "est_local_relief_m": 0.93,
            "max_positive_relief_m": 0.93,
            "centroid_local_relief_m": -0.45,
            "patch_min_relief_m": -0.60,
            "vlm_shape_verdict": "ROUNDED_MOUND_OR_RING",
            "is_known_monument": False,
            "target_domain": "roman",
        },
        {
            "easting": 465419.1,
            "northing": 211746.5,
            "confidence": 0.9794,
            "area_m2": 555.0,
            "est_local_relief_m": 1.13,
            "max_positive_relief_m": 1.13,
            "centroid_local_relief_m": 0.91,
            "patch_min_relief_m": -0.32,
            "est_surface_prominence_m": 0.10,
            "soilmark_core_lum_diff": -0.0989,
            "cropmark_ring_exg_diff": -0.1507,
            "vlm_shape_verdict": "ROUNDED_MOUND_OR_RING",
            "is_known_monument": False,
            "target_domain": "prehistoric",
        },
    ]
    annotate_candidates_with_spatial_context(lagoon_cands, None, target_domain="roman")
    for lc in lagoon_cands:
        assert lc["context_category"] == "flagged_modern_or_karst_confounder"
        assert lc["osm_confounder_matches"][0]["osm_id"] == "lidar/slurry_lagoon_or_bund"

    # Test Qwen3.8 VLM note downrating for obvious false positives vs uncertain vs plausible
    tier1, mult1, _ = classify_vlm_note_downrating(
        "The morphology strongly indicates a ring ditch or a modern military emplacement.",
        "ROUNDED_MOUND_OR_RING",
        {"predicted_class": "megalithic_tomb_long_barrow", "elongation": 1.19},
    )
    assert tier1 == "OBVIOUS_FALSE_POSITIVE"
    assert mult1 < 0.50

    tier2, mult2, _ = classify_vlm_note_downrating(
        "The model's long barrow classification is rejected due to the lack of rectilinear morphology.",
        "ROUNDED_MOUND_OR_RING",
        {"predicted_class": "megalithic_tomb_long_barrow", "elongation": 1.07},
    )
    assert tier2 == "UNCERTAIN_OR_DEGRADED"
    assert 0.70 < mult2 < 0.95

    tier3, mult3, _ = classify_vlm_note_downrating(
        "The 1m LiDAR close-up reveals a well-defined circular bowl barrow with a surrounding quarry ditch.",
        "ROUNDED_MOUND_OR_RING",
        {"predicted_class": "round_barrow_cairn", "elongation": 1.06},
    )
    assert tier3 == "PLAUSIBLE_MONUMENT"
    assert mult3 == 1.0


def test_footprint_geometry_class_conditioning_and_enclosure_blob_gate():
    # 1. Circular prehistoric mound (elongation < 1.45) must not be classified as megalithic_tomb_long_barrow
    prob_map = np.zeros((256, 256), dtype=np.float32)
    yy, xx = np.ogrid[:256, :256]
    circle_mask = (yy - 128) ** 2 + (xx - 128) ** 2 <= 9**2
    prob_map[circle_mask] = 0.96

    # Force pixel class head to prefer class 1 (long barrow) even though footprint is circular
    cls_prob_map = np.zeros((4, 256, 256), dtype=np.float32)
    cls_prob_map[1] = 0.80
    cls_prob_map[3] = 0.15

    raw_features = np.zeros((NUM_TERRAIN_CHANNELS, 256, 256), dtype=np.float32)
    raw_features[0, circle_mask] = 0.45  # +0.45m mound relief
    raw_features[3, circle_mask] = 0.06  # gentle slope

    cat = MonumentCatalog([])
    cands = extract_candidates_from_tile(
        prob_map=prob_map,
        cls_prob_map=cls_prob_map,
        raw_features=raw_features,
        e_min=400000.0,
        e_max=400256.0,
        n_min=130000.0,
        n_max=130256.0,
        catalog=cat,
        prob_threshold=0.70,
        target_domain="prehistoric",
    )
    assert len(cands) == 1
    assert cands[0]["elongation"] < 1.25
    assert cands[0]["predicted_class"] == "round_barrow_cairn"

    # 2. Oversized whole-tile enclosure blob (> 8,500 m^2) must be flagged as confounder
    enc_cands = [
        {
            "easting": 341344.0,
            "northing": 347344.0,
            "confidence": 0.97,
            "area_m2": 32708.0,
            "elongation": 1.33,
            "est_local_relief_m": 2.02,
            "max_positive_relief_m": 2.02,
            "est_surface_prominence_m": 0.1,
            "mean_slope_tangent": 0.045,
            "is_known_monument": False,
        },
        {
            "easting": 286384.0,
            "northing": 109152.0,
            "confidence": 0.86,
            "area_m2": 4712.0,
            "elongation": 1.03,
            "est_local_relief_m": 0.83,
            "max_positive_relief_m": 0.83,
            "patch_min_relief_m": -0.20,
            "est_surface_prominence_m": 0.27,
            "mean_slope_tangent": 0.045,
            "is_known_monument": False,
        },
    ]
    annotate_candidates_with_spatial_context(enc_cands, None, target_domain="enclosure")
    assert enc_cands[0]["context_category"] == "flagged_modern_or_karst_confounder"
    assert enc_cands[1]["context_category"] == "unidentified_candidate"


def test_v4_d4_geometric_moments_and_legacy_checkpoint_compat():
    # 1. Verify D4 invariance and shape discrimination of _compute_d4_geometric_moments
    yy, xx = np.ogrid[:256, :256]
    circle_logits = np.full((1, 1, 256, 256), -8.0, dtype=np.float32)
    circle_logits[0, 0, (yy - 128) ** 2 + (xx - 128) ** 2 <= 14**2] = 6.0

    elong_logits = np.full((1, 1, 256, 256), -8.0, dtype=np.float32)
    elong_logits[0, 0, ((yy - 128) / 32.0) ** 2 + ((xx - 128) / 7.0) ** 2 <= 1.0] = 6.0

    ring_logits = np.full((1, 1, 256, 256), -8.0, dtype=np.float32)
    r2 = (yy - 128) ** 2 + (xx - 128) ** 2
    ring_logits[0, 0, (r2 >= 18**2) & (r2 <= 26**2)] = 6.0

    t_circle = torch.from_numpy(circle_logits)
    t_elong = torch.from_numpy(elong_logits)
    t_ring = torch.from_numpy(ring_logits)

    g_circle = MegalithUNet._compute_d4_geometric_moments(t_circle)
    g_elong = MegalithUNet._compute_d4_geometric_moments(t_elong)
    g_elong_rot = MegalithUNet._compute_d4_geometric_moments(torch.rot90(t_elong, 1, [2, 3]))
    g_ring = MegalithUNet._compute_d4_geometric_moments(t_ring)

    # Exact D4 rotational invariance on all 4 geometric moments
    assert torch.allclose(g_elong, g_elong_rot, atol=1e-4)
    # Elongated long barrow has significantly higher log_elongation than circular barrow
    assert float(g_elong[0, 1]) > float(g_circle[0, 1]) + 0.5
    # Hollow ring-work has higher ring_hollowness than solid circular mound
    assert float(g_ring[0, 2]) > float(g_circle[0, 2])

    # 2. Verify legacy v3 state_dict (without geom_* keys) loads cleanly with strict=True
    model = MegalithUNet(in_channels=NUM_TERRAIN_CHANNELS, num_classes=4, pretrained=False)
    legacy_sd = {k: v for k, v in model.state_dict().items() if not k.startswith("geom_")}
    model.load_state_dict(legacy_sd, strict=True)


def test_scale_space_cropmark_features_d4_and_ring_ditch():
    from megalith_lidar.aerial_cropmarks import (
        compute_candidate_cropmark_metrics,
        compute_cropmark_features,
        render_cropmark_false_color,
    )

    # Synthetic arable field with a greener concentric ring-ditch cropmark around a parched central mound
    yy, xx = np.ogrid[:256, :256]
    dist = np.hypot(xx - 128.0, yy - 128.0)
    rgb = np.full((256, 256, 3), 110, dtype=np.uint8)
    # Central chalk/parched mound (r <= 6m): brighter red/green
    core = dist <= 6.0
    rgb[core, 0] = 145
    rgb[core, 1] = 135
    rgb[core, 2] = 105
    # Surrounding humic ring-ditch cropmark (7m <= r <= 12m): higher Excess-Green (ExG)
    ring = (dist >= 7.0) & (dist <= 12.0)
    rgb[ring, 0] = 85
    rgb[ring, 1] = 140
    rgb[ring, 2] = 75

    feats = compute_cropmark_features(rgb)
    assert feats.shape == (2, 256, 256)

    # Verify exact D4 rotational equivariance
    rgb_rot = np.rot90(rgb, k=1, axes=(0, 1))
    feats_rot = compute_cropmark_features(rgb_rot)
    assert np.allclose(feats_rot, np.rot90(feats, k=1, axes=(1, 2)), atol=1e-5)

    # Verify positive ring-ditch ΔExG anomaly and positive core ΔL soilmark contrast
    metrics = compute_candidate_cropmark_metrics(rgb, cx_px=128.0, cy_px=128.0, area_m2=333.0)
    assert metrics["cropmark_ring_exg_diff"] > 0.02
    assert metrics["soilmark_core_lum_diff"] > 0.01

    vis = render_cropmark_false_color(rgb)
    assert vis.shape == (256, 256, 3) and vis.dtype == np.uint8


def test_megalith_unet_8channel_cropmark_branch():
    """Verify 8-channel multi-modal forward/backward and exact 6-channel zero-init equivalence."""
    torch.manual_seed(42)
    model = MegalithUNet(pretrained=False)
    model.eval()
    x6 = torch.randn(2, 6, 128, 128)
    x8_zero = torch.cat([x6, torch.zeros(2, 2, 128, 128)], dim=1)
    with torch.no_grad():
        out6 = model(x6)
        out8_zero = model(x8_zero)
    # At zero-initialization of the cropmark output projections, 6-channel and 8-channel outputs match
    assert torch.allclose(out6["seg_logits"], out8_zero["seg_logits"], atol=1e-5)
    assert torch.allclose(out6["tile_logits"], out8_zero["tile_logits"], atol=1e-5)

    # Verify gradients flow through the cropmark branch during training
    model.train()
    x8 = torch.randn(2, 8, 128, 128, requires_grad=True)
    mask = torch.zeros(2, 128, 128, dtype=torch.long)
    mask[0, 48:64, 48:64] = 3
    tile_label = torch.tensor([1.0, 0.0])
    type_label = torch.tensor([3, 0], dtype=torch.long)
    criterion = MegalithLoss()
    out = model(x8)
    loss, _ = criterion(out, mask, tile_label, type_label)
    loss.backward()
    assert model.cropmark_seg_head[-1].weight.grad is not None
    assert model.cropmark_tile_out.weight.grad is not None


def test_environmental_and_geomorphic_false_positive_context():
    from megalith_lidar.arch_context import (
        build_deterministic_academic_context,
        classify_vlm_note_downrating,
        derive_false_positive_context_hints,
        infer_landscape_character_area,
    )
    from megalith_lidar.osm_context import (
        OSMParsedFeature,
        OSMSpatialIndex,
        annotate_candidates_with_spatial_context,
        is_active_holocene_coastal_dune_belt,
    )

    # 1. Ainsdale / Formby sand-dune belt recognition and downrating
    e_dune, n_dune = 329533.0, 409075.0
    assert is_active_holocene_coastal_dune_belt(e_dune, n_dune) is not None
    area_name = infer_landscape_character_area(e_dune, n_dune)
    assert "Ainsdale" in area_name and "sand-dune" in area_name

    dune_cand = {
        "os_grid_ref": "SD 29533 09075",
        "easting": e_dune,
        "northing": n_dune,
        "predicted_class": "round_barrow_cairn",
        "confidence": 0.96,
        "est_local_relief_m": 0.45,
        "mean_slope_tangent": 0.08,
        "area_m2": 320.0,
        "elongation": 1.15,
        "est_surface_prominence_m": 0.08,
    }
    annotated = annotate_candidates_with_spatial_context([dune_cand], osm_index=None)
    assert annotated[0]["context_category"] == "flagged_modern_or_karst_confounder"
    assert any(m["osm_id"] == "geomorph/coastal_sand_dunes" for m in annotated[0]["osm_confounder_matches"])

    dune_hints = derive_false_positive_context_hints(
        annotated[0],
        nearby_nhle=[],
        cluster={"landscape_area": area_name, "nhle_within_2km": 0, "prehistoric_within_2km": 0, "nhle_within_5km": 1},
    )
    assert any("sand-dune" in h and "cannot survive" in h for h in dune_hints)
    tier, mult, reason = classify_vlm_note_downrating("Test note.", "ROUNDED_MOUND_OR_RING", annotated[0])
    assert tier == "OBVIOUS_FALSE_POSITIVE" and mult == 0.35

    # 2. Meandering river scroll-bar / oxbow palaeochannel environmental context within 400 m
    river_feat = OSMParsedFeature(
        osm_id="way/999",
        category="osm_confounder",
        subtype="waterway=river",
        name="River Ribble",
        wikidata=None,
        coords_27700=[(365100.0, 435000.0), (365100.0, 435500.0)],
        is_closed_area=False,
        bbox_27700=(365100.0, 435000.0, 365100.0, 435500.0),
        match_radius_m=28.0,
    )
    idx = OSMSpatialIndex([river_feat])
    river_cand = {
        "os_grid_ref": "SD 65000 35200",
        "easting": 365000.0,
        "northing": 435200.0,
        "predicted_class": "round_barrow_cairn",
        "confidence": 0.94,
        "est_local_relief_m": 0.38,
        "mean_slope_tangent": 0.06,
        "area_m2": 290.0,
        "elongation": 1.30,
        "est_surface_prominence_m": 0.05,
        "ridge_continuation_span_m": 48.0,
        "adjacent_channel_span_m": 34.0,
    }
    annotate_candidates_with_spatial_context([river_cand], osm_index=idx)
    assert river_cand["context_category"] == "unidentified_candidate"
    assert len(river_cand["nearby_environmental_features"]) == 1
    assert river_cand["nearby_environmental_features"][0]["name"] == "River Ribble"

    cluster = {"landscape_area": infer_landscape_character_area(365000.0, 435200.0), "nhle_within_2km": 1, "prehistoric_within_2km": 1, "nhle_within_5km": 3}
    r_hints = derive_false_positive_context_hints(river_cand, nearby_nhle=[], cluster=cluster)
    assert any("meandering river" in h and "River Ribble" in h for h in r_hints)
    det_note = build_deterministic_academic_context(river_cand, nearby_nhle=[], cluster=cluster)
    assert "meandering river" in det_note and "River Ribble" in det_note


def test_community_and_historic_survey_annotations():
    from megalith_lidar.arch_context import (
        apply_community_annotations_to_candidates,
        load_community_annotations,
        parse_survey_annotations_payload,
    )

    # 1. Verify repository docs/community_annotations.json loads cleanly
    repo_anns = load_community_annotations()
    assert len(repo_anns) >= 5

    # 2. Verify GeoJSON FeatureCollection of historic survey data parses and matches by coordinate proximity
    geojson_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [216305.0, 81158.0]},
                "properties": {
                    "her_id": "MCO-9988",
                    "name": "Bodmin Moor Ring Cairn (1894 Survey)",
                    "source": "Cornwall HER",
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [351067.0, 537187.0]},
                "properties": {
                    "os_grid_ref": "NY 51067 37187",
                    "status": "field_rejected_confounder",
                    "comment": "Field survey confirmed farm pond",
                },
            },
        ],
    }
    parsed = parse_survey_annotations_payload(geojson_payload)
    assert len(parsed) == 2
    assert parsed[0]["status"] == "historic_survey_confirmed"
    assert parsed[0]["reference"] == "MCO-9988"

    cands = [
        {
            "os_grid_ref": "SX 16303 81157",
            "easting": 216302.9,
            "northing": 81157.1,
            "confidence": 0.9854,
            "context_category": "unidentified_candidate",
        },
        {
            "os_grid_ref": "NY 51067 37187",
            "easting": 351067.3,
            "northing": 537186.8,
            "confidence": 0.9610,
            "context_category": "unidentified_candidate",
        },
    ]
    matched = apply_community_annotations_to_candidates(cands, parsed)
    assert matched == 2
    assert cands[0]["user_survey_status"] == "historic_survey_confirmed"
    assert cands[0]["user_survey_reference"] == "MCO-9988"
    assert cands[1]["user_survey_status"] == "field_rejected_confounder"
    assert cands[1]["context_category"] == "flagged_modern_or_karst_confounder"
    assert cands[1]["confidence"] < 0.40









