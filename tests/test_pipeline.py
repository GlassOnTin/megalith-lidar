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



