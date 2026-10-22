# megalith-lidar

Scale-space terrain segmentation and calibrated national survey pipeline for recognising prehistoric earthworks and megalithic monuments in Environment Agency 1 m UK LiDAR data.

Inspired by landscape archaeologist Adam Morgan Ibbotson's video on LiDAR survey in the Thornborough Henges landscape ([https://youtu.be/D_R0bgNgY2o](https://youtu.be/D_R0bgNgY2o)). This repository provides statistical terrain screening, calibrated probability maps, and open GeoJSON/LiDAR diagnostic outputs so archaeologists and Historic Environment Record (HER) researchers can inspect and interpret the results.

Live out-of-sample survey map and diagnostic viewer: [https://glassontin.github.io/megalith-lidar/](https://glassontin.github.io/megalith-lidar/)

## Design and Bias Controls

Remote-sensing models for archaeology fail when they overfit to regional elevation, solar hillshade azimuth, synthetic target masks, or post-hoc score adjustments. This pipeline enforces six controls across feature extraction, dataset construction, and statistical evaluation:

1. Isotropic scale-space features (`megalith_lidar/terrain_features.py`): Raw elevation in metres OD is excluded so the network cannot memorise upland altitude. Traditional directional hillshading is also excluded because a fixed solar azimuth introduces compass bias. Each 1 m Digital Terrain Model (DTM) and Last-Return Digital Surface Model (LZ_DSM) pair is transformed into a 6-channel representation:
   - Fine local relief bandpass (`z - G_4(z)`, σ = 4 m) for orthostats, kerbs, and ring-ditch banks.
   - Monument-scale bandpass (`G_4(z) - G_16(z)`, σ₁ = 4 m, σ₂ = 16 m) for barrow mounds, henge banks, and chambered tombs (⌀ 8 m to 45 m).
   - Meso-scale topographic context (`G_16(z) - G_64(z)`, σ₁ = 16 m, σ₂ = 64 m) for ridge, saddle, and terrace siting.
   - Isotropic gradient magnitude (`|∇z| = sqrt((∂z/∂x)² + (∂z/∂y)²)`).
   - Isotropic Laplacian curvature (`∇²z = ∂²z/∂x² + ∂²z/∂y²`).
   - Log-compressed above-ground surface prominence (`log(1 + max(0, DSM - DTM))`).
   Unit tests in `tests/test_pipeline.py` verify altitude shift invariance and exact equivariance under the 8-element dihedral group D₄ (rotations by 0°, 90°, 180°, 270° and reflections).

2. Exact polygon supervision (`megalith_lidar/catalog.py`): Ground-truth masks are rasterised directly from the 20,003 Historic England National Heritage List for England (NHLE Layer 6) Scheduled Monument polygons in British National Grid (`EPSG:27700`) at 1 m resolution, with no synthetic circular radius inflation. Monument titles are categorised into three target morphological classes (`megalithic_tomb_long_barrow`, `stone_circle_henge_alignment`, `round_barrow_cairn`) and a non-target class (`other_scheduled_monument`) using generic morphological terms.

3. Hash-based 20 km spatial block split (`megalith_lidar/dataset.py`): Every 256 m × 256 m tile is assigned to `train` (75%), `val` (12.5%), or `test` (12.5%) purely from the SHA-256 digest of its 20 km × 20 km British National Grid block coordinates `(E // 20000, N // 20000)`. No manual overrides or pinned blocks exist. Positive tiles receive a deterministic uniform spatial offset in `[-56 m, +56 m]` so the network cannot learn a centre-of-tile prior.

4. Three negative null distributions (`megalith_lidar/dataset.py`): To test whether the model separates prehistoric monuments from background terrain rather than upland roughness, negatives are drawn from three explicit distributions:
   - `random_national` (`n = 420`): Sampled uniformly across all land-bearing 20 km × 20 km blocks of England (1 weight per block, independent of monument density), at least 500 m from any Scheduled Monument.
   - `matched_environment` (`n = 419`): Sampled inside the same 20 km × 20 km blocks as positive monuments (`400 m` to `1,600 m` away, outside all Scheduled Monuments), matching local geology, slope, and LiDAR flight characteristics.
   - `anthropogenic_non_target` (`n = 216`): Non-prehistoric earthworks and structures (medieval priories, industrial spoil heaps, World War II batteries, post-medieval quarries).

5. Validation-only Platt calibration and threshold locking (`megalith_lidar/stats.py`): Logistic Platt scaling (`P(y=1|s) = σ(a·s + b)`) and operating thresholds (`1%`, `5%`, and `10%` false positive rate, plus Youden's `J`) are fitted exclusively on the validation split (`n = 353` tiles) and evaluated once on the held-out test split (`n = 263` tiles) with 8-way D₄ test-time augmentation.

6. Unaltered model probabilities and OpenStreetMap spatial cross-check (`megalith_lidar/scanner.py`, `megalith_lidar/osm_context.py`): Connected components are extracted directly from the calibrated probability map without heuristic topographic bonuses. Each candidate is then cross-checked against NHLE Layer 6 polygons and the OpenStreetMap v0.6 API (`https://api.openstreetmap.org/api/0.6/map.json`) and labelled as one of four categories without mutating its probability score:
   - `rediscovered_nhle_scheduled` (within 45 m of a Scheduled Monument polygon)
   - `rediscovered_osm_archaeology` (within 45 m of an unscheduled OSM `man_made=cairn`, `historic=megalith|archaeological_site|tumulus|barrow`, or `megalith_type=*`)
   - `flagged_modern_or_karst_confounder` (inside or within 40 m of a mapped quarry, sinkhole, embankment, covered reservoir, golf feature, landscaped park building/folly, or railway earthwork)
   - `unidentified_candidate` (outside NHLE Scheduled Monuments, outside OSM recorded archaeology, and outside mapped OSM confounders)

## Held-Out Test Split Evaluation

Evaluated on `n = 263` held-out test tiles (`143` positives, `120` negatives across non-overlapping 20 km × 20 km BNG blocks) using 1,000 stratified bootstrap resamples for 95% confidence intervals (`checkpoints/evaluation_metrics_v2_unbiased.json`):

| Metric / Negative Null Comparison | Estimate | 95% Bootstrap CI | Sample Size |
| :--- | :---: | :---: | :---: |
| Overall Test ROC-AUC (all nulls combined) | `0.8393` | `[0.7909, 0.8821]` | `143 pos vs 120 neg` |
| Overall Test PR-AUC | `0.8902` | `[0.8572, 0.9216]` | `143 pos vs 120 neg` |
| Scalar local-relief baseline ROC-AUC | `0.5506` | `[0.4832, 0.6210]` | `143 pos vs 120 neg` |
| Positives vs `random_national` null ROC-AUC | `0.8745` | `[0.8220, 0.9203]` | `143 pos vs 53 neg` |
| Positives vs `matched_environment` null ROC-AUC | `0.8156` | `[0.7532, 0.8714]` | `143 pos vs 51 neg` |
| Positives vs `anthropogenic_non_target` null ROC-AUC | `0.7987` | `[0.6984, 0.8864]` | `143 pos vs 16 neg` |
| Calibration Brier score (test split) | `0.1590` | — | `n = 263` |
| Expected Calibration Error (ECE, 10 bins) | `0.0799` | — | `n = 263` |

### Out-of-Sample Operating Points (Locked on Validation Split)

| Validation Target | Locked Tile Threshold | Test TPR (Recall) | Test Overall FPR | FPR on `random_national` | FPR on `matched_env` | FPR on `anthropogenic` | Positive Likelihood Ratio (`LR+`) | Prior-Adjusted PPV (`π = 10⁻³`) | Prior-Adjusted PPV (`π = 10⁻²`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fpr_0.01` | `0.9428` | `39.16%` | `0.83%` | `0.00%` | `0.00%` | `6.25%` | `47.18×` | `4.51%` | `32.28%` |
| `fpr_0.05` | `0.8343` | `51.75%` | `3.33%` | `3.77%` | `1.96%` | `6.25%` | `15.54×` | `1.53%` | `13.57%` |
| `fpr_0.10` | `0.6593` | `62.24%` | `5.83%` | `3.77%` | `3.92%` | `18.75%` | `10.68×` | `1.06%` | `9.73%` |
| `max_youden_j` | `0.6617` | `62.24%` | `5.83%` | `3.77%` | `3.92%` | `18.75%` | `10.68×` | `1.06%` | `9.73%` |

## Limitations and Unverified Scope

1. Base-rate constraint in national screening: Even at `LR+ = 47.18×` (`fpr_0.01`, `Test FPR = 0.83%`), if true unrecorded monuments occur in only `π = 10⁻³` of 256 m × 256 m windows nationally, Bayes' rule bounds the raw single-window positive predictive value to `4.51%` before spatial context filtering.
2. Unmapped sub-decimetre confounders: While the OpenStreetMap v0.6 spatial cross-check flags mapped quarries, sinkholes, embankments, reservoirs, and landscaped garden buildings (such as the 18th-century Temple of Venus mound in West Wycombe Park), unmapped post-medieval clearance cairns, peat-cutting mounds, and unmapped karst shakeholes on limestone moorland cannot be separated from prehistoric round cairns by 1 m LiDAR geometry alone and require Historic Environment Record (HER) or field verification.
3. Geographic coverage: The current training catalog and WCS endpoints cover England (`Environment Agency 1 m Composite DTM / LZ_DSM` and `Historic England NHLE Layer 6`). Extending to Wales (`DataMapWales` / `Cadw`), Scotland (`Scottish Remote Sensing Portal` / `HES`), and Northern Ireland requires integrating their respective LiDAR WCS feeds and scheduled monument polygon registries.

## Repository Layout

- `megalith_lidar/terrain_features.py`: 6-channel isotropic scale-space bandpass pyramid and differential invariants.
- `megalith_lidar/catalog.py`: Historic England NHLE Layer 6 `EPSG:27700` polygon downloader, taxonomy classifier, and 1 m polygon rasteriser.
- `megalith_lidar/lidar_wcs.py`: Environment Agency 1 m DTM and 1 m LZ_DSM WCS client with disk caching and nodata validation.
- `megalith_lidar/dataset.py`: SHA-256 20 km BNG block splitter and multi-null dataset builder (`random_national`, `matched_environment`, `anthropogenic_non_target`).
- `megalith_lidar/model.py`: Reflection-padded `MegalithUNet` (`ResNet-34` encoder, multi-scale decoder, MIL negative-peak suppression).
- `megalith_lidar/stats.py`: Validation Platt scaling, operating-point selection, Bayes prior-adjusted PPV, and stratified bootstrap 95% CIs.
- `megalith_lidar/osm_context.py`: Live OpenStreetMap API v0.6 bounding-box spatial index for unscheduled archaeology and modern/karst confounder annotation.
- `megalith_lidar/scanner.py`: Sliding-window regional scanner with 8-way D₄ test-time augmentation, spatial non-maximum suppression, and 4-panel PNG diagnostics.
- `scripts/train.py`: End-to-end dataset verification, training, validation calibration, and held-out test evaluation.
- `scripts/run_national_batch.py`: Resumable national batch scanner over out-of-sample 2 km × 2 km test squares (`uniform_national_null` and `heldout_test_landscape`), exporting GeoJSON and summary statistics to `docs/` for GitHub Pages.
- `index.html`: Self-contained Leaflet map and candidate diagnostic inspector deployed via GitHub Pages.

## Running Locally

```bash
# Run the unit test suite (shift invariance, D4 equivariance, hash split, Platt calibration, OSM cross-check)
PYTHONPATH=. .venv/bin/pytest -v tests/test_pipeline.py

# Train and evaluate on the 20 km block-split dataset
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train.py --data-dir /mnt/bulk/megalith-lidar --epochs 30 --batch-size 8 --grad-accum 2 --device cuda:1

# Run a scheduled batch of out-of-sample 2 km x 2 km squares and update docs/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/run_national_batch.py --max-squares 4 --device cuda:1 --git-push
```

## License

Released under the [GNU Affero General Public License v3.0](LICENSE) (`AGPL-3.0-or-later`).
Contains Environment Agency Open LiDAR data and Historic England National Heritage List for England data licensed under the Open Government Licence v3.0, and OpenStreetMap data licensed under the Open Database Licence (ODbL).
