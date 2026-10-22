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

2. Exact polygon supervision with legal-buffer refinement (`megalith_lidar/catalog.py`, `megalith_lidar/dataset.py`): Ground-truth masks are rasterised directly from the 20,003 Historic England National Heritage List for England (NHLE Layer 6) Scheduled Monument polygons in British National Grid (`EPSG:27700`) at 1 m resolution. On large legal scheduling polygons (`> 600 m²`), active local-relief earthworks (`|z - G_16(z)|` above the 75th polygon percentile or `0.20 m`, with `nDSM < 2.5 m`) are retained as positive pixels while flat surrounding pasture inside the scheduling boundary is masked with `IGNORE_INDEX = 255` and tall woodland canopy (`nDSM >= 2.5 m`) is labelled background (`0`).

3. Hash-based 20 km spatial block split (`megalith_lidar/dataset.py`): Every 256 m × 256 m tile is assigned to `train` (75%), `val` (12.5%), or `test` (12.5%) purely from the SHA-256 digest of its 20 km × 20 km British National Grid block coordinates `(E // 20000, N // 20000)`. No manual overrides or pinned blocks exist. Positive tiles receive a deterministic uniform spatial offset in `[-56 m, +56 m]` so the network cannot learn a centre-of-tile prior.

4. Five negative null distributions (`megalith_lidar/dataset.py`, `scripts/train.py`): To test whether the model separates prehistoric monuments from background terrain and modern rural confounders, negatives (`n = 1,347` tiles) are drawn from five explicit distributions:
   - `random_national` (`n = 420`): Sampled uniformly across all land-bearing 20 km × 20 km blocks of England (1 weight per block, independent of monument density), outside all Scheduled Monuments.
   - `matched_environment` (`n = 419`): Sampled inside the same 20 km × 20 km blocks as positive monuments (`400 m` to `1,600 m` away, outside all Scheduled Monuments), matching local geology, slope, and LiDAR flight characteristics.
   - `anthropogenic_non_target` (`n = 216`): Non-prehistoric earthworks and structures (medieval priories, industrial spoil heaps, World War II batteries, post-medieval quarries).
   - `woodland_canopy_and_pit` (`n = 160`): Monument-free woodland canopy (`>= 20%` `nDSM > 2.5 m`) and sunken marl/chalk/quarry pits (`0.5th` percentile `16 m` relief `<= -0.95 m`).
   - `trackside_and_field_boundary` (`n = 132`): Open agricultural land with hedgerow, trackside, and field-corner boundary relief bumps.

5. Validation-only Platt calibration and threshold locking (`megalith_lidar/stats.py`): Logistic Platt scaling (`P(y=1|s) = σ(a·s + b)`) and operating thresholds (`1%`, `5%`, and `10%` false positive rate, plus Youden's `J`) are fitted exclusively on the validation split (`n = 386` tiles) and evaluated once on the held-out test split (`n = 303` tiles) with 8-way D₄ test-time augmentation.

6. Unaltered model probabilities and OpenStreetMap + intrinsic LiDAR confounder cross-check (`megalith_lidar/scanner.py`, `megalith_lidar/osm_context.py`): Connected components are extracted directly from the calibrated probability map without heuristic topographic bonuses. Each candidate is then cross-checked against NHLE Layer 6 polygons, the OpenStreetMap v0.6 API (`https://api.openstreetmap.org/api/0.6/map.json`), and intrinsic 1 m LiDAR observables (tall canopy `nDSM >= 2.8 m`, steep silage/manure clamps `nDSM >= 1.1 m` with `slope >= 0.18`, sunken pits `relief < -0.15 m`, or interpolated TIN voids) and labelled as one of four categories without mutating its probability score:
   - `rediscovered_nhle_scheduled` (within 45 m of a Scheduled Monument polygon)
   - `rediscovered_osm_archaeology` (within 45 m of an unscheduled OSM `man_made=cairn`, `historic=megalith|archaeological_site|tumulus|barrow`, or `megalith_type=*`)
   - `flagged_modern_or_karst_confounder` (inside or within feature-specific radii of mapped quarries, sinkholes, farmyards, tracks/roads, hedges/fences, power poles, woodlands, military ranges, or steep surface clamps/pits)
   - `unidentified_candidate` (outside NHLE Scheduled Monuments, outside OSM recorded archaeology, and outside mapped/intrinsic confounders)

## Held-Out Test Split Evaluation

Evaluated on `n = 303` held-out test tiles (`143` positives, `160` negatives across non-overlapping 20 km × 20 km BNG blocks) using 1,000 stratified bootstrap resamples for 95% confidence intervals (`checkpoints/evaluation_metrics_v2_unbiased.json`):

| Metric / Negative Null Comparison | Estimate | 95% Bootstrap CI | Sample Size |
| :--- | :---: | :---: | :---: |
| Overall Test ROC-AUC (all 5 nulls combined) | `0.8675` | `[0.8248, 0.9070]` | `143 pos vs 160 neg` |
| Overall Test PR-AUC | `0.8623` | `[0.8180, 0.9042]` | `143 pos vs 160 neg` |
| Scalar local-relief baseline ROC-AUC | `0.5329` | `[0.4696, 0.5988]` | `143 pos vs 160 neg` |
| Positives vs `random_national` null ROC-AUC | `0.8976` | `[0.8527, 0.9379]` | `143 pos vs 53 neg` |
| Positives vs `trackside_and_field_boundary` null ROC-AUC | `0.8899` | `[0.8140, 0.9479]` | `143 pos vs 20 neg` |
| Positives vs `woodland_canopy_and_pit` null ROC-AUC | `0.8836` | `[0.8220, 0.9360]` | `143 pos vs 20 neg` |
| Positives vs `anthropogenic_non_target` null ROC-AUC | `0.8459` | `[0.7432, 0.9266]` | `143 pos vs 16 neg` |
| Positives vs `matched_environment` null ROC-AUC | `0.8280` | `[0.7660, 0.8795]` | `143 pos vs 51 neg` |
| Calibration Brier score (test split) | `0.1412` | — | `n = 303` |
| Expected Calibration Error (ECE, 10 bins) | `0.0525` | — | `n = 303` |

### Out-of-Sample Operating Points (Locked on Validation Split)

| Validation Target | Locked Tile Threshold | Test TPR (Recall) | Test Overall FPR | FPR on `random_national` | FPR on `matched_env` | FPR on `anthropogenic` | FPR on `woodland_pit` | FPR on `trackside` | Positive Likelihood Ratio (`LR+`) | Prior-Adjusted PPV (`π = 10⁻³`) | Prior-Adjusted PPV (`π = 10⁻²`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fpr_0.01` | `0.9153` | `38.46%` | `0.63%` | `0.00%` | `0.00%` | `6.25%` | `0.00%` | `0.00%` | `61.05×` | `5.76%` | `38.14%` |
| `fpr_0.05` | `0.8071` | `51.05%` | `3.12%` | `1.89%` | `3.92%` | `6.25%` | `5.00%` | `0.00%` | `16.36×` | `1.61%` | `14.18%` |
| `fpr_0.10` | `0.6959` | `61.54%` | `3.75%` | `1.89%` | `5.88%` | `6.25%` | `5.00%` | `0.00%` | `16.41×` | `1.62%` | `14.22%` |
| `max_youden_j` | `0.7030` | `61.54%` | `3.75%` | `1.89%` | `5.88%` | `6.25%` | `5.00%` | `0.00%` | `16.41×` | `1.62%` | `14.22%` |

## Sampled National Survey Coverage (`408.0 km²` / `0.31%` of England)

This survey is a statistically sampled out-of-sample evaluation, not a completed wall-to-wall scan of Great Britain. The national coverage map on [https://glassontin.github.io/megalith-lidar/](https://glassontin.github.io/megalith-lidar/) plots the exact `102` held-out `2 km × 2 km` squares (`408.0 km²`, `0.31%` of England's `130,279 km²` land area) scanned so far:
- `47` `uniform_national_null` squares (`188.0 km²`): `4` unidentified candidates at the strict `1%` FPR operating point (`0.021 / km²`, or `1` per `47 km²`) and `37` at the `5%` FPR operating point (`0.197 / km²`).
- `55` `heldout_test_landscape` squares (`220.0 km²`): `81` rediscovered NHLE Scheduled Monuments, `3` rediscovered unscheduled OSM archaeological monuments, `20` unidentified candidates at the `1%` FPR operating point (`0.091 / km²`, `4.3×` the null background density), and `93` at the `5%` FPR operating point (`0.423 / km²`).

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
