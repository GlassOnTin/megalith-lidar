<h1 align="center">megalith-lidar</h1>

<p align="center">
  Scale-space 1 m LiDAR terrain segmentation, 16 m aerial cropmark/soilmark analysis, 1888–1913 Historic Ordnance Survey 6-Inch map verification, and calibrated national survey pipeline for prehistoric, Iron Age, and Romano-British earthworks across England
</p>

<p align="center">
  <a href="https://glassontin.github.io/megalith-lidar/"><img src="https://img.shields.io/badge/Live_Interactive_Survey_Map-glassontin.github.io%2Fmegalith--lidar-8f3d1e?style=for-the-badge&logo=openstreetmap&logoColor=white" alt="Live Interactive Survey Map" /></a>
  <a href="https://glassontin.github.io/megalith-lidar/docs/megalith-lidar.osf"><img src="https://img.shields.io/badge/OsmAnd_Android_Plugin-megalith--lidar.osf-15803d?style=for-the-badge&logo=android&logoColor=white" alt="OsmAnd Custom Plugin (.osf)" /></a>
</p>

<p align="center">
  <a href="https://glassontin.github.io/megalith-lidar/"><img src="https://img.shields.io/website?url=https%3A%2F%2Fglassontin.github.io%2Fmegalith-lidar%2F&style=flat-square&label=GitHub%20Pages&up_message=live&up_color=2f6b48" alt="GitHub Pages Live Viewer" /></a>
  <a href="https://glassontin.github.io/megalith-lidar/docs/megalith_lidar_osmand.gpx"><img src="https://img.shields.io/badge/GPS_Waypoints-megalith__lidar__osmand.gpx-1b5e6f?style=flat-square" alt="OsmAnd & Garmin GPX" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-orange?style=flat-square" alt="License" /></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/PyTorch-ResNet--34_U--Net-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CRS-EPSG%3A27700_(1_m_LiDAR)-4a607a?style=flat-square" alt="EPSG:27700 1m LiDAR" />
  <a href="https://github.com/GlassOnTin/megalith-lidar/commits/main"><img src="https://img.shields.io/github/last-commit/GlassOnTin/megalith-lidar?style=flat-square&label=last%20scan" alt="Last Commit" /></a>
  <a href="https://ko-fi.com/glassontin"><img src="https://img.shields.io/badge/Ko--fi-support-ff5e5b?style=flat-square&logo=ko-fi&logoColor=white" alt="Ko-fi" /></a>
</p>

<p align="center">
  <a href="https://glassontin.github.io/megalith-lidar/">Live Survey Map &amp; 7-Panel Inspector</a> &bull;
  <a href="docs/megalith-lidar.osf">OsmAnd Plugin (.osf)</a> &bull;
  <a href="docs/megalith_lidar_osmand.gpx">OsmAnd / GPS (.gpx)</a> &bull;
  <a href="docs/candidates.geojson">Prehistoric GeoJSON</a> &bull;
  <a href="docs/enclosure_candidates.geojson">Iron Age GeoJSON</a> &bull;
  <a href="docs/roman_candidates.geojson">Roman GeoJSON</a>
</p>

---

Inspired by landscape archaeologist Adam Morgan Ibbotson's video on LiDAR survey in the Thornborough Henges landscape ([https://youtu.be/D_R0bgNgY2o](https://youtu.be/D_R0bgNgY2o)). This repository provides statistical terrain screening, calibrated probability maps, and open GeoJSON / 7-panel diagnostic outputs (4-panel 1 m LiDAR, aerial orthophoto, 16 m scale-space cropmark, and 1888–1913 Historic Ordnance Survey 6-Inch map with [Megalithic Portal](https://www.megalithic.co.uk/) cross-referencing) so archaeologists and Historic Environment Record (HER) researchers can inspect and verify candidate earthworks.

## Multi-Domain Detectors

Three domain-specific `ResNet-34` U-Net detectors (`megalith_lidar/model.py`) with differentiable D₄-invariant 2D geometric moment heads (`area_frac`, `log_elongation`, `ring_hollowness`, `compactness`), perimeter-ring supervision, and validation Platt calibration scan each `256 m × 256 m` British National Grid (`EPSG:27700`) window:

| Detector | Target Monument Classes | Dataset Size (`train / val / test`) | Held-Out Test ROC-AUC (`95% CI`) | `random_national` Null ROC-AUC | `trackside_field` Null ROC-AUC | Strict `1%` FPR Test `LR+` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Prehistoric (`MegalithUNet`)** | Round barrows/cairns, long barrows/chambered tombs, henges/stone circles | `4,654` (`3,965 / 386 / 303`) | `0.8675` (`[0.8248, 0.9070]`) | `0.8976` | `0.8899` | `61.05×` (`FPR = 0.63%`) |
| **Iron Age & Enclosures (`EnclosureUNet` v3, 8-Ch)** | Hut circle & pound settlements, hillforts & banjo enclosures, cross-ridge dykes & coaxial fields | `4,323` (`3,743 / 315 / 265`) | `0.8654` (`[0.8128, 0.9119]`) | `0.8941` | `0.9614` | `54.76×` (`FPR = 0.55%`) |
| **Roman & Roads (`RomanUNet`)** | Roman villas/buildings, Roman road aggers/trackways, forts/camps/settlements | `3,928` (`3,358 / 282 / 288`) | `0.7967` (`[0.7406, 0.8495]`) | `0.7741` | `0.8647` | `7.00×` (`FPR = 2.23%`) |

## Design and Bias Controls

Remote-sensing models for archaeology fail when they overfit to regional elevation, solar hillshade azimuth, synthetic target masks, or post-hoc score adjustments. This pipeline enforces seven controls across feature extraction, dataset construction, and statistical evaluation:

1. **Isotropic scale-space LiDAR features (`megalith_lidar/terrain_features.py`)**: Raw elevation in metres OD is excluded so the network cannot memorise upland altitude. Traditional directional hillshading is also excluded because a fixed solar azimuth introduces compass bias. Each 1 m Digital Terrain Model (DTM) and Last-Return Digital Surface Model (LZ_DSM) pair is transformed into a 6-channel terrain representation:
   - Fine local relief bandpass (`z - G_4(z)`, σ = 4 m) for orthostats, kerbs, hut-circle walls, and ring-ditch banks.
   - Monument-scale bandpass (`G_4(z) - G_16(z)`, σ₁ = 4 m, σ₂ = 16 m) for barrow mounds, henge banks, ramparts, and road aggers (⌀ 8 m to 45 m).
   - Meso-scale topographic context (`G_16(z) - G_64(z)`, σ₁ = 16 m, σ₂ = 64 m) for ridge, saddle, and terrace siting.
   - Isotropic gradient magnitude (`|∇z| = sqrt((∂z/∂x)² + (∂z/∂y)²)`).
   - Isotropic Laplacian curvature (`∇²z = ∂²z/∂x² + ∂²z/∂y²`).
   - Log-compressed above-ground surface prominence (`log(1 + max(0, DSM - DTM))`).
   Unit tests in `tests/test_pipeline.py` verify altitude shift invariance and exact equivariance under the 8-element dihedral group D₄ (rotations by 0°, 90°, 180°, 270° and reflections).

2. **16 m Scale-Space Aerial Cropmark & Soilmark Channels (`megalith_lidar/aerial_cropmarks.py`)**: Ploughed-out ring-ditches, barrows, and enclosures in arable farmland often retain `< 0.25 m` of vertical LiDAR relief while producing differential crop growth (`+ΔExG` over moisture-retaining buried ditches) or soil albedo contrast (`+ΔL` over chalk/gravel banks). To prevent overfitting to regional soil colour or seasonal flight exposure, RGB orthophotos (`1 m/px` `EPSG:27700`) are converted into two zero-mean, illumination-normalised `16 m` scale-space bandpass channels (`G_1.2 - G_16`) gated to bare/crop ground (`nDSM < 2.0 m`):
   - `band_cropmark_exg_16m` (`ΔExG_16m`): Local Excess-Green vegetation index anomaly.
   - `band_soilmark_lum_16m` (`ΔL_16m`): Local luminance soilmark/parchmark contrast.
   In `MegalithUNet`, the spatial cropmark branch is gated by the `7 × 7` dilated LiDAR terrain candidate probability halo and trained with 50% modality dropout so that summer tractor tramlines on flat fields cannot trigger false-positive peaks.

3. **Exact polygon supervision with perimeter-ring refinement (`megalith_lidar/catalog.py`, `megalith_lidar/dataset.py`)**: Ground-truth masks are rasterised directly from the 20,003 Historic England National Heritage List for England (NHLE Layer 6) Scheduled Monument polygons in British National Grid (`EPSG:27700`) at 1 m resolution. On large legal scheduling polygons (`> 500–600 m²`), active local-relief earthworks (`|z - G_16(z)|` and slope scarps) are retained as positive pixels, a 3 m transition halo is marked `IGNORE_INDEX = 255`, and flat interior/buffer pasture is supervised as explicit background (`0`) so the decoder segments actual ramparts and mounds rather than flood-filling 256 m tiles.

4. **Hash-based 20 km spatial block split (`megalith_lidar/dataset.py`)**: Every 256 m × 256 m tile is assigned to `train` (75%), `val` (12.5%), or `test` (12.5%) purely from the SHA-256 digest of its 20 km × 20 km British National Grid block coordinates `(E // 20000, N // 20000)`. No manual overrides or pinned blocks exist. Positive tiles receive a deterministic uniform spatial offset in `[-56 m, +56 m]` with reflection-padded shift augmentation during training so the network cannot learn a centre-of-tile prior.

5. **Five negative null distributions (`megalith_lidar/dataset.py`, `scripts/train_and_upgrade_v3.py`)**: Negative tiles are drawn from five explicit distributions to test separation from both background terrain and rural confounders:
   - `random_national`: Sampled uniformly across all land-bearing 20 km × 20 km blocks of England (independent of monument density), outside all Scheduled Monuments.
   - `matched_environment`: Sampled inside the same 20 km × 20 km blocks as positive monuments (`400 m` to `1,600 m` away, outside all Scheduled Monuments), matching local geology, slope, and LiDAR flight characteristics.
   - `anthropogenic_non_target`: Non-target earthworks and structures (medieval priories, industrial spoil heaps, World War II batteries, post-medieval quarries).
   - `woodland_canopy_and_pit`: Monument-free woodland canopy (`>= 20%` `nDSM > 2.5 m`) and sunken marl/chalk/quarry pits (`0.5th` percentile `16 m` relief `<= -0.95 m`).
   - `trackside_and_field_boundary`: Open agricultural land with hedgerow, trackside, and field-corner boundary relief bumps.

6. **Validation-only Platt calibration and threshold locking (`megalith_lidar/stats.py`)**: Logistic Platt scaling (`P(y=1|s) = σ(a·s + b)`) and operating thresholds (`1%`, `5%`, and `10%` false positive rate, plus Youden's `J`) are fitted exclusively on the validation split and evaluated once on the held-out test split with 8-way D₄ test-time augmentation.

7. **OpenStreetMap, 1888–1913 Historic Ordnance Survey 6-Inch Map, Megalithic Portal, intrinsic LiDAR, and Vision-Language confounder cross-check (`megalith_lidar/scanner.py`, `megalith_lidar/osm_context.py`, `megalith_lidar/arch_context.py`, `scripts/vlm_audit_candidates.py`)**: Connected components are extracted from the calibrated probability map, conditioned on 2D footprint geometry (area, elongation, hollowness), and cross-checked against NHLE Layer 6 polygons, the OpenStreetMap v0.6 API, [Megalithic Portal](https://www.megalithic.co.uk/) monument records, intrinsic 1 m LiDAR observables (tall canopy, steep silage/manure clamps, sunken pits, river-meander/lynchet scarps, or interpolated TIN voids), and Qwen3.8 Vision 3-panel inspection (1 m LiDAR Relief + Aerial RGB + 1888–1913 Historic Ordnance Survey Six-Inch map):
   - `rediscovered_nhle_scheduled` (within 45 m of a Scheduled Monument polygon)
   - `rediscovered_osm_archaeology` (within 45 m of an unscheduled OSM `man_made=cairn`, `historic=megalith|archaeological_site|tumulus|barrow|hillfort|roman_villa`, or `megalith_type=*`)
   - `flagged_modern_or_karst_confounder` (inside or within feature-specific radii of mapped quarries, sinkholes, farmyards, tracks/roads, hedges/fences, power poles, woodlands, covered reservoirs, golf bunkers, or steep surface clamps/pits)
   - `unidentified_candidate` (outside NHLE Scheduled Monuments, outside OSM recorded archaeology, and outside mapped/intrinsic confounders)

## Held-Out Test Split Evaluation

### 1. Prehistoric Detector (`MegalithUNet`, `n = 303` held-out test tiles)
Evaluated on `143` positives and `160` negatives across non-overlapping 20 km × 20 km BNG blocks using 1,000 stratified bootstrap resamples (`docs/evaluation_metrics_v2_unbiased.json`):

| Metric / Negative Null Comparison | Estimate | 95% Bootstrap CI | Sample Size |
| :--- | :---: | :---: | :---: |
| Overall Test ROC-AUC (all 5 nulls combined) | `0.8675` | `[0.8248, 0.9070]` | `143 pos vs 160 neg` |
| Overall Test PR-AUC | `0.8818` | `[0.8417, 0.9176]` | `143 pos vs 160 neg` |
| Scalar local-relief baseline ROC-AUC | `0.5329` | `[0.4696, 0.5988]` | `143 pos vs 160 neg` |
| Positives vs `random_national` null ROC-AUC | `0.8976` | `[0.8527, 0.9379]` | `143 pos vs 53 neg` |
| Positives vs `trackside_and_field_boundary` null ROC-AUC | `0.8899` | `[0.8140, 0.9479]` | `143 pos vs 20 neg` |
| Positives vs `woodland_canopy_and_pit` null ROC-AUC | `0.8836` | `[0.8220, 0.9360]` | `143 pos vs 20 neg` |
| Positives vs `anthropogenic_non_target` null ROC-AUC | `0.8459` | `[0.7432, 0.9266]` | `143 pos vs 16 neg` |
| Positives vs `matched_environment` null ROC-AUC | `0.8280` | `[0.7660, 0.8795]` | `143 pos vs 51 neg` |
| Calibration Brier score (test split) | `0.1412` | — | `n = 303` |
| Expected Calibration Error (ECE, 10 bins) | `0.0525` | — | `n = 303` |

| Validation Target | Locked Tile Threshold | Test TPR (Recall) | Test Overall FPR | FPR on `random_national` | FPR on `matched_env` | FPR on `anthropogenic` | FPR on `woodland_pit` | FPR on `trackside` | Positive Likelihood Ratio (`LR+`) | Prior-Adjusted PPV (`π = 10⁻³`) | Prior-Adjusted PPV (`π = 10⁻²`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fpr_0.01` | `0.9153` | `38.46%` | `0.63%` | `0.00%` | `0.00%` | `6.25%` | `0.00%` | `0.00%` | `61.05×` | `5.76%` | `38.14%` |
| `fpr_0.05` | `0.8071` | `51.05%` | `3.12%` | `1.89%` | `3.92%` | `6.25%` | `5.00%` | `0.00%` | `16.36×` | `1.61%` | `14.18%` |
| `fpr_0.10` | `0.6959` | `61.54%` | `3.75%` | `1.89%` | `5.88%` | `6.25%` | `5.00%` | `0.00%` | `16.41×` | `1.62%` | `14.22%` |

### 2. Iron Age & Enclosure Detector (`EnclosureUNet` v3, 8-Channel LiDAR + Cropmarks, `n = 265` held-out test tiles)
Evaluated on `83` positives and `182` negatives (`docs/enclosure_evaluation_metrics.json`):
- **Overall Test ROC-AUC**: `0.8654` (`95% CI: [0.8128, 0.9119]`), PR-AUC `0.8002`, Brier score `0.1330`.
- **Subgroup ROC-AUCs**: `0.9614` vs `trackside_and_field_boundary`, `0.8941` vs `random_national`, `0.8503` vs `anthropogenic_non_enclosure`, `0.8301` vs `woodland_canopy_and_pit`, `0.8231` vs `matched_environment`.
- **Strict `1%` FPR Operating Point (`threshold = 0.8248`)**: Test Recall `30.12%` (`44.19%` on hillforts & banjo enclosures), Test FPR `0.55%` (`0.00%` on `random_national`, `matched_environment`, `woodland_pit`, and `trackside`), **`LR+ = 54.76×`**.

### 3. Roman & Romano-British Detector (`RomanUNet`, `n = 288` held-out test tiles)
Evaluated on `109` positives and `179` negatives (`docs/enclosure_summary.json`, `docs/roman_summary.json`):
- **Overall Test ROC-AUC**: `0.7967` (`95% CI: [0.7406, 0.8495]`), with `0.8647` vs `trackside_and_field_boundary`, `0.8330` vs `matched_environment`, and `0.7898` vs `anthropogenic_non_roman`.

## National Survey Progress (`7,669` Squares / `30,660.0 km²` / `23.5%` of England)

`7,669` valid `2 km × 2 km` squares (`30,660.0 km²`, `23.5%` of England's `130,279 km²` land area) have been scanned so far across the three detectors and enriched with `16 m` scale-space aerial cropmark/soilmark metrics, 1888–1913 Historic Ordnance Survey 6-Inch map verification, and [Megalithic Portal](https://www.megalithic.co.uk/) cross-references (`docs/survey_summary.json`, `docs/roman_summary.json`, `docs/enclosure_summary.json`):

1. **Prehistoric Survey (`docs/candidates.geojson`)**:
   - `2,708` `uniform_national_null` squares (`10,832.0 km²`): `275` unidentified candidates at the strict `1%` FPR operating point (`0.025 / km²`) and `1,269` at the `5%` FPR operating point (`0.117 / km²`).
   - `4,957` `heldout_test_landscape` squares (`19,828.0 km²`): **`1,598` rediscovered NHLE Scheduled Monuments**, **`86` rediscovered unscheduled OSM archaeological monuments**, **`808` unidentified candidates at the strict `1%` FPR operating point** (`0.041 / km²`), and `3,305` at the `5%` FPR operating point (`0.167 / km²`).
2. **Iron Age & Prehistoric Enclosure Survey (`docs/enclosure_candidates.geojson`)**:
   - Across `7,719` squares (`30,876.0 km²`): **`3,118` rediscovered NHLE Scheduled Monuments**, **`129` rediscovered OSM archaeological sites**, and **`11` unidentified enclosure/settlement candidates** (`4` passing the strict `1%` FPR operating point after 1888 OS 6-Inch map and geomorphic screening, with `7` explicitly marked `'Camp'`, `'Tumulus'`, or hachured on the 1888 OS map).
3. **Roman & Romano-British Survey (`docs/roman_candidates.geojson`)**:
   - Across `7,719` squares (`30,876.0 km²`): **`4,884` rediscovered NHLE Scheduled Monuments**, **`488` rediscovered OSM archaeological sites**, and **`36` unidentified candidates** (`11` passing the strict `1%` FPR operating point after geometric subtype and 1888 OS / intrinsic LiDAR screening).

## OsmAnd & Field Navigation (`docs/megalith-lidar.osf` & `docs/megalith_lidar_osmand.gpx`)

- **One-Tap OsmAnd Custom Plugin (`docs/megalith-lidar.osf`)**: Downloading and opening [`megalith-lidar.osf`](https://glassontin.github.io/megalith-lidar/docs/megalith-lidar.osf) on Android or iOS installs the **UK 1 m LiDAR Monument Survey** plugin into [OsmAnd](https://osmand.net/), bundling:
  - Categorised monument Favourites (`1888 OS Confirmed Antiquities ★`, `Prehistoric Candidates ●`, `Iron Age & Settlement ◆`, `Roman Candidates ■`) with earthwork relief, walkability notes, Qwen3.8 VLM / 1888 OS assessments, and clickable `#site=...` deep-links back to the 7-panel LiDAR inspector.
  - Searchable GPX tracks (`UK_Megalith_LiDAR_Candidates.gpx` and `UK_Roman_and_Ancient_Roads.gpx` covering `360` British Roman road and prehistoric ridgeway segments).
  - Custom online tile sources for **UK 1888–1913 OS 6-Inch Map** (`uk-osgb1888`) and **Digital Atlas of the Roman Empire (DARE)** so fieldwalkers can underlay 19th-century Ordnance Survey mapping beneath their live GPS position.
- **Standalone GPX &Filtered Export (`docs/megalith_lidar_osmand.gpx`)**: Compatible with OsmAnd, Garmin, Organic Maps, Memory-Map, and QGIS. The web viewer also includes an **Export Visible (.gpx)** button to export any filtered county or category subset on the fly.

## Limitations and Unverified Scope

1. **Base-rate constraint in national screening**: Even at `LR+ = 54.8×–61.1×` (`fpr_0.01`, `Test FPR = 0.55%–0.63%`), if true unrecorded monuments occur in only `π = 10⁻³` of 256 m × 256 m windows nationally, Bayes' rule bounds the raw single-window positive predictive value to `5.2%–5.8%` before spatial context and visual inspection.
2. **Unmapped rural confounders**: While OpenStreetMap cross-checking, geometric moment conditioning, and Qwen3.8 Vision audits filter out mapped quarries, sinkholes, reservoirs, U-bunds, golf bunkers, and landscaped garden buildings, unmapped post-medieval clearance cairns, tree-throw mounds, former dewponds, and unmapped karst shakeholes on limestone moorland cannot be separated from prehistoric earthworks by 1 m LiDAR and single-date aerial imagery alone. All `unidentified_candidate` entries are statistical anomalies requiring Historic Environment Record (HER), multi-date aerial, or field verification.
3. **Geographic coverage**: The current training catalog and WCS endpoints cover England (`Environment Agency 1 m Composite DTM / LZ_DSM` and `Historic England NHLE Layer 6`). Extending to Wales (`DataMapWales` / `Cadw`), Scotland (`Scottish Remote Sensing Portal` / `HES`), and Northern Ireland requires integrating their respective LiDAR WCS feeds and scheduled monument polygon registries.

## Repository Layout

- `megalith_lidar/terrain_features.py`: 6-channel isotropic scale-space LiDAR bandpass pyramid and differential invariants.
- `megalith_lidar/aerial_cropmarks.py`: `EPSG:27700` aerial orthophoto fetcher (ArcGIS World Imagery static zoom-17 pyramid with 4-corner BNG bilinear warp) and `16 m` scale-space cropmark/soilmark (`ΔExG_16m`, `ΔL_16m`) extractor.
- `megalith_lidar/catalog.py`: Historic England NHLE Layer 6 `EPSG:27700` polygon downloader, multi-domain taxonomy classifier (Prehistoric, Roman, Enclosure), and 1 m polygon rasteriser.
- `megalith_lidar/lidar_wcs.py`: Environment Agency 1 m DTM and 1 m LZ_DSM WCS client with disk caching and nodata validation.
- `megalith_lidar/dataset.py`: SHA-256 20 km BNG block splitter, perimeter-ring mask refinement, 50% cropmark modality dropout, and 5-null dataset builder.
- `megalith_lidar/model.py`: Reflection-padded `MegalithUNet` (`ResNet-34` encoder, multi-scale decoder, D₄-invariant 2D geometric moment head, LiDAR-halo-gated 8-channel cropmark branch, and MIL negative-peak + extent spill loss).
- `megalith_lidar/stats.py`: Validation Platt scaling, operating-point selection, Bayes prior-adjusted PPV, and stratified bootstrap 95% CIs.
- `megalith_lidar/osm_context.py`: OpenStreetMap API v0.6 bounding-box spatial index and intrinsic LiDAR confounder filters.
- `megalith_lidar/arch_context.py`: NHLE spatial cluster synthesis, 1888–1913 Historic Ordnance Survey 6-Inch map cropper (`uk-osgb1888`), [Megalithic Portal](https://www.megalithic.co.uk/) spatial cross-referencing, and 3-panel Qwen3.8 Vision candidate context generator.
- `megalith_lidar/scanner.py`: Sliding-window regional scanner with 8-way D₄ test-time augmentation, two-stage LiDAR + aerial cropmark inference, spatial non-maximum suppression, and 4-panel PNG diagnostics.
- `scripts/train_and_upgrade_v3.py`: Expanded `v3` manifest builder, warm-started 8-channel training, and held-out test evaluation across Prehistoric, Roman, and Enclosure domains.
- `scripts/vlm_audit_candidates.py`: Multi-panel Qwen3.8 Vision audit and archaeological landscape synthesis over candidate detections.
- `index.html`: Self-contained Leaflet map and 7-panel candidate diagnostic inspector (4-panel 1 m LiDAR + live `EPSG:27700` Aerial Orthophoto, `16 m` Scale-Space Cropmark, 1888–1913 Historic Ordnance Survey 6-Inch Map `<canvas>` panels, and Megalithic Portal / NLS side-by-side card) deployed via GitHub Pages.

## Running Locally

```bash
# Run the unit test suite (shift invariance, D4 equivariance, 8-channel cropmark zero-init & gradients, Platt calibration, OSM cross-check)
PYTHONPATH=. .venv/bin/pytest -v tests/test_pipeline.py

# Train and evaluate the v3 multi-domain models on the 20 km block-split dataset
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train_and_upgrade_v3.py --data-dir /mnt/bulk/megalith-lidar --domains enclosure,roman,prehistoric --epochs 6 --batch-size 4 --grad-accum 4 --device cuda:0

# Run the national batch scanner and update docs/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/run_national_batch.py --max-squares 4 --device cuda:0 --git-push
```

## License

Released under the [GNU Affero General Public License v3.0](LICENSE) (`AGPL-3.0-or-later`).
Contains Environment Agency Open LiDAR data and Historic England National Heritage List for England data licensed under the Open Government Licence v3.0, OpenStreetMap data licensed under the Open Database Licence (ODbL), and Esri/Maxar World Imagery basemap tiles used for visual and cropmark inspection.
