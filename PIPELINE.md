# CIH Farm Credit Assessment — Pipeline Documentation

## What this system does

A rural Moroccan farmer sends GPS coordinates (latitude, longitude) via WhatsApp.
The system fetches 24 months of satellite and climate data for that location, runs a machine-learning model on each monthly snapshot, and produces a **composite quality score (0–100)** that a CIH bank officer uses to assess creditworthiness.

No site visit. No paperwork. Fully automated from coordinates to score.

---

## Project structure

```
notebooks/
  02_improved_yield_model.ipynb   — Stage 1: train the yield estimator
  03_gee_fetcher.ipynb            — Stage 2: fetch satellite data from GEE
  04_quality_scorer.ipynb         — Stage 3: compute and output the quality score

saved_model/
  yield_model_v2.joblib           — trained XGBoost model (14 features → yield)
  feature_cols_v2.joblib          — ordered feature list used at inference

data/
  farm_profiles/                  — 24-month CSV profiles + JSON reports + score cards
  farm_images/                    — monthly Sentinel-2 RGB images (presentation)

yield_prediction_dataset.csv      — Kaggle training dataset (1,621 observations, 90 fields)
```

---

## Stage 1 — Yield Estimator (`02_improved_yield_model.ipynb`)

### Purpose
Train a regression model that maps a single satellite/climate snapshot of a field to a predicted crop yield. This model is later applied 24 times (once per month) at inference to build a temporal productivity curve.

### Training data
**Dataset:** "Crop Yield Prediction" — Kaggle  
1,621 observations across 90 fields, ~18 monthly snapshots per field (2023).  
Yield range: 26.9 – 75.7 tons/ha, mean 40.5 tons/ha.

### Key design choices

**Snapshot-per-row training** — Each row is a `(field conditions at time T) → (yield at time T)` pair. Yield varies with conditions at each snapshot (std ~7 t/ha within the same field), so training on individual snapshots — not field averages — is correct. Averaging features before training would destroy the signal.

**Field-level train/test split** — 72 fields train, 18 fields test. No farm appears in both sets. This is stricter than random row splitting and prevents data leakage.

**GroupKFold cross-validation** — 5 folds, grouping by `field_id`. Prevents the same field from appearing in both train and validation within a fold.

**Outlier removal** — Observations where yield deviates more than 3σ from the field mean are excluded (removed 4 rows out of 1,625).

### Features (14 total)

| Feature | Formula / Source | What it captures |
|---|---|---|
| `NDVI` | (B8−B4)/(B8+B4) — Rouse et al. (1974) | Overall plant greenness |
| `GNDVI` | (B8−B3)/(B8+B3) — Gitelson et al. (1996) | Chlorophyll content |
| `NDWI` | (B3−B8)/(B3+B8) — Gao (1996) | Leaf water content |
| `SAVI` | 1.5×(B8−B4)/(B8+B4+0.5) — Huete (1988), L=0.5 | Greenness corrected for bare soil |
| `soil_moisture` | SMAP SPL4SMGP/008, m³/m³ × 100 | Volumetric soil water content (%) |
| `temperature` | MODIS MOD11A1, raw × 0.02 − 273.15 | Land surface temperature (°C) |
| `rainfall` | CHIRPS daily, summed monthly — Funk et al. (2015) | Monthly precipitation (mm) |
| `month_sin` | sin(2π × month / 12) | Cyclic seasonal encoding |
| `month_cos` | cos(2π × month / 12) | Cyclic seasonal encoding |
| `veg_stress` | (NDVI − GNDVI) / (NDVI + GNDVI) | Nutrient deficiency signal |
| `water_stress` | NDVI / soil_moisture | Vegetation relative to water availability |
| `ndvi_x_moisture` | NDVI × soil_moisture | Non-linear coupling of health and water |
| `thermal_load` | \|temperature − 25\| | Deviation from general crop optimum |
| `rain_efficiency` | NDVI / rainfall | Vegetation gain per mm of rain |

Crop type was excluded — feature importance analysis showed it contributed minimally; the spectral indices already encode crop-type effects implicitly through reflectance patterns.

### Model

**XGBoost Regressor** — 400 trees, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8.

| Version | Features | Test MAE | Test R² |
|---|---|---|---|
| Baseline (01) | 10 | 1.250 t/ha | 0.883 |
| **Improved (02)** | **14** | **1.169 t/ha** | **0.901** |

GroupKFold CV: MAE = 0.961 ± 0.261 t/ha

> **Important:** Yield values are model-relative predictions on the Kaggle dataset scale. They are used only as a comparative signal (one month vs another, one farm vs another), not as absolute Moroccan crop benchmarks. The quality score is dimensionless.

---

## Stage 2 — GEE Farm Data Fetcher (`03_gee_fetcher.ipynb`)

### Purpose
Given GPS coordinates, fetch 24 monthly satellite and climate snapshots for that location, apply Stage 1 to each snapshot to produce a yield curve, and save the profile.

### Data sources

#### Sentinel-2 Surface Reflectance — `COPERNICUS/S2_SR_HARMONIZED`
- **Resolution:** 10 m
- **Filter:** cloud cover < 20% (`CLOUDY_PIXEL_PERCENTAGE`)
- **Aggregation:** median composite over the farm buffer (500 m radius)
- **Bands used:** B3 (green, 560 nm), B4 (red, 665 nm), B8 (NIR, 842 nm)
- **Derived indices:** NDVI, GNDVI, NDWI, SAVI — computed via pixel math on each image before taking the median

#### MODIS Land Surface Temperature — `MODIS/061/MOD11A1`
- **Resolution:** 1 km
- **Aggregation:** mean over 10 km radius buffer
- **Conversion:** raw digital number × 0.02 − 273.15 = °C
  (official scale factor from NASA MOD11A1 product documentation, Wan 2014)

#### CHIRPS Daily Precipitation — `UCSB-CHG/CHIRPS/DAILY`
- **Resolution:** ~5 km
- **Aggregation:** daily values summed to monthly total (mm/month) over 10 km buffer
- **Reference:** Funk et al. (2015), *Scientific Data*, doi:10.1038/sdata.2015.66

#### NASA SMAP Soil Moisture — `NASA/SMAP/SPL4SMGP/008`
- **Resolution:** ~11 km
- **Band:** `sm_surface` (0–5 cm depth)
- **Aggregation:** mean over 10 km buffer
- **Conversion:** m³/m³ × 100 = volumetric %
- **Reference:** Reichle et al. (2019), SMAP L4 Algorithm Theoretical Basis Document

### Inference logic

```
24 monthly snapshots (Apr 2024 → Mar 2026)
    ↓  for each month
    fetch Sentinel-2 indices + MODIS temp + CHIRPS rain + SMAP moisture
    ↓
    build 14 engineered features
    ↓
    apply Stage 1 XGBoost → yield_pred (tons/ha, model-relative)
    ↓
24-point yield curve + all raw features saved to CSV
```

Months with no cloud-free Sentinel-2 image are linearly interpolated from adjacent months.

---

## Stage 3 — Quality Scorer (`04_quality_scorer.ipynb`)

### Purpose
Aggregate the 24-month farm profile into a single **quality score (0–100)** across five dimensions.

### Scoring dimensions

Each dimension is linearly scaled between a low bound (score = 0) and a high bound (score = 100), then clipped. All bounds are derived from the training dataset statistics.

#### 1. Productivity (weight 30%)
Blended signal: 70% mean yield + 30% p90 yield over 24 months.  
Bounds: p5 = 29.01 t/ha → 0, p95 = 54.53 t/ha → 100 (all yield observations in training dataset).

#### 2. Consistency (weight 25%)
Coefficient of Variation of monthly yield predictions. Inverted: lower CV = more stable income = higher score.  
Bounds: max observed field CV = 0.269 → 0, min observed field CV = 0.028 → 100.

#### 3. Trend (weight 20%)
Ratio of year-2 mean yield to year-1 mean yield.  
Bounds: 0.85 → 0 (farm losing >15% year-over-year), 1.15 → 100 (gaining >15%).  
±15% is consistent with the training dataset's observed field CV range (0.028–0.269).

#### 4. Vegetation Health (weight 15%)
Mean NDVI over 24 months.  
Bounds: p10 = 0.18 → 0, p90 = 0.66 → 100 (NDVI > 0 observations in training dataset).  
Active-vegetation threshold NDVI > 0.30: Myneni et al. (1995), *J. Geophys. Res.*; standard in semi-arid agricultural remote sensing (USGS EROS center).

#### 5. Drought Resilience (weight 10%)
Mean yield during stress months (rainfall < 5 mm/month OR temperature > 35°C).  
Bounds: p10 = 28.5 t/ha → 0, p90 = 50.4 t/ha → 100 (stress-period yields in training dataset).  
Heat-stress threshold 35°C: FAO Irrigation & Drainage Paper No. 56 (Allen et al., 1998).

### Composite formula

```
quality_score = 0.30 × productivity
              + 0.25 × consistency
              + 0.20 × trend
              + 0.15 × vegetation
              + 0.10 × resilience
```

Weights follow the methodology of FAO composite agricultural indicators (FAO, 2012). No historical loan-default data was available to derive weights empirically.

### Output

The scorer produces three files per farm:

| File | Content |
|---|---|
| `data/farm_profiles/farm_{lat}_{lon}.csv` | 24-month feature table (raw + engineered + yield_pred) |
| `data/farm_profiles/score_card_{lat}_{lon}.png` | Radar chart + yield timeline + score card visualization |
| `data/farm_profiles/report_{lat}_{lon}.json` | Structured report: score, sub-scores, vegetation detail, climate summary, stress months |

---

## References

| Source | Reference |
|---|---|
| NDVI | Rouse et al. (1974). *Monitoring vegetation systems in the Great Plains with ERTS*. |
| GNDVI | Gitelson et al. (1996). *Use of a green channel in remote sensing of global vegetation*. Remote Sens. Environ. |
| NDWI | Gao (1996). *NDWI — a normalized difference water index*. Remote Sens. Environ. |
| SAVI | Huete (1988). *A soil-adjusted vegetation index (SAVI)*. Remote Sens. Environ. |
| MODIS LST | Wan (2014). *MOD11A1 MODIS/Terra Land Surface Temperature*. NASA EOSDIS LP DAAC. |
| CHIRPS | Funk et al. (2015). *The climate hazards infrared precipitation with stations*. Scientific Data. |
| SMAP | Reichle et al. (2019). *SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Rootzone Soil Moisture*. NASA NSIDC DAAC. |
| Active vegetation threshold | Myneni et al. (1995). *The interpretation of spectral vegetation indexes*. IEEE TGRS. |
| Heat stress threshold | Allen et al. (1998). *FAO Irrigation and Drainage Paper No. 56*. FAO Rome. |
| Composite indicator weights | FAO (2012). *The State of Food Insecurity in the World*. FAO Rome. |
