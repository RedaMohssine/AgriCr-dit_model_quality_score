# AgriCrédit: Farm Credit Assessment via Satellite

> CIH Bank Hackathon 2026: Automated rural credit scoring from GPS coordinates using satellite imagery and machine learning.

A rural Moroccan farmer sends their GPS location via **WhatsApp**. Within minutes, CIH bank officers receive a structured **credit quality score (0–100)** on an interactive dashboard: no site visit, no paperwork.

---

## How it works

```
Farmer sends GPS location via WhatsApp
        ↓
   n8n workflow receives it via Webhook
        ↓
   ── MODEL 1 (this repo) ──────────────────────────────
   POST /assess
   Google Earth Engine → 24 months satellite data
   XGBoost → monthly yield curve
   5-dimension quality score (0–100)
   JSON report produced
        ↓
   ── MODEL 2 (Farm Stage Classifier) ──────────────────
   POST /classify
   Claude Haiku reads the JSON report
   Returns: farm stage + confidence + analytical paragraph
        ↓
   n8n sends results back to farmer via WhatsApp
   Database updated with full assessment
        ↓
   CIH bank officer reviews score on the dashboard + stage for credit decision
```

---

## The Score

The quality score aggregates **24 months of satellite and climate data** into a single number across 5 dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Productivity | 30% | Average and peak yield potential |
| Consistency | 25% | Income stability (low variance = reliable repayment) |
| Trend | 20% | Year-over-year improvement |
| Vegetation Health | 15% | NDVI greenness: calibrated to Moroccan semi-arid conditions |
| Drought Resilience | 10% | Yield floor during heat/dry stress periods |

### Example JSON response

```json
{
  "farm": {
    "lat": 31.7917,
    "lon": -7.0926,
    "region": "Marrakech, Morocco"
  },
  "assessment_date": "2026-04-19",
  "quality_score": 54.3,
  "sub_scores": {
    "productivity": 87.8,
    "consistency": 24.5,
    "trend": 43.1,
    "vegetation": 39.1,
    "resilience": 34.9
  },
  "vegetation": {
    "mean_ndvi": 0.256,
    "mean_gndvi": 0.368,
    "peak_ndvi": 0.334,
    "green_months": 9,
    "ndvi_trend": "improving"
  },
  "climate": {
    "avg_temperature_c": 26.75,
    "avg_rainfall_mm_month": 41.48,
    "total_rainfall_mm": 995.5,
    "avg_soil_moisture_pct": 22.8,
    "stress_months_count": 7,
    "stress_months": ["2024-06", "2024-07", "2024-08", "2025-06", "2025-07", "2025-08", "2025-09"]
  },
  "resilience": {
    "avg_sm_during_stress": 15.31,
    "stress_months_count": 7
  },
  "farmerPhone": "+212600000001"
}
```

---

## Satellite Data Sources

| Source | Dataset | Resolution | What we extract |
|---|---|---|---|
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | NDVI, GNDVI, NDWI, SAVI |
| MODIS | `MODIS/061/MOD11A1` | 1 km | Land surface temperature (°C) |
| CHIRPS | `UCSB-CHG/CHIRPS/DAILY` | ~5 km | Monthly rainfall (mm) |
| NASA SMAP | `NASA/SMAP/SPL4SMGP/008` | ~11 km | Soil moisture (%) |

All data fetched via **Google Earth Engine** for the farm location (500 m radius buffer) over the last 24 months.

---

## ML Model

**XGBoost Regressor** trained on the [Kaggle Crop Yield Prediction dataset](https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction): 1,621 observations across 90 fields.

- 14 engineered features (spectral indices + climate + seasonal encoding)
- Field-level train/test split (no data leakage)
- GroupKFold cross-validation
- **Test MAE: 1.169 t/ha: Test R²: 0.901**

> Yield predictions are model-relative (used as a comparative signal between months/farms), not absolute Moroccan benchmarks.

---

## n8n Workflow

The entire end-to-end pipeline runs inside an **n8n** automation workflow: no manual steps. **Both models are called in sequence** for every location message received.

### Full workflow overview

![n8n workflow overview](https://github.com/user-attachments/assets/fff23b42-81ba-4266-8382-65f341decd88)

### Step-by-step breakdown

#### Step 1: Receive WhatsApp message
**Webhook (POST)** receives the incoming message from the WhatsApp integration.

#### Step 2: Parse message
**Code in JavaScript** extracts the message fields: phone number, message type, and coordinates.

#### Step 3: Route by message type (If)
- **true** (text/non-location) → Sends an acknowledgment WhatsApp reply immediately.
- **false** (location message) → Continues to the assessment pipeline.

![n8n workflow start](https://github.com/user-attachments/assets/ad5c6298-a873-4c3e-9066-27ceb196f994)

#### Step 4: Extract coordinates (Code in JavaScript1)
Validates and formats the latitude/longitude from the WhatsApp location payload.

#### Step 5: Check if assessment should run (If1)
- **true** → Calls Model 1.
- **false** → Routes to database handling (If2: create or update farmer record).

#### Step 6: Call Model 1: AgriCredit Scoring (HTTP Request)
`POST https://agricredit-model-quality-score.onrender.com/assess`

Sends the coordinates. This triggers the full 24-month satellite fetch from Google Earth Engine, XGBoost yield inference, and 5-dimension quality scoring. Returns the full JSON report.

- **Update a row**: logs the raw request to the database in parallel.

#### Step 7: Call Model 2: Farm Stage Classifier (HTTP Request1)
**Code in JavaScript2** passes the full JSON report from Model 1 directly into:

`POST https://cih-hackathon-model2-1.onrender.com/classify`

Claude Haiku reads the satellite report and returns:
- `stage`: current farm stage in French (e.g. *croissance végétative*)
- `confidence`: haute / moyenne / faible
- `paragraph`: analytical paragraph for the CIH credit team

#### Step 8: Save & deliver
- **Update a row1**: saves the complete assessment (score + stage + paragraph) to the database for the dashboard.
- **Send WhatsApp**: delivers the results back to the farmer's phone number.

![n8n workflow end](https://github.com/user-attachments/assets/0084a059-ad5d-4f95-bbc4-e804a16f64c4)

---

## Deployment

Hosted on **Render** (free tier). Auto-deploys on every push to `main`.

### Environment variables required on Render

| Variable | Value |
|---|---|
| `GEE_SERVICE_ACCOUNT` | `your-sa@your-project.iam.gserviceaccount.com` |
| `GEE_PRIVATE_KEY` | Full JSON key file content |
| `GEE_PROJECT` | Google Cloud project ID (e.g. `agritcredit`) |

### Test locally

```bash
# Health check
curl https://agricredit-model-quality-score.onrender.com/

# Full assessment
curl -X POST "https://agricredit-model-quality-score.onrender.com/assess" \
  -H "Content-Type: application/json" \
  -d '{"latitude":"31.7917","longitude":"-7.0926","farmerPhone":"+212600000001","messageType":"location","shouldProcess":true}'
```

---

## References

- Rouse et al. (1974): NDVI
- Gitelson et al. (1996): GNDVI
- Gao (1996): NDWI
- Huete (1988): SAVI
- Funk et al. (2015): CHIRPS precipitation
- Wan (2014): MODIS LST
- Reichle et al. (2019): NASA SMAP soil moisture
- Allen et al. (1998): FAO Irrigation & Drainage Paper No. 56 (heat stress threshold)
- FAO (2012): Composite indicator methodology
- Myneni et al. (1995): Active vegetation NDVI threshold
