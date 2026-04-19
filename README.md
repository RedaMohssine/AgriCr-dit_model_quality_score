# AgriCrédit — Farm Credit Assessment via Satellite

> CIH Bank Hackathon 2026 — Automated rural credit scoring from GPS coordinates using satellite imagery and machine learning.

A rural Moroccan farmer sends their GPS location via **WhatsApp**. Within minutes, CIH bank officers receive a structured **credit quality score (0–100)** — no site visit, no paperwork.

---

## How it works

```
Farmer sends WhatsApp location
        ↓
   n8n workflow captures coordinates
        ↓
   POST /assess → Render API
        ↓
   Google Earth Engine fetches 24 months of satellite data
        ↓
   XGBoost model predicts monthly yield for each snapshot
        ↓
   5-dimension quality score computed
        ↓
   JSON report returned to n8n → forwarded to bank officer
```

---

## The Score

The quality score aggregates **24 months of satellite and climate data** into a single number across 5 dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Productivity | 30% | Average and peak yield potential |
| Consistency | 25% | Income stability (low variance = reliable repayment) |
| Trend | 20% | Year-over-year improvement |
| Vegetation Health | 15% | NDVI greenness — calibrated to Moroccan semi-arid conditions |
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

**XGBoost Regressor** trained on the [Kaggle Crop Yield Prediction dataset](https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction) — 1,621 observations across 90 fields.

- 14 engineered features (spectral indices + climate + seasonal encoding)
- Field-level train/test split (no data leakage)
- GroupKFold cross-validation
- **Test MAE: 1.169 t/ha — Test R²: 0.901**

> Yield predictions are model-relative (used as a comparative signal between months/farms), not absolute Moroccan benchmarks.

---

## n8n Workflow

The entire end-to-end pipeline runs inside an **n8n** automation workflow — no manual steps.

### Full workflow overview

![n8n workflow overview](screenshots/n8n_overview.png)

### Step-by-step breakdown

#### Step 1 — Receive WhatsApp message
**Webhook (POST)** receives the incoming message from the WhatsApp integration. Every message sent to the farmer's dedicated number triggers this webhook.

#### Step 2 — Parse message
**Code in JavaScript** extracts the message fields: phone number, message type, coordinates (if present), and any text content.

#### Step 3 — Route by message type (If)
Checks whether the message contains a GPS location or is just a text message.
- **true** (text/non-location) → Sends an acknowledgment WhatsApp message back to the farmer immediately.
- **false** (location message) → Continues to the assessment pipeline.

![n8n workflow start](screenshots/n8n_start.png)

#### Step 4 — Extract coordinates (Code in JavaScript1)
Formats and validates the latitude/longitude extracted from the WhatsApp location payload.

#### Step 5 — Check if assessment should run (If1)
Determines whether this location qualifies for a full satellite assessment.
- **true** → Calls the AgriCredit API.
- **false** → Routes to database handling (If2).

#### Step 6 — Call AgriCredit API (HTTP Request)
`POST https://agricredit-model-quality-score.onrender.com/assess`

Sends the farmer's coordinates to the Render-hosted API. This triggers the full 24-month satellite data fetch, XGBoost inference, and quality scoring pipeline.

#### Step 7 — Format & deliver results
- **Code in JavaScript2** — formats the JSON score response into a readable WhatsApp message.
- **HTTP Request1** — posts the formatted result to a secondary CIH backend endpoint for logging.
- **Update a row1** — saves the assessment result to the database.
- **Send WhatsApp** — delivers the credit quality score back to the farmer's phone.

#### Step 8 — Database routing (If2)
For messages that don't trigger a new assessment:
- **true** (farmer record exists) → **Update a row2** → Send WhatsApp confirmation.
- **false** (new farmer) → **Create a row** → Send WhatsApp welcome message.

![n8n workflow end](screenshots/n8n_end.png)

### HTTP Request node configuration

```
Method : POST
URL    : https://agricredit-model-quality-score.onrender.com/assess
Body   : {
           "latitude":      "{{ $json.latitude }}",
           "longitude":     "{{ $json.longitude }}",
           "farmerPhone":   "{{ $json.farmerPhone }}",
           "messageType":   "location",
           "shouldProcess": true
         }
```

---

## Project Structure

```
app.py                            — FastAPI server (deployed on Render)
requirements.txt                  — Python dependencies
.python-version                   — Python 3.11.9 (for Render)

notebooks/
  02_improved_yield_model.ipynb   — Train the XGBoost yield estimator
  03_gee_fetcher.ipynb            — Fetch 24-month satellite profiles via GEE
  04_quality_scorer.ipynb         — Compute quality score from farm profile

saved_model/
  yield_model_v2.joblib           — Trained model (14 features → yield)
  feature_cols_v2.joblib          — Feature column order for inference

yield_prediction_dataset.csv      — Training dataset (Kaggle)
PIPELINE.md                       — Full technical pipeline documentation
```

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

- Rouse et al. (1974) — NDVI
- Gitelson et al. (1996) — GNDVI
- Gao (1996) — NDWI
- Huete (1988) — SAVI
- Funk et al. (2015) — CHIRPS precipitation
- Wan (2014) — MODIS LST
- Reichle et al. (2019) — NASA SMAP soil moisture
- Allen et al. (1998) — FAO Irrigation & Drainage Paper No. 56 (heat stress threshold)
- FAO (2012) — Composite indicator methodology
- Myneni et al. (1995) — Active vegetation NDVI threshold
