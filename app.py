"""
CIH Farm Credit Assessment API
POST /assess  →  receives farmer location JSON, returns quality score report
"""

import os, json, tempfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import ee
from datetime import date
from dateutil.relativedelta import relativedelta


# ── Startup: load model ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL      = joblib.load(os.path.join(BASE_DIR, "saved_model", "yield_model_v2.joblib"))
FEAT_COLS  = joblib.load(os.path.join(BASE_DIR, "saved_model", "feature_cols_v2.joblib"))


# ── GEE authentication ─────────────────────────────────────────────────────────
# Lazy init: called on first /assess request, not at startup.
# On Render: set env vars GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY
# Locally:   uses your cached OAuth token (~/.config/earthengine/credentials)
_gee_initialized = False

def ensure_gee():
    global _gee_initialized
    if _gee_initialized:
        return
    sa   = os.environ.get("GEE_SERVICE_ACCOUNT")
    key  = os.environ.get("GEE_PRIVATE_KEY")
    proj = os.environ.get("GEE_PROJECT", "")
    if sa and key:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(key)
            key_path = f.name
        credentials = ee.ServiceAccountCredentials(sa, key_path)
        ee.Initialize(credentials, project=proj)
    else:
        ee.Initialize(project=proj or None)
    _gee_initialized = True


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="CIH Farm Credit Assessment", version="1.0")


class FarmerMessage(BaseModel):
    messageType:  Optional[str] = "location"
    farmerPhone:  Optional[str] = ""
    messageText:  Optional[str] = ""
    hasLocation:  Optional[str] = None
    latitude:     str
    longitude:    str
    shouldProcess: Optional[bool] = True


# ── GEE fetch helpers ──────────────────────────────────────────────────────────
def _fetch_month(lat, lon, year, month, radius_m=500):
    start = f"{year}-{month:02d}-01"
    end   = (date(year, month, 1) + relativedelta(months=1)).strftime("%Y-%m-%d")
    pt    = ee.Geometry.Point([lon, lat])
    farm  = pt.buffer(radius_m)
    wide  = pt.buffer(10_000)

    def add_indices(img):
        nir, red, grn = img.select("B8"), img.select("B4"), img.select("B3")
        return img.addBands([
            nir.subtract(red).divide(nir.add(red).add(1e-9)).rename("NDVI"),
            nir.subtract(grn).divide(nir.add(grn).add(1e-9)).rename("GNDVI"),
            grn.subtract(nir).divide(grn.add(nir).add(1e-9)).rename("NDWI"),
            nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI"),
        ])

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(farm).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B3","B4","B8"]))
    s2_stats = (s2.map(add_indices).select(["NDVI","GNDVI","NDWI","SAVI"])
                  .median()
                  .reduceRegion(ee.Reducer.mean(), farm, 10, maxPixels=1e9)
                  .getInfo())
    if s2_stats.get("NDVI") is None:
        return None

    lst  = (ee.ImageCollection("MODIS/061/MOD11A1").filterDate(start, end)
              .select("LST_Day_1km").mean().multiply(0.02).subtract(273.15)
              .reduceRegion(ee.Reducer.mean(), wide, 1000, maxPixels=1e9).getInfo())
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end)
              .select("precipitation").sum()
              .reduceRegion(ee.Reducer.mean(), wide, 5000, maxPixels=1e9).getInfo())
    smap = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(start, end)
              .select("sm_surface").mean().multiply(100)
              .reduceRegion(ee.Reducer.mean(), wide, 11000, maxPixels=1e9).getInfo())

    def safe(d, k):
        v = d.get(k)
        return float(v) if v is not None else np.nan

    return {
        "year": year, "month": month,
        "NDVI":          safe(s2_stats, "NDVI"),
        "GNDVI":         safe(s2_stats, "GNDVI"),
        "NDWI":          safe(s2_stats, "NDWI"),
        "SAVI":          safe(s2_stats, "SAVI"),
        "temperature":   safe(lst,  "LST_Day_1km"),
        "rainfall":      safe(rain, "precipitation"),
        "soil_moisture": safe(smap, "sm_surface"),
    }


def fetch_profile(lat, lon, n_months=24):
    today = date.today()
    # Use last complete month
    end = date(today.year, today.month, 1) - relativedelta(months=1)
    cur = end
    months = []
    for _ in range(n_months):
        months.append((cur.year, cur.month))
        cur -= relativedelta(months=1)
    months.reverse()

    rows = []
    for y, m in months:
        row = _fetch_month(lat, lon, y, m)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.interpolate(method="linear", limit_direction="both")
    df["year"]  = df["year"].round().astype(int)
    df["month"] = df["month"].round().astype(int)
    return df


def build_features(df):
    d = df.copy()
    d["month_sin"]       = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"]       = np.cos(2 * np.pi * d["month"] / 12)
    d["veg_stress"]      = (d["NDVI"] - d["GNDVI"]) / (d["NDVI"] + d["GNDVI"] + 1e-9)
    d["water_stress"]    = d["NDVI"] / (d["soil_moisture"] + 1e-9)
    d["ndvi_x_moisture"] = d["NDVI"] * d["soil_moisture"]
    d["thermal_load"]    = np.abs(d["temperature"] - 25.0)
    d["rain_efficiency"] = d["NDVI"] / (d["rainfall"] + 1e-9)
    return d


# ── Scoring ────────────────────────────────────────────────────────────────────
BOUNDS = {
    "productivity": (29.01, 54.53),
    "consistency":  (0.269, 0.028),
    "trend":        (0.85,  1.15),
    "vegetation":   (0.10,  0.50),
    "resilience":   (28.5,  50.4),
}
WEIGHTS = {
    "productivity": 0.30,
    "consistency":  0.25,
    "trend":        0.20,
    "vegetation":   0.15,
    "resilience":   0.10,
}

def clip_score(raw, low, high):
    return float(np.clip((raw - low) / (high - low) * 100, 0, 100))

def compute_score(df):
    mean_y = df["yield_pred"].mean()
    peak_y = df["yield_pred"].quantile(0.90)
    s_prod = clip_score(0.70*mean_y + 0.30*peak_y, *BOUNDS["productivity"])

    cv = df["yield_pred"].std() / (df["yield_pred"].mean() + 1e-9)
    s_cons = clip_score(cv, *BOUNDS["consistency"])

    n = len(df)
    yr1 = df.iloc[:n//2]["yield_pred"].mean()
    yr2 = df.iloc[n//2:]["yield_pred"].mean()
    s_trend = clip_score(yr2 / (yr1 + 1e-9), *BOUNDS["trend"])

    s_veg = clip_score(df["NDVI"].mean(), *BOUNDS["vegetation"])

    stress = (df["rainfall"] < 5) | (df["temperature"] > 35)
    s_res  = clip_score(df[stress]["yield_pred"].mean(), *BOUNDS["resilience"]) if stress.sum() > 0 else 100.0

    sub = {"productivity": s_prod, "consistency": s_cons, "trend": s_trend,
           "vegetation": s_veg, "resilience": s_res}
    quality = sum(sub[k] * WEIGHTS[k] for k in sub)
    return round(quality, 1), {k: round(v, 1) for k, v in sub.items()}


def build_report(lat, lon, df, quality_score, sub_scores):
    r0, r1 = df.iloc[0], df.iloc[-1]
    ndvi_slope  = float(np.polyfit(np.arange(len(df)), df["NDVI"].values, 1)[0])
    peak_ndvi_r = df.loc[df["NDVI"].idxmax()]
    stress_mask = (df["rainfall"] < 5) | (df["temperature"] > 35)
    stress_list = [f"{int(r['year'])}-{int(r['month']):02d}" for _, r in df[stress_mask].iterrows()]
    dry         = df[df["rainfall"] < 10]

    return {
        "farm": {"lat": lat, "lon": lon},
        "assessment_date": str(date.today()),
        "data_window": {
            "from":   f"{int(r0['year'])}-{int(r0['month']):02d}",
            "to":     f"{int(r1['year'])}-{int(r1['month']):02d}",
            "months": len(df),
        },
        "quality_score": quality_score,
        "sub_scores": sub_scores,
        "vegetation": {
            "mean_ndvi":            round(float(df["NDVI"].mean()), 4),
            "mean_gndvi":           round(float(df["GNDVI"].mean()), 4),
            "peak_ndvi":            round(float(df["NDVI"].max()), 4),
            "peak_ndvi_month":      f"{int(peak_ndvi_r['year'])}-{int(peak_ndvi_r['month']):02d}",
            "green_months":         int((df["NDVI"] > 0.30).sum()),
            "ndvi_slope_per_month": round(ndvi_slope, 6),
            "ndvi_trend":           "improving" if ndvi_slope > 0 else "declining",
        },
        "climate": {
            "avg_temperature_c":     round(float(df["temperature"].mean()), 2),
            "avg_rainfall_mm_month": round(float(df["rainfall"].mean()), 2),
            "total_rainfall_mm":     round(float(df["rainfall"].sum()), 1),
            "avg_soil_moisture_pct": round(float(df["soil_moisture"].mean()), 2),
            "dry_season_sm_pct":     round(float(dry["soil_moisture"].mean()), 2) if len(dry) > 0 else None,
            "stress_months_count":   int(stress_mask.sum()),
            "stress_months":         stress_list,
        },
        "resilience": {
            "avg_sm_during_stress": round(float(df[stress_mask]["soil_moisture"].mean()), 2) if stress_mask.sum() > 0 else None,
            "stress_months_count":  int(stress_mask.sum()),
        },
    }


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "CIH Farm Credit Assessment"}


@app.post("/assess")
def assess(payload: FarmerMessage):
    if not payload.shouldProcess:
        raise HTTPException(status_code=400, detail="shouldProcess is false")

    try:
        lat = float(payload.latitude)
        lon = float(payload.longitude)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid latitude/longitude")

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Coordinates out of range")

    try:
        ensure_gee()
        raw_df  = fetch_profile(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GEE fetch failed: {str(e)}")

    if len(raw_df) < 6:
        raise HTTPException(status_code=422, detail="Too few cloud-free satellite images for this location")

    feat_df = build_features(raw_df)
    feat_df["yield_pred"] = MODEL.predict(feat_df[FEAT_COLS].values)

    quality_score, sub_scores = compute_score(feat_df)
    report = build_report(lat, lon, feat_df, quality_score, sub_scores)

    if payload.farmerPhone:
        report["farmerPhone"] = payload.farmerPhone

    return JSONResponse(content=report)
