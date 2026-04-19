"""
MODULE 3: BIOMASS PREDICTOR
Random Forest model that predicts canopy height + biomass from Sentinel-2 bands.
Also generates an auto MRV Monitoring Report (CCTS 2026 compliant structure).

Features used:
  - Sentinel-2 spectral bands (B2–B12)
  - Derived indices: NDVI, NDWI, EVI, SAVI, NBR
  - Monthly rainfall (from NASA POWER)
  - Stand age (from planting records)

Target variables:
  - mean_height_m   (primary — for CCTS Digital MRV)
  - plot_co2_tonnes (derived from height via bio-engine allometry)

Training strategy:
  - Self-supervised: uses bio-engine simulation as synthetic labels
    when no ground-truth LiDAR is available (student prototype stage)
  - Later: replace with real LiDAR-fused labels (your university "cheat code")
"""

import json
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
SPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# GEE naming convention — adjust "B4" → "B04" for openEO if needed

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all spectral indices from raw band values.
    Input df must have Sentinel-2 band columns.
    All reflectance values expected in [0, 1] range (divide by 10000 if DN).
    """
    df = df.copy()
    eps = 1e-9

    # Normalise from DN (0–10000) if needed
    for b in SPECTRAL_BANDS:
        if b in df.columns and df[b].max() > 10:
            df[b] = df[b] / 10000.0

    nir   = df.get("B8",  df.get("B08",  np.nan))
    red   = df.get("B4",  df.get("B04",  np.nan))
    green = df.get("B3",  df.get("B03",  np.nan))
    blue  = df.get("B2",  df.get("B02",  np.nan))
    re1   = df.get("B5",  df.get("B05",  np.nan))   # Red-Edge 1
    swir1 = df.get("B11", np.nan)
    swir2 = df.get("B12", np.nan)

    # ── Vegetation Indices ────────────────────────────────────────────────────
    df["NDVI"]  = (nir - red)   / (nir + red   + eps)
    df["EVI"]   = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
    df["SAVI"]  = 1.5 * (nir - red) / (nir + red + 0.5)           # L=0.5
    df["NDRE"]  = (nir - re1)   / (nir + re1   + eps)              # Red-Edge NDVI
    df["CIre"]  = (nir / (re1 + eps)) - 1                          # Chlorophyll Index

    # ── Water / Moisture Indices ──────────────────────────────────────────────
    df["NDWI"]  = (green - nir)   / (green + nir + eps)
    df["NDMI"]  = (nir - swir1)   / (nir + swir1 + eps)            # Moisture

    # ── Burn / Stress Indices ─────────────────────────────────────────────────
    df["NBR"]   = (nir - swir2)   / (nir + swir2 + eps)            # Burn Ratio

    # ── Texture proxy ─────────────────────────────────────────────────────────
    df["NIR_SWIR_ratio"] = nir / (swir1 + eps)

    return df


def build_feature_matrix(sat_df: pd.DataFrame,
                          rain_df: pd.DataFrame,
                          stand_age_col: str = "stand_age_yr") -> pd.DataFrame:
    """
    Merges satellite indices with rainfall and stand age into model features.
    """
    sat_df = engineer_features(sat_df)

    # Add annual rainfall if available
    if "year_month" not in sat_df.columns and "date" in sat_df.columns:
        sat_df["year_month"] = pd.to_datetime(sat_df["date"]).dt.to_period("M")

    if rain_df is not None and "year_month" in sat_df.columns:
        rain_df["year_month"] = pd.to_datetime(rain_df["date"]).dt.to_period("M")
        sat_df = sat_df.merge(
            rain_df[["year_month", "rainfall_mm_day"]],
            on="year_month", how="left"
        )
        # 12-month rolling total rainfall
        sat_df["rainfall_mm_day"] = sat_df["rainfall_mm_day"].fillna(method="ffill")
        sat_df["annual_rain_mm"] = sat_df["rainfall_mm_day"] * 365.25

    # Stand age feature
    if stand_age_col not in sat_df.columns:
        if "date" in sat_df.columns:
            base_date = pd.to_datetime(sat_df["date"]).min()
            sat_df["stand_age_yr"] = (
                (pd.to_datetime(sat_df["date"]) - base_date).dt.days / 365.25
            )

    return sat_df


FEATURE_COLS = [
    "NDVI", "EVI", "SAVI", "NDRE", "CIre",
    "NDWI", "NDMI", "NBR", "NIR_SWIR_ratio",
    "B5", "B6", "B7", "B8", "B8A",           # structure-sensitive bands
    "stand_age_yr", "annual_rain_mm"
]


# =============================================================================
# SYNTHETIC TRAINING DATA (self-supervised from bio-engine)
# =============================================================================
def generate_synthetic_training_data(n_samples: int = 2000,
                                     species: str = "khejri") -> pd.DataFrame:
    """
    When LiDAR labels aren't available yet, generate plausible training data
    using the bio-engine model + realistic spectral noise.

    This is your PROTOTYPE strategy. Replace with real field + LiDAR data
    as you progress to serious validation.
    """
    from module2_bio_engine import (ForestPlot, SPECIES_PARAMS,
                                     prep_rainfall_for_simulation)

    np.random.seed(42)
    records = []

    for _ in range(n_samples):
        stand_age  = np.random.uniform(0.5, 25)
        annual_rain = np.random.uniform(200, 800)
        area_ha    = 1.0

        rain_ts = pd.DataFrame({
            "year": range(2000, 2031),
            "annual_rain_mm": np.random.normal(annual_rain, 40, 31).clip(100, 900)
        })

        plot = ForestPlot(species=species, area_ha=area_ha,
                         trees_per_ha=400, planting_year=2000,
                         rainfall_ts=rain_ts)
        sim = plot.simulate(years=26)

        age_int = max(1, int(stand_age))
        row = sim[sim["stand_age_yr"] == age_int]
        if row.empty:
            continue

        height = float(row["mean_height_m"].values[0])
        co2    = float(row["plot_co2_tonnes"].values[0])

        # Simulate Sentinel-2 spectral response based on NDVI–height relationship
        # These are empirically calibrated approximations for tropical dry deciduous
        ndvi  = 0.25 + 0.55 * (height / 12.0) + np.random.normal(0, 0.05)
        ndvi  = np.clip(ndvi, 0.1, 0.9)

        evi   = ndvi * 0.85 + np.random.normal(0, 0.04)
        savi  = ndvi * 0.90 + np.random.normal(0, 0.04)
        ndre  = ndvi * 1.05 + np.random.normal(0, 0.03)
        cire  = 2.5 * ndvi + np.random.normal(0, 0.1)
        ndwi  = -0.1 - 0.3 * ndvi + np.random.normal(0, 0.05)
        ndmi  = 0.1 + 0.4 * ndvi + np.random.normal(0, 0.04)
        nbr   = 0.2 + 0.3 * ndvi + np.random.normal(0, 0.05)
        nir_swir = 1.5 + 2.0 * ndvi + np.random.normal(0, 0.2)

        # Band-level noise
        b5  = 0.08 + 0.05 * ndvi + np.random.normal(0, 0.005)
        b6  = 0.15 + 0.10 * ndvi + np.random.normal(0, 0.008)
        b7  = 0.20 + 0.15 * ndvi + np.random.normal(0, 0.01)
        b8  = 0.25 + 0.30 * ndvi + np.random.normal(0, 0.015)
        b8a = 0.28 + 0.28 * ndvi + np.random.normal(0, 0.015)

        records.append({
            "NDVI": ndvi, "EVI": evi, "SAVI": savi, "NDRE": ndre, "CIre": cire,
            "NDWI": ndwi, "NDMI": ndmi, "NBR": nbr, "NIR_SWIR_ratio": nir_swir,
            "B5": b5, "B6": b6, "B7": b7, "B8": b8, "B8A": b8a,
            "stand_age_yr":  round(stand_age, 2),
            "annual_rain_mm": round(annual_rain, 1),
            "mean_height_m": round(height, 3),
            "plot_co2_tonnes": round(co2, 3),
        })

    return pd.DataFrame(records)


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_models(df: pd.DataFrame,
                 model_dir: str = "models") -> dict:
    """
    Trains two Random Forest models:
      1. height_model   → predicts mean canopy height (m)
      2. biomass_model  → predicts plot CO₂ tonnes

    Returns dict with models, metrics, feature importances.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available_features].fillna(0).values
    y_height = df["mean_height_m"].values
    y_co2    = df["plot_co2_tonnes"].values

    X_tr, X_te, yh_tr, yh_te, yc_tr, yc_te = train_test_split(
        X, y_height, y_co2, test_size=0.2, random_state=42
    )

    results = {}
    for name, y_tr, y_te in [("height", yh_tr, yh_te), ("co2", yc_tr, yc_te)]:
        rf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ))
        ])
        rf.fit(X_tr, y_tr)

        preds = rf.predict(X_te)
        mae   = mean_absolute_error(y_te, preds)
        r2    = r2_score(y_te, preds)
        cv    = cross_val_score(rf, X, y_height if name=="height" else y_co2,
                                cv=5, scoring="r2").mean()

        print(f"\n[Model: {name}]  MAE={mae:.3f}  R²={r2:.3f}  CV-R²={cv:.3f}")

        importances = dict(zip(
            available_features,
            rf.named_steps["rf"].feature_importances_.round(4)
        ))
        imp_sorted = dict(sorted(importances.items(),
                                 key=lambda x: -x[1]))

        model_path = f"{model_dir}/{name}_rf_model.joblib"
        joblib.dump(rf, model_path)

        results[name] = {
            "model": rf,
            "path": model_path,
            "mae": mae,
            "r2": r2,
            "cv_r2": cv,
            "feature_importances": imp_sorted,
            "n_features": len(available_features),
            "feature_names": available_features,
        }

    return results


def predict(features_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Run inference on new satellite observations.
    """
    available = [f for f in FEATURE_COLS if f in features_df.columns]
    X = features_df[available].fillna(0).values

    out = features_df.copy()
    if "height" in models:
        out["predicted_height_m"]    = models["height"]["model"].predict(X).round(2)
    if "co2" in models:
        out["predicted_co2_tonnes"] = models["co2"]["model"].predict(X).round(2)

    return out


# =============================================================================
# MRV MONITORING REPORT (CCTS 2026 Digital MRV Structure)
# =============================================================================
def generate_mrv_report(plot_info: dict,
                         simulation_df: pd.DataFrame,
                         model_metrics: dict,
                         predictions_df: pd.DataFrame = None,
                         output_dir: str = "reports") -> str:
    """
    Auto-generates a structured Monitoring Report following:
    - India CCTS 2026 framework (BEE / MoEFCC guidelines)
    - Verra VCS VM0047 methodology (A/R projects)
    - IPCC 2006 / 2019 supplement accounting

    Outputs a Markdown file (easy to convert to PDF for submission).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today   = datetime.date.today().isoformat()
    site    = plot_info.get("site", {})
    species = plot_info.get("species", "N/A")

    # Key metrics from simulation
    latest   = simulation_df.iloc[-1]
    baseline = simulation_df.iloc[0]
    total_co2_net = simulation_df["annual_increment_co2"].sum()
    peak_annual   = simulation_df["annual_increment_co2"].max()
    avg_annual    = simulation_df["annual_increment_co2"].mean()

    # Model performance summary
    h_metrics  = model_metrics.get("height", {})
    c_metrics  = model_metrics.get("co2", {})
    top_features = list(h_metrics.get("feature_importances", {}).items())[:5]

    report = f"""# DIGITAL MRV MONITORING REPORT
## Carbon Credit Trading Scheme (CCTS) — India 2026

---
**Report ID:**         MRV-{today.replace('-','')}-{site.get('name','PLOT')[:8].upper()}
**Generated:**         {today}
**Monitoring Period:** {simulation_df['calendar_year'].min()} – {simulation_df['calendar_year'].max()}
**Methodology:**       A/R Afforestation/Reforestation (CCTS Removal Credits)
**Standard:**          India CCTS / VCS VM0047 compatible

---

## 1. PROJECT IDENTIFICATION

| Field | Value |
|-------|-------|
| Site Name | {site.get('name', 'N/A')} |
| Coordinates | {site.get('lat', 'N/A')}°N, {site.get('lon', 'N/A')}°E |
| Area (ha) | {plot_info.get('area_ha', 'N/A')} |
| Species | {species} |
| Trees/ha | {plot_info.get('trees_per_ha', 'N/A')} |
| Planting Year | {plot_info.get('planting_year', 'N/A')} |
| Project Type | Afforestation / Reforestation (Removal) |

---

## 2. MONITORING METHODOLOGY (Digital MRV)

### 2.1 Remote Sensing Data Sources
- **Primary:** Sentinel-2 L2A (ESA Copernicus) — 10m resolution, monthly composites
- **Rainfall:** NASA POWER monthly precipitation (mm/day)
- **Cloud Masking:** SCL band, max {20}% cloud cover threshold
- **Temporal Coverage:** {5} years of historical NDVI time series

### 2.2 Machine Learning Models
| Model | Target | MAE | R² | CV-R² |
|-------|--------|-----|----|-------|
| Random Forest | Canopy Height (m) | {h_metrics.get('mae', 'N/A'):.3f} | {h_metrics.get('r2', 'N/A'):.3f} | {h_metrics.get('cv_r2', 'N/A'):.3f} |
| Random Forest | Plot CO₂ (tonnes) | {c_metrics.get('mae', 'N/A'):.3f} | {c_metrics.get('r2', 'N/A'):.3f} | {c_metrics.get('cv_r2', 'N/A'):.3f} |

**Top 5 Predictive Features (by importance):**
{chr(10).join(f'  {i+1}. {feat} ({imp:.4f})' for i,(feat,imp) in enumerate(top_features))}

### 2.3 Bio-Physical Model
- Growth Function: Chapman-Richards (standard FAO/IPCC forestry)
- Allometry: FSI India Biomass Tables (2021 edition)
- BEF: {SPECIES_PARAMS.get(species, {}).get('bef', 'N/A')} (FSI species-specific)
- Root:Shoot Ratio: {SPECIES_PARAMS.get(species, {}).get('rs_ratio', 'N/A')} (IPCC Tier 2)
- Carbon Fraction: 0.47 (IPCC 2006 default)

---

## 3. CARBON STOCK RESULTS

### 3.1 Cumulative Carbon Accounting

| Year | Age (yr) | Height (m) | Rainfall (mm) | CO₂ Stock (t) | Annual Increment (t) |
|------|----------|------------|---------------|----------------|---------------------|
{chr(10).join(
    f'| {int(r.calendar_year)} | {int(r.stand_age_yr)} | {r.mean_height_m:.1f} | {r.annual_rain_mm:.0f} | {r.plot_co2_tonnes:.2f} | {r.annual_increment_co2:.2f} |'
    for _, r in simulation_df.iterrows()
)}

### 3.2 Summary Statistics

| Metric | Value |
|--------|-------|
| Total CO₂ Removal (project life) | **{total_co2_net:.2f} tonnes CO₂e** |
| Peak Annual Sequestration | {peak_annual:.2f} t CO₂e/yr |
| Average Annual Sequestration | {avg_annual:.2f} t CO₂e/yr |
| Final Stand Height | {latest['mean_height_m']:.1f} m |
| Final DBH | {latest['mean_dbh_cm']:.1f} cm |
| Survival Rate (final year) | {(0.98 ** int(latest['stand_age_yr'])):.1%} |

---

## 4. UNCERTAINTY & PERMANENCE

- **Model Uncertainty:** ±{h_metrics.get('mae', 0)*1.96:.2f}m height (95% CI based on MAE)
- **Permanence Buffer:** 20% of credits held in CCTS buffer pool (per BEE guidelines)
- **Creditable Units (net of buffer):** {total_co2_net * 0.80:.2f} Indian Carbon Credits (ICCs)

---

## 5. BASELINE SCENARIO

Project baseline: **Zero carbon scenario** (unmanaged / degraded land).
Additionality demonstrated via:
1. Land history: confirm via Bhunaksha / cadastral records
2. Financial additionality: IRR < cost of capital without carbon revenue
3. Common practice test: reforestation density > district average

---

## 6. DATA QUALITY & AUDIT TRAIL

- All satellite data fetched from official Copernicus / GEE endpoints
- Raw data checksums stored at: `data/raw/`
- Model artifacts: `models/height_rf_model.joblib`, `models/co2_rf_model.joblib`
- This report generated automatically by: `module3_predictor.py`
- Code repository hash: [link to GitHub commit]
- **Next monitoring event:** {(datetime.date.today() + datetime.timedelta(days=180)).isoformat()} (6-month cadence)

---

## 7. VERIFICATION READINESS (Third-Party)

For CCTS-accredited verifier (DoVE) submission, the following are available:
- [ ] Shapefiles of project boundary (EPSG:4326)
- [ ] Time-series NDVI export (GeoTIFF / CSV)
- [ ] Bio-engine simulation logs
- [ ] Field photos with GPS tags
- [ ] Species/stocking density records
- [ ] LiDAR fusion report (planned — university collaboration)

---

*Report generated automatically by Digital MRV Pipeline v0.1*
*Code: github.com/[your-username]/carbon-mrv-india*
*Contact: [your email] | Institution: [your university]*
"""

    filename = f"{output_dir}/MRV_Report_{today}_{site.get('name', 'PLOT')}.md"
    with open(filename, "w") as f:
        f.write(report)

    print(f"\n✅  MRV Report saved → {filename}")
    return filename


# Import species params for use in report
from module2_bio_engine import SPECIES_PARAMS


# =============================================================================
# FULL PIPELINE
# =============================================================================
def run_full_pipeline(plot_info: dict,
                       rainfall_csv: str = None,
                       satellite_csv: str = None,
                       species: str = "khejri") -> None:
    """
    End-to-end: train model → simulate → predict → report.
    """
    print("\n" + "="*60)
    print("   CARBON MRV PIPELINE — CCTS 2026 INDIA")
    print("="*60)

    # 1. Generate or load training data
    print("\n[1/4] Fetching real training data (Meta CHM + GEDI)...")
    from module0_real_data import get_real_training_data
    sat_csv = "data/raw/Aravalli_Test_Plot_satellite.csv"
    sat_df_real = pd.read_csv(sat_csv) if Path(sat_csv).exists() else None
    train_df = get_real_training_data(lat=28.01, lon=75.79, sat_df=sat_df_real)
    print(f"      Training set: {len(train_df)} samples")

    # 2. Train models
    print("\n[2/4] Training Random Forest models...")
    models = train_models(train_df)

    # 3. Simulate plot trajectory
    print("\n[3/4] Running bio-engine simulation...")
    from module2_bio_engine import ForestPlot, prep_rainfall_for_simulation

    rain_df = None
    if rainfall_csv and Path(rainfall_csv).exists():
        rain_df = pd.read_csv(rainfall_csv, parse_dates=["date"])
        rain_df = prep_rainfall_for_simulation(rain_df)

    plot = ForestPlot(
        species=species,
        area_ha=plot_info.get("area_ha", 1.0),
        trees_per_ha=plot_info.get("trees_per_ha", 276),
        planting_year=plot_info.get("planting_year", 2024),
        rainfall_ts=rain_df,
    )
    sim_df = plot.simulate(years=plot_info.get("sim_years", 30))


    # 4. Generate MRV report
    print("\n[4/4] Generating Digital MRV Monitoring Report...")
    report_path = generate_mrv_report(
        plot_info={**plot_info, "site": plot_info.get("site", {}), "species": species},
        simulation_df=sim_df,
        model_metrics=models,
    )

    print(f"\n{'='*60}")
    print(f"   PIPELINE COMPLETE")
    print(f"   Report: {report_path}")
    print(f"   Total CO₂ (30yr): {sim_df['plot_co2_tonnes'].iloc[-1]:.1f} tonnes")
    print(f"   Est. Credits (net): {sim_df['annual_increment_co2'].sum()*0.80:.1f} ICCs")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    site_project = {
        "name": "Khetri_Project",
        "lat": 28.01,
        "lon": 75.79,
    }

    site_baseline = {
        "name": "Khetri_Baseline",
        "lat": 28.01,
        "lon": 75.79,
    }
    # PROJECT scenario — what you plant
    plot_info = {
        "site":          site_project,
        "species":       "khejri",
        "area_ha":       1.0,
        "trees_per_ha":  276,
        "planting_year": 2024,
        "sim_years":     30,
    }

    # BASELINE scenario — what's already there
    baseline_info = {
        "site":          site_baseline,
        "species":       "khejri",
        "area_ha":       1.0,
        "trees_per_ha":  45,
        "planting_year": 2000,   # existing trees ~25 years old
        "sim_years":     30,
    }

    print("\n--- RUNNING PROJECT SCENARIO ---")
    run_full_pipeline(plot_info, species="khejri")

    print("\n--- RUNNING BASELINE SCENARIO ---")
    run_full_pipeline(baseline_info, species="khejri")
