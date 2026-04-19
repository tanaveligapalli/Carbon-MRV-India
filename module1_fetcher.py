"""
MODULE 1: SATELLITE DATA FETCHER
Pulls 5 years of Sentinel-2 NDVI + spectral bands for a given plot in India.
Supports both Google Earth Engine and Copernicus openEO backends.
"""

import json
import datetime
from pathlib import Path
import numpy as np

# ── GEE backend ──────────────────────────────────────────────────────────────
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# ── openEO backend ───────────────────────────────────────────────────────────
try:
    import openeo
    OPENEO_AVAILABLE = True
except ImportError:
    OPENEO_AVAILABLE = False

# ── pandas/requests for rainfall ─────────────────────────────────────────────
import pandas as pd
import requests


# =============================================================================
# CONFIGURATION — Edit these for your test site
# =============================================================================
DEFAULT_SITE = {
    "name": "Aravalli_Test_Plot",
    "lat": 28.3670,   # Pilani, Rajasthan (change to your actual plot)
    "lon": 75.5880,
    "buffer_m": 50,   # 50m radius ≈ ~1 ha plot
}

YEARS_BACK = 1
END_DATE   = datetime.date.today().isoformat()
START_DATE = (datetime.date.today() - datetime.timedelta(days=365 * YEARS_BACK)).isoformat()

SENTINEL2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CLOUD_COVER_MAX = 20   # % threshold


# =============================================================================
# HELPER: Compute NDVI (also NDWI, EVI for richer features)
# =============================================================================
def compute_indices(nir: np.ndarray, red: np.ndarray,
                    green: np.ndarray = None, swir: np.ndarray = None,
                    blue: np.ndarray = None) -> dict:
    """
    NDVI  = (NIR - Red) / (NIR + Red)         → vegetation density
    NDWI  = (Green - NIR) / (Green + NIR)     → water content / stress
    EVI   = 2.5 * (NIR-Red)/(NIR+6*Red-7.5*Blue+1) → enhanced veg index
    """
    eps = 1e-9
    results = {}

    results["NDVI"] = (nir - red) / (nir + red + eps)

    if green is not None:
        results["NDWI"] = (green - nir) / (green + nir + eps)

    if blue is not None:
        results["EVI"] = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)

    return results


# =============================================================================
# BACKEND A: Google Earth Engine
# =============================================================================
def fetch_via_gee(site: dict = DEFAULT_SITE) -> pd.DataFrame:
    """
    Authenticates with GEE, pulls monthly median Sentinel-2 composites,
    returns a DataFrame with dates + band values + NDVI.

    Run once in terminal: `earthengine authenticate`
    """
    if not GEE_AVAILABLE:
        raise ImportError("Install google-earth-engine: pip install earthengine-api")

    ee.Initialize()

    point  = ee.Geometry.Point([site["lon"], site["lat"]])
    region = point.buffer(site["buffer_m"])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_MAX))
        .select(SENTINEL2_BANDS)
    )

    def extract_image(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            maxPixels=1e6
        )
        ndvi_img = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndvi_val = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            maxPixels=1e6
        )
        return ee.Feature(None, stats.combine(ndvi_val)
                          .set("date", img.date().format("YYYY-MM-dd")))

    feature_collection = ee.FeatureCollection(collection.map(extract_image))
    records = feature_collection.getInfo()["features"]

    rows = [r["properties"] for r in records]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[GEE] Fetched {len(df)} Sentinel-2 observations for {site['name']}")
    return df


# =============================================================================
# BACKEND B: Copernicus openEO (free tier)
# =============================================================================
def fetch_via_openeo(site: dict = DEFAULT_SITE) -> pd.DataFrame:
    """
    Uses Copernicus Data Space openEO endpoint (free, no credit card).
    Sign up at: https://dataspace.copernicus.eu/

    Auth: openeo.authenticate_oidc() — browser pop-up on first run.
    """
    if not OPENEO_AVAILABLE:
        raise ImportError("Install openeo: pip install openeo")

    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()

    bbox = {
        "west": site["lon"] - 0.02,
        "east": site["lon"] + 0.02,
        "south": site["lat"] - 0.02,
        "north": site["lat"] + 0.02,
        "crs": "EPSG:4326"
    }
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[START_DATE, END_DATE],
        bands=["B04", "B08", "B03", "B02", "B11"],
        max_cloud_cover=CLOUD_COVER_MAX,
    )

    # Compute NDVI as a new band
    ndvi_cube = cube.ndvi(nir="B08", red="B04", target_band="NDVI")

    # Download as NetCDF then convert — more reliable than aggregate_spatial
    import tempfile, os
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "ndvi_timeseries.nc")
        ndvi_cube.download(output_path, format="NetCDF")
        ds = xr.open_dataset(output_path)
    numeric_vars = [v for v in ds.data_vars if ds[v].dtype.kind in ('f', 'i')]
    df = ds[numeric_vars].mean(dim=["x", "y"]).to_dataframe().reset_index()



    df["date"] = pd.to_datetime(df.get("t", df.get("time", df.index)))
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[openEO] Fetched {len(df)} observations for {site['name']}")
    return df


# =============================================================================
# RAINFALL: NASA POWER API (free, no key needed)
# =============================================================================
def fetch_rainfall(site: dict = DEFAULT_SITE) -> pd.DataFrame:
    """
    Pulls monthly precipitation (mm/day) from NASA POWER API.
    Docs: https://power.larc.nasa.gov/api/temporal/monthly/point
    """
    start_yr = int(START_DATE[:4])
    end_yr = min(int(END_DATE[:4]), 2025)

    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=PRECTOTCORR"
        f"&community=AG"
        f"&longitude={site['lon']}&latitude={site['lat']}"
        f"&start={start_yr}&end={end_yr}"
        f"&format=JSON"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["properties"]["parameter"]["PRECTOTCORR"]

    records = []
    for yyyymm, val in raw.items():
        year, month = int(yyyymm[:4]), int(yyyymm[4:])
        if month < 1 or month > 12:
            continue
        records.append({
            "date": pd.Timestamp(year=year, month=month, day=1),
            "rainfall_mm_day": val
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"[NASA POWER] Fetched {len(df)} monthly rainfall records")
    return df


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================
def fetch_all(site: dict = DEFAULT_SITE,
              backend: str = "gee",
              output_dir: str = "data/raw") -> dict:
    """
    Master fetch function. Returns dict with satellite_df and rainfall_df.
    Also saves CSVs to output_dir for offline use.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Satellite data
    if backend == "gee":
        sat_df = fetch_via_gee(site)
    elif backend == "openeo":
        sat_df = fetch_via_openeo(site)
    else:
        raise ValueError("backend must be 'gee' or 'openeo'")

    # Rainfall data
    rain_df = fetch_rainfall(site)

    # Merge on month
    sat_df["year_month"] = sat_df["date"].dt.to_period("M")
    rain_df["year_month"] = rain_df["date"].dt.to_period("M")
    merged = sat_df.merge(rain_df[["year_month", "rainfall_mm_day"]],
                          on="year_month", how="left")

    # Save
    sat_path  = f"{output_dir}/{site['name']}_satellite.csv"
    rain_path = f"{output_dir}/{site['name']}_rainfall.csv"
    sat_df.to_csv(sat_path, index=False)
    rain_df.to_csv(rain_path, index=False)

    print(f"\n✅  Saved satellite data  → {sat_path}")
    print(f"✅  Saved rainfall data   → {rain_path}")

    return {"satellite": merged, "rainfall": rain_df, "site": site}


if __name__ == "__main__":
    # Quick test with openEO (no GEE setup needed)
    data = fetch_all(backend="openeo")
    print("\nSample rows:")
    print(data["satellite"].head())
