# Carbon MRV India — Digital MRV Pipeline (CCTS 2026)

A Python-based Minimum Viable Model for carbon credit monitoring in Indian
reforestation projects. Built for the India Carbon Credit Trading Scheme (CCTS).

## Architecture

```
carbon_mrv/
├── module1_fetcher.py    ← Sentinel-2 NDVI + NASA POWER rainfall
├── module2_bio_engine.py ← Chapman-Richards growth + allometric biomass
├── module3_predictor.py  ← Random Forest predictor + MRV report generator
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Authenticate (choose one backend)

**Google Earth Engine** (most powerful):
```bash
earthengine authenticate
# Follow browser instructions, paste token
```

**Copernicus openEO** (free, no credit card):
```python
import openeo
conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
conn.authenticate_oidc()   # Browser popup
```

### 3. Run the full pipeline
```bash
python module3_predictor.py
```

This will:
1. Generate synthetic training data using the bio-engine
2. Train Random Forest models (height + CO₂)
3. Run a 30-year growth simulation for your plot
4. Output a CCTS-structured MRV Monitoring Report → `reports/`

## Customise for Your Plot

Edit the `plot_info` dict in `module3_predictor.py`:

```python
plot_info = {
    "site": {
        "name": "Your_Plot_Name",
        "lat":  28.367,    # Your GPS coordinates
        "lon":  75.588,
    },
    "species":       "khejri",   # teak | khejri | neem | generic_tropical
    "area_ha":       1.0,
    "trees_per_ha":  400,
    "planting_year": 2024,
    "sim_years":     30,
}
```

## Species Supported

| Species | Region | Rain Optimum | Wood Density |
|---------|--------|-------------|--------------|
| Teak | Peninsular India | 1200 mm | 0.65 g/cm³ |
| Khejri | Rajasthan / Arid | 350 mm | 0.85 g/cm³ |
| Neem | Pan-India | 600 mm | 0.56 g/cm³ |
| Generic tropical | Fallback | 900 mm | 0.60 g/cm³ |

## CCTS 2026 Compliance Notes

- Project type: **Afforestation/Reforestation (A/R)** — "Removal" credits
- Follows BEE / MoEFCC Digital MRV requirements
- 20% permanence buffer deducted automatically
- Compatible with VCS VM0047 methodology
- Report output structured for DoVE (Designated Operational Verifier Entity) submission

## Roadmap to $10M Startup

| Stage | What to add | Where to get it |
|-------|------------|-----------------|
| Now | Synthetic training data (this codebase) | ✅ Done |
| 3 months | Real field measurements (height, DBH) | Field surveys |
| 6 months | LiDAR fusion | University credentials + ISRO Bhuvan |
| 12 months | IoT sensor integration (soil moisture) | NSRCEL / C-CAMP grant |
| 18 months | First verified credit issuance | CCTS registry (BEE) |

## University Cheat Codes

- **LiDAR data**: Apply via ISRO's NRSC Data Centre with university credentials
- **Incubation**: NSRCEL Climate CoE (IIM-B), C-CAMP Climate Solutions call
- **Legal/regulatory**: MoEFCC CCTS helpdesk (free for registered projects)
- **Computing**: Use Google Earth Engine Education license (free with .edu email)
