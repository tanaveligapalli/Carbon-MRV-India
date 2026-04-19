"""
MODULE 2: BIO-ENGINE
Mathematical simulation of tree growth adjusted for local rainfall.
Based on:
  - Chapman-Richards growth function (standard forestry)
  - Rainfall-adjusted growth modifier (Thornthwaite water-balance concept)
  - Allometric equations for Indian tropical dry deciduous / semi-arid species
    (source: FSI biomass equations, ICFRE 2021)

Species supported:
  - Teak (Tectona grandis)          ← plantation, high credit value
  - Khejri (Prosopis cineraria)     ← Rajasthan native, agroforestry
  - Neem (Azadirachta indica)       ← common, fast grower
  - Generic tropical hardwood       ← fallback
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# SPECIES DATABASE
# =============================================================================
SPECIES_PARAMS = {
    "teak": {
        "A": 35.0,       # asymptotic height (m) — maximum achievable height
        "k": 0.055,      # growth rate constant (yr⁻¹)
        "m": 0.6,        # shape parameter (Chapman-Richards)
        "wue": 2.8,      # water use efficiency index (relative)
        "rain_opt": 1200, # optimal annual rainfall (mm)
        "rain_min": 500,
        "rain_max": 2500,
        "wood_density": 0.65,  # g/cm³
        "bef": 1.4,      # Biomass Expansion Factor (FSI)
        "rs_ratio": 0.26 # root:shoot ratio (IPCC Tier 2 for tropical)
    },
    "khejri": {
        "A": 12.0,
        "k": 0.08,
        "m": 0.5,
        "wue": 4.2,      # very drought tolerant
        "rain_opt": 350,
        "rain_min": 150,
        "rain_max": 800,
        "wood_density": 0.85,
        "bef": 1.6,
        "rs_ratio": 0.40  # deep roots for arid species
    },
    "neem": {
        "A": 20.0,
        "k": 0.09,
        "m": 0.55,
        "wue": 3.5,
        "rain_opt": 600,
        "rain_min": 250,
        "rain_max": 1500,
        "wood_density": 0.56,
        "bef": 1.35,
        "rs_ratio": 0.28
    },
    "generic_tropical": {
        "A": 25.0,
        "k": 0.06,
        "m": 0.58,
        "wue": 3.0,
        "rain_opt": 900,
        "rain_min": 400,
        "rain_max": 2000,
        "wood_density": 0.60,
        "bef": 1.45,
        "rs_ratio": 0.30
    }
}


# =============================================================================
# RAINFALL MODIFIER
# =============================================================================
def rainfall_growth_modifier(annual_rain_mm: float, params: dict) -> float:
    """
    Returns a scalar [0, 1] that multiplies the baseline growth rate.
    Uses a Gaussian response curve centred on the species' optimal rainfall.

    Below rain_min or above rain_max → severe stress (modifier < 0.1)
    At rain_opt → modifier = 1.0
    """
    mu  = params["rain_opt"]
    lo  = params["rain_min"]
    hi  = params["rain_max"]

    if annual_rain_mm < lo:
        # Linear stress below minimum — drought kill zone
        modifier = max(0.0, 0.1 * annual_rain_mm / lo)
    elif annual_rain_mm > hi:
        # Waterlogging stress above maximum
        modifier = max(0.0, 1.0 - 0.3 * (annual_rain_mm - hi) / hi)
    else:
        # Gaussian curve between lo and hi
        sigma = (hi - lo) / 4.0
        modifier = np.exp(-0.5 * ((annual_rain_mm - mu) / sigma) ** 2)

    return float(np.clip(modifier, 0.0, 1.0))


# =============================================================================
# CHAPMAN-RICHARDS GROWTH FUNCTION
# =============================================================================
def chapman_richards_height(t: float, A: float, k: float, m: float,
                             rain_modifier: float = 1.0) -> float:
    """
    H(t) = A * [1 - exp(-k_eff * t)]^(1/(1-m))

    where k_eff = k * rain_modifier  (rainfall scales the rate, not the asymptote)

    Parameters
    ----------
    t              : stand age in years
    A              : asymptotic height (m)
    k              : intrinsic growth rate (yr⁻¹)
    m              : shape parameter (0 < m < 1)
    rain_modifier  : scalar from rainfall_growth_modifier()
    """
    k_eff = k * rain_modifier
    if t <= 0:
        return 0.0
    return A * (1 - np.exp(-k_eff * t)) ** (1 / (1 - m))


# =============================================================================
# ALLOMETRIC BIOMASS EQUATIONS (FSI / ICFRE)
# =============================================================================
def height_to_dbh(height_m: float, species: str = "generic_tropical") -> float:
    """
    DBH (cm) from height using H-D allometry.
    Coefficients from Forest Survey of India (2021) biomass tables.
    DBH = exp(a + b * ln(H))
    """
    # Coefficients [a, b] per species
    hd_coeff = {
        "teak":             [0.95, 0.82],
        "khejri":           [0.75, 0.78],
        "neem":             [0.88, 0.80],
        "generic_tropical": [0.90, 0.81],
    }
    a, b = hd_coeff.get(species, hd_coeff["generic_tropical"])
    if height_m <= 0:
        return 0.0
    return np.exp(a + b * np.log(max(height_m, 0.01)))


def dbh_to_agb(dbh_cm: float, wood_density: float) -> float:
    """
    Above-Ground Biomass (kg) via Chave et al. (2014) pantropical equation:
    AGB = 0.0673 * (ρ * DBH² * H)^0.976

    We use the simplified 2-parameter form common in Indian forestry:
    AGB = wood_density * exp(-2.977 + 2.619 * ln(DBH))
    """
    if dbh_cm <= 0:
        return 0.0
    agb_kg = wood_density * np.exp(-2.977 + 2.619 * np.log(dbh_cm))
    return max(agb_kg, 0.0)


def agb_to_total_biomass(agb_kg: float, bef: float, rs_ratio: float) -> dict:
    """
    Total biomass components per tree:
      - AGB (above-ground biomass)
      - BGB (below-ground biomass) = AGB * rs_ratio
      - Dead wood (10% of AGB for managed plantations per IPCC default)
      - Total biomass
      - Carbon stock (biomass × 0.47 — IPCC default carbon fraction)
      - CO₂ equivalent (C × 44/12)
    """
    bgb = agb_kg * rs_ratio
    dead_wood = agb_kg * 0.10
    total_biomass = agb_kg + bgb + dead_wood
    carbon_kg  = total_biomass * 0.47
    co2_eq_kg  = carbon_kg * (44 / 12)

    return {
        "agb_kg":        round(agb_kg, 3),
        "bgb_kg":        round(bgb, 3),
        "dead_wood_kg":  round(dead_wood, 3),
        "total_biomass_kg": round(total_biomass, 3),
        "carbon_kg":     round(carbon_kg, 3),
        "co2_eq_kg":     round(co2_eq_kg, 3),
    }


# =============================================================================
# MAIN SIMULATION CLASS
# =============================================================================
@dataclass
class ForestPlot:
    """
    Represents a reforestation / A-R plot.

    Parameters
    ----------
    species       : species key (see SPECIES_PARAMS)
    area_ha       : plot area in hectares
    trees_per_ha  : stocking density
    planting_year : calendar year of planting
    rainfall_ts   : DataFrame with columns [year, annual_rain_mm]
                    (from Module 1's NASA POWER data)
    """
    species:       str
    area_ha:       float
    trees_per_ha:  int
    planting_year: int
    rainfall_ts:   Optional[pd.DataFrame] = None

    # Derived at post-init
    params:        dict = field(init=False)
    total_trees:   int  = field(init=False)

    def __post_init__(self):
        if self.species not in SPECIES_PARAMS:
            raise ValueError(f"Unknown species '{self.species}'. "
                             f"Choose from: {list(SPECIES_PARAMS.keys())}")
        self.params      = SPECIES_PARAMS[self.species]
        self.total_trees = int(self.area_ha * self.trees_per_ha)

    def _get_annual_rain(self, year: int) -> float:
        """Look up annual rainfall for a given year, or use species optimal as fallback."""
        if self.rainfall_ts is not None and "year" in self.rainfall_ts.columns:
            row = self.rainfall_ts[self.rainfall_ts["year"] == year]
            if not row.empty:
                return float(row["annual_rain_mm"].values[0])
        return self.params["rain_opt"]   # fallback to optimum

    def simulate(self, years: int = 30) -> pd.DataFrame:
        """
        Run year-by-year growth simulation.
        Returns a DataFrame with one row per year.
        """
        records = []
        for t in range(1, years + 1):
            cal_year   = self.planting_year + t
            rain       = self._get_annual_rain(cal_year)
            rain_mod   = rainfall_growth_modifier(rain, self.params)

            height_m   = chapman_richards_height(
                t, self.params["A"], self.params["k"],
                self.params["m"], rain_mod
            )
            dbh_cm     = height_to_dbh(height_m, self.species)
            biomass    = dbh_to_agb(dbh_cm, self.params["wood_density"])
            components = agb_to_total_biomass(
                biomass, self.params["bef"], self.params["rs_ratio"]
            )

            # Survival rate: ~8% annual mortality in managed plantations
            survival_rate = 0.94 ** t
            living_trees  = int(self.total_trees * survival_rate)

            # Scale to plot level
            plot_co2_tonnes = (components["co2_eq_kg"] * living_trees) / 1000.0

            records.append({
                "stand_age_yr":     t,
                "calendar_year":    cal_year,
                "annual_rain_mm":   round(rain, 1),
                "rain_modifier":    round(rain_mod, 3),
                "mean_height_m":    round(height_m, 2),
                "mean_dbh_cm":      round(dbh_cm, 2),
                "agb_per_tree_kg":  components["agb_kg"],
                "co2_per_tree_kg":  components["co2_eq_kg"],
                "living_trees":     living_trees,
                "plot_co2_tonnes":  round(plot_co2_tonnes, 2),
                "annual_increment_co2": 0.0,   # filled below
            })

        df = pd.DataFrame(records)

        # Annual CO₂ increment (for MRV crediting periods)
        df["annual_increment_co2"] = df["plot_co2_tonnes"].diff().fillna(
            df["plot_co2_tonnes"].iloc[0]
        ).round(2)

        return df


# =============================================================================
# RAINFALL PREPROCESSING (bridges Module 1 output)
# =============================================================================
def prep_rainfall_for_simulation(rain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts monthly NASA POWER rainfall (mm/day) to annual totals (mm/year).
    Expects columns: [date, rainfall_mm_day]
    """
    rain_df = rain_df.copy()
    rain_df["date"] = pd.to_datetime(rain_df["date"])
    rain_df["year"] = rain_df["date"].dt.year

    # mm/day × days_in_month → monthly mm
    rain_df["days_in_month"] = rain_df["date"].dt.days_in_month
    rain_df["monthly_mm"] = rain_df["rainfall_mm_day"] * rain_df["days_in_month"]

    annual = (
        rain_df.groupby("year")["monthly_mm"]
        .sum()
        .reset_index()
        .rename(columns={"monthly_mm": "annual_rain_mm"})
    )
    return annual


# =============================================================================
# QUICK DEMO
# =============================================================================
if __name__ == "__main__":
    # Simulate Khejri plantation in Rajasthan — realistic for Pilani region
    sample_rain = pd.DataFrame({
        "year":           range(2024, 2055),
        "annual_rain_mm": np.random.normal(380, 60, 31).clip(150, 700)
    })

    plot = ForestPlot(
        species="khejri",
        area_ha=1.0,
        trees_per_ha=50,
        planting_year=2024,
        rainfall_ts=sample_rain
    )

    results = plot.simulate(years=30)
    print(results[["stand_age_yr", "mean_height_m", "mean_dbh_cm",
                   "plot_co2_tonnes", "annual_increment_co2"]].to_string(index=False))
