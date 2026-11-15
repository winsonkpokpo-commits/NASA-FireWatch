"""Shared constants for the FireWatch pipeline and app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BOUNDARIES = ROOT / "data" / "boundaries" / "natural_earth_110m_benin_neighbours.geojson"

POINTS_PATH = PROCESSED_DIR / "hotspots.parquet"
CELL_MONTH_PATH = PROCESSED_DIR / "cell_month.parquet"
AVAILABILITY_PATH = PROCESSED_DIR / "availability.json"
VALIDATION_PATH = PROCESSED_DIR / "validation.json"

# Study domain: the FIRMS download box around Benin.
LAT_MIN, LAT_MAX = 6.0, 12.5
LON_MIN, LON_MAX = 0.75, 3.75  # whole 0.25-degree cells inside the 0.7-3.9 download box
CELL_DEG = 0.25

MODIS_SATS = ("Terra", "Aqua")
VIIRS_SATS = ("SNPP", "N20", "N21")
SAT_NAMES = {"Terra": "MODIS Terra", "Aqua": "MODIS Aqua", "SNPP": "VIIRS Suomi-NPP",
             "N20": "VIIRS NOAA-20", "N21": "VIIRS NOAA-21"}

# Detection filters (documented in the README).
MODIS_MIN_CONFIDENCE = 30          # MODIS confidence is 0-100; <30 is "low"
VIIRS_KEEP_CONFIDENCE = ("n", "h")  # VIIRS: drop "l" (low)

# Model training window: both sensors overlap and MODIS orbits had not yet drifted.
TRAIN_START = "2012-02"
TRAIN_END = "2021-12"
# First month in which harmonized values come from observed VIIRS.
VIIRS_ERA_START = "2012-02"
