# NASA-FireWatch: harmonized burning-activity calendar for Benin

FireWatch joins 26 years of NASA FIRMS hotspots over Benin, from MODIS (2000–) and VIIRS (2012–), into **one consistent record**, so that the satellite of origin no longer matters. It presents that record as an interactive **burning-activity calendar**: a web app where users pick an area and a period and see the fire history, the normal fire season, anomalies and trends, and can download the results.

Built for the 2026 NASA Space Apps Challenge, *Harmonization of MODIS and VIIRS Hot Spots*.

## What the app shows

- **Area of interest:** the whole of Benin, a department (if department boundaries are in `data/boundaries/`, see below), a shape drawn on the map, a point and radius, or a latitude/longitude box.
- **Burning calendar:** a year × month heatmap (calendar years or West-African fire seasons, Oct → Sep), with a weekly view. It can show the number of hotspots or the **anomaly** against the baseline median.
- **Now vs history:**
  - month-to-date and season-to-date counts compared with the same days in every past year (median, rank);
  - the current season plotted against the historical median and 10–90th percentile band.
- **Map:** detections for any period on OpenStreetMap, as a heat map or as points coloured by sensor.
- **Trends:** yearly totals, the Theil–Sen trend with a Mann–Kendall test, and the most and least active seasons.
- **Harmonization:** the naive MODIS→VIIRS join compared with the harmonized record, plus the validation metrics.
- **Downloads (CSV):** monthly and daily harmonized series, the calendar table, raw detections, and the area of interest as GeoJSON.

Colour scales are colour-blind safe: YlOrRd for counts, and RdBu for anomalies (blue = below normal, red = above).

## Method

| Step | What we do |
|---|---|
| Data | FIRMS MODIS C6.1 (Terra from Nov 2000, Aqua from Jul 2002) and VIIRS 375 m C2 (Suomi-NPP from Jan 2012, NOAA-20 from Apr 2018, NOAA-21 from Mar 2023), archive + NRT after the archive ends (`data/raw/`). |
| Cleaning | Vegetation fires only (FIRMS type 0; static industrial sources removed, also from NRT). MODIS confidence ≥ 30; VIIRS nominal/high. |
| Common unit | **VIIRS-equivalent hotspots**: what *one* VIIRS instrument detects. Each day, the VIIRS count is divided by the number of VIIRS satellites operating. Going from 1 to 3 satellites, or losing Suomi-NPP after 2 Nov 2026, therefore creates no jumps. |
| Model | On a 0.25° grid and monthly steps, a gradient-boosting regressor (Poisson loss) predicts the VIIRS-equivalent count. Its inputs are the MODIS Terra and Aqua counts, the MODIS counts of the 3×3 neighbouring cells, the median FRP, the share of night detections, the month and the location. It is trained on Feb 2012 – Dec 2021, when both sensors overlap and before the MODIS orbits drift. A Terra-only model covers Nov 2000 – Jul 2002. |
| Harmonized record | Nov 2000 – Jan 2012: MODIS → model. From Feb 2012: observed VIIRS. |
| Any area | Each MODIS detection gets a weight (model value ÷ MODIS count of its cell-month), and each VIIRS detection gets 1 ÷ the number of satellites operating. Summing the weights inside any polygon gives the harmonized count for that polygon. The ~1–2 % predicted where MODIS saw nothing is spread by area. |

### Validation: leave-one-year-out, 2012–2021

| Method | R² cell-month | R² monthly total | R² yearly total | Bias of total |
|---|---:|---:|---:|---:|
| Raw MODIS (no harmonization) | 0.34 | 0.06 | < 0 | −79 % |
| Single ratio | 0.79 | 0.93 | 0.73 | +0.1 % |
| Ratio per calendar month | 0.87 | 0.985 | 0.91 | +0.2 % |
| **Gradient boosting (used)** | **0.92** | **0.989** | **0.91** | **−0.5 %** |
| Gradient boosting, Terra only | 0.87 | 0.984 | 0.89 | −0.8 % |

We also checked the model after the training window, while the MODIS orbits drift. The ratio of predicted to observed yearly totals stays between 0.96 and 1.06 for 2022–2025. The harmonized record does not depend on this, because it uses observed VIIRS from 2012 onward.

### Main findings for Benin

- **The fire season is concentrated:** December and January hold about 60 % of a normal year's hotspots, and almost all burning happens between November and March.
- **Burning activity has declined significantly:** about −4 % per year from the 2002–03 season to the 2025–26 season (Theil–Sen slope, Mann–Kendall p < 0.001).
- **This season is below normal so far:** as of 15 Nov 2026, the season-to-date total is below the 2001–2025 median for the same days.

## Run it

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

The processed files in `data/processed/` are committed, so the app starts right away. To rebuild them from the raw FIRMS archives (about 40 s):

```bash
python scripts/build_data.py
```

```bash
python -m pytest -q tests
```

### Deploy on Streamlit Community Cloud

Push this repository to GitHub. At [share.streamlit.io](https://share.streamlit.io), create an app from this repository with main file `app.py`. `requirements.txt` is picked up automatically.

### Better country and department borders (optional)

The repository ships the coarse Natural Earth 1:110m outline (public domain). If you add a geoBoundaries GeoJSON for Benin, the app uses it automatically:
- **ADM0** gives the country outline;
- **ADM1** gives the country outline and a **Department** choice for the area of interest.

Name the files `data/boundaries/geoBoundaries-BEN-ADM1.geojson` (and/or `geoBoundaries-BEN-ADM0.geojson`).

## Repository layout

```
app.py                     Streamlit web app
firewatch/ingest.py        raw FIRMS ZIPs -> cleaned detections (data/processed/hotspots.parquet)
firewatch/harmonize.py     grid, model, cross-validation, weights (cell_month.parquet, validation.json)
firewatch/analytics.py     area of interest, calendar, anomalies, trend
scripts/build_data.py      rebuild everything
data/raw/                  original FIRMS archives (see data/raw/README.md)
data/boundaries/           country outlines
tests/                     unit tests
```

## Limits

- Hotspots are satellite detections of active fires, not burned area or fire counts.
- Clouds, smoke and overpass times affect detection.
- Areas smaller than a 0.25° cell (~28 km) inherit the model of their cell before 2012.

## Attribution

We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations (LANCE) and NASA's Earth Science Data and Information System (ESDIS). Country outlines: [Natural Earth](https://www.naturalearthdata.com/) (public domain). Map tiles: © OpenStreetMap contributors.

## Disclaimer

NASA FIRMS data are provided "as is." Users are responsible for their use. FIRMS cautions that the spatial resolution and other characteristics make the data unsuitable for tactical decision-making or informing about conditions at a local scale. See the [NASA FIRMS data citation and disclaimer guidance](https://www.earthdata.nasa.gov/earth-observation-data/near-real-time/citation#ed-firms-citation).

This project is independent and is not affiliated with or endorsed by NASA.
