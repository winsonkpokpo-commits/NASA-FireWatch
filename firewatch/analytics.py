"""Area-of-interest analytics on the harmonized hotspot record."""
import json

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box, shape

from . import config as C

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # West African fire season year


def load_points():
    return pd.read_parquet(C.POINTS_PATH)


def load_cell_month():
    return pd.read_parquet(C.CELL_MONTH_PATH)


def load_validation():
    return json.loads(C.VALIDATION_PATH.read_text())


def load_countries():
    return json.loads(C.BOUNDARIES.read_text())


def _first_existing(patterns):
    for pat in patterns:
        hits = sorted((C.ROOT / "data" / "boundaries").glob(pat))
        if hits:
            return hits[0]
    return None


def departments():
    """Benin departments from a geoBoundaries ADM1 GeoJSON, if one is in data/boundaries."""
    path = _first_existing(["geoBoundaries-BEN-ADM1*.geojson", "*BEN*ADM1*.geojson", "*benin*adm1*.geojson"])
    if path is None:
        return {}
    feats = json.loads(path.read_text())["features"]
    out = {}
    for f in feats:
        p = f["properties"]
        name = p.get("shapeName") or p.get("NAME_1") or p.get("name") or p.get("ADM1_FR")
        out[str(name)] = shape(f["geometry"]).buffer(0)
    return dict(sorted(out.items()))


def country_polygon(name="Benin"):
    """Benin outline: geoBoundaries ADM0/ADM1 when present, else Natural Earth 1:110m."""
    if name == "Benin":
        adm0 = _first_existing(["geoBoundaries-BEN-ADM0*.geojson", "*BEN*ADM0*.geojson"])
        if adm0 is not None:
            return shapely.union_all([shape(f["geometry"]).buffer(0)
                                      for f in json.loads(adm0.read_text())["features"]])
        deps = departments()
        if deps:
            return shapely.union_all(list(deps.values()))
    for f in load_countries()["features"]:
        if f["properties"]["name"] == name:
            return shape(f["geometry"])
    raise KeyError(name)


def domain_polygon():
    return box(C.LON_MIN, C.LAT_MIN, C.LON_MAX, C.LAT_MAX)


def circle_polygon(lat, lon, radius_km):
    """Approximate circle (in degrees, corrected for latitude)."""
    t = np.linspace(0, 2 * np.pi, 72)
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * np.cos(np.radians(lat)))
    return shapely.Polygon(np.c_[lon + dlon * np.cos(t), lat + dlat * np.sin(t)])


def clip_to_domain(poly):
    return poly.intersection(domain_polygon())


def points_in(pts, poly):
    """Detections inside the polygon (bounding-box prefilter, then exact test)."""
    x0, y0, x1, y1 = poly.bounds
    sub = pts[(pts.longitude >= x0) & (pts.longitude <= x1)
              & (pts.latitude >= y0) & (pts.latitude <= y1)]
    inside = shapely.contains_xy(poly, sub.longitude.values, sub.latitude.values)
    return sub[inside]


def cell_fractions(cm, poly):
    """Fraction of each grid cell's area inside the polygon."""
    cells = cm.drop_duplicates("cell")[["cell", "lat", "lon"]]
    h = C.CELL_DEG / 2
    lon, lat = cells.lon.values.astype(float), cells.lat.values.astype(float)
    boxes = shapely.box(lon - h, lat - h, lon + h, lat + h)
    frac = np.asarray(shapely.area(shapely.intersection(boxes, poly))) / (C.CELL_DEG ** 2)
    return pd.Series(frac, index=cells.cell.values)


def daily_series(pts_aoi, cm, poly, end_date, start_date="2000-11-01"):
    """Harmonized daily VIIRS-equivalent counts, plus raw counts per sensor."""
    days = pd.date_range(start_date, pd.Timestamp(end_date), freq="D")
    h = pts_aoi.groupby("date").weight.sum().reindex(days, fill_value=0.0)

    # Spread the small no-MODIS predictions evenly over the days of each month.
    frac = cell_fractions(cm, poly)
    r = cm[cm.pred_no_modis > 0]
    r = r.assign(v=r.pred_no_modis * r.cell.map(frac).fillna(0).values)
    r = r.groupby("ym").v.sum()
    if len(r):
        per_day = r / r.index.days_in_month
        h = h + per_day.reindex(days.to_period("M").to_timestamp()).fillna(0).values

    raw = pts_aoi.groupby(["date", "sensor"], observed=True).size().unstack(fill_value=0)
    raw = raw.reindex(index=days, columns=["MODIS", "VIIRS"], fill_value=0)
    out = pd.DataFrame({"harmonized": h.values, "modis_raw": raw.MODIS.values,
                        "viirs_raw": raw.VIIRS.values}, index=days)
    out.index.name = "date"
    return out


def monthly(daily):
    m = daily.resample("MS").sum()
    m["year"], m["month"] = m.index.year, m.index.month
    m["season"] = np.where(m.month >= 10, m.year + 1, m.year)  # season "2026" = Oct 2025-Sep 2026
    m["method"] = np.where(m.index >= pd.Timestamp(C.VIIRS_ERA_START), "VIIRS observed",
                           "MODIS -> model")
    return m


def calendar_matrix(m, season=False, value="harmonized"):
    """Year x month table for the heatmap."""
    ycol = "season" if season else "year"
    order = SEASON_MONTHS if season else list(range(1, 13))
    # One row per month, so pivot (not pivot_table, whose sum turns missing values into 0).
    t = m.pivot(index=ycol, columns="month", values=value)
    t = t.reindex(columns=order)
    t.columns = [MONTHS[c - 1] for c in order]
    return t


def climatology(m, baseline):
    """Median and percentiles of each calendar month over the baseline years."""
    b = m[(m.year >= baseline[0]) & (m.year <= baseline[1])]
    g = b.groupby("month").harmonized
    return pd.DataFrame({"median": g.median(), "mean": g.mean(), "p10": g.quantile(0.1),
                         "p90": g.quantile(0.9), "std": g.std()})


def anomaly_matrix(m, clim, season=False, min_share=0.005):
    """Ratio of each month to the baseline median of the same calendar month.

    Off-season months whose median is below `min_share` of the peak month's median
    are left blank: with a handful of fires a ratio says nothing.
    """
    med = clim["median"].where(clim["median"] >= min_share * clim["median"].max())
    mm = m.assign(ratio=m.harmonized / m.month.map(med).replace(0, np.nan))
    return calendar_matrix(mm, season, "ratio")


def to_date_comparison(daily, baseline, end_date):
    """Month-to-date (and season-to-date) total vs the same window in baseline years."""
    end = pd.Timestamp(end_date)
    res = {}
    for label, start_of in (("month", lambda d: d.replace(day=1)),
                            ("season", lambda d: pd.Timestamp(d.year if d.month >= 10 else d.year - 1, 10, 1))):
        start = start_of(end)
        cur = float(daily.loc[start:end, "harmonized"].sum())
        hist = {}
        for y in range(baseline[0], baseline[1] + 1):
            off = pd.DateOffset(years=end.year - y)
            s, e = start - off, end - off
            if s < daily.index[0] or e >= start:
                continue
            hist[y] = float(daily.loc[s:e, "harmonized"].sum())
        vals = np.array(list(hist.values()))
        med = float(np.median(vals)) if len(vals) else np.nan
        res[label] = {"start": start, "end": end, "current": cur, "median": med,
                      "ratio": cur / med if med else np.nan,
                      "rank": int((vals > cur).sum() + 1) if len(vals) else None,
                      "n_years": len(vals), "history": hist}
    return res


def theil_sen(years, values):
    """Theil-Sen slope (per year) and Mann-Kendall S statistic with a normal p-value."""
    x, y = np.asarray(years, float), np.asarray(values, float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    i, j = np.triu_indices(n, 1)
    slope = float(np.median((y[j] - y[i]) / (x[j] - x[i])))
    s = np.sign(y[j] - y[i]).sum()
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    from math import erf, sqrt
    p = float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))
    return slope, p


def week_matrix(daily, years):
    """Year x ISO-like week (1-53) table, week = day-of-year // 7 + 1."""
    d = daily.loc[f"{years[0]}-01-01":f"{years[1]}-12-31", ["harmonized"]].copy()
    d["year"], d["week"] = d.index.year, (d.index.dayofyear - 1) // 7 + 1
    return d.pivot_table(index="year", columns="week", values="harmonized", aggfunc="sum")
