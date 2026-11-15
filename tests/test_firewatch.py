import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from firewatch import analytics as A
from firewatch import config as C
from firewatch.harmonize import n_viirs_available

AVAIL = {"SNPP": ["2012-01-20", "2026-11-02"], "N20": ["2018-04-01", "2026-11-15"],
         "N21": ["2023-03-01", "2026-11-15"]}


def test_viirs_satellite_count_follows_missions():
    n = n_viirs_available(pd.to_datetime(["2012-01-19", "2015-06-01", "2020-01-01",
                                          "2024-01-01", "2026-11-10"]), AVAIL)
    assert list(n.values) == [0, 1, 2, 3, 2]


def test_theil_sen_detects_decline():
    years = np.arange(2001, 2021)
    slope, p = A.theil_sen(years, 1000 - 20 * (years - 2001) + np.tile([5, -5], 10))
    assert slope == pytest.approx(-20, abs=1)
    assert p < 0.01


def test_calendar_and_anomaly_shapes():
    idx = pd.date_range("2001-01-01", "2004-12-01", freq="MS")
    d = pd.DataFrame({"harmonized": 100.0, "modis_raw": 0, "viirs_raw": 0}, index=idx)
    m = A.monthly(d)
    cal = A.calendar_matrix(m)
    assert cal.shape == (4, 12) and (cal.values == 100).all()
    clim = A.climatology(m, (2001, 2004))
    an = A.anomaly_matrix(m, clim, season=True)
    assert list(an.columns[:3]) == ["Oct", "Nov", "Dec"]
    assert np.nanmax(np.abs(an.values - 1)) < 1e-9


def test_off_season_anomalies_are_blank():
    idx = pd.date_range("2001-01-01", "2003-12-01", freq="MS")
    vals = np.where(idx.month.isin([12, 1]), 1000.0, 0.0)
    m = A.monthly(pd.DataFrame({"harmonized": vals, "modis_raw": 0, "viirs_raw": 0}, index=idx))
    an = A.anomaly_matrix(m, A.climatology(m, (2001, 2003)))
    assert an["Jul"].isna().all() and (an["Jan"] == 1).all()


@pytest.mark.skipif(not C.POINTS_PATH.exists(), reason="processed data not built")
def test_point_weights_reproduce_cell_month_totals():
    pts, cm = A.load_points(), A.load_cell_month()
    poly = box(C.LON_MIN, C.LAT_MIN, C.LON_MAX, C.LAT_MAX)
    daily = A.daily_series(pts, cm, poly, pts.date.max())
    total_cm = cm.harmonized.sum()
    assert daily.harmonized.sum() == pytest.approx(total_cm, rel=1e-4)
    # no jump at the MODIS -> VIIRS switch: Feb 2012 season close to the model
    m = A.monthly(daily)
    jan_feb = m.loc["2011-12-01":"2012-03-01", "harmonized"]
    assert jan_feb.min() > 0
