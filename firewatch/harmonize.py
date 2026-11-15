"""Harmonize MODIS and VIIRS hotspot counts on a 0.25-degree monthly grid.

Target unit: "VIIRS-equivalent hotspots", i.e. the number of hotspots one VIIRS
instrument would detect. From February 2012 the harmonized value is the observed
VIIRS count, averaged over the VIIRS satellites operating that day (Suomi-NPP,
NOAA-20, NOAA-21), so adding or losing a satellite does not create a jump.
Before 2012 the value is predicted from MODIS (Terra/Aqua) by a gradient-boosted
model trained where both sensors overlap (2012-02 to 2021-12).
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .ingest import cell_id

FEATURES_FULL = ["lat", "lon", "month", "log_terra", "log_aqua", "log_modis_nb",
                 "frp_median", "night_frac"]
FEATURES_TERRA = ["lat", "lon", "month", "log_terra", "log_terra_nb", "frp_median", "night_frac"]


def load_inputs():
    pts = pd.read_parquet(C.POINTS_PATH)
    avail = json.loads(C.AVAILABILITY_PATH.read_text())
    return pts, avail


def n_viirs_available(dates, avail):
    """Number of VIIRS satellites delivering data on each date."""
    dates = pd.DatetimeIndex(dates)
    n = np.zeros(len(dates), dtype=np.int8)
    for s in C.VIIRS_SATS:
        if s in avail:
            a, b = pd.Timestamp(avail[s][0]), pd.Timestamp(avail[s][1])
            n += ((dates >= a) & (dates <= b)).astype(np.int8)
    return pd.Series(n, index=dates)


def viirs_daily_mean(pts, avail, by=("cell",)):
    """Daily VIIRS count per single instrument (sum over satellites / satellites available)."""
    v = pts[pts.sensor == "VIIRS"]
    g = v.groupby(list(by) + ["date"], observed=True).size().rename("n").reset_index()
    nav = n_viirs_available(g["date"].unique(), avail)
    g["n_sat"] = g["date"].map(nav).values
    g = g[g.n_sat > 0]
    g["viirs"] = g["n"] / g["n_sat"]
    return g.drop(columns=["n", "n_sat"])


def grid_cells():
    ii = np.arange(int(round(C.LAT_MIN / C.CELL_DEG)), int(round(C.LAT_MAX / C.CELL_DEG)))
    jj = np.arange(int(round(C.LON_MIN / C.CELL_DEG)), int(round(C.LON_MAX / C.CELL_DEG)))
    I, J = np.meshgrid(ii, jj, indexing="ij")
    cells = pd.DataFrame({"i": I.ravel(), "j": J.ravel()})
    cells["cell"] = cell_id(cells.i, cells.j)
    cells["lat"] = (cells.i + 0.5) * C.CELL_DEG
    cells["lon"] = (cells.j + 0.5) * C.CELL_DEG
    return cells


def build_cell_month(pts, avail):
    pts = pts.assign(ym=pts["date"].dt.to_period("M"))
    first = pd.Period(min(v[0] for v in avail.values()), "M")
    last = pd.Period(max(v[1] for v in avail.values()), "M")
    months = pd.period_range(first, last, freq="M")
    cells = grid_cells()
    base = cells.merge(pd.DataFrame({"ym": months}), how="cross")

    m = pts[pts.sensor == "MODIS"]
    cnt = m.groupby(["cell", "ym", "satellite"], observed=True).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=list(C.MODIS_SATS), fill_value=0).rename(columns=str.lower)
    stats = m.groupby(["cell", "ym"], observed=True).agg(
        frp_median=("frp", "median"),
        night_frac=("daynight", lambda s: float((s == "N").mean())))
    vd = viirs_daily_mean(pts, avail)
    vd["ym"] = vd["date"].dt.to_period("M")
    vm = vd.groupby(["cell", "ym"])["viirs"].sum().rename("viirs_obs")

    df = base.merge(cnt.reset_index(), on=["cell", "ym"], how="left")
    df = df.merge(stats.reset_index(), on=["cell", "ym"], how="left")
    df = df.merge(vm.reset_index(), on=["cell", "ym"], how="left")
    df[["terra", "aqua"]] = df[["terra", "aqua"]].fillna(0)
    viirs_era = df.ym >= pd.Period(C.VIIRS_ERA_START, "M")
    df.loc[viirs_era, "viirs_obs"] = df.loc[viirs_era, "viirs_obs"].fillna(0)
    df.loc[~viirs_era, "viirs_obs"] = np.nan
    aqua_start = pd.Period(avail["Aqua"][0], "M") + 1  # first full Aqua month
    df["aqua_available"] = df.ym >= aqua_start
    df["month"] = df.ym.dt.month
    df["modis"] = df.terra + df.aqua

    # 3x3 neighbourhood sums give the model the spatial context of each cell.
    for col, out in (("modis", "modis_nb"), ("terra", "terra_nb")):
        wide = df.pivot_table(index="ym", columns=["i", "j"], values=col, aggfunc="sum")
        acc = pd.Series(0.0, index=df.index)
        key = df.set_index(["ym", "i", "j"]).index
        stacked = wide.stack(["i", "j"], future_stack=True)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                shifted = key.map(lambda k, di=di, dj=dj: (k[0], k[1] + di, k[2] + dj))
                acc += stacked.reindex(shifted).fillna(0).values
        df[out] = acc.values
    for col in ("terra", "aqua", "modis_nb", "terra_nb"):
        df["log_" + col] = np.log1p(df[col])
    return df


def make_model():
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.06, max_iter=500,
                                         max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)


def training_mask(df):
    return (df.ym >= pd.Period(C.TRAIN_START, "M")) & (df.ym <= pd.Period(C.TRAIN_END, "M"))


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return {"r2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else None,
            "mae": float(np.abs(y - p).mean()),
            "bias_pct": float(100 * (p.sum() - y.sum()) / y.sum()) if y.sum() else None}


def cross_validate(df):
    """Leave-one-year-out CV on the overlap period, against simple baselines."""
    tr = df[training_mask(df)].copy()
    years = tr.ym.dt.year
    preds = {k: np.zeros(len(tr)) for k in ("raw_modis", "global_ratio", "monthly_ratio",
                                            "gbm", "gbm_terra_only")}
    for y in sorted(years.unique()):
        te, fit = (years == y).values, (years != y).values
        a, b = tr[fit], tr[te]
        preds["raw_modis"][te] = b.modis
        preds["global_ratio"][te] = b.modis * a.viirs_obs.sum() / a.modis.sum()
        r = a.groupby("month").viirs_obs.sum() / a.groupby("month").modis.sum()
        preds["monthly_ratio"][te] = b.modis * b.month.map(r).values
        preds["gbm"][te] = make_model().fit(a[FEATURES_FULL], a.viirs_obs).predict(b[FEATURES_FULL])
        preds["gbm_terra_only"][te] = make_model().fit(a[FEATURES_TERRA], a.viirs_obs).predict(b[FEATURES_TERRA])
    tr = tr.assign(**{"pred_" + k: v for k, v in preds.items()})

    out = {}
    dom = tr.groupby("ym")[["viirs_obs"] + ["pred_" + k for k in preds]].sum()
    ann = tr.groupby(tr.ym.dt.year)[["viirs_obs"] + ["pred_" + k for k in preds]].sum()
    for k in preds:
        out[k] = {"cell_month": _metrics(tr.viirs_obs, tr["pred_" + k]),
                  "domain_month": _metrics(dom.viirs_obs, dom["pred_" + k]),
                  "domain_year": _metrics(ann.viirs_obs, ann["pred_" + k])}
    return out, tr


def drift_check(df, model):
    """How the MODIS->VIIRS relation holds after the training window (MODIS orbit drift)."""
    late = df[(df.ym > pd.Period(C.TRAIN_END, "M")) & df.viirs_obs.notna()]
    p = model.predict(late[FEATURES_FULL])
    ann = pd.DataFrame({"obs": late.viirs_obs.values, "pred": p, "year": late.ym.dt.year.values})
    ann = ann.groupby("year")[["obs", "pred"]].sum()
    return {int(y): {"observed": float(r.obs), "predicted_from_modis": float(r.pred),
                     "ratio": float(r.pred / r.obs) if r.obs else None} for y, r in ann.iterrows()}


def point_weights(pts, df, avail):
    """Give each detection its contribution to the harmonized count.

    VIIRS era: each VIIRS detection weighs 1 / (VIIRS satellites operating that day).
    MODIS era: each MODIS detection weighs pred / MODIS count of its cell-month, so the
    detections of a cell-month sum to the model's VIIRS-equivalent value. Cell-months
    with no MODIS detection keep their (small) prediction in cell_month.parquet as
    "pred_no_modis" and are spread over the area of interest by area.
    Detections that do not feed the harmonized series get weight 0.
    """
    pts = pts.drop(columns=[c for c in ("weight",) if c in pts]).copy()
    ym = pts["date"].dt.to_period("M")
    era = ym >= pd.Period(C.VIIRS_ERA_START, "M")
    w = np.zeros(len(pts), dtype=np.float32)

    v = (pts.sensor == "VIIRS").values & era.values
    nav = n_viirs_available(pts.loc[v, "date"].unique(), avail)
    w[v] = 1.0 / pts.loc[v, "date"].map(nav).values

    m = (pts.sensor == "MODIS").values & ~era.values
    f = df[df.modis > 0].set_index(["cell", "ym"])
    factor = (f.pred / f.modis)
    key = pd.MultiIndex.from_arrays([pts.loc[m, "cell"].values, ym[m].values])
    w[m] = factor.reindex(key).values
    pts["weight"] = w
    return pts


def run():
    pts, avail = load_inputs()
    df = build_cell_month(pts, avail)
    cv, cv_rows = cross_validate(df)

    tr = df[training_mask(df)]
    full = make_model().fit(tr[FEATURES_FULL], tr.viirs_obs)
    terra = make_model().fit(tr[FEATURES_TERRA], tr.viirs_obs)
    df["pred"] = np.where(df.aqua_available, full.predict(df[FEATURES_FULL]),
                          terra.predict(df[FEATURES_TERRA]))
    era = df.viirs_obs.notna()
    df["harmonized"] = np.where(era, df.viirs_obs, df.pred)
    df["method"] = np.where(era, "VIIRS observed",
                            np.where(df.aqua_available, "MODIS Terra+Aqua -> model",
                                     "MODIS Terra -> model"))

    df["pred_no_modis"] = np.where(~era & (df.modis == 0), df.pred, 0.0)
    keep = ["cell", "i", "j", "lat", "lon", "ym", "month", "terra", "aqua", "modis",
            "viirs_obs", "pred", "pred_no_modis", "harmonized", "method"]
    out = df[keep].copy()
    out["ym"] = out.ym.dt.to_timestamp()
    out["method"] = out["method"].astype("category")
    for c in ("terra", "aqua", "modis"):
        out[c] = out[c].astype(np.int32)
    for c in ("viirs_obs", "pred", "pred_no_modis", "harmonized", "lat", "lon"):
        out[c] = out[c].astype(np.float32)
    out.to_parquet(C.CELL_MONTH_PATH, index=False, compression="zstd")
    point_weights(pts, df, avail).to_parquet(C.POINTS_PATH, index=False, compression="zstd")

    # Out-of-fold predictions for the app's validation chart (domain monthly totals).
    oof = cv_rows.groupby("ym")[["viirs_obs", "modis", "pred_monthly_ratio", "pred_gbm"]].sum()
    validation = {
        "unit": "VIIRS-equivalent hotspots (one VIIRS instrument)",
        "train_window": [C.TRAIN_START, C.TRAIN_END],
        "cross_validation": "leave-one-year-out",
        "metrics": cv,
        "drift_after_training": drift_check(df, full),
        "oof_domain_month": {str(k): {kk: float(vv) for kk, vv in r.items()}
                             for k, r in oof.iterrows()},
        "availability": avail,
    }
    C.VALIDATION_PATH.write_text(json.dumps(validation, indent=2))
    return out, validation


if __name__ == "__main__":
    _, v = run()
    for k, m in v["metrics"].items():
        print(f"{k:16s} cell-month R2={m['cell_month']['r2']:.3f}  "
              f"domain-month R2={m['domain_month']['r2']:.3f}  "
              f"domain-year bias={m['domain_year']['bias_pct']:+.1f}%  R2={m['domain_year']['r2']:.3f}")
    print(json.dumps(v["drift_after_training"], indent=1))
