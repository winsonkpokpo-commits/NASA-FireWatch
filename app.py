"""FireWatch Benin: harmonized MODIS + VIIRS burning-activity calendar (Streamlit app)."""
import io
import json

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw, HeatMap
from shapely.geometry import box, mapping, shape
from streamlit_folium import st_folium

from firewatch import analytics as A
from firewatch import config as C

st.set_page_config(page_title="FireWatch Benin", page_icon="🔥", layout="wide")

UNIT = "VIIRS-equivalent hotspots"
COUNT_SCALE = "YlOrRd"   # sequential, colour-blind safe
ANOM_SCALE = "RdBu_r"    # diverging, colour-blind safe; blue = below normal, red = above


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading harmonized hotspot record…")
def load():
    return A.load_points(), A.load_cell_month(), A.load_validation()


pts, cm, val = load()
LAST_DATE = pd.Timestamp(pts.date.max())
FIRST_DATE = pd.Timestamp("2000-11-01")


@st.cache_data(show_spinner="Computing harmonized series for the area…", max_entries=16)
def aoi_series(wkt):
    import shapely
    poly = shapely.from_wkt(wkt)
    sub = A.points_in(pts, poly)
    daily = A.daily_series(sub, cm, poly, LAST_DATE)
    return daily, A.monthly(daily), len(sub)


def aoi_points(wkt, start, end):
    import shapely
    poly = shapely.from_wkt(wkt)
    sub = A.points_in(pts, poly)
    return sub[(sub.date >= start) & (sub.date <= end)]


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔥 FireWatch Benin")
st.sidebar.caption("Harmonized MODIS + VIIRS burning-activity calendar, 2000 – today")

DEPTS = st.cache_resource(A.departments)()
modes = ["Benin (whole country)"] + (["Department"] if DEPTS else []) + \
        ["Draw on the map", "Point and radius", "Bounding box"]
mode = st.sidebar.radio("Area of interest", modes)
if mode == "Department":
    dep = st.sidebar.selectbox("Department", list(DEPTS), index=0)
    aoi = DEPTS[dep]
    aoi_name = f"{dep} department"
elif mode == "Point and radius":
    c1, c2 = st.sidebar.columns(2)
    lat = c1.number_input("Latitude", 6.0, 12.5, 10.30, 0.05, format="%.3f")
    lon = c2.number_input("Longitude", 0.75, 3.75, 1.90, 0.05, format="%.3f")
    radius = st.sidebar.slider("Radius (km)", 5, 150, 40)
    aoi = A.circle_polygon(lat, lon, radius)
    aoi_name = f"{radius} km around {lat:.2f}°N, {lon:.2f}°E"
elif mode == "Bounding box":
    c1, c2 = st.sidebar.columns(2)
    s = c1.number_input("South", 6.0, 12.5, 9.5, 0.1)
    n = c2.number_input("North", 6.0, 12.5, 12.4, 0.1)
    w = c1.number_input("West", 0.75, 3.75, 0.9, 0.1)
    e = c2.number_input("East", 0.75, 3.75, 3.7, 0.1)
    aoi = box(min(w, e), min(s, n), max(w, e), max(s, n))
    aoi_name = f"Box {min(s, n):.1f}–{max(s, n):.1f}°N, {min(w, e):.1f}–{max(w, e):.1f}°E"
elif mode == "Draw on the map":
    drawn = st.session_state.get("drawn_geom")
    if drawn:
        aoi = shape(drawn)
        aoi_name = "Drawn area"
    else:
        aoi = A.country_polygon("Benin")
        aoi_name = "Benin (draw a shape on the map to change)"
else:
    aoi = A.country_polygon("Benin")
    aoi_name = "Benin"

aoi = A.clip_to_domain(aoi)
if aoi.is_empty or aoi.area < 1e-5:
    st.error("The selected area is outside the data domain (6–12.5°N, 0.75–3.75°E). "
             "Choose an area inside Benin.")
    st.stop()

st.sidebar.markdown("---")
season_mode = st.sidebar.toggle("Fire-season years (Oct → Sep)", value=True,
                                help="In West Africa the burning season runs from about "
                                     "November to March, across two calendar years.")
years = st.sidebar.slider("Years shown", 2001, LAST_DATE.year, (2001, LAST_DATE.year))
baseline = st.sidebar.slider("Baseline for 'normal' (median)", 2001, LAST_DATE.year - 1,
                             (2001, LAST_DATE.year - 1),
                             help="Years used for the historical median and percentiles. "
                                  "Fire activity in Benin has declined since the 2000s; "
                                  "pick the last 10 years for a 'recent normal'.")
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: NASA FIRMS, MODIS C6.1 and VIIRS 375 m (S-NPP, NOAA-20, NOAA-21). "
                   f"Last detection in record: **{LAST_DATE:%d %b %Y}**.")

daily, monthly, n_pts = aoi_series(aoi.wkt)
clim = A.climatology(monthly, baseline)
todate = A.to_date_comparison(daily, baseline, LAST_DATE)


def season_label(y):
    return f"{y - 1}–{str(y)[-2:]}"


# ---------------------------------------------------------------- header + KPIs
st.title("FireWatch Benin — burning-activity calendar")
st.markdown(f"**Area:** {aoi_name} · **Unit:** {UNIT} · one consistent record from "
            f"Nov 2000 to {LAST_DATE:%d %b %Y}, whichever satellite observed it.")


def pct(r):
    return "n/a" if not np.isfinite(r) else f"{(r - 1) * 100:+.0f}% vs median"


k1, k2, k3, k4 = st.columns(4)
mt, se = todate["month"], todate["season"]
k1.metric(f"{mt['start']:%-d}–{mt['end']:%-d %b %Y} (month to date)", f"{mt['current']:,.0f}",
          pct(mt["ratio"]), delta_color="inverse",
          help=f"Median of the same days in {baseline[0]}–{baseline[1]}: {mt['median']:,.0f}. "
               f"Rank {mt['rank']} of {mt['n_years'] + 1} years (1 = most fires).")
k2.metric(f"Season to date (since {se['start']:%-d %b %Y})", f"{se['current']:,.0f}",
          pct(se["ratio"]), delta_color="inverse",
          help=f"Median of the same window in baseline years: {se['median']:,.0f}. "
               f"Rank {se['rank']} of {se['n_years'] + 1}.")
peak = clim["median"].sort_values(ascending=False)
k3.metric("Critical period (median)", f"{A.MONTHS[peak.index[0] - 1]} & {A.MONTHS[peak.index[1] - 1]}",
          f"{peak.iloc[:2].sum() / clim['median'].sum():.0%} of a normal year", delta_color="off", delta_arrow="off")
# Complete years / seasons only. Season "2026" = Oct 2025 - Sep 2026.
ycol = "season" if season_mode else "year"
current_period = (LAST_DATE.year + (LAST_DATE.month >= 10)) if season_mode else LAST_DATE.year
first_period = 2002 if season_mode else 2001
annual = monthly.groupby(ycol).harmonized.sum()
annual = annual[(annual.index >= max(years[0], first_period)) &
                (annual.index <= min(years[1], current_period - 1))]
slope, p = A.theil_sen(annual.index, annual.values)
k4.metric("Long-term trend", f"{slope / annual.mean() * 100:+.1f}% / year" if len(annual) > 3 else "n/a",
          ("significant (p < 0.05)" if p < 0.05 else "not significant") if np.isfinite(p) else "",
          delta_color="off", delta_arrow="off", help="Theil–Sen slope of yearly totals, Mann–Kendall test.")

# A radio instead of st.tabs: only the chosen view is computed, and the Leaflet map
# renders correctly (maps inside hidden tabs get zero size).
VIEWS = ["🔥 Burning calendar", "📈 Now vs history", "🗺️ Map", "📊 Trends",
         "🛰️ Harmonization", "⬇️ Downloads", "ℹ️ Method"]
view_name = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")
st.markdown("---")

# ---------------------------------------------------------------- calendar
if view_name == VIEWS[0]:
    view = st.radio("Show", ["Number of hotspots", "Anomaly vs baseline median"], horizontal=True)
    lo = years[0]
    if view == "Number of hotspots":
        mat = A.calendar_matrix(monthly, season_mode)
        mat = mat.loc[(mat.index >= lo) & (mat.index <= years[1] + (1 if season_mode else 0))]
        z, text = mat.values, np.vectorize(lambda v: "" if np.isnan(v) else f"{v:,.0f}")(mat.values)
        heat = go.Heatmap(z=np.log10(z + 1), x=mat.columns, y=[season_label(y) if season_mode else str(y) for y in mat.index],
                          text=text, texttemplate="%{text}", textfont={"size": 9},
                          customdata=z, hovertemplate="%{y} %{x}: %{customdata:,.0f} hotspots<extra></extra>",
                          colorscale=COUNT_SCALE, xgap=1, ygap=1,
                          colorbar=dict(title="hotspots", tickvals=[0, 1, 2, 3, 4, 5],
                                        ticktext=["0", "10", "100", "1k", "10k", "100k"]))
    else:
        mat = A.anomaly_matrix(monthly, clim, season_mode)
        mat = mat.loc[(mat.index >= lo) & (mat.index <= years[1] + (1 if season_mode else 0))]
        z = np.log2(mat.values.astype(float))
        text = np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.1f}×")(mat.values.astype(float))
        heat = go.Heatmap(z=np.clip(z, -2, 2), x=mat.columns,
                          y=[season_label(y) if season_mode else str(y) for y in mat.index],
                          text=text, texttemplate="%{text}", textfont={"size": 9},
                          customdata=mat.values, hovertemplate="%{y} %{x}: %{customdata:.2f}× the median<extra></extra>",
                          colorscale=ANOM_SCALE, zmid=0, zmin=-2, zmax=2, xgap=1, ygap=1,
                          colorbar=dict(title="× median", tickvals=[-2, -1, 0, 1, 2],
                                        ticktext=["¼×", "½×", "1×", "2×", "4×"]))
    fig = go.Figure(heat)
    fig.update_layout(height=max(420, 22 * len(mat) + 120), margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(autorange="reversed", title="fire season" if season_mode else "year"),
                      xaxis=dict(side="top"))
    st.plotly_chart(fig, use_container_width=True)
    if view != "Number of hotspots":
        st.caption("Blank cells in the anomaly view are off-season months with too few fires "
                   "for a meaningful ratio.")
    st.caption(f"Each cell is the harmonized number of hotspots in that month. "
               f"{LAST_DATE:%B %Y} is partial (to {LAST_DATE:%-d %b}). "
               "Before Feb 2012 values are modelled from MODIS; from Feb 2012 they are observed by VIIRS.")

    with st.expander("Weekly calendar (finer detail)"):
        wk = A.week_matrix(daily, years)
        f2 = go.Figure(go.Heatmap(z=np.log10(wk.values + 1), x=wk.columns, y=wk.index.astype(str),
                                  customdata=wk.values, colorscale=COUNT_SCALE,
                                  hovertemplate="%{y}, week %{x}: %{customdata:,.0f}<extra></extra>",
                                  colorbar=dict(title="hotspots", tickvals=[0, 1, 2, 3, 4],
                                                ticktext=["0", "10", "100", "1k", "10k"])))
        f2.update_layout(height=max(380, 16 * len(wk) + 100), xaxis_title="week of year",
                         yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(f2, use_container_width=True)

# ---------------------------------------------------------------- now vs history
if view_name == VIEWS[1]:
    order = A.SEASON_MONTHS if season_mode else list(range(1, 13))
    xl = [A.MONTHS[m - 1] for m in order]
    c = clim.reindex(order)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xl, y=c.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xl, y=c.p10, fill="tonexty", fillcolor="rgba(120,120,120,0.25)",
                             line=dict(width=0), name=f"10–90th percentile {baseline[0]}–{baseline[1]}"))
    fig.add_trace(go.Scatter(x=xl, y=c["median"], line=dict(color="#555", dash="dash"),
                             name="historical median"))
    cur_season = LAST_DATE.year + (1 if LAST_DATE.month >= 10 else 0)
    for sy, color, name in ((cur_season, "#d7301f", "current"), (cur_season - 1, "#fc8d59", "previous")):
        if season_mode:
            s = monthly[monthly.season == sy]
            lab = f"{name} season {season_label(sy)}"
        else:
            yy = LAST_DATE.year - (0 if name == "current" else 1)
            s = monthly[monthly.year == yy]
            lab = f"{name} year {yy}"
        s = s.set_index("month").reindex(order)
        fig.add_trace(go.Scatter(x=xl, y=s.harmonized, mode="lines+markers", line=dict(color=color, width=3),
                                 name=lab))
    fig.update_layout(height=420, yaxis_title=f"{UNIT} per month", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"The current month ({LAST_DATE:%B}) is partial; use the bars below for a like-for-like comparison.")

    h = mt["history"]
    hist = pd.Series(h)
    bars = go.Figure(go.Bar(x=[str(y) for y in hist.index] + [str(LAST_DATE.year)],
                            y=list(hist.values) + [mt["current"]],
                            marker_color=["#bdbdbd"] * len(hist) + ["#d7301f"]))
    bars.add_hline(y=mt["median"], line_dash="dash", annotation_text="median")
    bars.update_layout(title=f"Hotspots between {mt['start']:%-d %b} and {mt['end']:%-d %b}, each year",
                       height=320, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=UNIT)
    st.plotly_chart(bars, use_container_width=True)

# ---------------------------------------------------------------- map
if view_name == VIEWS[2]:
    c1, c2 = st.columns([3, 1])
    with c2:
        default_start = max(FIRST_DATE, LAST_DATE - pd.Timedelta(days=30))
        rng = st.date_input("Detections between", (default_start.date(), LAST_DATE.date()),
                            min_value=FIRST_DATE.date(), max_value=LAST_DATE.date())
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            m_start, m_end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
        else:
            m_start, m_end = default_start, LAST_DATE
        sensors = st.multiselect("Sensors", ["MODIS", "VIIRS"], ["MODIS", "VIIRS"])
        style = st.radio("Style", ["Heat map", "Points"], horizontal=True)
        if mode == "Draw on the map":
            st.info("Use the ■ or ⬟ tool on the map to draw your area. The whole app updates.")
            if st.session_state.get("drawn_geom") and st.button("Clear drawn area"):
                st.session_state.pop("drawn_geom")
                st.rerun()
    sel = aoi_points(aoi.wkt, m_start, m_end)
    sel = sel[sel.sensor.isin(sensors)]
    with c2:
        st.metric("Detections in view", f"{len(sel):,}")
    with c1:
        fmap = folium.Map(location=[aoi.centroid.y, aoi.centroid.x], zoom_start=7, tiles="OpenStreetMap")
        folium.GeoJson(A.load_countries(), name="countries",
                       style_function=lambda f: {"color": "#444", "weight": 1, "fillOpacity": 0},
                       tooltip=folium.GeoJsonTooltip(["name"])).add_to(fmap)
        folium.GeoJson(mapping(aoi), name="area of interest",
                       style_function=lambda f: {"color": "#2166ac", "weight": 3, "fillOpacity": 0.05}).add_to(fmap)
        show = sel.sample(min(len(sel), 25000), random_state=0) if len(sel) else sel
        if len(show):
            if style == "Heat map":
                HeatMap(show[["latitude", "longitude"]].values.tolist(), radius=8, blur=10,
                        min_opacity=0.3).add_to(fmap)
            else:
                colors = {"MODIS": "#7b3294", "VIIRS": "#e66101"}
                for r in show.head(4000).itertuples():
                    folium.CircleMarker([r.latitude, r.longitude], radius=2, color=colors[r.sensor],
                                        fill=True, weight=0, fill_opacity=0.7,
                                        tooltip=f"{r.date:%Y-%m-%d} {C.SAT_NAMES[r.satellite]} FRP {r.frp:.0f} MW").add_to(fmap)
        if mode == "Draw on the map":
            Draw(draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False,
                               "polygon": True, "rectangle": True}, edit_options={"edit": False}).add_to(fmap)
        out = st_folium(fmap, height=560, use_container_width=True,
                        returned_objects=["last_active_drawing"], key="map")
        if mode == "Draw on the map" and out and out.get("last_active_drawing"):
            g = out["last_active_drawing"]["geometry"]
            if g != st.session_state.get("drawn_geom"):
                st.session_state["drawn_geom"] = g
                st.rerun()
    st.caption("Map shows individual detections (up to 25,000 sampled for speed). "
               "Purple = MODIS, orange = VIIRS in point view.")

# ---------------------------------------------------------------- trends
if view_name == VIEWS[3]:
    labels = [season_label(y) if season_mode else str(y) for y in annual.index]
    med = annual.median()
    fig = go.Figure(go.Bar(x=labels, y=annual.values, name="total",
                           marker=dict(color=np.log2(annual.values / med), colorscale=ANOM_SCALE,
                                       cmid=0, cmin=-1.5, cmax=1.5,
                                       colorbar=dict(title="vs median", tickvals=[-1, 0, 1],
                                                     ticktext=["½×", "1×", "2×"])),
                           hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    if np.isfinite(slope):
        xs = annual.index.values
        fit = np.median(annual.values - slope * xs) + slope * xs
        fig.add_trace(go.Scatter(x=labels, y=fit, mode="lines", name="Theil–Sen trend",
                                 line=dict(color="black", dash="dash")))
    fig.update_layout(height=420, yaxis_title=f"{UNIT} per {'season' if season_mode else 'year'}",
                      margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    top = annual.sort_values(ascending=False)
    c1, c2 = st.columns(2)
    c1.markdown("**Most active " + ("seasons" if season_mode else "years") + "**")
    c1.dataframe(pd.DataFrame({"period": [season_label(y) if season_mode else y for y in top.index[:5]],
                               "hotspots": top.values[:5].round(0)}), hide_index=True)
    c2.markdown("**Least active**")
    c2.dataframe(pd.DataFrame({"period": [season_label(y) if season_mode else y for y in top.index[-5:][::-1]],
                               "hotspots": top.values[-5:][::-1].round(0)}), hide_index=True)
    if np.isfinite(slope):
        st.markdown(f"Trend: **{slope:+,.0f} hotspots per year** ({slope / annual.mean() * 100:+.1f}% of the "
                    f"average per year), Mann–Kendall p = {p:.3g}. Only complete "
                    f"{'seasons' if season_mode else 'years'} are included.")

# ---------------------------------------------------------------- harmonization
if view_name == VIEWS[4]:
    st.markdown("#### Why harmonize?")
    st.markdown("VIIRS (375 m pixels) detects about **5 times more hotspots** than MODIS (1 km) over Benin, "
                "and the number of VIIRS satellites changed from 1 (2012) to 3 (2023). "
                "Joining the raw records gives a false jump in 2012. The harmonized line removes it.")
    mm = monthly.copy()
    naive = np.where(mm.index < pd.Timestamp(C.VIIRS_ERA_START), mm.modis_raw, mm.viirs_raw)
    ann = pd.DataFrame({"naive": naive, "harmonized": mm.harmonized, "modis": mm.modis_raw,
                        "viirs": mm.viirs_raw}, index=mm.index)
    ann = ann[(ann.index >= pd.Timestamp("2001-01-01")) & (ann.index < pd.Timestamp(LAST_DATE.year, 1, 1))].resample("YS").sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ann.index.year, y=ann.naive, name="naive join: MODIS, then all VIIRS satellites",
                             line=dict(color="#999", dash="dot", width=3)))
    fig.add_trace(go.Scatter(x=ann.index.year, y=ann.modis, name="MODIS raw (Terra + Aqua)",
                             line=dict(color="#7b3294")))
    fig.add_trace(go.Scatter(x=ann.index.year, y=ann.harmonized, name="harmonized (VIIRS-equivalent)",
                             line=dict(color="#d7301f", width=4)))
    fig.add_vline(x=2012, line_dash="dash", annotation_text="VIIRS starts")
    fig.update_layout(height=400, yaxis_title="hotspots per year", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Validation (whole study area, leave-one-year-out, 2012–2021)")
    names = {"raw_modis": "Raw MODIS count (no harmonization)", "global_ratio": "Single ratio",
             "monthly_ratio": "Ratio per calendar month", "gbm": "Gradient boosting (used, Terra+Aqua)",
             "gbm_terra_only": "Gradient boosting, Terra only (used for 2000–02)"}
    rows = []
    for k, n in names.items():
        mtr = val["metrics"][k]
        rows.append({"method": n, "R² cell-month": mtr["cell_month"]["r2"],
                     "R² monthly total": mtr["domain_month"]["r2"],
                     "R² yearly total": mtr["domain_year"]["r2"],
                     "bias of total": f"{mtr['domain_year']['bias_pct']:+.1f}%"})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    oof = pd.DataFrame(val["oof_domain_month"]).T
    oof.index = pd.PeriodIndex(oof.index, freq="M").to_timestamp()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oof.index, y=oof.viirs_obs, name="VIIRS observed", line=dict(color="#e66101", width=3)))
    fig.add_trace(go.Scatter(x=oof.index, y=oof.pred_gbm, name="predicted from MODIS (held-out year)",
                             line=dict(color="#5e3c99", dash="dash")))
    fig.update_layout(height=320, yaxis_title="hotspots per month", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    d = val["drift_after_training"]
    st.markdown("**Check after the training period** (MODIS orbits drift from 2022): predicted ÷ observed = " +
                ", ".join(f"{y}: {v['ratio']:.2f}" for y, v in d.items()) +
                ". Because the harmonized record uses observed VIIRS after 2012, this drift does not affect it.")

# ---------------------------------------------------------------- downloads
if view_name == VIEWS[5]:
    st.markdown("All files cover the selected area of interest.")
    mo = monthly.drop(columns=["year", "month", "season"]).rename(columns={
        "harmonized": "harmonized_viirs_equivalent", "modis_raw": "modis_detections",
        "viirs_raw": "viirs_detections_all_satellites"})
    mo.index = mo.index.strftime("%Y-%m")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Monthly harmonized series (CSV)", mo.round(2).to_csv().encode(),
                       "firewatch_monthly.csv", "text/csv")
    dd = daily.rename(columns={"harmonized": "harmonized_viirs_equivalent"})
    c2.download_button("Daily harmonized series (CSV)", dd.round(3).to_csv().encode(),
                       "firewatch_daily.csv", "text/csv")
    c3.download_button("Calendar table (CSV)", A.calendar_matrix(monthly, season_mode).round(1).to_csv().encode(),
                       "firewatch_calendar.csv", "text/csv")
    st.markdown("##### Hotspot detections")
    rng2 = st.date_input("Period", (pd.Timestamp(LAST_DATE.year if LAST_DATE.month >= 10 else LAST_DATE.year - 1, 10, 1).date(),
                                    LAST_DATE.date()),
                         min_value=FIRST_DATE.date(), max_value=LAST_DATE.date(), key="dl_range")
    if isinstance(rng2, (tuple, list)) and len(rng2) == 2:
        det = aoi_points(aoi.wkt, pd.Timestamp(rng2[0]), pd.Timestamp(rng2[1]))
        det = det.assign(satellite_name=det.satellite.map(C.SAT_NAMES)).rename(
            columns={"weight": "harmonized_weight"}).drop(columns=["cell"])
        st.write(f"{len(det):,} detections")
        if len(det) > 300_000:
            st.warning("Too many detections for one file; choose a shorter period.")
        else:
            st.download_button("Hotspot detections (CSV)", det.to_csv(index=False).encode(),
                               "firewatch_hotspots.csv", "text/csv")
    st.download_button("Area of interest (GeoJSON)", json.dumps(mapping(aoi)).encode(),
                       "firewatch_aoi.geojson", "application/geo+json")

# ---------------------------------------------------------------- method
if view_name == VIEWS[6]:
    st.markdown(f"""
**Data.** NASA FIRMS active-fire detections over Benin: MODIS Collection 6.1 (Terra from Nov 2000,
Aqua from Jul 2002) and VIIRS 375 m Collection 2 (Suomi-NPP from Jan 2012, NOAA-20 from Apr 2018,
NOAA-21 from Mar 2023), standard archive plus near-real-time records after the archive ends.

**Cleaning.** Vegetation-fire detections only (FIRMS type 0; static industrial sources removed).
MODIS confidence ≥ {C.MODIS_MIN_CONFIDENCE}; VIIRS nominal or high confidence.

**Common unit.** *VIIRS-equivalent hotspots*: what one VIIRS instrument detects. Each day the VIIRS
count is divided by the number of VIIRS satellites operating, so adding NOAA-20/21 or losing
Suomi-NPP (no data after {val['availability']['SNPP'][1]}) does not create jumps.

**Harmonization model.** On a 0.25° grid and monthly steps, a gradient-boosting model (Poisson loss)
learns the VIIRS-equivalent count from MODIS Terra and Aqua counts, the neighbouring cells' MODIS
counts, fire radiative power, the share of night detections, the month and the location. It is trained
on {C.TRAIN_START} to {C.TRAIN_END}, when both sensors overlap and before MODIS orbit drift.
A Terra-only version covers Nov 2000 – Jul 2002. Each MODIS detection receives a weight so that
the detections of a cell-month add up to the model value; this lets any drawn area use the model.

**Harmonized record.** Nov 2000 – Jan 2012: MODIS → model. From Feb 2012: observed VIIRS.

**Calendar and anomalies.** Monthly totals per year (or fire season, Oct–Sep). Anomaly = month ÷
median of the same month in the baseline years. Month-to-date compares the same calendar days.
Trend = Theil–Sen slope with Mann–Kendall test.

**Limits.** Hotspots are detections, not burned area. Clouds, smoke and overpass times affect
detection. Small areas (< one 0.25° cell, ~28 km) inherit the model of their cell. FIRMS data are
not suitable for tactical, local-scale decisions.

We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management
System (FIRMS), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations
(LANCE) and NASA's Earth Science Data and Information System (ESDIS). Country outlines: Natural Earth.
This project is independent and not endorsed by NASA.
""")
