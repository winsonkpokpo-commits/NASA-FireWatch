"""Read the raw FIRMS ZIP archives into one clean table of hotspot detections."""
import json
import zipfile

import numpy as np
import pandas as pd

from . import config as C

USECOLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
           "confidence", "frp", "daynight", "type"]


def _read_zip(path):
    frames = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            df = pd.read_csv(zf.open(name), usecols=lambda c: c in USECOLS)
            df["source"] = "nrt" if "_nrt_" in name else "archive"
            frames.append(df)
    return frames


def cell_index(lat, lon):
    """Row/column of the global 0.25-degree grid cell that holds each point."""
    i = np.floor(np.asarray(lat) / C.CELL_DEG).astype(np.int32)
    j = np.floor(np.asarray(lon) / C.CELL_DEG).astype(np.int32)
    return i, j


def cell_id(i, j):
    return (np.asarray(i) * 10000 + np.asarray(j)).astype(np.int32)


def load_raw():
    frames = []
    for path in sorted(C.RAW_DIR.glob("*.zip")):
        frames.extend(_read_zip(path))
    df = pd.concat(frames, ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


def clean(df):
    """Apply confidence/type filters and drop NRT rows already covered by the archive."""
    df = df.copy()
    df["sensor"] = np.where(df["satellite"].isin(C.MODIS_SATS), "MODIS", "VIIRS")

    # NRT rows only for dates after the end of each satellite's archive record.
    arch_end = df[df.source == "archive"].groupby("satellite")["acq_date"].max()
    end = df["satellite"].map(arch_end)
    df = df[(df.source == "archive") | (df["acq_date"] > end)]

    # Keep vegetation fires / other land fires (type 0). NRT has no type column:
    # drop NRT points at locations flagged as static land sources in the archive.
    static = df[(df.source == "archive") & (df["type"] == 2)]
    static_keys = set(zip(static.latitude.round(2), static.longitude.round(2)))
    is_static_nrt = (df.source == "nrt") & pd.Series(
        list(zip(df.latitude.round(2), df.longitude.round(2))), index=df.index).isin(static_keys)
    df = df[((df.source == "archive") & (df["type"] == 0)) | ((df.source == "nrt") & ~is_static_nrt)]

    conf = df["confidence"].astype(str)
    modis_ok = (df.sensor == "MODIS") & (pd.to_numeric(conf, errors="coerce") >= C.MODIS_MIN_CONFIDENCE)
    viirs_ok = (df.sensor == "VIIRS") & conf.isin(C.VIIRS_KEEP_CONFIDENCE)
    df = df[modis_ok | viirs_ok]

    # Keep points inside whole grid cells of the study domain.
    df = df[(df.latitude >= C.LAT_MIN) & (df.latitude < C.LAT_MAX)
            & (df.longitude >= C.LON_MIN) & (df.longitude < C.LON_MAX)]

    i, j = cell_index(df.latitude, df.longitude)
    out = pd.DataFrame({
        "date": df["acq_date"].values.astype("datetime64[D]"),
        "acq_time": df["acq_time"].astype(np.int16).values,
        "satellite": pd.Categorical(df["satellite"], categories=list(C.MODIS_SATS + C.VIIRS_SATS)),
        "sensor": pd.Categorical(df["sensor"], categories=["MODIS", "VIIRS"]),
        "latitude": df["latitude"].astype(np.float32).values,
        "longitude": df["longitude"].astype(np.float32).values,
        "frp": df["frp"].astype(np.float32).values,
        "confidence": df["confidence"].astype(str).values,
        "daynight": pd.Categorical(df["daynight"], categories=["D", "N"]),
        "source": pd.Categorical(df["source"], categories=["archive", "nrt"]),
        "cell": cell_id(i, j),
    })
    return out.sort_values(["date", "acq_time"]).reset_index(drop=True)


# Start of each satellite's FIRMS record. Taken from the mission records instead of
# the first detection, because the first detection can fall weeks later in the wet season.
KNOWN_START = {"Terra": "2000-11-01", "Aqua": "2002-07-04", "SNPP": "2012-01-20",
               "N20": "2018-04-01", "N21": "2023-03-01"}


def availability(raw):
    """Observation window of each satellite: known start, last detection in the download.

    The end is the last detection because a satellite can stop delivering data
    (e.g. Suomi-NPP after 2026-11-02 in this download) while the others continue.
    """
    g = raw.groupby("satellite")["acq_date"]
    first, last = g.min(), g.max()
    return {s: [KNOWN_START.get(s, str(first[s].date())), str(last[s].date())]
            for s in last.index}


def run():
    raw = load_raw()
    avail = availability(raw)
    pts = clean(raw)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(C.POINTS_PATH, index=False, compression="zstd")
    C.AVAILABILITY_PATH.write_text(json.dumps(avail, indent=2))
    print(f"raw rows {len(raw):,} -> kept {len(pts):,}")
    print(pts.groupby("satellite", observed=True).size().to_string())
    return pts, avail


if __name__ == "__main__":
    run()
