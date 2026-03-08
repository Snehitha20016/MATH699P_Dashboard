"""
data_loader.py — CASTNET Dashboard Data Bridge
================================================
Reads outputs written by the four Jupyter notebooks.
Falls back to synthetic data for any file that hasn't been generated yet.

Expected files in  ./dashboard_data/  (auto-created by notebook export cells):
  site_metadata.json        ← data extraction notebook
  ozone_predictions.parquet ← model evaluation notebook
  annual_vegetation.parquet ← model evaluation notebook
  model_metrics.json        ← model training notebook
  feature_importance.json   ← model training notebook
  coverage_audit.csv        ← data extraction notebook
  pipeline_status.json      ← written by every notebook on completion
"""

import os, json, hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "dashboard_data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SITE REGISTRY  (canonical — used as fallback if site_metadata.json absent)
# ─────────────────────────────────────────────────────────────────────────────
SITES_DEFAULT = {
    "CAN407": {"name":"Cabrillo NM",           "lat":32.67,"lon":-117.24,"elev":80,
               "coast_km":19.1,  "marine":"COASTAL",    "region":"Southern Coast",
               "status":"ACTIVE",     "yr_start":1995,"yr_end":2025},
    "CON186": {"name":"Converse Basin",         "lat":36.86,"lon":-118.81,"elev":2195,
               "coast_km":221.0, "marine":"INLAND",     "region":"Sierra Nevada",
               "status":"TRAIN_ONLY","yr_start":2003,"yr_end":2011},
    "DEV412": {"name":"Devils Postpile",        "lat":37.62,"lon":-119.08,"elev":2300,
               "coast_km":215.0, "marine":"INLAND",     "region":"Sierra Nevada",
               "status":"NO_TRAIN",  "yr_start":1995,"yr_end":2025},
    "JOT403": {"name":"Joshua Tree NP",         "lat":34.06,"lon":-116.39,"elev":1244,
               "coast_km":179.7, "marine":"TRANSITION", "region":"Mojave Desert",
               "status":"ACTIVE",     "yr_start":1996,"yr_end":2025},
    "LAV410": {"name":"Lassen Volcanic NP",     "lat":40.54,"lon":-121.58,"elev":1756,
               "coast_km":225.5, "marine":"INLAND",     "region":"Cascades",
               "status":"ACTIVE",     "yr_start":1992,"yr_end":2025},
    "LPO010": {"name":"Lassen Pass",            "lat":40.44,"lon":-121.50,"elev":1829,
               "coast_km":222.0, "marine":"INLAND",     "region":"Cascades",
               "status":"TRAIN_ONLY","yr_start":2005,"yr_end":2015},
    "PIN414": {"name":"Pinnacles NP",           "lat":36.49,"lon":-121.18,"elev":335,
               "coast_km":64.5,  "marine":"COASTAL",    "region":"Central Coast",
               "status":"ACTIVE",     "yr_start":2008,"yr_end":2025},
    "SEK402": {"name":"Sequoia NP",             "lat":36.49,"lon":-118.83,"elev":1920,
               "coast_km":221.2, "marine":"INLAND",     "region":"Sierra Nevada S.",
               "status":"TRAIN_ONLY","yr_start":1997,"yr_end":2004},
    "SEK430": {"name":"Sequoia (lower)",         "lat":36.56,"lon":-118.77,"elev":1210,
               "coast_km":215.0, "marine":"INLAND",     "region":"Sierra Nevada S.",
               "status":"ACTIVE",     "yr_start":2003,"yr_end":2025},
    "SND152": {"name":"San Bernardino NF",      "lat":34.19,"lon":-116.77,"elev":1737,
               "coast_km":149.5, "marine":"TRANSITION", "region":"S. CA Mountains",
               "status":"ACTIVE",     "yr_start":1994,"yr_end":2025},
    "YOS204": {"name":"Yosemite (Turtleback)",  "lat":37.71,"lon":-119.74,"elev":1603,
               "coast_km":227.0, "marine":"INLAND",     "region":"Sierra Nevada C.",
               "status":"TRAIN_ONLY","yr_start":2012,"yr_end":2013},
    "YOS404": {"name":"Yosemite NP",            "lat":37.71,"lon":-119.71,"elev":1603,
               "coast_km":229.7, "marine":"INLAND",     "region":"Sierra Nevada C.",
               "status":"ACTIVE",     "yr_start":1988,"yr_end":2025},
}

FIRE_YEARS = {2018, 2020, 2021}
NAAQS      = 70.0


# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHETIC FALLBACK GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def _seed(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def _synth_ozone(sid, days=120):
    rng  = np.random.default_rng(_seed(sid))
    info = SITES_DEFAULT[sid]
    n    = days * 24
    t    = np.arange(n)
    base = 30 + info["elev"] * 0.006 + (8 if info["marine"] == "INLAND" else 0)
    doy  = (t / 24) % 365
    ozone = np.clip(
        base
        + 12 * np.sin(2 * np.pi * (doy - 80) / 365)
        + 10 * np.maximum(0, np.sin(np.pi * ((t % 24) - 6) / 12))
        + rng.normal(0, 3.5, n),
        5, 115,
    ).astype("float32")
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="h")

    def pred(h):
        err  = 2.2 * (1 + h / 9)
        bias = -0.4 * (h / 24)
        return np.clip(ozone + bias + rng.normal(0, err, n), 5, 130).astype("float32")

    return pd.DataFrame({
        "DATE_TIME": dates, "SITE_ID": sid, "OZONE": ozone,
        "pred_t1": pred(1), "pred_t8": pred(8), "pred_t24": pred(24),
    })


def _synth_annual(sid):
    rng  = np.random.default_rng(_seed(sid + "v"))
    info = SITES_DEFAULT[sid]
    years = list(range(max(info["yr_start"], 2000), min(info["yr_end"] + 1, 2025)))
    base  = 22 + info["elev"] * 0.022 + (12 if info["marine"] == "INLAND" else 0)
    rows  = []
    for yr in years:
        trend = -0.55 * (yr - 2000)
        fire  = -4.2 if yr in FIRE_YEARS else 0
        w126  = max(4, base + trend + fire + float(rng.normal(0, 2.5)))
        aot40 = max(0, w126 * 640 + float(rng.normal(0, 1400)))
        rows.append({"SITE_ID": sid, "year": yr,
                     "w126": round(w126, 2), "aot40": round(aot40, 0),
                     "is_fire_year": yr in FIRE_YEARS,
                     "pct_complete": round(float(rng.uniform(72, 100)), 1)})
    return pd.DataFrame(rows)


def _synth_metrics(sid):
    rng     = np.random.default_rng(_seed(sid + "m"))
    info    = SITES_DEFAULT[sid]
    penalty = 0.8 if info["status"] == "NO_TRAIN" else (0.25 if info["status"] == "TRAIN_ONLY" else 0)
    out = {}
    for H in [1, 8, 24]:
        r2 = max(0.05, 0.965 - H * 0.021 - penalty * 0.14 + float(rng.uniform(-0.01, 0.01)))
        if sid == "LAV410" and H == 24:
            r2 = 0.40
        miss = round(0.642 + float(rng.uniform(-0.04, 0.04)), 3) if H == 24 \
               else round(0.28 + float(rng.uniform(-0.05, 0.05)), 3)
        out[str(H)] = {
            "mae":       round(1.8 + H * 0.21 + penalty + float(rng.uniform(-0.1, 0.1)), 3),
            "rmse":      round(2.5 + H * 0.30 + penalty + float(rng.uniform(-0.1, 0.1)), 3),
            "r2":        round(r2, 3),
            "recall":    round(max(0.0, (0.44 if H != 24 else 0.04) - penalty * 0.08 + float(rng.uniform(-0.03, 0.03))), 3),
            "precision": round(max(0.1, 0.60 - penalty * 0.09 + float(rng.uniform(-0.03, 0.03))), 3),
            "miss_rate": miss,
            "n_test":    int(SITES_DEFAULT[sid].get("n_test", 30000)),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC LOADERS  — each returns (data, source_label)
#  source_label is "live" when loaded from real notebook output, else "synthetic"
# ─────────────────────────────────────────────────────────────────────────────

def load_site_metadata():
    path = DATA_DIR / "site_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f), "live"
    return SITES_DEFAULT, "synthetic"


def load_ozone_predictions(sites):
    path = DATA_DIR / "ozone_predictions.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
        return df, "live"
    frames = [_synth_ozone(sid) for sid in sites]
    return pd.concat(frames, ignore_index=True), "synthetic"


def load_annual_vegetation(sites):
    path = DATA_DIR / "annual_vegetation.parquet"
    if path.exists():
        return pd.read_parquet(path), "live"
    frames = [_synth_annual(sid) for sid in sites]
    return pd.concat(frames, ignore_index=True), "synthetic"


def load_model_metrics(sites):
    path = DATA_DIR / "model_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f), "live"
    return {sid: _synth_metrics(sid) for sid in sites}, "synthetic"


def load_feature_importance():
    path = DATA_DIR / "feature_importance.json"
    if path.exists():
        with open(path) as f:
            return json.load(f), "live"
    default = {
        "1":  {"Autoregressive (lag/rmean)": 61.2, "Met / Chem / PM": 36.6, "Calendar / Static": 2.2},
        "8":  {"Autoregressive (lag/rmean)": 55.5, "Met / Chem / PM": 36.5, "Calendar / Static": 7.7},
        "24": {"Autoregressive (lag/rmean)": 54.7, "Met / Chem / PM": 40.4, "Calendar / Static": 4.8},
    }
    return default, "synthetic"


def load_coverage_audit(sites):
    path = DATA_DIR / "coverage_audit.csv"
    if path.exists():
        return pd.read_csv(path), "live"
    rows = []
    for sid, info in SITES_DEFAULT.items():
        if sid not in sites:
            continue
        nt = info.get("n_train", 0) if info["status"] != "NO_TRAIN" else 0
        nv = info.get("n_val",   0) if info["status"] == "ACTIVE" else 0
        ne = info.get("n_test",  0) if info["status"] == "ACTIVE" else 0
        rows.append({
            "SITE_ID":   sid, "name": info["name"],
            "yr_start":  info["yr_start"], "yr_end": info["yr_end"],
            "n_train":   nt, "n_val": nv, "n_test": ne,
            "status":    info["status"],
        })
    return pd.DataFrame(rows), "synthetic"


def load_pipeline_status():
    path = DATA_DIR / "pipeline_status.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "data_extraction":    {"status": "not_run", "timestamp": None, "rows": None},
        "feature_engineering":{"status": "not_run", "timestamp": None, "features": None},
        "model_training":     {"status": "not_run", "timestamp": None, "sites": None},
        "model_evaluation":   {"status": "not_run", "timestamp": None, "test_rows": None},
    }
