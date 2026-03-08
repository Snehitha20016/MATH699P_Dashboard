from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# --------
# Config
# --------
DATA_DIR = Path(__file__).parent / "dashboard_data"
DATA_DIR.mkdir(exist_ok=True)

NAAQS_PPB_DEFAULT = 70.0

# Minimal default site schema (used only if metadata is missing)
DEFAULT_SITE_FIELDS = {
    "name": "Unknown",
    "lat": np.nan,
    "lon": np.nan,
    "elev": np.nan,
    "coast_km": np.nan,
    "marine": "",
    "region": "",
    "status": "ACTIVE",  # ACTIVE / TRAIN_ONLY / NO_TRAIN
    "yr_start": 0,
    "yr_end": 0,
    "n_rows": 0,
    "pct_valid": np.nan,
}

STATUS_ORDER = {"ACTIVE": 0, "TRAIN_ONLY": 1, "NO_TRAIN": 2}


@dataclass(frozen=True)
class LoadResult:
    data: object
    source: str          # "live" or "missing"
    path: Optional[Path] # where it came from


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _safe_load_json(path: Path) -> LoadResult:
    if not path.exists():
        return LoadResult({}, "missing", None)
    try:
        return LoadResult(_read_json(path), "live", path)
    except Exception:
        return LoadResult({}, "missing", path)


def _safe_load_csv(path: Path) -> LoadResult:
    if not path.exists():
        return LoadResult(pd.DataFrame(), "missing", None)
    try:
        return LoadResult(pd.read_csv(path), "live", path)
    except Exception:
        return LoadResult(pd.DataFrame(), "missing", path)


def _safe_load_parquet(path: Path) -> LoadResult:
    if not path.exists():
        return LoadResult(pd.DataFrame(), "missing", None)
    try:
        df = pd.read_parquet(path)
        return LoadResult(df, "live", path)
    except Exception:
        return LoadResult(pd.DataFrame(), "missing", path)


def load_pipeline_status() -> LoadResult:
    return _safe_load_json(DATA_DIR / "pipeline_status.json")


def load_site_metadata() -> LoadResult:
    return _safe_load_json(DATA_DIR / "site_metadata.json")


def load_ozone_predictions() -> LoadResult:
    lr = _safe_load_parquet(DATA_DIR / "ozone_predictions.parquet")
    df = lr.data if isinstance(lr.data, pd.DataFrame) else pd.DataFrame()

    if len(df) == 0:
        return lr

    # Normalize core columns
    if "DATE_TIME" in df.columns:
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")

    # Ensure standard pred columns exist
    for col in ["pred_t1", "pred_t8", "pred_t24"]:
        if col not in df.columns:
            df[col] = np.nan

    if "is_extrapolation" not in df.columns:
        df["is_extrapolation"] = False

    # Keep only expected columns if present
    keep = [c for c in ["DATE_TIME", "SITE_ID", "OZONE", "pred_t1", "pred_t8", "pred_t24", "is_extrapolation"] if c in df.columns]
    df = df[keep].dropna(subset=["DATE_TIME", "SITE_ID"]).sort_values(["SITE_ID", "DATE_TIME"])
    return LoadResult(df, lr.source, lr.path)


def load_annual_vegetation() -> LoadResult:
    lr = _safe_load_parquet(DATA_DIR / "annual_vegetation.parquet")
    df = lr.data if isinstance(lr.data, pd.DataFrame) else pd.DataFrame()
    if len(df) == 0:
        return lr

    # Normalize
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    for col in ["w126", "aot40"]:
        if col not in df.columns:
            df[col] = np.nan

    if "is_fire_year" not in df.columns:
        df["is_fire_year"] = False

    keep = [c for c in df.columns if c in {
        "SITE_ID","year","is_fire_year","w126","aot40",
        "mean_o3","p95_o3","max_o3",
        "hours_above_40","hours_above_60","hours_above_80",
        "valid_hours","pct_complete","sufficient_data"
    }]
    df = df[keep].dropna(subset=["SITE_ID", "year"]).sort_values(["SITE_ID", "year"])
    return LoadResult(df, lr.source, lr.path)


def load_model_metrics() -> LoadResult:
    return _safe_load_json(DATA_DIR / "model_metrics.json")


def load_feature_importance() -> LoadResult:
    return _safe_load_json(DATA_DIR / "feature_importance.json")


def load_coverage_audit() -> LoadResult:
    return _safe_load_csv(DATA_DIR / "coverage_audit.csv")


def load_site_coverage_audit() -> LoadResult:
    return _safe_load_csv(DATA_DIR / "site_coverage_audit.csv")


def load_gap_metadata() -> LoadResult:
    return _safe_load_json(DATA_DIR / "gap_metadata.json")


def load_eval_gap_summary() -> LoadResult:
    return _safe_load_json(DATA_DIR / "eval_gap_summary.json")


def load_clf_metadata() -> LoadResult:
    return _safe_load_json(DATA_DIR / "clf_metadata.json")


def derive_site_index(
    site_meta: dict,
    pipeline: dict,
    coverage: pd.DataFrame,
    site_cov: pd.DataFrame,
    preds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one canonical site table used everywhere (map, dropdowns, coverage tables).
    Fixes the common failure mode: site_metadata missing sites that exist in other exports.
    """
    # Collect sites from every source we have
    sites = set()

    # pipeline_status lists expected sites
    for k in ["data_extraction", "model_training", "model_evaluation"]:
        v = pipeline.get(k, {})
        for s in v.get("sites", []) or []:
            sites.add(s)

    if isinstance(coverage, pd.DataFrame) and "SITE_ID" in coverage.columns:
        sites |= set(coverage["SITE_ID"].astype(str).unique())

    if isinstance(site_cov, pd.DataFrame) and "SITE_ID" in site_cov.columns:
        sites |= set(site_cov["SITE_ID"].astype(str).unique())

    if isinstance(preds, pd.DataFrame) and "SITE_ID" in preds.columns:
        sites |= set(preds["SITE_ID"].astype(str).unique())

    # site_metadata can contain extras; include them too
    sites |= set(site_meta.keys())

    rows = []
    cov_idx = None
    if isinstance(coverage, pd.DataFrame) and len(coverage) and "SITE_ID" in coverage.columns:
        cov_idx = coverage.set_index("SITE_ID")

    sc_idx = None
    if isinstance(site_cov, pd.DataFrame) and len(site_cov) and "SITE_ID" in site_cov.columns:
        sc_idx = site_cov.set_index("SITE_ID")

    for sid in sorted(sites):
        meta = dict(DEFAULT_SITE_FIELDS)
        meta.update(site_meta.get(sid, {}))

        # attach coverage fields if present
        if cov_idx is not None and sid in cov_idx.index:
            r = cov_idx.loc[sid]
            # if your "first/last" are YYYY-MM, keep them
            if "first" in r: meta["first"] = r["first"]
            if "last" in r:  meta["last"] = r["last"]
            if "n_rows" in r: meta["n_rows"] = int(r["n_rows"])
            if "pct_valid" in r: meta["pct_valid"] = float(r["pct_valid"])

        # attach split counts if present
        if sc_idx is not None and sid in sc_idx.index:
            r = sc_idx.loc[sid]
            for c in ["n_train", "n_val", "n_test"]:
                if c in r:
                    meta[c] = int(r[c])

        rows.append({"SITE_ID": sid, **meta})

    out = pd.DataFrame(rows)

    # Normalize status values
    out["status"] = out["status"].astype(str).str.upper().replace({
        "TRAINONLY": "TRAIN_ONLY",
        "NO TRAIN": "NO_TRAIN",
        "NOTRAIN": "NO_TRAIN",
    })

    # Sort stable
    out["status_rank"] = out["status"].map(STATUS_ORDER).fillna(9)
    out = out.sort_values(["status_rank", "SITE_ID"]).drop(columns=["status_rank"])
    return out
