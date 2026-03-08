# 🌬️ CASTNET California Ozone Dashboard

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard runs immediately with **synthetic placeholder data**.  
To use real pipeline outputs, follow the steps below.

---

## Connecting Your Jupyter Notebooks

Each notebook exports one JSON/Parquet/CSV file to `./dashboard_data/`.  
Add the export cell at the **bottom** of each notebook, then run it.  
Click **🔄 Refresh Data** in the sidebar to reload.

---

### 1. Data Extraction Notebook (`ca_data_extraction_combined.ipynb`)

Paste this cell at the end:

```python
# ── DASHBOARD EXPORT ──────────────────────────────────────────
import json, os
from pathlib import Path

DASH_DIR = Path("dashboard_data")   # adjust path if needed
DASH_DIR.mkdir(exist_ok=True)

# Build site metadata dict from your CA_SITE_REGISTRY
site_export = {}
for sid, meta in CA_SITE_REGISTRY.items():
    site_export[sid] = {
        "name":      meta["display_name"],
        "lat":       meta["lat"],
        "lon":       meta["lon"],
        "elev":      meta["elevation_m"],
        "coast_km":  meta["coast_dist_km"],
        "marine":    meta["marine_influence"],
        "region":    meta.get("region", ""),
        "status":    "ACTIVE",          # update manually for TRAIN_ONLY/NO_TRAIN sites
        "yr_start":  2000,              # update from your data
        "yr_end":    2025,
        "n_train":   0,
        "n_val":     0,
        "n_test":    0,
    }

with open(DASH_DIR / "site_metadata.json", "w") as f:
    json.dump(site_export, f, indent=2)

# Coverage audit
cov_df.to_csv(DASH_DIR / "coverage_audit.csv", index=False)

# Pipeline status
status = {}
try:
    with open(DASH_DIR / "pipeline_status.json") as f:
        status = json.load(f)
except FileNotFoundError:
    pass
status["data_extraction"] = {
    "status": "complete",
    "timestamp": pd.Timestamp.now().isoformat(),
    "rows": len(df_ozone),
}
with open(DASH_DIR / "pipeline_status.json", "w") as f:
    json.dump(status, f, indent=2)

print("✅ Data extraction exported to dashboard_data/")
```

---

### 2. Feature Engineering Notebook

```python
# ── DASHBOARD EXPORT ──────────────────────────────────────────
import json
from pathlib import Path
DASH_DIR = Path("dashboard_data"); DASH_DIR.mkdir(exist_ok=True)

status = {}
try:
    with open(DASH_DIR / "pipeline_status.json") as f: status = json.load(f)
except FileNotFoundError: pass
status["feature_engineering"] = {
    "status": "complete",
    "timestamp": pd.Timestamp.now().isoformat(),
    "features": len(FEATURE_COLS),
}
with open(DASH_DIR / "pipeline_status.json", "w") as f:
    json.dump(status, f, indent=2)

print("✅ Feature engineering status exported")
```

---

### 3. Model Training Notebook (`california_models_ets.ipynb`)

```python
# ── DASHBOARD EXPORT ──────────────────────────────────────────
import json
from pathlib import Path
DASH_DIR = Path("dashboard_data"); DASH_DIR.mkdir(exist_ok=True)

# Model metrics  {sid: {horizon_str: {mae, rmse, r2, recall, precision, miss_rate, n_test}}}
# Build from your site_reg_eval and global_site_eval dicts
metrics_export = {}
for sid in ca_sites:
    metrics_export[sid] = {}
    for H in [1, 8, 24]:
        e = site_reg_eval.get(sid, {}).get(H) or global_site_eval.get(H, {}).get(sid, {})
        if e:
            metrics_export[sid][str(H)] = {
                "mae":       round(float(e.get("mae", 0)), 4),
                "rmse":      round(float(e.get("rmse", 0)), 4),
                "r2":        round(float(e.get("r2", 0)), 4),
                "recall":    round(float(e.get("recall", 0)), 4),
                "precision": round(float(e.get("precision", 0)), 4),
                "miss_rate": round(float(e.get("miss_rate", 0.642)), 4),
                "n_test":    int(e.get("n_test", 0)),
            }

with open(DASH_DIR / "model_metrics.json", "w") as f:
    json.dump(metrics_export, f, indent=2)

# Feature importance (from global_reg)
feat_imp_export = {}
for H in [1, 8, 24]:
    if H not in global_reg: continue
    imp  = global_reg[H].feature_importance(importance_type="gain")
    feat = global_reg[H].feature_name()
    s    = pd.Series(imp, index=feat) / sum(imp)
    lag_frac    = s[[f for f in LAG_FEATURES    if f in s]].sum()
    static_frac = s[[f for f in STATIC_FEATURES if f in s]].sum()
    other_frac  = s[[f for f in OTHER_FEATURES  if f in s]].sum()
    feat_imp_export[str(H)] = {
        "Autoregressive (lag/rmean)": round(float(lag_frac)*100, 2),
        "Met / Chem / PM":            round(float(other_frac)*100, 2),
        "Calendar / Static":          round(float(static_frac)*100, 2),
    }

with open(DASH_DIR / "feature_importance.json", "w") as f:
    json.dump(feat_imp_export, f, indent=2)

status = {}
try:
    with open(DASH_DIR / "pipeline_status.json") as f: status = json.load(f)
except FileNotFoundError: pass
status["model_training"] = {
    "status": "complete",
    "timestamp": pd.Timestamp.now().isoformat(),
    "sites": ca_sites,
}
with open(DASH_DIR / "pipeline_status.json", "w") as f:
    json.dump(status, f, indent=2)

print("✅ Model training exported to dashboard_data/")
```

---

### 4. Model Evaluation Notebook (`ca_evaluation_gaps.ipynb`)

```python
# ── DASHBOARD EXPORT ──────────────────────────────────────────
import json
from pathlib import Path
DASH_DIR = Path("dashboard_data"); DASH_DIR.mkdir(exist_ok=True)

# Ozone predictions — hourly test-set DataFrame
# Needs columns: DATE_TIME, SITE_ID, OZONE, pred_t1, pred_t8, pred_t24
pred_export = pd.concat([
    test_preds[1] [["DATE_TIME","SITE_ID","OZONE","y_hat"]].rename(columns={"y_hat":"pred_t1"}),
    test_preds[8] [["DATE_TIME","SITE_ID","y_hat"]].rename(columns={"y_hat":"pred_t8"}),
    test_preds[24][["DATE_TIME","SITE_ID","y_hat"]].rename(columns={"y_hat":"pred_t24"}),
], axis=1)
# Remove duplicate columns from concat
pred_export = pred_export.loc[:, ~pred_export.columns.duplicated()]
pred_export.to_parquet(DASH_DIR / "ozone_predictions.parquet", index=False)

# Annual vegetation metrics (from ca_vegetation_exposure_gaps notebook)
# Needs columns: SITE_ID, year, w126, aot40, is_fire_year, pct_complete
annual[annual["sufficient_data"]].to_parquet(DASH_DIR / "annual_vegetation.parquet", index=False)

status = {}
try:
    with open(DASH_DIR / "pipeline_status.json") as f: status = json.load(f)
except FileNotFoundError: pass
status["model_evaluation"] = {
    "status":    "complete",
    "timestamp": pd.Timestamp.now().isoformat(),
    "test_rows": int(len(pred_export)),
}
with open(DASH_DIR / "pipeline_status.json", "w") as f:
    json.dump(status, f, indent=2)

print("✅ Model evaluation exported to dashboard_data/")
```

---

## dashboard_data/ File Reference

| File | Written by | Contents |
|------|-----------|----------|
| `site_metadata.json` | Data extraction | Site registry (lat/lon, elev, status) |
| `coverage_audit.csv` | Data extraction | Train/val/test row counts per site |
| `ozone_predictions.parquet` | Model evaluation | Hourly observed + t+1/8/24h predictions |
| `annual_vegetation.parquet` | Model evaluation | Annual W126, AOT40 per site |
| `model_metrics.json` | Model training | MAE/RMSE/R² per site × horizon |
| `feature_importance.json` | Model training | Feature group importance by horizon |
| `pipeline_status.json` | All notebooks | Timestamps + completion status |

---

## Deployment

**Streamlit Community Cloud:**
1. Push this folder to GitHub (ensure `dashboard_data/` is gitignored or populated)
2. Connect repo at share.streamlit.io
3. Set main file to `app.py`

**Local:**
```bash
streamlit run app.py --server.port 8501
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
