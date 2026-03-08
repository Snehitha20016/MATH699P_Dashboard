from __future__ import annotations

import math
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_loader import (
    load_pipeline_status, load_site_metadata, load_ozone_predictions,
    load_annual_vegetation, load_model_metrics, load_feature_importance,
    load_coverage_audit, load_site_coverage_audit, load_gap_metadata,
    load_eval_gap_summary, load_clf_metadata, derive_site_index, NAAQS_PPB_DEFAULT,
)

st.set_page_config(page_title="CASTNET CA Ozone Dashboard", page_icon="🌬️", layout="wide")

# -----------------------
# Cached load (refreshable)
# -----------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_all():
    pipeline_lr = load_pipeline_status()
    site_lr     = load_site_metadata()
    preds_lr    = load_ozone_predictions()
    annual_lr   = load_annual_vegetation()
    metrics_lr  = load_model_metrics()
    feat_lr     = load_feature_importance()
    cov_lr      = load_coverage_audit()
    scov_lr     = load_site_coverage_audit()
    gap_lr      = load_gap_metadata()
    egap_lr     = load_eval_gap_summary()
    clf_lr      = load_clf_metadata()

    pipeline = pipeline_lr.data or {}
    site_meta = site_lr.data or {}

    preds  = preds_lr.data if isinstance(preds_lr.data, pd.DataFrame) else pd.DataFrame()
    annual = annual_lr.data if isinstance(annual_lr.data, pd.DataFrame) else pd.DataFrame()
    cov    = cov_lr.data if isinstance(cov_lr.data, pd.DataFrame) else pd.DataFrame()
    scov   = scov_lr.data if isinstance(scov_lr.data, pd.DataFrame) else pd.DataFrame()

    site_index = derive_site_index(site_meta, pipeline, cov, scov, preds)

    # NAAQS threshold for lines/alerts
    clf = clf_lr.data or {}
    naaqs = float(clf.get("naaqs_threshold_ppb", NAAQS_PPB_DEFAULT))

    sources = {
        "pipeline_status": pipeline_lr.source,
        "site_metadata": site_lr.source,
        "predictions": preds_lr.source,
        "annual_veg": annual_lr.source,
        "model_metrics": metrics_lr.source,
        "feature_importance": feat_lr.source,
        "coverage_audit": cov_lr.source,
        "site_coverage_audit": scov_lr.source,
        "gap_metadata": gap_lr.source,
        "eval_gap_summary": egap_lr.source,
        "clf_metadata": clf_lr.source,
    }

    return {
        "pipeline": pipeline,
        "site_meta_raw": site_meta,
        "site_index": site_index,
        "preds": preds,
        "annual": annual,
        "metrics": metrics_lr.data or {},
        "feat": feat_lr.data or {},
        "coverage": cov,
        "site_coverage": scov,
        "gap_meta": gap_lr.data or {},
        "eval_gap": egap_lr.data or {},
        "clf": clf,
        "naaqs": naaqs,
        "sources": sources,
    }


# -----------------------
# Sidebar
# -----------------------
def sidebar(data):
    st.sidebar.markdown("## 🌬️ CA Ozone")
    st.sidebar.caption("CASTNET Monitor Dashboard")

    # refresh
    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pipeline status")
    pipeline = data["pipeline"]
    for step in ["data_extraction", "feature_engineering", "model_training", "gap_fixes", "model_evaluation", "vegetation_exposure"]:
        info = pipeline.get(step, {})
        status = info.get("status", "not_run")
        ts = info.get("timestamp")
        ts_str = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
        st.sidebar.write(f"- **{step}**: `{status}`  \n  {ts_str}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data sources")
    for k, v in data["sources"].items():
        st.sidebar.write(f"- {k}: **{v.upper()}**")


# -----------------------
# Page helpers
# -----------------------
STATUS_COLOR = {"ACTIVE": "#2ecc71", "TRAIN_ONLY": "#f39c12", "NO_TRAIN": "#e74c3c"}

def site_map_tab(site_index: pd.DataFrame):
    st.subheader("🗺️ Site Map")
    df = site_index.copy()
    df = df.dropna(subset=["lat", "lon"], how="any")

    if len(df) == 0:
        st.warning("No mappable sites found (missing lat/lon). Check site_metadata.json.")
        return

    fig = go.Figure()
    for status, g in df.groupby("status", dropna=False):
        fig.add_trace(go.Scattermapbox(
            lat=g["lat"], lon=g["lon"],
            mode="markers+text",
            text=g["SITE_ID"],
            textposition="top right",
            marker=dict(size=14, color=STATUS_COLOR.get(status, "#3498db")),
            name=str(status),
            hovertemplate="<b>%{text}</b><br>" +
                          "name=%{customdata[0]}<br>" +
                          "region=%{customdata[1]}<br>" +
                          "yrs=%{customdata[2]}–%{customdata[3]}<extra></extra>",
            customdata=np.stack([
                g["name"].fillna(""),
                g["region"].fillna(""),
                g["yr_start"].fillna(0).astype(int),
                g["yr_end"].fillna(0).astype(int),
            ], axis=1),
        ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=37.3, lon=-119.6), zoom=5.3),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Site registry (canonical)")
    st.dataframe(
        site_index.sort_values(["status", "SITE_ID"]),
        use_container_width=True,
        hide_index=True
    )


def forecast_tab(site_index: pd.DataFrame, preds: pd.DataFrame, metrics: dict, naaqs: float):
    st.subheader("📈 Forecast: Observed vs Predicted")

    if len(preds) == 0:
        st.warning("Missing ozone_predictions.parquet")
        return

    # Only allow sites that exist in preds
    pred_sites = sorted(preds["SITE_ID"].astype(str).unique().tolist())
    label_map = {r["SITE_ID"]: f'{r["SITE_ID"]} — {r.get("name","")}' for _, r in site_index.iterrows()}
    options = [label_map.get(s, s) for s in pred_sites]
    inv = {label_map.get(s, s): s for s in pred_sites}

    c1, c2, c3 = st.columns([2, 1, 1])
    sel_label = c1.selectbox("Site", options, index=0)
    sid = inv[sel_label]
    horizon = c2.radio("Horizon", [1, 8, 24], horizontal=True, format_func=lambda h: f"t+{h}h")
    days = c3.slider("Days", 7, 120, 30, 1)

    pcol = f"pred_t{horizon}"
    sdf = preds[preds["SITE_ID"] == sid].sort_values("DATE_TIME").tail(days * 24)

    if len(sdf) == 0:
        st.warning("No rows for this site.")
        return

    # metrics card if present
    m = (metrics.get(sid, {}) or {}).get(str(horizon), {})
    if m:
        a, b, c = st.columns(3)
        a.metric("MAE (ppb)", f'{m.get("mae", float("nan")):.2f}')
        b.metric("RMSE (ppb)", f'{m.get("rmse", float("nan")):.2f}')
        c.metric("R²", f'{m.get("r2", float("nan")):.3f}')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sdf["DATE_TIME"], y=sdf["OZONE"],
        name="Observed", mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=sdf["DATE_TIME"], y=sdf[pcol],
        name=f"Predicted {pcol}", mode="lines"
    ))
    fig.add_hline(y=naaqs, line_dash="dot", annotation_text=f"NAAQS {naaqs:.0f} ppb")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # quick exceedance summary
    exc_obs = int((sdf["OZONE"] >= naaqs).sum())
    exc_pred = int((sdf[pcol] >= naaqs).sum())
    st.caption(f"Observed exceedance hours in window: **{exc_obs}** | Predicted exceedance hours: **{exc_pred}**")


def exceedance_tab(site_index: pd.DataFrame, preds: pd.DataFrame, clf: dict, naaqs: float):
    st.subheader("⚠️ Exceedance Alerts (simple, data-driven)")
    if len(preds) == 0:
        st.warning("Missing ozone_predictions.parquet")
        return

    thr = float(clf.get("best_threshold", 0.5))
    st.info(f"Classifier metadata loaded (best_threshold={thr:.3f}, NAAQS={naaqs:.0f} ppb).")

    # A pragmatic “risk” score without needing a separate prob column:
    # use fraction of next-24 predicted hours exceeding NAAQS on t+24 series.
    rows = []
    for sid in sorted(preds["SITE_ID"].unique()):
        sdf = preds[preds["SITE_ID"] == sid].sort_values("DATE_TIME").tail(24)
        if len(sdf) < 6:
            continue
        frac = float((sdf["pred_t24"] >= naaqs).mean())
        mx = float(sdf["pred_t24"].max())
        rows.append({"SITE_ID": sid, "risk_frac": frac, "max_pred_t24": mx})

    out = pd.DataFrame(rows)
    if len(out) == 0:
        st.warning("Not enough recent prediction rows to compute alerts.")
        return

    out = out.merge(site_index[["SITE_ID", "name", "status"]], on="SITE_ID", how="left")
    out = out.sort_values(["risk_frac", "max_pred_t24"], ascending=False)

    st.dataframe(out, use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(x=out["SITE_ID"], y=out["risk_frac"], text=(out["risk_frac"] * 100).round(0).astype(int).astype(str) + "%"))
    fig.update_layout(height=340, yaxis_title="Fraction of next 24h ≥ NAAQS (from pred_t24)")
    st.plotly_chart(fig, use_container_width=True)


def vegetation_tab(site_index: pd.DataFrame, annual: pd.DataFrame):
    st.subheader("🌿 Vegetation Exposure")
    if len(annual) == 0:
        st.warning("Missing annual_vegetation.parquet")
        return

    sites = sorted(annual["SITE_ID"].astype(str).unique().tolist())
    label_map = {r["SITE_ID"]: f'{r["SITE_ID"]} — {r.get("name","")}' for _, r in site_index.iterrows()}
    options = [label_map.get(s, s) for s in sites]
    inv = {label_map.get(s, s): s for s in sites}

    c1, c2 = st.columns([2, 1])
    sel_label = c1.selectbox("Site", options, index=0, key="veg_site")
    sid = inv[sel_label]
    metric = c2.radio("Metric", ["w126", "aot40"], horizontal=True)

    sdf = annual[annual["SITE_ID"] == sid].sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sdf["year"], y=sdf[metric], mode="lines+markers", name=metric))
    if "is_fire_year" in sdf.columns:
        fires = sdf[sdf["is_fire_year"] == True]
        fig.add_trace(go.Scatter(
            x=fires["year"], y=fires[metric],
            mode="markers", name="fire year", marker=dict(size=10, symbol="diamond")
        ))
    fig.update_layout(height=450, xaxis_title="year", yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sdf, use_container_width=True, hide_index=True)


def coverage_tab(coverage: pd.DataFrame, site_cov: pd.DataFrame):
    st.subheader("📋 Data Coverage")
    if len(coverage) == 0 and len(site_cov) == 0:
        st.warning("Missing coverage_audit.csv and site_coverage_audit.csv")
        return

    if len(coverage):
        st.markdown("#### Coverage audit (raw)")
        st.dataframe(coverage, use_container_width=True, hide_index=True)

    if len(site_cov):
        st.markdown("#### Split coverage (train/val/test)")
        st.dataframe(site_cov, use_container_width=True, hide_index=True)


def diagnostics_tab(metrics: dict, feat: dict, eval_gap: dict, gap_meta: dict):
    st.subheader("🔬 Diagnostics")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Eval gap summary**")
    c1.json(eval_gap)

    c2.markdown("**Gap metadata**")
    c2.json(gap_meta)

    c3.markdown("**Feature importance**")
    c3.json(feat)

    st.markdown("### Model metrics (raw json)")
    st.json(metrics)


def main():
    data = load_all()
    sidebar(data)

    st.title("🌬️ CASTNET California Ozone Dashboard")
    st.caption(f"Last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tabs = st.tabs(["Site Map", "Forecast", "Alerts", "Vegetation", "Coverage", "Diagnostics"])

    with tabs[0]:
        site_map_tab(data["site_index"])
    with tabs[1]:
        forecast_tab(data["site_index"], data["preds"], data["metrics"], data["naaqs"])
    with tabs[2]:
        exceedance_tab(data["site_index"], data["preds"], data["clf"], data["naaqs"])
    with tabs[3]:
        vegetation_tab(data["site_index"], data["annual"])
    with tabs[4]:
        coverage_tab(data["coverage"], data["site_coverage"])
    with tabs[5]:
        diagnostics_tab(data["metrics"], data["feat"], data["eval_gap"], data["gap_meta"])


if __name__ == "__main__":
    main()
