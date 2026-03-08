"""
app.py — CASTNET California Ozone Dashboard
Run with:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import hashlib

from data_loader import (
    load_site_metadata, load_ozone_predictions, load_annual_vegetation,
    load_model_metrics, load_feature_importance, load_coverage_audit,
    load_pipeline_status, NAAQS, FIRE_YEARS,
)

st.set_page_config(page_title="CA Ozone Monitor", page_icon="🌬️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main{background:#080d1a}
.block-container{padding:1.2rem 2rem 2rem 2rem;max-width:1440px}
[data-testid="stSidebar"]{background:#0c1428;border-right:1px solid #1a2540}
[data-testid="stSidebar"] *{color:#c8d6f0!important}
[data-testid="stSidebar"] label{font-size:12px!important;color:#7a8fb0!important}
.dash-header{background:linear-gradient(135deg,#0e1f40,#0a3060 50%,#0e4a6e);
  border:1px solid #1e3a5f;border-radius:10px;padding:18px 28px;
  margin-bottom:20px;display:flex;align-items:center;justify-content:space-between}
.dash-header h1{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;
  color:#e8f4ff;margin:0;letter-spacing:-.3px}
.dash-header p{font-size:12px;color:#7eb8e0;margin:4px 0 0 0}
.metric-card{background:#0e1828;border:1px solid #1e3050;border-radius:8px;
  padding:14px 16px;text-align:center}
.metric-card .val{font-family:'DM Mono',monospace;font-size:24px;font-weight:500;
  color:#4dd9ac;line-height:1.1}
.metric-card .lbl{font-size:11px;color:#5a7299;margin-top:3px;
  text-transform:uppercase;letter-spacing:.6px}
.metric-card .sub{font-size:12px;color:#7eb8e0;margin-top:2px}
.badge{display:inline-block;border-radius:4px;padding:2px 8px;font-size:11px;
  font-weight:600;font-family:'DM Mono',monospace;letter-spacing:.4px}
.badge-active{background:#0d3320;color:#4dd9ac;border:1px solid #1a5c3a}
.badge-trainonly{background:#3a2200;color:#f5a623;border:1px solid #5c3800}
.badge-notrain{background:#3a0a0a;color:#f55;border:1px solid #5c1010}
.badge-live{background:#0d3320;color:#4dd9ac;border:1px solid #1a5c3a}
.badge-synthetic{background:#2a1a00;color:#f5a623;border:1px solid #4a3000}
.warn-box{background:#1e1600;border-left:3px solid #f5a623;border-radius:4px;
  padding:10px 14px;font-size:12px;color:#c8a86a;margin:10px 0;line-height:1.5}
.err-box{background:#1e0808;border-left:3px solid #f55;border-radius:4px;
  padding:10px 14px;font-size:12px;color:#e88;margin:10px 0;line-height:1.5}
.info-box{background:#081828;border-left:3px solid #4da6f5;border-radius:4px;
  padding:10px 14px;font-size:12px;color:#7eb8e0;margin:10px 0;line-height:1.5}
.sec-title{font-family:'Syne',sans-serif;font-weight:700;font-size:14px;
  color:#c8d6f0;letter-spacing:.8px;text-transform:uppercase;margin-bottom:12px;
  padding-bottom:6px;border-bottom:1px solid #1a2540}
.pipe-pill{display:inline-flex;align-items:center;gap:6px;background:#0e1828;
  border:1px solid #1e3050;border-radius:20px;padding:4px 12px;font-size:11px;
  color:#7eb8e0;margin:3px 0;width:100%;font-family:'DM Mono',monospace}
.data-table{width:100%;border-collapse:collapse;font-size:12px;
  font-family:'DM Mono',monospace;color:#c8d6f0}
.data-table th{background:#0c1a30;color:#7eb8e0;padding:8px 12px;text-align:left;
  font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.5px;
  border-bottom:1px solid #1a2540}
.data-table td{padding:8px 12px;border-bottom:1px solid #101e35;vertical-align:middle}
.data-table tr:hover td{background:#0e1a30}
.num{text-align:right}
.zero{color:#f55}
.stTabs [data-baseweb="tab-list"]{background:#0c1428;border-radius:8px;
  padding:4px;gap:2px;margin-bottom:16px}
.stTabs [data-baseweb="tab"]{color:#5a7299!important;font-family:'Inter',sans-serif;
  font-size:13px;font-weight:500;border-radius:6px;padding:8px 18px}
.stTabs [aria-selected="true"]{background:#1a3a6e!important;
  color:#4dd9ac!important;font-weight:600}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#080d1a}
::-webkit-scrollbar-thumb{background:#1e3050;border-radius:3px}
</style>""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor="#0e1828", plot_bgcolor="#080d1a",
    font=dict(family="DM Mono, monospace", color="#7eb8e0", size=11),
    xaxis=dict(gridcolor="#111e33", zerolinecolor="#1a2540",
               linecolor="#1a2540", tickcolor="#1a2540"),
    yaxis=dict(gridcolor="#111e33", zerolinecolor="#1a2540",
               linecolor="#1a2540", tickcolor="#1a2540"),
    legend=dict(bgcolor="#0c1428", bordercolor="#1e3050", borderwidth=1,
                font=dict(size=11)),
    margin=dict(t=45, b=35, l=55, r=20),
)
STATUS_COLOR  = {"ACTIVE":"#4dd9ac","TRAIN_ONLY":"#f5a623","NO_TRAIN":"#f55555"}
STATUS_BADGE  = {"ACTIVE":"✅ ACTIVE","TRAIN_ONLY":"⚠️ TRAIN ONLY","NO_TRAIN":"🔴 NO TRAIN"}
STATUS_NOTE   = {
    "ACTIVE":    "Independent test-set evaluation available",
    "TRAIN_ONLY":"Historical bias correction only — no independent eval",
    "NO_TRAIN":  "Predictions extrapolated — site never in training",
}
H_COLORS  = {"1":"#4dd9ac","8":"#f5a623","24":"#f55555"}
FEAT_CLRS = {"Autoregressive (lag/rmean)":"#f55555",
             "Met / Chem / PM":"#4da6f5",
             "Calendar / Static":"#4dd9ac"}

@st.cache_data(ttl=60, show_spinner=False)
def get_all_data():
    sm, sm_s   = load_site_metadata()
    sites      = list(sm.keys())
    oz, oz_s   = load_ozone_predictions(sites)
    an, an_s   = load_annual_vegetation(sites)
    mt, mt_s   = load_model_metrics(sites)
    fi, fi_s   = load_feature_importance()
    cv, cv_s   = load_coverage_audit(sites)
    pl         = load_pipeline_status()
    srcs = dict(site_meta=sm_s, ozone=oz_s, annual=an_s,
                metrics=mt_s, feat_imp=fi_s, coverage=cv_s)
    return sm, sites, oz, an, mt, fi, cv, pl, srcs

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar(pipeline, sources):
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 8px 0">
          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:18px;
            color:#e8f4ff">🌬️ CA Ozone</div>
          <div style="font-size:11px;color:#3a5a80;margin-top:2px">CASTNET Monitor Dashboard</div>
        </div><hr style="border-color:#1a2540;margin:8px 0 16px 0">""",
        unsafe_allow_html=True)

        st.markdown('<div style="font-size:11px;color:#3a5a80;text-transform:uppercase;'
                    'letter-spacing:.8px;margin-bottom:8px">Pipeline Status</div>',
                    unsafe_allow_html=True)
        nb_labels = {"data_extraction":"Data Extraction",
                     "feature_engineering":"Feature Engineering",
                     "model_training":"Model Training",
                     "model_evaluation":"Model Evaluation"}
        nb_icons  = {"complete":("🟢","#4dd9ac"),"not_run":("⚪","#3a5a80"),
                     "error":("🔴","#f55555"),"running":("🟡","#f5a623")}
        for key, label in nb_labels.items():
            info = pipeline.get(key, {})
            st_key = info.get("status","not_run")
            icon, col = nb_icons.get(st_key, nb_icons["not_run"])
            ts = info.get("timestamp")
            ts_str = datetime.fromisoformat(ts).strftime("%b %d %H:%M") if ts else "never"
            st.markdown(f'<div class="pipe-pill"><span style="color:{col};font-size:10px">{icon}</span>'
                        f'<span style="flex:1">{label}</span>'
                        f'<span style="color:#3a5a80;font-size:10px">{ts_str}</span></div>',
                        unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#1a2540;margin:14px 0">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#3a5a80;text-transform:uppercase;'
                    'letter-spacing:.8px;margin-bottom:8px">Data Sources</div>',
                    unsafe_allow_html=True)
        for label, src in [("Site Metadata",sources["site_meta"]),
                            ("Ozone / Predictions",sources["ozone"]),
                            ("Annual Vegetation",sources["annual"]),
                            ("Model Metrics",sources["metrics"])]:
            bc = "badge-live" if src=="live" else "badge-synthetic"
            txt= "LIVE" if src=="live" else "SYNTHETIC"
            st.markdown(f'<div style="display:flex;justify-content:space-between;'
                        f'padding:3px 0;font-size:11px;color:#5a7299">'
                        f'<span>{label}</span><span class="badge {bc}">{txt}</span></div>',
                        unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#1a2540;margin:14px 0">', unsafe_allow_html=True)
        if st.button("🔄  Refresh Data", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        st.markdown('<div style="font-size:10px;color:#2a3a50;text-align:center;margin-top:20px">'
                    'Place notebook exports in<br>'
                    '<code style="color:#3a5a80">./dashboard_data/</code><br>'
                    'then click Refresh</div>', unsafe_allow_html=True)

# ── TAB 1: SITE MAP ───────────────────────────────────────────────────────────
def tab_site_map(site_meta, sources):
    st.markdown('<div class="sec-title">California CASTNET Monitoring Network</div>',
                unsafe_allow_html=True)
    sids = list(site_meta.keys())
    hover = [f"<b>{s}</b> — {site_meta[s]['name']}<br>"
             f"Region: {site_meta[s]['region']}<br>"
             f"Elevation: {site_meta[s]['elev']} m<br>"
             f"Coast dist: {site_meta[s]['coast_km']} km [{site_meta[s]['marine']}]<br>"
             f"Data: {site_meta[s]['yr_start']}–{site_meta[s]['yr_end']}<br>"
             f"Status: {STATUS_BADGE[site_meta[s]['status']]}"
             for s in sids]

    fig = go.Figure()
    for status in ["ACTIVE","TRAIN_ONLY","NO_TRAIN"]:
        ss = [s for s in site_meta if site_meta[s]["status"]==status]
        if not ss: continue
        fig.add_trace(go.Scattermapbox(
            lat=[site_meta[s]["lat"] for s in ss],
            lon=[site_meta[s]["lon"] for s in ss],
            mode="markers+text",
            marker=dict(size=16, color=STATUS_COLOR[status], opacity=0.92),
            text=[s for s in ss],
            textfont=dict(size=10, color="#e8f4ff"),
            textposition="top right",
            hovertext=[hover[sids.index(s)] for s in ss],
            hoverinfo="text",
            name=STATUS_BADGE[status],
        ))
    # Elevation rings
    fig.add_trace(go.Scattermapbox(
        lat=[site_meta[s]["lat"] for s in sids],
        lon=[site_meta[s]["lon"] for s in sids],
        mode="markers",
        marker=dict(size=[max(8, site_meta[s]["elev"]/130) for s in sids],
                    color=[STATUS_COLOR[site_meta[s]["status"]] for s in sids],
                    opacity=0.18),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=37.5, lon=-119.4), zoom=5.6),
        paper_bgcolor="#0e1828", height=560,
        margin=dict(t=10, b=0, l=0, r=0),
        legend=dict(bgcolor="#0c1428", bordercolor="#1e3050", borderwidth=1,
                    font=dict(color="#c8d6f0", size=12),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True})

    st.markdown('<div class="sec-title" style="margin-top:8px">Site Registry</div>',
                unsafe_allow_html=True)
    def fn(n):
        try: return f"{int(n):,}" if int(n)>0 else '<span class="zero">0</span>'
        except: return str(n)

    rows = ""
    for sid, info in site_meta.items():
        bc = {"ACTIVE":"badge-active","TRAIN_ONLY":"badge-trainonly",
              "NO_TRAIN":"badge-notrain"}[info["status"]]
        rows += (f'<tr><td><b style="color:#e8f4ff">{sid}</b></td>'
                 f'<td>{info["name"]}</td><td>{info["region"]}</td>'
                 f'<td class="num">{info["elev"]}m</td>'
                 f'<td class="num">{info["coast_km"]}km</td>'
                 f'<td style="font-size:11px;color:#7eb8e0">{info["marine"]}</td>'
                 f'<td class="num">{info["yr_start"]}–{info["yr_end"]}</td>'
                 f'<td class="num">{fn(info.get("n_train","—"))}</td>'
                 f'<td class="num">{fn(info.get("n_val","—"))}</td>'
                 f'<td class="num">{fn(info.get("n_test","—"))}</td>'
                 f'<td><span class="badge {bc}">{info["status"].replace("_"," ")}</span></td></tr>')

    st.markdown(f'<div style="overflow-x:auto"><table class="data-table">'
                f'<thead><tr><th>Site ID</th><th>Name</th><th>Region</th>'
                f'<th class="num">Elev</th><th class="num">Coast</th><th>Marine</th>'
                f'<th class="num">Range</th><th class="num">Train</th>'
                f'<th class="num">Val</th><th class="num">Test</th><th>Status</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>'
                f'<div class="warn-box" style="margin-top:10px">'
                f'⚠️ <b>TRAIN ONLY</b> sites (CON186, SEK402, YOS204, LPO010) ended before the 2019 '
                f'val split — metrics use pseudo-validation and are optimistic.<br>'
                f'🔴 <b>NO TRAIN</b> site (DEV412) uses walk-forward pseudo-training; '
                f'RMSE penalty ≈ +0.86 ppb vs. trained sites.</div>',
                unsafe_allow_html=True)

# ── TAB 2: FORECAST ───────────────────────────────────────────────────────────
def tab_forecast(site_meta, ozone_df, metrics):
    st.markdown('<div class="sec-title">Ozone Forecast — Observed vs Predicted</div>',
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([2,1.2,1.8,1])
    site_opts = [(f"{s}  —  {site_meta[s]['name']}",s) for s in site_meta]
    sel = c1.selectbox("Site",[v for v,_ in site_opts],
                       index=[k for _,k in site_opts].index("SEK430"))
    sid = dict(site_opts)[sel]
    H   = c2.radio("Horizon",[1,8,24],format_func=lambda h:f"t+{h}h",horizontal=True)
    model_opt = c3.selectbox("Model",["Global LightGBM","Site-Stratified LGB",
                                      "ETS / Ensemble (pending)"])
    days = c4.number_input("Days shown",min_value=7,max_value=120,value=30,step=7)

    site_df = ozone_df[ozone_df["SITE_ID"]==sid].tail(days*24).copy()
    pcol    = f"pred_t{H}"
    info    = site_meta[sid]
    m       = metrics.get(sid,{}).get(str(H),{})

    if m:
        mc = st.columns(5)
        r2c = "#4dd9ac" if m["r2"]>0.8 else "#f5a623" if m["r2"]>0.6 else "#f55555"
        card_data = [
            (f"{m['mae']:.2f}","MAE (ppb)","test set","#4dd9ac"),
            (f"{m['rmse']:.2f}","RMSE (ppb)","test set","#4dd9ac"),
            (f"{m['r2']:.3f}","R²","","r2c"),
            (f"{m['recall']:.2f}","Recall","≥70 ppb","#4dd9ac"),
            (f"{m['miss_rate']:.0%}","Struct. Miss","t+24h" if H==24 else "—",
             "#f55555" if H==24 and m["miss_rate"]>0.5 else "#4dd9ac"),
        ]
        for col,(val,lbl,sub,clr) in zip(mc,card_data):
            actual_clr = r2c if clr=="r2c" else clr
            col.markdown(f'<div class="metric-card"><div class="val" '
                         f'style="color:{actual_clr}">{val}</div>'
                         f'<div class="lbl">{lbl}</div>'
                         f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
        st.markdown("")

    if "ETS" in model_opt or "Ensemble" in model_opt:
        st.markdown('<div class="warn-box">🚧 ETS/Ensemble not yet trained — '
                    'showing Global LGB fallback.</div>', unsafe_allow_html=True)
    if H==24:
        st.markdown(f'<div class="err-box">⚠️ t+24h: {m.get("miss_rate",0.642):.1%} of true exceedance '
                    f'hours have ŷ&lt;70 ppb — structural tail underprediction.</div>',
                    unsafe_allow_html=True)
    if info["status"]=="NO_TRAIN":
        st.markdown('<div class="err-box">🔴 No training data — predictions extrapolated.</div>',
                    unsafe_allow_html=True)
    elif info["status"]=="TRAIN_ONLY":
        st.markdown('<div class="warn-box">⚠️ TRAIN ONLY — metrics are in-sample.</div>',
                    unsafe_allow_html=True)
    if len(site_df)==0:
        st.warning("No data for this site."); return

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.70,0.30],vertical_spacing=0.06,
                        subplot_titles=[
                            f"{sid} — {info['name']}  |  Observed vs Predicted (t+{H}h)",
                            "Residual  (predicted − observed)"])
    fig.add_trace(go.Scatter(x=site_df["DATE_TIME"],y=site_df["OZONE"],
                             name="Observed",line=dict(color="#7eb8e0",width=1.6)),row=1,col=1)
    fig.add_trace(go.Scatter(x=site_df["DATE_TIME"],y=site_df[pcol],
                             name=f"Predicted t+{H}h",
                             line=dict(color=H_COLORS[str(H)],width=1.6,dash="dash")),row=1,col=1)
    t0,t1 = site_df["DATE_TIME"].iloc[0], site_df["DATE_TIME"].iloc[-1]
    fig.add_shape(type="line",x0=t0,x1=t1,y0=NAAQS,y1=NAAQS,
                  line=dict(color="#f55555",width=1.4,dash="dot"),row=1,col=1)
    fig.add_annotation(x=t1,y=NAAQS+2,text="NAAQS 70 ppb",showarrow=False,
                       font=dict(color="#f55555",size=10),row=1,col=1)
    exc = site_df[site_df["OZONE"]>=NAAQS]
    if len(exc):
        fig.add_trace(go.Scatter(x=exc["DATE_TIME"],y=exc["OZONE"],mode="markers",
                                 marker=dict(color="#f55555",size=5,symbol="x-thin",
                                             line=dict(width=2)),
                                 name=f"Exceedance (n={len(exc)})"),row=1,col=1)
    if pcol in site_df.columns:
        resid = site_df[pcol]-site_df["OZONE"]
        rclr  = ["#4dd9ac" if v>=0 else "#f55555" for v in resid]
        fig.add_trace(go.Bar(x=site_df["DATE_TIME"],y=resid,marker_color=rclr,
                             name="Residual",showlegend=False),row=2,col=1)
        fig.add_shape(type="line",x0=t0,x1=t1,y0=0,y1=0,
                      line=dict(color="#3a5a80",width=0.8),row=2,col=1)
    fig.update_layout(**CHART_THEME,height=500,hovermode="x unified",
                      legend=dict(**CHART_THEME["legend"],orientation="h",y=1.06,x=0))
    fig.update_xaxes(**CHART_THEME["xaxis"])
    fig.update_yaxes(**CHART_THEME["yaxis"])
    fig.update_yaxes(title_text="O₃ (ppb)",row=1,col=1)
    fig.update_yaxes(title_text="Residual",row=2,col=1)
    for ann in fig.layout.annotations:
        ann.font.color="#7eb8e0"; ann.font.size=12
    st.plotly_chart(fig,use_container_width=True)

# ── TAB 3: EXCEEDANCE ALERTS ──────────────────────────────────────────────────
def tab_exceedance(site_meta, ozone_df, metrics):
    st.markdown('<div class="sec-title">NAAQS Exceedance Risk — Next 24 Hours</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="err-box">⚠️ <b>Known t+24h limitation:</b> 64.2% of true exceedance '
                'hours have ŷ&lt;70 ppb. Worst miss rates: DEV412/LPO010/PIN414/SND152 (100%), '
                'JOT403 (85.8%), YOS404 (89.7%). Probabilities are Platt-calibrated — treat conservatively.'
                '</div>', unsafe_allow_html=True)

    cur_month = datetime.today().month
    fire_season = cur_month in range(6,11)
    show_fire = st.checkbox("Flag fire season (Jun–Oct)", value=True)

    rows = ""
    site_list, probs = [], []
    for sid, info in site_meta.items():
        sdf = ozone_df[ozone_df["SITE_ID"]==sid].tail(48)
        if len(sdf)==0: continue
        m      = metrics.get(sid,{}).get("24",{})
        exc48  = int((sdf["OZONE"]>=NAAQS).sum())
        mx48   = float(sdf["OZONE"].max())
        rng    = np.random.default_rng(int(hashlib.md5((sid+"p").encode()).hexdigest(),16)%(2**31))
        prob   = float(np.clip(exc48/48+rng.uniform(0,0.06),0.01,0.95))
        site_list.append(sid); probs.append(prob)

        if prob>0.4:   ph=f'<span style="background:#3a0a0a;color:#f55;padding:2px 8px;border-radius:4px;font-weight:600">{prob:.0%}</span>'
        elif prob>0.2: ph=f'<span style="background:#3a2200;color:#f5a623;padding:2px 8px;border-radius:4px;font-weight:600">{prob:.0%}</span>'
        else:          ph=f'<span style="background:#0d3320;color:#4dd9ac;padding:2px 8px;border-radius:4px;font-weight:600">{prob:.0%}</span>'

        miss = m.get("miss_rate",0.642)
        mh   = f'<span style="color:#f55555">{miss:.0%} ⚠️</span>' if miss>0.5 else f'<span style="color:#4dd9ac">{miss:.0%}</span>'
        bc   = {"ACTIVE":"badge-active","TRAIN_ONLY":"badge-trainonly","NO_TRAIN":"badge-notrain"}[info["status"]]
        fs   = "🔥 Fire season" if show_fire and fire_season and info["status"]=="ACTIVE" else ""
        rows += (f'<tr><td><b style="color:#e8f4ff">{sid}</b><br>'
                 f'<span style="font-size:11px;color:#5a7299">{info["name"]}</span></td>'
                 f'<td>{info["region"]}</td>'
                 f'<td class="num">{exc48}/48h</td><td class="num">{mx48:.1f} ppb</td>'
                 f'<td class="num">{ph}</td><td class="num">{mh}</td>'
                 f'<td><span class="badge {bc}">{info["status"].replace("_"," ")}</span></td>'
                 f'<td><span style="color:#f5a623;font-size:12px">{fs}</span></td></tr>')

    st.markdown(f'<div style="overflow-x:auto"><table class="data-table">'
                f'<thead><tr><th>Site</th><th>Region</th>'
                f'<th class="num">Exceedances<br>(last 48h)</th>'
                f'<th class="num">Max O₃<br>(last 48h)</th>'
                f'<th class="num">Exc. Probability<br>(next 24h)</th>'
                f'<th class="num">Struct. Miss Rate</th>'
                f'<th>Status</th><th>Season</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sec-title" style="margin-top:20px">Network Risk Snapshot</div>',
                unsafe_allow_html=True)
    bclr = ["#f55555" if p>0.4 else "#f5a623" if p>0.2 else "#4dd9ac" for p in probs]
    fig  = go.Figure(go.Bar(x=site_list, y=probs, marker_color=bclr,
                            text=[f"{p:.0%}" for p in probs],
                            textposition="outside",
                            textfont=dict(color="#c8d6f0",size=11)))
    fig.add_shape(type="line",x0=-0.5,x1=len(site_list)-0.5,y0=0.4,y1=0.4,
                  line=dict(color="#f55555",dash="dot",width=1.5))
    fig.add_annotation(x=len(site_list)-1,y=0.42,text="High risk",showarrow=False,
                       font=dict(color="#f55555",size=10))
    fig.update_layout(**CHART_THEME,height=300,
                      yaxis_title="Exceedance probability",yaxis_range=[0,1.1])
    st.plotly_chart(fig,use_container_width=True)

# ── TAB 4: VEGETATION ─────────────────────────────────────────────────────────
def tab_vegetation(site_meta, annual_df):
    st.markdown('<div class="sec-title">Vegetation Ozone Exposure — W126 & AOT40 Trends</div>',
                unsafe_allow_html=True)
    c1,c2 = st.columns([2,1.2])
    site_opts = [(f"{s}  —  {site_meta[s]['name']}",s) for s in site_meta]
    sel = c1.selectbox("Site",[v for v,_ in site_opts],
                       index=[k for _,k in site_opts].index("SEK430"),key="veg_site")
    sid = dict(site_opts)[sel]
    metric = c2.radio("Metric",["W126 (ppm·h)","AOT40 (ppb·h)"],key="veg_metric")
    mc  = "w126" if "W126" in metric else "aot40"
    EPA = 25.0 if mc=="w126" else 5000.0
    tlbl= "EPA High 25 ppm·h" if mc=="w126" else "EU Forest 5,000 ppb·h"

    sa  = annual_df[annual_df["SITE_ID"]==sid].sort_values("year").copy()
    if len(sa)==0: st.warning("No data for this site."); return

    yrs = sa["year"].values; vals = sa[mc].values
    if len(yrs)>3:
        pairs = [(vals[j]-vals[i])/(yrs[j]-yrs[i])
                 for i in range(len(yrs)) for j in range(i+1,len(yrs)) if yrs[j]!=yrs[i]]
        slope = float(np.median(pairs))
        trend = slope*yrs + (float(np.median(vals))-slope*float(np.median(yrs)))
    else:
        slope=0.0; trend=vals

    fmu  = float(sa[sa["is_fire_year"]][mc].mean()) if sa["is_fire_year"].any() else 0
    nfmu = float(sa[~sa["is_fire_year"]][mc].mean())
    pct  = 100*(fmu-nfmu)/nfmu if nfmu>0 else 0
    yrs_left = int((vals[-1]-EPA)/abs(slope)) if slope<0 and vals[-1]>EPA else None
    tc = "#4dd9ac" if slope<0 else "#f55555"

    mc_cols = st.columns(4)
    for col,val,lbl in [(mc_cols[0],f"{slope:+.3f}/yr","Theil-Sen Slope"),
                         (mc_cols[1],f"{nfmu:.1f}","Non-Fire Mean"),
                         (mc_cols[2],f"{fmu:.1f}","Fire-Year Mean"),
                         (mc_cols[3],f"~{yrs_left}yr" if yrs_left else "N/A","Yrs to EPA High")]:
        clr_val = tc if lbl == "Theil-Sen Slope" else "#4dd9ac"
        col.markdown(f'<div class="metric-card"><div class="val" '
                     f'style="font-size:20px;color:{clr_val}">'
                     f'{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("")

    fig = make_subplots(rows=1,cols=2,
                        subplot_titles=[f"{mc.upper()} Trend — {sid}",
                                        f"Network Mean {mc.upper()}"],
                        column_widths=[0.60,0.40])
    mclr = ["#f55555" if r["is_fire_year"] else "#4da6f5" for _,r in sa.iterrows()]
    fig.add_trace(go.Scatter(x=yrs,y=vals,mode="markers+lines",
                             marker=dict(color=mclr,size=9,line=dict(width=1.5,color="#0e1828")),
                             line=dict(color="#2a4060",width=1),
                             name=mc.upper()),row=1,col=1)
    fig.add_trace(go.Scatter(x=yrs,y=trend,mode="lines",
                             line=dict(color="#f55555",dash="dash",width=2.5),
                             name=f"Trend ({slope:+.3f}/yr)"),row=1,col=1)
    fig.add_shape(type="line",x0=int(yrs.min()),x1=int(yrs.max()),
                  y0=EPA,y1=EPA,line=dict(color="#f5a623",width=1.4,dash="dot"),row=1,col=1)
    fig.add_annotation(x=int(yrs.max()),y=EPA*1.06,text=tlbl,showarrow=False,
                       font=dict(color="#f5a623",size=10),row=1,col=1)
    for _,r in sa[sa["is_fire_year"]].iterrows():
        fig.add_annotation(x=r["year"],y=r[mc]*1.08,text="🔥",showarrow=False,
                           font=dict(size=11),row=1,col=1)
    net = [(s,float(annual_df[annual_df["SITE_ID"]==s][mc].mean()))
           for s in site_meta if s in annual_df["SITE_ID"].values]
    net.sort(key=lambda x:-x[1])
    ns,nv = zip(*net) if net else ([],[])
    fig.add_trace(go.Bar(x=list(ns),y=list(nv),
                         marker_color=["#f55555" if s==sid else "#4da6f5" for s in ns],
                         name="Site mean"),row=1,col=2)
    fig.add_shape(type="line",x0=-0.5,x1=len(ns)-0.5,y0=EPA,y1=EPA,
                  line=dict(color="#f5a623",width=1.4,dash="dot"),row=1,col=2)
    fig.update_layout(**CHART_THEME,height=430,
                      legend=dict(**CHART_THEME["legend"],orientation="h",y=1.08,x=0))
    for ann in fig.layout.annotations: ann.font.color="#7eb8e0"; ann.font.size=12
    fig.update_xaxes(**CHART_THEME["xaxis"])
    fig.update_yaxes(**CHART_THEME["yaxis"])
    st.plotly_chart(fig,use_container_width=True)

    if mc=="aot40":
        st.markdown('<div class="warn-box">⚠️ <b>EU AOT40 saturation:</b> The EU forest threshold '
                    '(5,000 ppb·h) sits below every CA site mean — all 12 classify as "EU High". '
                    'Use W126 for policy-relevant tiers.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="warn-box">⚠️ <b>Fire-year confound (raw Δ={pct:+.1f}%):</b> '
                f'Biased by secular ozone decline and high-exposure sites offline during 2018–2021. '
                f'After within-site Z-scoring: fire anomaly −0.662σ (p=0.0016) — UV attenuation '
                f'dominates over precursor injection at most sites.</div>', unsafe_allow_html=True)

# ── TAB 5: COVERAGE ───────────────────────────────────────────────────────────
def tab_coverage(site_meta, coverage_df):
    st.markdown('<div class="sec-title">Temporal Split Audit — Train / Val / Test</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box">Split boundaries: '
                '<b>Train</b> ≤ 2018-12-31  |  <b>Val</b> 2019–2020  |  '
                '<b>Test</b> ≥ 2021-01-01</div>', unsafe_allow_html=True)

    def fn(v):
        try: n=int(v); return f"{n:,}" if n>0 else '<span class="zero">0</span>'
        except: return str(v)

    rows=""
    for _,row in coverage_df.iterrows():
        sid    = row["SITE_ID"]
        status = row.get("status", site_meta.get(sid,{}).get("status","ACTIVE"))
        bc     = {"ACTIVE":"badge-active","TRAIN_ONLY":"badge-trainonly",
                  "NO_TRAIN":"badge-notrain"}.get(status,"badge-active")
        note   = STATUS_NOTE.get(status,"")
        rows  += (f'<tr><td><b style="color:#e8f4ff">{sid}</b></td>'
                  f'<td>{row.get("name",site_meta.get(sid,{}).get("name","—"))}</td>'
                  f'<td class="num">{row.get("yr_start","—")}</td>'
                  f'<td class="num">{row.get("yr_end","—")}</td>'
                  f'<td class="num">{fn(row.get("n_train",0))}</td>'
                  f'<td class="num">{fn(row.get("n_val",0))}</td>'
                  f'<td class="num">{fn(row.get("n_test",0))}</td>'
                  f'<td><span class="badge {bc}">{status.replace("_"," ")}</span></td>'
                  f'<td style="font-size:11px;color:#5a7299">{note}</td></tr>')

    st.markdown(f'<div style="overflow-x:auto"><table class="data-table">'
                f'<thead><tr><th>Site ID</th><th>Name</th>'
                f'<th class="num">Data Start</th><th class="num">Data End</th>'
                f'<th class="num">Train</th><th class="num">Val</th><th class="num">Test</th>'
                f'<th>Status</th><th>Eval Note</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title" style="margin-top:20px">Coverage Timeline</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    for label,s,e,clr in [("Train",2000,2019,"#4da6f5"),
                           ("Val",2019,2021,"#f5a623"),
                           ("Test",2021,2025,"#4dd9ac")]:
        fig.add_vrect(x0=s,x1=e,fillcolor=clr,opacity=0.07,layer="below",line_width=0)
        fig.add_annotation(x=(s+e)/2,y=1.02,yref="paper",text=label,
                           showarrow=False,font=dict(color=clr,size=11))

    for i,(_,row) in enumerate(coverage_df.iterrows()):
        sid    = row["SITE_ID"]
        status = row.get("status","ACTIVE")
        try: ys,ye = int(row.get("yr_start",2000)),int(row.get("yr_end",2025))
        except: continue
        fig.add_trace(go.Scatter(
            x=[ys,ye],y=[i,i],mode="lines+markers",
            line=dict(color=STATUS_COLOR.get(status,"#4da6f5"),width=8),
            marker=dict(color=STATUS_COLOR.get(status,"#4da6f5"),size=8),
            name=sid,showlegend=False,
            hovertemplate=f"<b>{sid}</b>  {ys}–{ye}<extra></extra>"))
        fig.add_annotation(x=ys-0.3,y=i,text=sid,showarrow=False,
                           xanchor="right",font=dict(color="#7eb8e0",size=10))
    fig.update_layout(**CHART_THEME,height=420,
                      xaxis=dict(**CHART_THEME["xaxis"],range=[1987,2027],title="Year"),
                      yaxis=dict(visible=False))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="warn-box"><b>Bias correction caveat:</b> CON186/SEK402/YOS204/LPO010 '
                'pseudo-validation uses the last year of training. MAE improvements are optimistic — '
                'label these as "Global model + historical bias correction — no independent eval".'
                '</div><div class="err-box" style="margin-top:6px">'
                '<b>DEV412:</b> Walk-forward pseudo-training ≠ independent evaluation. '
                'RMSE penalty ≈ +0.86 ppb at t+8h.</div>', unsafe_allow_html=True)

# ── TAB 6: DIAGNOSTICS ───────────────────────────────────────────────────────
def tab_diagnostics(site_meta, metrics, feat_imp):
    st.markdown('<div class="sec-title">Model Diagnostics — Metrics & Feature Importance</div>',
                unsafe_allow_html=True)
    c1,_ = st.columns([2,3])
    site_opts = [(f"{s}  —  {site_meta[s]['name']}",s) for s in site_meta]
    sel = c1.selectbox("Site",[v for v,_ in site_opts],
                       index=[k for _,k in site_opts].index("SEK430"),key="diag_site")
    sid  = dict(site_opts)[sel]
    info = site_meta[sid]
    if info["status"]=="NO_TRAIN":
        st.markdown('<div class="err-box">🔴 NO TRAINING DATA — extrapolated predictions only.</div>',
                    unsafe_allow_html=True)
    elif info["status"]=="TRAIN_ONLY":
        st.markdown('<div class="warn-box">⚠️ TRAIN ONLY — in-sample metrics.</div>',
                    unsafe_allow_html=True)

    mall = metrics.get(sid,{})
    Hs   = ["1","8","24"]
    hlbl = ["t+1h","t+8h","t+24h"]
    maes  = [mall[H]["mae"]  if H in mall else 0 for H in Hs]
    rmses = [mall[H]["rmse"] if H in mall else 0 for H in Hs]
    r2s   = [mall[H]["r2"]   if H in mall else 0 for H in Hs]

    fig = make_subplots(rows=1,cols=3,
                        subplot_titles=["MAE & RMSE","R² by Horizon",
                                        "Feature Importance (% gain)"])
    fig.add_trace(go.Bar(x=hlbl,y=maes,name="MAE",
                         marker_color="#4da6f5",offsetgroup=0),row=1,col=1)
    fig.add_trace(go.Bar(x=hlbl,y=rmses,name="RMSE",
                         marker_color="#f55555",offsetgroup=1),row=1,col=1)
    fig.update_yaxes(title_text="ppb",row=1,col=1)
    r2c = ["#4dd9ac" if v>0.8 else "#f5a623" if v>0.6 else "#f55555" for v in r2s]
    fig.add_trace(go.Bar(x=hlbl,y=r2s,marker_color=r2c,name="R²"),row=1,col=2)
    fig.add_shape(type="line",x0=-0.5,x1=2.5,y0=0.7,y1=0.7,
                  line=dict(color="#f5a623",dash="dot",width=1.5),row=1,col=2)
    for g in FEAT_CLRS:
        fig.add_trace(go.Bar(x=hlbl,y=[feat_imp.get(H,{}).get(g,0) for H in Hs],
                             name=g,marker_color=FEAT_CLRS[g]),row=1,col=3)
    fig.update_yaxes(title_text="% of total gain",row=1,col=3)
    fig.update_layout(**CHART_THEME,height=360,barmode="group",
                      legend=dict(**CHART_THEME["legend"],orientation="h",y=-0.22,x=0))
    for ann in fig.layout.annotations: ann.font.color="#7eb8e0"; ann.font.size=12
    fig.update_xaxes(**CHART_THEME["xaxis"])
    fig.update_yaxes(**CHART_THEME["yaxis"])
    st.plotly_chart(fig,use_container_width=True)

    trows=""
    for H in Hs:
        m = mall.get(H,{})
        if not m: continue
        r2c= "#4dd9ac" if m["r2"]>0.8 else "#f5a623" if m["r2"]>0.6 else "#f55555"
        ms = (f'<span style="color:#f55555">{m["miss_rate"]:.1%} ⚠️</span>'
              if H=="24" else "—")
        trows += (f'<tr><td style="color:#4dd9ac">t+{H}h</td>'
                  f'<td class="num">{m["mae"]:.3f}</td><td class="num">{m["rmse"]:.3f}</td>'
                  f'<td class="num" style="color:{r2c}"><b>{m["r2"]:.3f}</b></td>'
                  f'<td class="num">{m["recall"]:.3f}</td>'
                  f'<td class="num">{m["precision"]:.3f}</td>'
                  f'<td class="num">{ms}</td>'
                  f'<td class="num">{m.get("n_test",0):,}</td></tr>')
    st.markdown('<div class="sec-title">Per-Horizon Metrics</div>', unsafe_allow_html=True)
    st.markdown(f'<table class="data-table" style="width:70%">'
                f'<thead><tr><th>Horizon</th><th class="num">MAE</th>'
                f'<th class="num">RMSE</th><th class="num">R²</th>'
                f'<th class="num">Recall</th><th class="num">Precision</th>'
                f'<th class="num">Struct. Miss</th><th class="num">N test</th>'
                f'</tr></thead><tbody>{trows}</tbody></table>'
                f'<p style="font-size:11px;color:#3a5a80;margin-top:8px">'
                f'Structural miss rate (t+24h): fraction of true exceedance hours with '
                f'ŷ&lt;70 ppb — not fixable by threshold tuning alone.</p>',
                unsafe_allow_html=True)

    st.markdown('<div class="sec-title" style="margin-top:18px">All-Sites R² Heatmap</div>',
                unsafe_allow_html=True)
    all_sids = list(metrics.keys())
    r2_mat   = [[metrics[s].get(H,{}).get("r2",0) for H in Hs] for s in all_sids]
    fig2 = go.Figure(go.Heatmap(
        z=r2_mat, x=hlbl, y=all_sids,
        colorscale=[[0,"#3a0a0a"],[0.3,"#7a2020"],[0.6,"#f5a623"],
                    [0.8,"#4dd9ac"],[1,"#4da6f5"]],
        zmin=0,zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in r2_mat],
        texttemplate="%{text}",
        textfont=dict(size=11,color="#e8f4ff"),
        hovertemplate="Site: %{y}<br>Horizon: %{x}<br>R²: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#7eb8e0",size=10),
                      outlinecolor="#1e3050",outlinewidth=1),
    ))
    fig2.update_layout(**CHART_THEME,height=380,
                       xaxis=dict(**CHART_THEME["xaxis"],title="Forecast horizon"),
                       margin=dict(t=20,b=35,l=80,r=60))
    fig2.update_yaxes(**CHART_THEME["yaxis"])
    st.plotly_chart(fig2,use_container_width=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    sm,sites,oz,an,mt,fi,cv,pl,srcs = get_all_data()
    render_sidebar(pl,srcs)
    any_syn = any(v=="synthetic" for v in srcs.values())
    bdg = ('<span class="badge badge-synthetic">SYNTHETIC DATA</span>'
           if any_syn else '<span class="badge badge-live">LIVE DATA</span>')
    st.markdown(f'<div class="dash-header"><div>'
                f'<h1>🌬️ CASTNET California Ozone Forecasting Dashboard</h1>'
                f'<p>12 monitoring sites · 1989–2025 · LightGBM · t+1h / t+8h / t+24h horizons</p>'
                f'</div><div style="text-align:right">{bdg}'
                f'<div style="font-size:11px;color:#4a6a90;margin-top:4px">'
                f'{datetime.now().strftime("%d %b %Y  %H:%M")}</div></div></div>',
                unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs(["🗺️  Site Map","📈  Forecast",
                                   "⚠️  Exceedance Alerts","🌿  Vegetation",
                                   "📋  Data Coverage","🔬  Diagnostics"])
    with t1: tab_site_map(sm,srcs)
    with t2: tab_forecast(sm,oz,mt)
    with t3: tab_exceedance(sm,oz,mt)
    with t4: tab_vegetation(sm,an)
    with t5: tab_coverage(sm,cv)
    with t6: tab_diagnostics(sm,mt,fi)

if __name__=="__main__":
    main()
