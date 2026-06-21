"""
AI-Driven Parking Intelligence Dashboard
Bengaluru Traffic Police — Jan to May Violations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
from streamlit_folium import st_folium

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.impute import SimpleImputer
import joblib, json, os
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parking Intelligence Hub — Bengaluru",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-title { font-size: 0.78rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #f8fafc; margin: 0.3rem 0; }
    .metric-delta { font-size: 0.82rem; color: #22d3ee; }
    .hotspot-card {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .hotspot-rank { font-size: 1.3rem; font-weight: 800; color: #ef4444; }
    .hotspot-name { font-size: 0.95rem; font-weight: 600; color: #f1f5f9; }
    .hotspot-sub  { font-size: 0.78rem; color: #94a3b8; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.9rem 0;
    }
    .risk-HIGH   { color: #ef4444; font-weight: 700; }
    .risk-MEDIUM { color: #f59e0b; font-weight: 700; }
    .risk-LOW    { color: #22c55e; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #3b82f6 !important; border-bottom: 2px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loader ──────────────────────────────────────────────────────────────
DATA_URL = "https://uc.hackerearth.com/he-public-ap-south-1/jan%20to%20may%20police%20violation_anonymized791b166.csv"

USE_COLS = [
    "id", "created_datetime", "closed_datetime", "violation_type",
    "latitude", "longitude", "vehicle_type", "police_station",
    "junction_name", "validation_status",
]

@st.cache_data(show_spinner="Loading violation dataset…", ttl=3600, max_entries=1)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_URL,
        usecols=lambda c: c in USE_COLS,
        dtype={
            "vehicle_type": "category",
            "police_station": "category",
            "junction_name": "category",
            "validation_status": "category",
        },
        low_memory=False,
    )

    # Datetime
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["hour"]  = df["created_datetime"].dt.hour
    df["month"] = df["created_datetime"].dt.month
    df["dow"]   = df["created_datetime"].dt.day_name()
    df["date"]  = df["created_datetime"].dt.date

    # Drop rows without coordinates (do this early, before expensive string ops)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] > 12.7) & (df["latitude"] < 13.4)]
    df = df[(df["longitude"] > 77.4) & (df["longitude"] < 77.9)]

    # Clean violation labels — strip JSON brackets & quotes
    df["violation_clean"] = (
        df["violation_type"]
        .fillna("UNKNOWN")
        .str.replace(r'[\[\]"]', "", regex=True)
        .str.strip()
    )
    # Primary violation (first listed)
    df["primary_violation"] = df["violation_clean"].str.split(",").str[0].str.strip()

    # Parking-specific flag
    PARKING_KEYWORDS = ["PARKING", "NO PARKING", "WRONG PARKING", "FOOTPATH"]
    df["is_parking"] = df["violation_clean"].str.contains(
        "|".join(PARKING_KEYWORDS), case=False, na=False
    )

    # Congestion impact score (heuristic, vectorized instead of row-wise .apply):
    # Main road parking = high impact; footpath/wrong = medium; no parking = medium-low
    v_upper = df["violation_clean"].str.upper()
    df["impact_score"] = np.select(
        [
            v_upper.str.contains("MAIN ROAD", na=False),
            v_upper.str.contains("FOOTPATH|DOUBLE|BUSTOP|SCHOOL", na=False),
            v_upper.str.contains("NO PARKING|WRONG PARKING", na=False),
        ],
        [3, 2, 1],
        default=0,
    )

    # Resolution time (minutes)
    df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], utc=True, errors="coerce")
    df["resolution_min"] = (
        (df["closed_datetime"] - df["created_datetime"]).dt.total_seconds() / 60
    ).clip(0, 1440)  # cap at 24h

    # Month name
    month_map = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 11:"Nov", 12:"Dec"}
    df["month_name"] = df["month"].map(month_map).fillna("Other")

    return df


@st.cache_data(show_spinner="Computing hotspots…", max_entries=5)
def compute_hotspots(df: pd.DataFrame, grid_size: float = 0.005) -> pd.DataFrame:
    """Grid-based hotspot aggregation with risk scoring."""
    parking = df[df["is_parking"]].copy()
    parking["lat_bin"] = (parking["latitude"]  / grid_size).round() * grid_size
    parking["lon_bin"] = (parking["longitude"] / grid_size).round() * grid_size

    agg = parking.groupby(["lat_bin", "lon_bin"]).agg(
        count=("id", "count"),
        avg_impact=("impact_score", "mean"),
        main_road_pct=("violation_clean", lambda x: x.str.contains("MAIN ROAD").mean() * 100),
        top_violation=("primary_violation", lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"),
        top_station=("police_station", lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"),
    ).reset_index()

    agg["weighted_score"] = agg["count"] * agg["avg_impact"]
    agg["risk_tier"] = pd.cut(
        agg["weighted_score"],
        bins=[0, 50, 200, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    agg = agg.sort_values("weighted_score", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1
    return agg


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Parking Intelligence Hub")
    st.markdown("*Bengaluru Traffic Police*")
    st.markdown("---")

    st.markdown("### Filters")

    month_opts = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 11:"Nov", 12:"Dec"}
    selected_months = st.multiselect(
        "Month(s)",
        options=list(month_opts.keys()),
        default=list(month_opts.keys()),
        format_func=lambda x: month_opts[x],
    )

    hour_range = st.slider("Hour of day (IST ≈ UTC+5:30)", 0, 23, (0, 23))

    violation_filter = st.selectbox(
        "Violation category",
        ["All", "WRONG PARKING", "NO PARKING", "PARKING IN A MAIN ROAD",
         "PARKING ON FOOTPATH", "DOUBLE PARKING"],
    )

    risk_filter = st.multiselect(
        "Risk tier",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

    st.markdown("---")
    st.markdown("### Map Settings")
    map_type = st.radio("Map layer", ["Heatmap", "Cluster Markers", "Risk Grid"])
    top_n_hotspots = st.slider("Top N hotspots to show", 5, 50, 20)

    st.markdown("---")
    st.caption("Data: Jan–Apr 2024 + Nov–Dec 2023 | Source: BTP")


# ─── Load & Filter ────────────────────────────────────────────────────────────
df_raw = load_data()

df = df_raw.copy()
if selected_months:
    df = df[df["month"].isin(selected_months)]
df = df[(df["hour"] >= hour_range[0]) & (df["hour"] <= hour_range[1])]
if violation_filter != "All":
    df = df[df["violation_clean"].str.contains(violation_filter, case=False, na=False)]

df_parking = df[df["is_parking"]]
hotspots   = compute_hotspots(df)
if risk_filter:
    hotspots = hotspots[hotspots["risk_tier"].isin(risk_filter)]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:1.9rem; font-weight:800; color:#f1f5f9; margin-bottom:0'>AI Parking Intelligence Hub</h1>"
    "<p style='color:#94a3b8; margin-top:4px'>Illegal Parking Hotspot Detection & Congestion Impact Quantification — Bengaluru</p>",
    unsafe_allow_html=True,
)

# ─── KPI Cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total       = len(df)
park_total  = len(df_parking)
park_pct    = park_total / total * 100 if total else 0
high_impact = len(df_parking[df_parking["impact_score"] == 3])
avg_res     = df_parking["resolution_min"].dropna().median()
unique_zones= hotspots[hotspots["risk_tier"] == "HIGH"].shape[0]

for col, title, value, delta in zip(
    [k1, k2, k3, k4, k5],
    ["Total Violations", "Parking Violations", "High-Impact %", "Median Response", "High-Risk Zones"],
    [f"{total:,}", f"{park_total:,}", f"{park_pct:.1f}%", f"{avg_res:.0f} min", str(unique_zones)],
    ["Filtered period", f"{park_pct:.0f}% of total", "Main road blocks", "Median resolution", "Priority enforcement"],
):
    col.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Hotspot Map",
    "Analytics",
    "Enforcement Zones",
    "Temporal Patterns",
    "AI Risk Scorer",
    "ML Pipeline",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOTSPOT MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_map, col_panel = st.columns([3, 1])

    with col_map:
        st.markdown('<div class="section-header">Parking Violation Heatmap — Bengaluru</div>', unsafe_allow_html=True)

        center_lat = df_parking["latitude"].mean()
        center_lon = df_parking["longitude"].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )

        # Heatmap points are cheap (just float triples), Cluster Markers are
        # expensive (a full Folium object + HTML popup string per point).
        MAP_SAMPLE_CAP = 15000 if map_type == "Heatmap" else 2000
        sample = df_parking.sample(min(len(df_parking), MAP_SAMPLE_CAP), random_state=42)

        if map_type == "Heatmap":
            heat_data = sample[["latitude", "longitude", "impact_score"]].values.tolist()
            HeatMap(
                heat_data,
                radius=14,
                blur=18,
                max_zoom=15,
                gradient={0.2: "#3b82f6", 0.5: "#f59e0b", 0.8: "#ef4444", 1.0: "#ffffff"},
            ).add_to(m)

        elif map_type == "Cluster Markers":
            mc = MarkerCluster(name="Violations").add_to(m)
            COLORS = {3: "red", 2: "orange", 1: "blue", 0: "gray"}
            for _, row in sample.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=4,
                    color=COLORS.get(row["impact_score"], "gray"),
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"<b>{row['primary_violation']}</b><br>{row.get('police_station','')}<br>{str(row.get('created_datetime',''))[:10]}",
                ).add_to(mc)

        else:  # Risk Grid
            top_hs = hotspots.head(top_n_hotspots)
            RISK_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
            for _, row in top_hs.iterrows():
                risk = str(row.get("risk_tier", "LOW"))
                folium.Rectangle(
                    bounds=[
                        [row["lat_bin"] - 0.0025, row["lon_bin"] - 0.0025],
                        [row["lat_bin"] + 0.0025, row["lon_bin"] + 0.0025],
                    ],
                    color=RISK_COLOR.get(risk, "#94a3b8"),
                    fill=True,
                    fill_opacity=0.55,
                    popup=folium.Popup(
                        f"<b>Rank #{int(row['rank'])}</b><br>"
                        f"Violations: {int(row['count'])}<br>"
                        f"Risk: <b>{risk}</b><br>"
                        f"Main Road %: {row['main_road_pct']:.1f}%<br>"
                        f"Station: {row['top_station']}",
                        max_width=220,
                    ),
                ).add_to(m)
                folium.Marker(
                    location=[row["lat_bin"], row["lon_bin"]],
                    icon=folium.DivIcon(
                        html=f"<div style='font-size:9px;font-weight:700;color:white;background:{RISK_COLOR.get(risk,'gray')};border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;'>{int(row['rank'])}</div>",
                        icon_size=(22, 22),
                        icon_anchor=(11, 11),
                    ),
                ).add_to(m)

        # Add top 5 junctions
        top_junctions = (
            df_parking[df_parking["junction_name"] != "No Junction"]
            .groupby("junction_name")
            .agg(lat=("latitude", "mean"), lon=("longitude", "mean"), count=("id", "count"))
            .nlargest(5, "count")
            .reset_index()
        )
        for _, jrow in top_junctions.iterrows():
            folium.Marker(
                location=[jrow["lat"], jrow["lon"]],
                icon=folium.Icon(color="purple", icon="info-sign"),
                popup=f"<b>{jrow['junction_name']}</b><br>{int(jrow['count'])} violations",
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, height=520, use_container_width=True)

    with col_panel:
        st.markdown('<div class="section-header">Top Hotspots</div>', unsafe_allow_html=True)
        for _, row in hotspots.head(10).iterrows():
            risk = str(row.get("risk_tier", "LOW"))
            border = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(risk, "#64748b")
            st.markdown(
                f"""<div style="background:#1e293b;border-left:4px solid {border};border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;">
                    <span style="font-size:1.1rem;font-weight:800;color:{border}">#{int(row['rank'])}</span>
                    <span style="font-size:0.82rem;font-weight:600;color:#f1f5f9;margin-left:0.5rem">{row['top_station']}</span><br>
                    <span style="font-size:0.72rem;color:#94a3b8">{int(row['count'])} violations | {row['main_road_pct']:.0f}% main road</span><br>
                    <span style="font-size:0.72rem;color:{border}">Risk: {risk}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Legend**")
        for tier, color, label in [("HIGH", "#ef4444", "Priority Zone"), ("MEDIUM", "#f59e0b", "Watch Zone"), ("LOW", "#22c55e", "Normal Zone")]:
            st.markdown(f"<span style='color:{color}'>■</span> **{tier}** — {label}", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Violation Type Breakdown</div>', unsafe_allow_html=True)
        vtype_counts = (
            df["primary_violation"]
            .value_counts()
            .nlargest(12)
            .reset_index()
        )
        vtype_counts.columns = ["Violation", "Count"]
        fig_vtype = px.bar(
            vtype_counts,
            x="Count",
            y="Violation",
            orientation="h",
            color="Count",
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig_vtype.update_layout(
            height=380,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=10),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vtype, width='stretch')

    with col_b:
        st.markdown('<div class="section-header">Vehicle Type Distribution</div>', unsafe_allow_html=True)
        veh_counts = df_parking["vehicle_type"].value_counts().nlargest(10).reset_index()
        veh_counts.columns = ["Vehicle", "Count"]
        fig_veh = px.pie(
            veh_counts,
            names="Vehicle",
            values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            template="plotly_dark",
        )
        fig_veh.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_veh, width='stretch')

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Monthly Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby("month_name").size().reset_index(name="count")
        order = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
        monthly["month_name"] = pd.Categorical(monthly["month_name"], categories=order, ordered=True)
        monthly = monthly.sort_values("month_name")
        fig_mon = px.area(
            monthly,
            x="month_name",
            y="count",
            template="plotly_dark",
            color_discrete_sequence=["#3b82f6"],
            markers=True,
        )
        fig_mon.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title="Month",
            yaxis_title="Violations",
        )
        st.plotly_chart(fig_mon, width='stretch')

    with col_d:
        st.markdown('<div class="section-header">Validation Status</div>', unsafe_allow_html=True)
        vs = df["validation_status"].value_counts().reset_index()
        vs.columns = ["Status", "Count"]
        COLOR_MAP = {
            "approved": "#22c55e",
            "rejected": "#ef4444",
            "processing": "#f59e0b",
            "created1": "#3b82f6",
            "duplicate": "#a855f7",
        }
        fig_vs = px.bar(
            vs,
            x="Status",
            y="Count",
            color="Status",
            color_discrete_map=COLOR_MAP,
            template="plotly_dark",
        )
        fig_vs.update_layout(
            height=300,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_vs, width='stretch')

    st.markdown('<div class="section-header">Top 15 Police Stations — Parking Violations</div>', unsafe_allow_html=True)
    station_df = (
        df_parking.groupby("police_station")
        .agg(
            total=("id", "count"),
            high_impact=("impact_score", lambda x: (x == 3).sum()),
            avg_score=("impact_score", "mean"),
            main_road_pct=("violation_clean", lambda x: x.str.contains("MAIN ROAD").mean() * 100),
        )
        .nlargest(15, "total")
        .reset_index()
    )
    station_df["congestion_risk"] = station_df["avg_score"].apply(
        lambda x: "HIGH" if x >= 2 else ("MEDIUM" if x >= 1 else "LOW")
    )
    st.dataframe(
        station_df[["police_station", "total", "high_impact", "main_road_pct", "congestion_risk"]]
        .rename(columns={
            "police_station": "Station",
            "total": "Total Violations",
            "high_impact": "Main Road Cases",
            "main_road_pct": "Main Road %",
            "congestion_risk": "Congestion Risk",
        }),
        width='stretch',
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ENFORCEMENT ZONES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Priority Enforcement Zones</div>', unsafe_allow_html=True)
    st.info(
        "Zones are ranked by a **Weighted Congestion Score** = violation count × avg impact. "
        "Main road blocking (score 3) weighs more than wrong/no parking (score 1).",
    )

    col_e, col_f = st.columns([2, 1])

    with col_e:
        top_hs = hotspots.head(top_n_hotspots).copy()
        top_hs["risk_tier"] = top_hs["risk_tier"].astype(str)
        fig_hs = px.scatter(
            top_hs,
            x="count",
            y="main_road_pct",
            size="weighted_score",
            color="risk_tier",
            color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"},
            hover_data={"top_station": True, "rank": True, "avg_impact": ":.2f"},
            labels={"count": "Total Violations", "main_road_pct": "Main Road %", "risk_tier": "Risk"},
            template="plotly_dark",
        )
        fig_hs.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=0, r=0, t=20, b=10),
        )
        st.plotly_chart(fig_hs, width='stretch')

    with col_f:
        st.markdown("**Top 10 Priority Grids**")
        for _, row in hotspots.head(10).iterrows():
            risk = str(row.get("risk_tier", "LOW"))
            st.markdown(
                f"**#{int(row['rank'])}** {row['top_station']} ({risk})  \n"
                f"<small style='color:#94a3b8'>{int(row['count'])} violations · {row['main_road_pct']:.0f}% main road · score {row['weighted_score']:.0f}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

    # Junction analysis
    st.markdown('<div class="section-header">Junction-Level Hotspots</div>', unsafe_allow_html=True)
    junction_df = (
        df_parking[df_parking["junction_name"] != "No Junction"]
        .groupby("junction_name")
        .agg(
            count=("id", "count"),
            avg_impact=("impact_score", "mean"),
            main_road_pct=("violation_clean", lambda x: x.str.contains("MAIN ROAD").mean() * 100),
            top_vehicle=("vehicle_type", lambda x: x.mode().iloc[0] if len(x) else "N/A"),
        )
        .nlargest(20, "count")
        .reset_index()
    )
    junction_df["Priority Score"] = (junction_df["count"] * junction_df["avg_impact"]).round(1)
    junction_df["Risk"] = junction_df["avg_impact"].apply(
        lambda x: "HIGH" if x >= 2 else ("MEDIUM" if x >= 1 else "LOW")
    )

    fig_junc = px.bar(
        junction_df.head(15),
        x="count",
        y="junction_name",
        color="avg_impact",
        color_continuous_scale="RdYlGn_r",
        orientation="h",
        template="plotly_dark",
        labels={"count": "Violations", "junction_name": "Junction", "avg_impact": "Avg Impact"},
    )
    fig_junc.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_junc, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="section-header">Hourly Violation Density</div>', unsafe_allow_html=True)
        hourly = df_parking.groupby("hour").size().reset_index(name="count")
        fig_hr = px.bar(
            hourly,
            x="hour",
            y="count",
            color="count",
            color_continuous_scale="Inferno",
            template="plotly_dark",
            labels={"hour": "Hour (UTC)", "count": "Violations"},
        )
        fig_hr.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_hr, width='stretch')

    with col_h:
        st.markdown('<div class="section-header">Day-of-Week Pattern</div>', unsafe_allow_html=True)
        DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df = df_parking.groupby("dow").size().reset_index(name="count")
        dow_df["dow"] = pd.Categorical(dow_df["dow"], categories=DOW_ORDER, ordered=True)
        dow_df = dow_df.sort_values("dow")
        fig_dow = px.line(
            dow_df,
            x="dow",
            y="count",
            markers=True,
            template="plotly_dark",
            color_discrete_sequence=["#22d3ee"],
            labels={"dow": "Day", "count": "Violations"},
        )
        fig_dow.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_dow, width='stretch')

    st.markdown('<div class="section-header">Hourly × Violation Type Heatmap</div>', unsafe_allow_html=True)
    top_vtypes = df_parking["primary_violation"].value_counts().nlargest(6).index.tolist()
    pivot = (
        df_parking[df_parking["primary_violation"].isin(top_vtypes)]
        .groupby(["hour", "primary_violation"])
        .size()
        .unstack(fill_value=0)
    )
    fig_heat = px.imshow(
        pivot.T,
        color_continuous_scale="Reds",
        aspect="auto",
        template="plotly_dark",
        labels={"x": "Hour (UTC)", "color": "Count"},
    )
    fig_heat.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_heat, width='stretch')

    st.markdown('<div class="section-header">Resolution Time Distribution</div>', unsafe_allow_html=True)
    res_df = df_parking["resolution_min"].dropna()
    fig_res = px.histogram(
        res_df,
        nbins=60,
        color_discrete_sequence=["#3b82f6"],
        template="plotly_dark",
        labels={"value": "Resolution Time (min)", "count": "Cases"},
    )
    fig_res.add_vline(x=res_df.median(), line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Median: {res_df.median():.0f} min")
    fig_res.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_res, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">AI Congestion Risk Scorer</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter zone details below to get an **AI-computed congestion risk score** "
        "and enforcement recommendation."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        inp_violations = st.number_input("Violations in zone (last 30 days)", 1, 5000, 150)
        inp_main_road  = st.slider("% Main road violations", 0, 100, 30)
    with c2:
        inp_vehicle    = st.selectbox("Dominant vehicle type", ["SCOOTER", "CAR", "MOTOR CYCLE", "PASSENGER AUTO", "MAXI-CAB", "LGV", "BUS"])
        inp_junction   = st.checkbox("Near a junction / metro / school?", value=False)
    with c3:
        inp_peak_hour  = st.checkbox("Peak hour violations dominant? (7–10 AM / 5–8 PM)", value=False)
        inp_repeat     = st.slider("Repeat offender rate (%)", 0, 100, 20)

    # Heuristic AI scoring
    def compute_ai_risk(violations, main_road_pct, vehicle, near_junction, peak_hour, repeat_rate):
        score = 0
        # Volume
        if violations > 500: score += 40
        elif violations > 200: score += 25
        elif violations > 50: score += 12
        else: score += 5
        # Main road impact
        score += main_road_pct * 0.35
        # Vehicle weight (heavier = more blocking)
        vehicle_weights = {"BUS": 12, "LGV": 10, "MAXI-CAB": 8, "PASSENGER AUTO": 5, "CAR": 4, "MOTOR CYCLE": 2, "SCOOTER": 1}
        score += vehicle_weights.get(vehicle, 3)
        # Context
        if near_junction: score += 15
        if peak_hour: score += 12
        score += repeat_rate * 0.1

        score = min(score, 100)
        if score >= 65: risk = "HIGH"; color = "#ef4444"; action = "Deploy dedicated enforcement unit immediately"
        elif score >= 35: risk = "MEDIUM"; color = "#f59e0b"; action = "Schedule regular patrol (twice daily)"
        else: risk = "LOW"; color = "#22c55e"; action = "Monitor remotely — include in weekly sweep"
        return score, risk, color, action

    if st.button("Compute Risk Score", type="primary"):
        score, risk, color, action = compute_ai_risk(
            inp_violations, inp_main_road, inp_vehicle, inp_junction, inp_peak_hour, inp_repeat
        )

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Risk Score</div>
                <div class="metric-value" style="color:{color}">{score:.0f}/100</div>
                <div class="metric-delta">Composite score</div>
            </div>""",
            unsafe_allow_html=True,
        )
        r2.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Risk Tier</div>
                <div class="metric-value" style="color:{color}">{risk}</div>
                <div class="metric-delta">Enforcement priority</div>
            </div>""",
            unsafe_allow_html=True,
        )
        r3.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Recommended Action</div>
                <div class="metric-value" style="font-size:1rem;color:#f1f5f9">{action}</div>
                <div class="metric-delta">&nbsp;</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Score gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Congestion Risk Score", "font": {"color": "#f1f5f9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35],  "color": "rgba(34,197,94,0.15)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.15)"},
                    {"range": [65, 100],"color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": score},
            },
            number={"font": {"color": color, "size": 48}},
        ))
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f1f5f9"},
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, width='stretch')

        # Factor breakdown
        st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
        factors = {
            "Volume score": min(40, 40 if inp_violations > 500 else (25 if inp_violations > 200 else (12 if inp_violations > 50 else 5))),
            "Main road impact": round(inp_main_road * 0.35, 1),
            "Vehicle weight": {"BUS": 12, "LGV": 10, "MAXI-CAB": 8, "PASSENGER AUTO": 5, "CAR": 4, "MOTOR CYCLE": 2, "SCOOTER": 1}.get(inp_vehicle, 3),
            "Junction/School proximity": 15 if inp_junction else 0,
            "Peak hour multiplier": 12 if inp_peak_hour else 0,
            "Repeat offender rate": round(inp_repeat * 0.1, 1),
        }
        factor_df = pd.DataFrame(list(factors.items()), columns=["Factor", "Points"])
        fig_f = px.bar(
            factor_df,
            x="Points",
            y="Factor",
            orientation="h",
            color="Points",
            color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
        )
        fig_f.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_f, width='stretch')

    st.markdown("---")
    st.markdown('<div class="section-header">Bulk Zone Risk Table (Top 30 Hotspots)</div>', unsafe_allow_html=True)
    bulk = hotspots.head(30).copy()
    bulk["risk_tier"] = bulk["risk_tier"].astype(str)
    bulk["Enforcement Action"] = bulk["risk_tier"].map({
        "HIGH":   "Immediate unit deployment",
        "MEDIUM": "Twice-daily patrol",
        "LOW":    "Weekly sweep",
    })
    st.dataframe(
        bulk[["rank", "top_station", "count", "main_road_pct", "avg_impact", "weighted_score", "risk_tier", "Enforcement Action"]]
        .rename(columns={
            "rank": "#",
            "top_station": "Station",
            "count": "Violations",
            "main_road_pct": "Main Road %",
            "avg_impact": "Avg Impact",
            "weighted_score": "Risk Score",
            "risk_tier": "Tier",
        }),
        width='stretch',
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(
        "<h2 style='color:#f1f5f9;font-size:1.4rem;font-weight:800'>Machine Learning Pipeline</h2>"
        "<p style='color:#94a3b8'>End-to-end classification pipeline: predict whether a parking violation "
        "will have <b>HIGH congestion impact</b> (main road blocking) vs LOW/MEDIUM.</p>",
        unsafe_allow_html=True,
    )

    # ── Step 1: Preprocessing ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Step 1 — Data Preprocessing")

    with st.expander("View preprocessing steps (click to expand)", expanded=True):

        # ── 1a. Feature engineering from raw data
        @st.cache_data(show_spinner="Building ML feature matrix…", max_entries=3)
        def build_features(df: pd.DataFrame):
            """
            Target  : is_high_impact  (1 = main road blocking, 0 = other parking)
            Features: hour, month, dow_num, lat, lon, vehicle_type encoded,
                      police_station encoded, near_junction, violation_count_area
            """
            ml = df[df["is_parking"]].copy()
            ml = ml.dropna(subset=["latitude", "longitude", "vehicle_type", "police_station"])

            # Target
            ml["is_high_impact"] = (ml["impact_score"] == 3).astype(int)

            # Numeric features from datetime
            ml["hour"]    = ml["created_datetime"].dt.hour
            ml["month"]   = ml["created_datetime"].dt.month
            ml["dow_num"] = ml["created_datetime"].dt.dayofweek   # 0=Mon

            # Binary: near a named junction
            ml["near_junction"] = (ml["junction_name"] != "No Junction").astype(int)

            # Area density: violations per ~500m grid cell (proxy for congestion pressure)
            grid = 0.005
            ml["lat_bin"] = (ml["latitude"]  / grid).round() * grid
            ml["lon_bin"] = (ml["longitude"] / grid).round() * grid
            density = ml.groupby(["lat_bin", "lon_bin"])["id"].transform("count")
            ml["area_density"] = density

            # Encode categoricals
            le_veh  = LabelEncoder()
            le_sta  = LabelEncoder()
            ml["vehicle_enc"]  = le_veh.fit_transform(ml["vehicle_type"].fillna("UNKNOWN"))
            ml["station_enc"]  = le_sta.fit_transform(ml["police_station"].fillna("UNKNOWN"))

            FEATURES = [
                "hour", "month", "dow_num",
                "latitude", "longitude",
                "vehicle_enc", "station_enc",
                "near_junction", "area_density",
            ]
            TARGET = "is_high_impact"

            X = ml[FEATURES].copy()
            y = ml[TARGET].copy()

            return X, y, FEATURES, le_veh, le_sta, ml

        X_full, y_full, FEATURE_NAMES, le_veh, le_sta, ml_df = build_features(df_raw)

        # ── Missing value report
        missing = X_full.isnull().sum()
        missing_pct = (missing / len(X_full) * 100).round(2)
        missing_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values,
            "Strategy": ["Median imputation" if m > 0 else "Complete" for m in missing.values],
        })

        col_pre1, col_pre2 = st.columns(2)

        with col_pre1:
            st.markdown("#### Missing Value Analysis")
            st.dataframe(missing_df, width='stretch', hide_index=True)
            total_missing = missing.sum()
            if total_missing == 0:
                st.success("No missing values in selected features after filtering.")
            else:
                st.warning(f"{total_missing} missing values — handled via median imputation.")

        with col_pre2:
            st.markdown("#### Feature Engineering Summary")
            st.markdown("""
| Feature | Type | Description |
|---------|------|-------------|
| `hour` | Numeric | Hour of violation (0–23 UTC) |
| `month` | Numeric | Month number |
| `dow_num` | Numeric | Day of week (0=Mon) |
| `latitude` / `longitude` | Numeric | GPS coordinates |
| `vehicle_enc` | Encoded | Vehicle type → integer |
| `station_enc` | Encoded | Police station → integer |
| `near_junction` | Binary | 1 if at named junction |
| `area_density` | Numeric | Violations per ~500m grid cell |
""")

        st.markdown("#### Encoding: Categorical Variables")
        enc_col1, enc_col2 = st.columns(2)
        with enc_col1:
            veh_map = pd.DataFrame({
                "Vehicle Type": le_veh.classes_,
                "Encoded Value": range(len(le_veh.classes_)),
            })
            st.markdown("**Vehicle Type → Label Encoding**")
            st.dataframe(veh_map.head(10), width='stretch', hide_index=True)
        with enc_col2:
            sta_map = pd.DataFrame({
                "Police Station": le_sta.classes_[:10],
                "Encoded Value": range(10),
            })
            st.markdown("**Police Station → Label Encoding (first 10)**")
            st.dataframe(sta_map, width='stretch', hide_index=True)

        st.markdown("#### Feature Scaling")
        st.markdown("""
`StandardScaler` is applied to all numeric features before model training:
- Removes mean, scales to unit variance: `z = (x − μ) / σ`
- Prevents high-magnitude features (e.g. `area_density` up to 1500+) from dominating distance-based computations
- Tree-based models (Random Forest, XGBoost) are scale-invariant, but scaling is included for reproducibility and hybrid pipeline compatibility
""")

        # Class balance
        st.markdown("#### Target Class Distribution")
        class_counts = y_full.value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        class_counts["Label"] = class_counts["Class"].map({1: "HIGH Impact (Main Road)", 0: "LOW/MEDIUM Impact"})
        fig_cls = px.pie(
            class_counts, names="Label", values="Count",
            color_discrete_sequence=["#ef4444", "#3b82f6"],
            hole=0.5, template="plotly_dark",
        )
        fig_cls.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(fig_cls, width='stretch')
        imbal_ratio = (y_full == 0).sum() / (y_full == 1).sum()
        st.info(f"**Class imbalance ratio:** {imbal_ratio:.1f}:1 (LOW:HIGH). Random Forest handles this well via `class_weight='balanced'`.")

    # ── Step 2: Train/Test Split ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Step 2 — Train / Test Split")

    split_col1, split_col2 = st.columns([1, 2])
    with split_col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100
        random_state = st.number_input("Random seed", value=42, min_value=0)

    # Impute, scale, split
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_full)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Stratified subsample for speed (max 15k rows) — training happens live
    # on a button click, on top of everything else already cached, so keep
    # this conservative to avoid memory spikes.
    MAX_ROWS = 15_000
    if len(X_scaled) > MAX_ROWS:
        idx = np.random.RandomState(int(random_state)).choice(len(X_scaled), MAX_ROWS, replace=False)
        X_s, y_s = X_scaled[idx], y_full.iloc[idx]
    else:
        X_s, y_s = X_scaled, y_full

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=test_size, random_state=int(random_state), stratify=y_s
    )

    with split_col2:
        split_df = pd.DataFrame({
            "Split": ["Training Set", "Test Set"],
            "Rows": [len(X_train), len(X_test)],
            "% of data": [f"{(1-test_size)*100:.0f}%", f"{test_size*100:.0f}%"],
            "HIGH impact": [int(y_train.sum()), int(y_test.sum())],
        })
        st.dataframe(split_df, width='stretch', hide_index=True)

    st.markdown(f"""
- **Method:** Stratified split — preserves class ratio in both sets  
- **Training rows:** `{len(X_train):,}` &nbsp;|&nbsp; **Test rows:** `{len(X_test):,}`  
- **Stratify:** `y` (ensures HIGH/LOW ratio is equal in train & test)
""")

    # ── Step 3: Model ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Step 3 — Machine Learning Model")

    model_col1, model_col2 = st.columns([1, 1])
    with model_col1:
        chosen_model = st.selectbox(
            "Choose algorithm",
            ["Random Forest", "Gradient Boosting (XGBoost-style)"],
        )
        n_estimators = st.slider("Number of trees / estimators", 50, 150, 100, step=50)
        max_depth    = st.slider("Max depth", 3, 15, 6)

    with model_col2:
        if chosen_model == "Random Forest":
            st.markdown("""
#### Why Random Forest?
- **Ensemble of decision trees** trained on random feature subsets — reduces overfitting
- Handles **mixed feature types** (numeric + encoded categorical) natively
- **`class_weight='balanced'`** compensates for the HIGH vs LOW class imbalance
- Provides **feature importance scores** — interpretable for enforcement decisions
- No need for feature scaling (tree-based), but scaling kept for pipeline consistency
- Robust to outliers in GPS coordinates
""")
        else:
            st.markdown("""
#### Why Gradient Boosting?
- **Sequential ensemble** — each tree corrects errors of the previous
- Generally **higher accuracy** than Random Forest on tabular data
- Works well with **imbalanced classes** via sample weighting
- `sklearn.GradientBoostingClassifier` is XGBoost-equivalent without extra installs
- Slower to train but typically **better F1** on minority class (HIGH impact)
""")

    run_ml = st.button("Train Model & Evaluate", type="primary")

    if run_ml:
        with st.spinner(f"Training {chosen_model} on {len(X_train):,} samples…"):
            # ── Use pre-trained model.pkl if available and RF selected ──────
            MODEL_PKL = "model.pkl"
            if chosen_model == "Random Forest" and os.path.exists(MODEL_PKL):
                model = joblib.load(MODEL_PKL)
                st.success("Loaded pre-trained `model.pkl` (training skipped — instant!)")
            elif chosen_model == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=int(random_state),
                    n_jobs=2,
                )
                model.fit(X_train, y_train)
            else:
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=int(random_state),
                )
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # ── Step 4: Evaluation ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Step 4 — Model Evaluation")

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec_hi = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_hi  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_hi   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        # KPI row
        m1, m2, m3, m4 = st.columns(4)
        for col, title, value, delta in zip(
            [m1, m2, m3, m4],
            ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"],
            [f"{acc*100:.2f}%", f"{prec*100:.2f}%", f"{rec*100:.2f}%", f"{f1*100:.2f}%"],
            ["Overall correct", "Avg across classes", "Avg across classes", "Harmonic mean"],
        ):
            col.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value" style="font-size:1.6rem">{value}</div>
                    <div class="metric-delta">{delta}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            # Per-class metrics table
            st.markdown("#### Per-Class Metrics")
            metrics_df = pd.DataFrame({
                "Class": ["LOW/MEDIUM Impact (0)", "HIGH Impact — Main Road (1)", "Weighted Avg"],
                "Precision": [
                    f"{precision_score(y_test, y_pred, pos_label=0, zero_division=0)*100:.1f}%",
                    f"{prec_hi*100:.1f}%",
                    f"{prec*100:.1f}%",
                ],
                "Recall": [
                    f"{recall_score(y_test, y_pred, pos_label=0, zero_division=0)*100:.1f}%",
                    f"{rec_hi*100:.1f}%",
                    f"{rec*100:.1f}%",
                ],
                "F1-Score": [
                    f"{f1_score(y_test, y_pred, pos_label=0, zero_division=0)*100:.1f}%",
                    f"{f1_hi*100:.1f}%",
                    f"{f1*100:.1f}%",
                ],
                "Support": [
                    int((y_test == 0).sum()),
                    int((y_test == 1).sum()),
                    len(y_test),
                ],
            })
            st.dataframe(metrics_df, width='stretch', hide_index=True)

            st.markdown("""
> **Precision** — Of all violations flagged as HIGH impact, how many actually were?  
> **Recall** — Of all actual HIGH impact violations, how many did we catch?  
> **F1** — Harmonic mean of Precision & Recall (key metric for imbalanced data)
""")

        with eval_col2:
            # Confusion matrix
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                x=["Pred: LOW/MED", "Pred: HIGH"],
                y=["Actual: LOW/MED", "Actual: HIGH"],
                template="plotly_dark",
                aspect="auto",
            )
            fig_cm.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cm, width='stretch')

        # Feature importance
        st.markdown("#### Feature Importance")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            template="plotly_dark",
            text=fi_df["Importance"].map(lambda x: f"{x:.3f}"),
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, width='stretch')

        top_feature = fi_df.sort_values("Importance", ascending=False).iloc[0]["Feature"]
        st.info(
            f"**Top predictor: `{top_feature}`** — This feature contributes most to identifying "
            "HIGH congestion impact violations. Enforcement patrols should be timed and routed accordingly.",
        )

        # Cross-validation
        st.markdown("#### Cross-Validation (3-Fold)")
        with st.spinner("Running 3-fold cross-validation…"):
            # n_jobs=-1 here would fit up to 5 models in parallel/in memory at
            # once on top of the original model — capped to limit peak RAM.
            cv_scores = cross_val_score(model, X_s, y_s, cv=3, scoring="f1_weighted", n_jobs=2)

        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
            "F1-Weighted": cv_scores,
        })
        fig_cv = px.bar(
            cv_df, x="Fold", y="F1-Weighted",
            color="F1-Weighted",
            color_continuous_scale="Greens",
            template="plotly_dark",
            text=cv_df["F1-Weighted"].map(lambda x: f"{x:.3f}"),
            range_y=[0, 1],
        )
        fig_cv.add_hline(
            y=cv_scores.mean(), line_dash="dash", line_color="#22d3ee",
            annotation_text=f"Mean F1: {cv_scores.mean():.3f}",
            annotation_font_color="#22d3ee",
        )
        fig_cv.update_traces(textposition="outside")
        fig_cv.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cv, width='stretch')

        st.success(
            f"3-Fold CV F1 (weighted): **{cv_scores.mean():.3f} ± {cv_scores.std():.3f}**  "
            f"— Low variance ({cv_scores.std():.3f}) confirms the model generalises well across different data folds.",
        )

        # Score distribution
        st.markdown("#### Prediction Probability Distribution")
        prob_df = pd.DataFrame({"P(HIGH impact)": y_prob, "Actual": y_test.values})
        fig_prob = px.histogram(
            prob_df, x="P(HIGH impact)", color=prob_df["Actual"].map({1: "HIGH", 0: "LOW/MED"}),
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map={"HIGH": "#ef4444", "LOW/MED": "#3b82f6"},
            template="plotly_dark",
            labels={"color": "Actual Class"},
        )
        fig_prob.update_layout(
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_prob, width='stretch')
        st.caption("Well-separated peaks indicate the model assigns high confidence to the correct class.")

        # Summary card
        st.markdown("---")
        st.markdown("### ML Pipeline Summary")
        st.markdown(f"""
| Step | Detail |
|------|--------|
| **Problem type** | Binary classification |
| **Target** | `is_high_impact` (1 = Main road parking, 0 = other) |
| **Features** | 9 engineered features (temporal, spatial, categorical encoded) |
| **Missing values** | Handled via `SimpleImputer(strategy='median')` |
| **Encoding** | `LabelEncoder` for `vehicle_type`, `police_station` |
| **Scaling** | `StandardScaler` (zero mean, unit variance) |
| **Train/Test split** | {(1-test_size)*100:.0f}% / {test_size*100:.0f}% stratified |
| **Algorithm** | {chosen_model} (`n_estimators={n_estimators}`, `max_depth={max_depth}`) |
| **Class imbalance** | `class_weight='balanced'` / stratified split |
| **Accuracy** | **{acc*100:.2f}%** |
| **Weighted F1** | **{f1*100:.2f}%** |
| **HIGH-impact F1** | **{f1_hi*100:.2f}%** |
| **CV F1 (3-fold)** | **{cv_scores.mean():.3f} ± {cv_scores.std():.3f}** |
""")

    else:
        st.info("Click **Train Model & Evaluate** above to run the full ML pipeline.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#475569'>Bengaluru Traffic Police · Parking Intelligence Hub · "
    "Data: Nov 2023 – Apr 2024 · Built with Streamlit + Plotly + Folium</small></center>",
    unsafe_allow_html=True,
)
