# AI-Driven Parking Intelligence Hub — Bengaluru

> HackerEarth / Flipkart Gridlock Hackathon submission — Problem: Poor visibility on parking-induced congestion

Detects illegal parking hotspots, quantifies their traffic impact, and predicts high-congestion violations using machine learning, to support data-driven enforcement for Bengaluru Traffic Police.

---

## Project Structure

This repo contains only the files needed to run the dashboard — there is no `data/` folder; the dataset is streamed live from a hosted CSV URL at runtime (see [Dataset](#dataset) below).

```
blr-parking-hotspot-ai/
├── app.py              ← Streamlit dashboard (all tabs, ML pipeline, maps)
├── model.pkl            ← Pre-trained Random Forest model (loaded instantly in the ML Pipeline tab)
├── notebook.ipynb        ← Exploratory analysis / model development notebook
├── requirements.txt       ← Python dependencies
└── README.md             ← This file
```

---

## Quick Start (local)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd blr-parking-hotspot-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| Hotspot Map | Interactive Folium map — Heatmap, Cluster Markers, or Risk Grid layers |
| Analytics | Violation types, vehicle distribution, monthly trends |
| Enforcement Zones | Priority zones ranked by weighted congestion score |
| Temporal Patterns | Hourly, day-of-week, and resolution-time analysis |
| AI Risk Scorer | Interactive zone risk calculator |
| ML Pipeline | Live preprocessing, train/test split, model training, and evaluation walkthrough |

Sidebar filters (month, hour range, violation category, risk tier, map layer, top-N hotspots) apply across all tabs.

---

## Machine Learning Pipeline

### Problem framing
Binary classification — predict whether a parking violation causes **HIGH** congestion impact (main-road blocking) vs **LOW/MEDIUM**.

### Features (9 total)
| Feature | Description |
|---|---|
| `hour` | Hour of violation (UTC) |
| `month` | Month number |
| `dow_num` | Day of week (0 = Monday) |
| `latitude` / `longitude` | GPS coordinates |
| `vehicle_enc` | Vehicle type, label-encoded |
| `station_enc` | Police station, label-encoded |
| `near_junction` | 1 if at a named junction |
| `area_density` | Violations per ~500m grid cell |

### Preprocessing
- Missing values → `SimpleImputer(strategy='median')`
- Categorical encoding → `LabelEncoder` (vehicle type, police station)
- Scaling → `StandardScaler`
- Class imbalance → `class_weight='balanced'`
- Split → 80/20 stratified train/test
- Training is capped at a 15,000-row stratified subsample for performance on Streamlit Cloud's free tier

### Algorithm: Random Forest (default) or Gradient Boosting
The ML Pipeline tab lets you choose the algorithm and tune `n_estimators` / `max_depth` live. If Random Forest is selected, the app loads the pre-trained `model.pkl` instantly instead of retraining from scratch.

### Reference model performance (from `model.pkl` / notebook)

| Metric | Score |
|---|---|
| Accuracy | 87.88% |
| Precision (weighted) | 89.67% |
| Recall (weighted) | 87.88% |
| F1-score (weighted) | 89.67% |
| F1 — HIGH impact class | 50.29% |
| AUC-ROC | 0.9159 |
| CV F1 (3-fold mean) | ~0.89 |

---

## Congestion Impact Score Logic

| Score | Condition | Interpretation |
|---|---|---|
| 3 — HIGH | Parking in main road | Blocks carriageway, major congestion |
| 2 — MEDIUM | Footpath / double parking / near bus stop or school | Partial obstruction |
| 1 — LOW | Wrong parking / no parking | Side-street, low impact |
| 0 — NONE | Non-parking violation | — |

**Hotspot weighted score** = `violation_count × avg_impact_score`

**Risk tiers:**
- HIGH → weighted score > 200 → immediate enforcement unit
- MEDIUM → 50–200 → twice-daily patrol
- LOW → < 50 → weekly sweep

---

## Dataset

- **Source:** Bengaluru Traffic Police, via HackerEarth / Flipkart Gridlock Hackathon
- **Records:** ~298,450 violations
- **Period:** November 2023 – April 2024
- **Key columns used:** `id`, `latitude`, `longitude`, `violation_type`, `vehicle_type`, `police_station`, `junction_name`, `created_datetime`, `closed_datetime`, `validation_status`

The app loads this CSV directly from its hosted URL at runtime via `pd.read_csv`, cached with `@st.cache_data`, rather than bundling it in the repo.

---

## Requirements

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
folium>=0.15.0
streamlit-folium>=0.20.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

**Note:** these are unpinned minimum versions. Streamlit Cloud will install the latest compatible release of each (e.g. pandas 3.x, numpy 2.x) at deploy time, which may behave differently from whatever version you tested locally. If you hit deployment issues that don't reproduce locally, pin exact versions (e.g. `pandas==2.2.2`) to match your local environment.

---

## Streamlit Cloud Deployment

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud auto-installs `requirements.txt`.

### Memory notes (free tier = 1 GB RAM)
This app does real-time CSV loading, Folium map rendering, and **on-demand model training** in the ML Pipeline tab — all of which compete for the same 1 GB ceiling. If you see a "resource limits" / out-of-memory error:
- Use the **Random Forest** option in the ML Pipeline tab where possible — it loads the pre-trained `model.pkl` instead of training live.
- Avoid running **Gradient Boosting** with a high `n_estimators` repeatedly in the same session.
- Use **Heatmap** mode rather than **Cluster Markers** for large filtered views — heatmap points are far cheaper to render.
- If the app crashes immediately on load (before any interaction), check the full Streamlit Cloud logs (Manage app → logs) for a Python traceback or "Killed" message — this points to the CSV load step itself rather than the interactive tabs.

---

*Built with Streamlit, Plotly, Folium, and scikit-learn.*
