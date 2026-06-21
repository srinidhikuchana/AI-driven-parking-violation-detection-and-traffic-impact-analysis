# 🚦 AI-Driven Parking Intelligence Hub — Bengaluru

> **HackerEarth Submission** | Problem: Poor Visibility on Parking-Induced Congestion

Detect illegal parking hotspots, quantify their traffic impact, and predict high-congestion violations using Machine Learning — enabling data-driven enforcement for Bengaluru Traffic Police.

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── violations.csv          ← Raw dataset (298K rows, Jan–Apr 2024)
│   ├── imputer.pkl             ← Fitted SimpleImputer
│   ├── scaler.pkl              ← Fitted StandardScaler
│   ├── le_vehicle.pkl          ← LabelEncoder for vehicle_type
│   ├── le_station.pkl          ← LabelEncoder for police_station
│   ├── model_metadata.json     ← Model metrics & config
│   ├── eda_charts.png          ← EDA visualizations
│   ├── correlation_heatmap.png ← Feature correlation matrix
│   ├── model_evaluation.png    ← Confusion matrix + ROC curve
│   └── cross_validation.png    ← 5-fold CV scores
│
├── notebook.ipynb              ← Full ML pipeline (executed, with outputs)
├── model.pkl                   ← Trained Random Forest model
├── app.py                      ← Streamlit dashboard
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🗺️ Hotspot Map | Interactive Folium map — Heatmap / Cluster / Risk Grid layers |
| 📊 Analytics | Violation types, vehicle distribution, monthly trends |
| 🔥 Enforcement Zones | Priority zones ranked by weighted congestion score |
| ⏱️ Temporal Patterns | Hourly, day-of-week, resolution time analysis |
| 🤖 AI Risk Scorer | Interactive zone risk calculator with gauge chart |
| 🧪 ML Pipeline | Full preprocessing → training → evaluation walkthrough |

---

## 🤖 Machine Learning Pipeline

### Problem Framing
**Binary Classification** — Predict whether a parking violation causes HIGH congestion impact (main road blocking) vs LOW/MEDIUM.

### Features (9 total)
| Feature | Description |
|---------|-------------|
| `hour` | Hour of violation (UTC) |
| `month` | Month number |
| `dow_num` | Day of week (0=Mon) |
| `latitude` / `longitude` | GPS coordinates |
| `vehicle_enc` | Vehicle type (Label Encoded) |
| `station_enc` | Police station (Label Encoded) |
| `near_junction` | 1 if at a named junction/metro |
| `area_density` | Violations per ~500m grid cell |

### Preprocessing
- **Missing values** → `SimpleImputer(strategy='median')`
- **Categorical encoding** → `LabelEncoder` (vehicle type, police station)
- **Scaling** → `StandardScaler` (zero mean, unit variance)
- **Class imbalance** → `class_weight='balanced'`
- **Split** → 80/20 stratified train/test

### Algorithm: Random Forest Classifier
**Why Random Forest?**
- Handles mixed numeric + encoded categorical features natively
- `class_weight='balanced'` compensates for ~5.5:1 LOW:HIGH imbalance
- Feature importance scores are interpretable for enforcement decisions
- Robust to GPS coordinate outliers
- No severe hyperparameter sensitivity compared to SVM/Neural Nets at this data scale

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **87.88%** |
| **Precision (weighted)** | **89.67%** |
| **Recall (weighted)** | **87.88%** |
| **F1-Score (weighted)** | **89.67%** |
| **F1 — HIGH Impact class** | **50.29%** |
| **AUC-ROC** | **0.9159** |
| **CV F1 (5-fold mean)** | **0.8935 ± 0.0024** |

> AUC-ROC of **0.916** means the model has excellent discrimination ability between HIGH and LOW impact violations. Low CV std dev (0.0024) confirms strong generalisation.

---

## 🏗️ Congestion Impact Score Logic

| Score | Condition | Interpretation |
|-------|-----------|----------------|
| 3 — HIGH | Parking in Main Road | Blocks carriageway, major congestion |
| 2 — MEDIUM | Footpath / Double / Near Bustop | Partial obstruction |
| 1 — LOW | Wrong Parking / No Parking | Side-street, low impact |
| 0 — NONE | Non-parking violation | — |

**Hotspot Weighted Score** = `violation_count × avg_impact_score`

**Risk Tiers:**
- 🔴 HIGH → weighted score > 200 → Immediate enforcement unit
- 🟡 MEDIUM → 50–200 → Twice-daily patrol
- 🟢 LOW → < 50 → Weekly sweep

---

## 🗃️ Dataset

- **Source:** Bengaluru Traffic Police via HackerEarth
- **Records:** 298,450 violations
- **Period:** November 2023 – April 2024
- **Key columns:** `latitude`, `longitude`, `violation_type`, `vehicle_type`, `police_station`, `junction_name`, `created_datetime`, `validation_status`

---

## 📦 Requirements

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

---

## ☁️ Streamlit Cloud Deployment

1. Push this repo to GitHub (include `data/violations.csv` or use Git LFS for large files)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo
3. Set **Main file path** to `app.py`
4. Click **Deploy** — Streamlit Cloud auto-installs `requirements.txt`

---

*Built with Streamlit · Plotly · Folium · scikit-learn*
