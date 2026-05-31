# DSLab_Project_Crime_Data_Analytics_Brinda_Soneji_u23cs029
# Chicago Crime Data Analytics

> A comprehensive machine learning and spatial analytics project on 7.8 million crime incidents spanning 2001–2023, uncovering temporal patterns, geographic hotspots, and predictive insights to support data-driven public safety decisions.

---

## Project Overview

This project performs end-to-end analytics on the **Chicago Crime Dataset** sourced from the City of Chicago Data Portal. It covers the full data science pipeline — from raw ingestion and cleaning of 7.84 million records, through exploratory analysis, spatial and temporal modeling, clustering, and finally predictive machine learning with model interpretation via SHAP.

The core objective is to **identify crime trends, spatial hotspots, and predictive patterns** that can inform law enforcement resource allocation and policy decisions.

| Attribute | Value |
|---|---|
| Dataset | Chicago Crime Dataset (2001–2023) |
| Source | [data.cityofchicago.org](https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD) |
| Total Records | 7,842,937 incident records |
| Columns | 22 features per record |
| Memory Footprint | ~5,602 MB |
| Unique Crime Types | 36 (grouped into 5 super-categories) |
| Overall Arrest Rate | 26.0% |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Processing** | Pandas, NumPy, PyArrow |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |
| **Clustering** | DBSCAN (Haversine metric), K-Means |
| **Model Interpretation** | SHAP (TreeExplainer), Partial Dependence Plots |
| **Visualization** | Seaborn, Matplotlib, Plotly |
| **Geospatial Mapping** | Folium (CartoDB Positron basemap) |
| **Data Access** | Sodapy (Chicago Open Data API) |

---

## Key Findings

### Temporal Patterns
- Crime in Chicago has **declined overall from 2001 to 2023**, with a sharp dip in 2020 attributable to COVID-19 mobility restrictions.
- A strong **summer seasonality** is present — June through August consistently see the highest crime volumes across all years.
- **Property crimes** peak during business hours (8 AM–6 PM); **violent crimes** peak late at night on weekends (10 PM–2 AM).
- Arrest probability is **highest at 11 AM (55.1%)** and lowest at midnight (30.6%), reflecting daytime court and booking availability.
- Overall arrest rate has **declined from ~30%+ in early 2000s to below 25%** in recent years.

### Spatial Patterns
- Crime is geographically concentrated in the **South and West sides** of Chicago.
- Violent crime hotspots are **more spatially concentrated** than property crime hotspots, which are dispersed across commercial corridors.
- DBSCAN identified **4 continuous geographic clusters** and 32 anomalous outlier grid cells.
- K-Means clustering revealed **3 distinct zone typologies**: Nighttime Violent Hotspots, Daytime Property Crime Zones, and Night Activity Zones.

### Crime Type Insights
- **Theft, Battery, and Criminal Damage** are the three most frequent crime types.
- High-frequency crimes (Theft, Criminal Damage) have very **low arrest rates (~6–7%)**.
- Proactive enforcement crimes (Narcotics, Prostitution) have **near-100% arrest rates** — they only enter the record when police make an arrest.
- Domestic Violence incidents showed a **100% arrest rate** in the dataset.

### Predictive Modeling Results

| Task | Best Model | Key Metric |
|---|---|---|
| Crime Category Classification | XGBoost | Accuracy: 56.1%, Weighted F1: 0.503 |
| Arrest Prediction (Binary) | Random Forest | Accuracy: 84.0%, ROC-AUC: 0.830 |
| Daily Crime Count Regression | XGBoost / RF / LightGBM | MAE: 0.29, MAPE: 20.6% |

---

## Features

- **Large-scale data ingestion** with Parquet caching and Sodapy-based streaming API fallback
- **Systematic missing value audit** and spatial bounding-box filtering (Chicago: lat 41.6–42.1, lon -87.9 to -87.5)
- **Deduplication** by Case Number (549 duplicates removed)
- **Interactive visualizations** using Plotly and Folium (heatmaps, dual-panel charts, district comparisons)
- **Spatial crime heatmaps** for city-wide, violent-only, and property-only incidents
- **Clustering dashboards** with Folium map overlays of K-Means zone labels
- **SHAP beeswarm plots** for model explainability on both classification models
- **Partial Dependence Plot** for arrest probability vs. hour of day
- **Time-series prediction plots** showing actual vs. predicted daily crime counts per district

---

## Methodology

### 1. Data Preparation
- Loaded 7.84M records from Chicago Data Portal; applied PyArrow Parquet caching for repeat runs
- Handled missingness: spatial subset (`df_spatial`, 7.73M rows) for geographic analysis; full temporal set (`df_temporal`, 7.84M rows) for trend analysis
- Filled `Location Description` nulls with `'Unknown'`; removed 549 duplicate Case Numbers

### 2. Feature Engineering

| Feature Group | Features Created |
|---|---|
| Temporal | `hour`, `dow`, `month`, `week`, `is_weekend`, `season`, `time_bucket` |
| Cyclic Encoding | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` |
| Crime Super-Labels | 5 categories: Violent, Property, Drugs, Quality of Life, Other |
| Location Bucketing | 5 buckets: Residential, Commercial, Transport, Outdoor, Other |
| Spatial Grid | `grid_lat`, `grid_lon`, `grid_id` (2 decimal places → 699 cells) |
| Target Encoding | Community Area, District, loc_bucket encoded by mean target (train set only, Gaussian noise std=0.01) |

### 3. Exploratory Data Analysis
- Top crime types and arrest rates (dual-panel chart)
- Top 20 crime locations by incident count
- Domestic vs. non-domestic volume and arrest rate comparison
- District-level interactive Plotly dual-bar chart
- Pearson correlation heatmap (8 numeric features)
- Arrest rate heatmap (Hour × Day-of-Week)
- Seasonal crime volume stacked bar chart

### 4. Temporal Analysis
- Year-over-year trend (2001–2023) with super-category breakdown
- Month × Year heatmap (12 × 23 grid, YlOrRd palette)
- Three-panel Hour × Day heatmaps (All / Violent / Property)
- Dual-axis arrest rate trend over years

### 5. Spatial Analysis
- City-wide Folium heatmap (80,000-point sample)
- Separate violent vs. property heatmaps (50,000-point samples each)
- Community Area-level 3-panel analysis (count, arrest rate, violent share)

### 6. Clustering
- **DBSCAN** on geographic coordinates (Haversine, eps=1.5 km, min_samples=8): 4 clusters, 32 noise cells; separate day/night runs
- **K-Means** on 7-feature normalized grid vectors (k=3 selected via silhouette score); clusters named via 60th-percentile thresholds

### 7. Predictive Modeling

**Model A — Crime Category Classification (5-class)**
- Features: 11 temporal + geographic features
- 80/20 stratified split; SMOTE applied (k_neighbors=3) → 13.23M training rows
- XGBoost (n_estimators=300, max_depth=7, lr=0.08) vs. Random Forest (200 trees, max_depth=20, 500K-row subsample)

**Model B — Arrest Prediction (Binary)**
- Features: 12 (same as A + crime_category)
- LightGBM (histogram-based, early stopping, patience=30) vs. Random Forest (500K subsample)
- ROC curves plotted for both models

**Model C — Daily Crime Count Regression**
- Target: Daily incident count per District
- Lag features: lag_1, lag_7, lag_14, lag_28; rolling features: rolling_7, rolling_28 (shift=1 to prevent leakage)
- XGBoost, Random Forest, LightGBM — all converged to identical metrics (MAE=0.29, RMSE=0.51, MAPE=20.6%)

### 8. Model Interpretation
- XGBoost feature importance bar chart (Domestic flag dominates: 0.829 score)
- SHAP TreeExplainer on 5,000-row test sample (crime category)
- SHAP beeswarm on 3,000-row test sample (arrest prediction)
- Partial Dependence Plot: arrest probability vs. hour (grid of 24 synthetic inputs at median feature values)

---

## Data Provenance and Description

### Source
- **Dataset Name:** Chicago Crime Data (2001–Present)
- **Provider:** City of Chicago — Department of Police
- **Portal:** [data.cityofchicago.org](https://data.cityofchicago.org)
- **Access Method:** Sodapy API (Chicago Open Data Portal) with local Parquet cache
- **License:** Public domain (City of Chicago Open Data License)

### Coverage
- **Temporal Range:** January 2001 – December 2023 (23 years)
- **Geographic Scope:** City of Chicago, Illinois, USA
- **Record Count:** 7,842,937 incidents (7,842,388 after deduplication)

### Key Columns

| Column | Type | Description |
|---|---|---|
| `ID` | Integer | Unique record identifier |
| `Case Number` | String | CPD case number (dedup key) |
| `Date` | Datetime | Full timestamp of incident |
| `Block` | String | Street-level address |
| `Primary Type` | String | Crime category (36 unique values) |
| `Description` | String | Granular crime sub-type |
| `Location Description` | String | Physical environment (street, residence, etc.) |
| `Arrest` | Boolean | Whether an arrest was made |
| `Domestic` | Boolean | Whether incident was domestic |
| `District` | Integer | CPD police district |
| `Ward` | Integer | City council ward (7.84% missing) |
| `Community Area` | Integer | Official Chicago community area (7.82% missing) |
| `Latitude` / `Longitude` | Float | WGS84 coordinates (1.12% missing) |
| `X Coordinate` / `Y Coordinate` | Float | Illinois State Plane coordinates |

### Known Limitations
- **Unreported crimes:** The dataset only captures reported incidents — true crime rates are likely higher
- **Arrest rate bias:** High arrest rates for proactive crimes (Narcotics, Prostitution) reflect policing patterns, not underlying prevalence
- **Geographic missingness:** ~1.1% of records lack coordinates; ~7.8% lack Ward/Community Area
- **Recording changes:** Shifts in arrest rate over time may reflect policy or reporting practice changes rather than actual crime dynamics

---

*Built with Python · Data via City of Chicago Open Data Portal · ML with XGBoost, LightGBM, Scikit-learn · Maps with Folium*
