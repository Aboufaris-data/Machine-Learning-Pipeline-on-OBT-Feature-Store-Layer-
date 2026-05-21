# 🏡 Moroccan Real Estate — ML Regression Pipeline
### Feature Store Layer on OBT (One Big Table)

> An end-to-end machine learning pipeline that extracts real estate listing data from a PostgreSQL data warehouse, engineers a rich feature store, and trains a **Random Forest Regressor** to predict property prices in Morocco.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model & Metrics](#model--metrics)
- [Logs & Artifacts](#logs--artifacts)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements a **modular ML pipeline** built on top of a One Big Table (OBT) architecture. The OBT (`ml_schema.obt_annonce`) serves as the central feature store, consolidating all real estate listing attributes into a single denormalized table — enabling fast, reproducible model training without complex joins.

The pipeline covers the full ML lifecycle:

```
PostgreSQL OBT  →  Extraction  →  Preprocessing  →  Feature Engineering  →  Model Training  →  Saved Artifacts
```

---

## Project Structure

```
├── main.py                    # Pipeline orchestrator (entry point)
├── requirements.txt           # Python dependencies
├── .env                       # DB credentials & config (not committed)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── extraction.py          # DB extraction + CSV fallback loader
│   ├── preprocessing.py       # Cleaning, encoding, train/test split
│   ├── feature_eng.py         # Feature store construction (log, ratios, interactions)
│   ├── train.py               # Scaling, RandomForest training, metrics & artifact saving
│   └── logger.py              # Dual-output logger (console + daily log file)
│
├── data/
│   ├── raw/
│   │   └── obt_raw.csv        # Snapshot extracted from the OBT
│   └── processed/
│       ├── X_train_raw.csv
│       ├── X_test_raw.csv
│       ├── X_train_engineered.csv
│       ├── X_test_engineered.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── regression_model.pkl   # Trained RandomForestRegressor
│   ├── feature_scaler.pkl     # Fitted StandardScaler
│   └── regression_metrics.json
│
└── logs/
    └── YYYY-MM-DD.log         # Daily rotating log file
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│                  STEP 1: EXTRACTION                 │
│  PostgreSQL OBT  ──►  extract_obt()                 │
│  (Fallback: obt_raw.csv if DB is offline)           │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│               STEP 2: PREPROCESSING                 │
│  Drop leakage cols  →  Train/Test Split (80/20)     │
│  Median imputation  →  Label Encoding (ville, quartier) │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│            STEP 3: FEATURE ENGINEERING              │
│  log(surface)  ·  surface_per_room                  │
│  bath_room_ratio  ·  surface_x_rooms                │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│              STEP 4: MODEL TRAINING                 │
│  StandardScaler  →  RandomForestRegressor           │
│  Evaluate  →  Save model, scaler & metrics          │
└─────────────────────────────────────────────────────┘
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- PostgreSQL (optional — pipeline falls back to CSV if DB is unavailable)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root (already included but **not committed** for security):

```env
# Database Credentials
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=avito_db

# Data Warehouse Settings
OBT_TABLE=ml_schema.obt_annonce
```

> ⚠️ **Never commit your `.env` file.** Make sure it is listed in `.gitignore`.

If no database connection is available, place `obt_raw.csv` in the project root or at `data/raw/obt_raw.csv` — the pipeline will automatically detect and use it as a fallback.

---

## Running the Pipeline

```bash
python main.py
```

The orchestrator will execute all four steps sequentially and log progress to both the console and a daily log file under `logs/`.

---

## Feature Engineering

The feature store layer (`src/feature_eng.py`) constructs the following derived features on top of the raw OBT columns:

| Feature | Formula | Rationale |
|---|---|---|
| `log_surface` | `log1p(surface)` | Normalizes right-skewed surface distribution |
| `surface_per_room` | `surface / (rooms + 1)` | Density proxy per room |
| `bath_room_ratio` | `baths / (rooms + 1)` | Amenity quality signal |
| `surface_x_rooms` | `surface × rooms` | Interaction term for total living capacity |

All transformations are applied to both training and test sets **after** the train/test split to prevent data leakage.

---

## Model & Metrics

| Parameter | Value |
|---|---|
| Algorithm | `RandomForestRegressor` |
| Estimators | 100 trees |
| Random State | 42 |
| Scaling | `StandardScaler` |
| Target | `price` (in DH) |
| Train/Test Split | 80% / 20% |

Saved artifacts after training:

```
models/
├── regression_model.pkl      # Serialized trained model (joblib)
├── feature_scaler.pkl        # Fitted scaler for inference
└── regression_metrics.json   # Evaluation results
```

---

## Logs & Artifacts

All pipeline runs produce a dated log file:

```
logs/2026-05-21.log
```

Log entries include timestamps, severity levels, row counts, feature shapes, and any fallback warnings. Both `INFO` (console) and `DEBUG` (file) levels are captured.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical transformations |
| `scikit-learn` | Preprocessing, model training & evaluation |
| `SQLAlchemy` | PostgreSQL ORM / connection engine |
| `psycopg2-binary` | PostgreSQL adapter |
| `python-dotenv` | Environment variable management |
| `joblib` | Model serialization |

---

## License

This project is for educational and portfolio purposes. Feel free to fork and adapt.