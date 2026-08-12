<div align="center">

#  Diabetes Patient Readmission Risk Analyser

**A full data-science pipeline — from raw EHR CSVs to a Random Forest risk model — wrapped in a Tkinter desktop GUI.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)


</div>

---

##  Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Project Structure](#-project-structure)
4. [Architecture](#-architecture)
5. [Installation & Setup](#-installation--setup)
6. [How to Run](#-how-to-run)
7. [Dataset](#-dataset)
8. [How to Use the Application](#-how-to-use-the-application)
9. [Visualisations](#-visualisations)
10. [Machine Learning Model](#-machine-learning-model)
11. [Expert Module (Regex)](#-expert-module-regex)
12. [Libraries Used](#-libraries-used)
13. [File Outputs](#-file-outputs)
14. [Team](#-team)
15. [License](#-license)

---

##  Project Overview

This application analyses electronic health records from **130 US hospitals (1999–2008)** to identify the factors most strongly correlated with **early patient readmission** (within 30 days of discharge).

The project covers the full data science pipeline end to end:

-  Raw CSV loading and smart data cleaning
-  Regex-based ICD-9 medical code classification (Expert Module)
-  Statistical analysis and feature engineering
-  Six interactive Matplotlib/Seaborn visualisations (embedded in GUI)
-  Random Forest machine learning model with cross-validation
-  A fully featured Tkinter desktop GUI

---

##  Features

| Feature | Description |
|---|---|
| **GUI Application** | Full Tkinter desktop app with sidebar, tabbed layout, KPI cards, and embedded charts |
| **Smart Data Cleaning** | Drops high-null columns, imputes with median/mode — no blind data loss |
| **Expert Module (Regex)** | ICD-9 diagnosis codes classified into clinical categories using `re.match` |
| **6 Visualisations** | Pie, bar, scatter, boxplot, bar chart, and correlation heatmap — all embedded in-app |
| **ML Prediction** | Random Forest with 5-fold cross-validation and confusion matrix |
| **CSV Export** | Cleaned data saved as `cleaned_diabetes_data.csv` for a reproducible pipeline |
| **Background Threading** | All heavy operations run in daemon threads — GUI stays responsive |

---

##  Project Structure

```
project/
│
├── Python_Proj_Final.py           ← Main application (all code)
├── README.md                      ← This file
├── Technical_Documentation.docx   ← Full code & algorithm documentation
│
├── (generated on run)
├── cleaned_diabetes_data.csv      ← Exported clean dataset
├── chart1_readmission_pie.png
├── chart2_age_hospital_bar.png
├── chart3_lab_meds_scatter.png
├── chart4_gender_boxplot.png
├── chart5_top_diagnoses_bar.png
├── chart6_correlation_heatmap.png
└── chart_ml_confusion_matrix.png
```

---

##  Architecture

The application follows the **Single Responsibility Principle** — each class does exactly one job.

```
┌─────────────────────────────────────────────────────────────┐
│                           AppGUI                             │
│    Tkinter interface: sidebar, tabs, KPI cards, log panel    │
└────────────┬────────────┬───────────────┬────────────────────┘
             │            │               │
      ┌──────▼──────┐ ┌───▼────────┐ ┌───▼──────────┐
      │ DataLoader  │ │DataCleaner │ │  MLPredictor │
      │             │ │            │ │              │
      │ Reads CSV   │ │ Cleans,    │ │ RandomForest │
      │ from disk   │ │ encodes,   │ │ + 5-fold CV  │
      │             │ │ regex ICD9 │ │              │
      └─────────────┘ └─────┬──────┘ └──────────────┘
                             │
                      ┌──────▼──────┐
                      │  Visualizer │
                      │             │
                      │  6 Charts   │
                      │  (embedded) │
                      └─────────────┘
```

**Class summary:**

| Class | Responsibility |
|---|---|
| `DataLoader` | Reads raw CSV, replaces `?` with `NaN` |
| `DataCleaner` | Cleans data, applies regex ICD-9 classifier, encodes features |
| `Visualizer` | Generates 6 Matplotlib/Seaborn figures |
| `MLPredictor` | Trains Random Forest, runs cross-validation, evaluates model |
| `AppGUI` | Full Tkinter GUI — layout, buttons, callbacks, chart embedding |

---

##  Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

> **Note:** `tkinter` ships with standard Python on Windows and macOS.
> On Linux (Ubuntu/Debian): `sudo apt-get install python3-tk`

### Verify installation

```bash
python -c "import numpy, pandas, matplotlib, seaborn, sklearn, tkinter; print('All OK')"
```

---

##  How to Run

```bash
python Python_Proj_Final.py
```

The GUI window opens automatically.

---

##  Dataset

| Property | Value |
|---|---|
| **Name** | UCI Diabetes 130-US Hospitals for Years 1999–2008 |
| **Source** | [archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| **File** | `diabetic_data.csv` (extracted from `dataset.zip`) |
| **Size** | ~100,000 patient records, 50 columns |
| **Target variable** | `readmitted` — values: `<30`, `>30`, `NO` |

### Key columns used

| Column | Description |
|---|---|
| `gender` | Patient gender |
| `age` | Age bracket (e.g. `[50-60)`) |
| `time_in_hospital` | Number of days admitted |
| `num_lab_procedures` | Number of lab tests performed |
| `num_medications` | Number of medications administered |
| `number_emergency` | Number of prior emergency visits |
| `number_inpatient` | Number of prior inpatient visits |
| `diag_1` | Primary ICD-9 diagnosis code |
| `readmitted` | Target: `<30`, `>30`, or `NO` |

> ⚠️ `diabetic_data.csv` is **not** bundled in this repo (dataset licensing) — download it from the UCI link above and place it alongside `Python_Proj_Final.py` before loading it in the app.

---

##  How to Use the Application

**1. Load CSV**
- Click `📂 Load CSV File` in the sidebar
- Select `diabetic_data.csv` from your downloads

**2. Process Data**
- Click `⚙️ Process Data`
- The pipeline cleans, encodes, and classifies the data
- KPI cards on the Dashboard tab update automatically

**3. Export Cleaned CSV** *(optional but recommended)*
- Click `💾 Export Clean CSV`
- Saves `cleaned_diabetes_data.csv` — a reproducible pipeline artifact

**4. Generate Charts**
- Click `📊 Generate Charts`
- All 6 charts are embedded in the **Charts tab**
- Use the dropdown to switch between charts

**5. Run ML Prediction**
- Click `🤖 Run ML Prediction`
- Trains a Random Forest model
- Shows test accuracy, 5-fold CV score, classification report, and confusion matrix

---

##  Visualisations

| # | Chart | Type | Insight |
|---|---|---|---|
| 1 | Readmission Rate Distribution | Pie chart | Proportion of `<30`, `>30`, `NO` readmissions |
| 2 | Hospital Stay by Age Group | Horizontal bar | Which age groups stay longest |
| 3 | Lab Procedures vs. Medications | Scatter + trend line | Correlation between tests and medications |
| 4 | Hospital Stay by Gender | Boxplot | Distribution differences across genders |
| 5 | Top 5 Diagnosis Categories | Bar chart | Most frequent ICD-9 categories (regex-derived) |
| 6 | Feature Correlation Heatmap | Heatmap | Pearson correlations between all numeric features |

---

##  Machine Learning Model

### Algorithm: Random Forest Classifier

A Random Forest is an **ensemble** of decision trees. Each tree is trained on a random subset of the data and features; the final prediction is decided by majority vote across all trees.

### Configuration

```python
RandomForestClassifier(
    n_estimators     = 150,   # 150 decision trees
    max_depth        = 8,     # max depth per tree (prevents overfitting)
    min_samples_leaf = 10,    # min 10 samples per leaf
    random_state     = 42,    # reproducibility seed
    n_jobs           = -1,    # use all CPU cores
)
```

### Evaluation

- **80/20 stratified train/test split** — `stratify=y` preserves class proportions
- **5-fold cross-validation** — splits data 5 ways; more reliable than a single split
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

### Features used

`time_in_hospital`, `num_lab_procedures`, `num_medications`, `number_emergency`, `number_inpatient`, `age_num`, `gender_enc`


##  Expert Module (Regex)

The `_classify_diagnosis()` static method in `DataCleaner` maps raw ICD-9 codes to clinical categories using `re.match()`.

```python
patterns = [
    (r"^250",                          "Diabetes"),         # codes 250.xx
    (r"^(390|39[1-9]|4[0-5]\d)",       "Circulatory"),      # 390–459
    (r"^(460|4[6-9]\d|5[0-1]\d)",      "Respiratory"),      # 460–519
    (r"^(520|5[2-9]\d|6[0-2]\d)",      "Digestive"),        # 520–629
    (r"^(580|5[89]\d|63\d|64[0-9])",   "Genitourinary"),    # 580–649
    (r"^(14[0-9]|1[5-9]\d|2[0-3]\d)",  "Neoplasms"),        # 140–239
    (r"^(800|8[0-9]\d|9[0-4]\d)",      "Injury"),           # 800–949
    (r"^[EV]",                          "External/Suppl."), # E/V codes
    (r"^\d{3}",                         "Other Numeric"),   # catch-all
]
```

`re.sub(r"\s+", "", code)` is also used to strip whitespace from raw codes before matching.

##  Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.21 | Array operations, `mean`, `std`, `polyfit` for trend lines |
| `pandas` | ≥1.3 | DataFrame operations, CSV I/O, `groupby`, `value_counts` |
| `matplotlib` | ≥3.4 | All chart rendering; `FigureCanvasTkAgg` for GUI embedding |
| `seaborn` | ≥0.11 | Boxplot, heatmap, themed styling |
| `scikit-learn` | ≥0.24 | `RandomForestClassifier`, `train_test_split`, `cross_val_score`, `LabelEncoder`, metrics |
| `tkinter` | stdlib | GUI: windows, buttons, labels, tabs, scrolled text |
| `re` | stdlib | Regex ICD-9 classifier (Expert Module) |
| `threading` | stdlib | Background daemon threads to keep GUI responsive |
| `datetime` | stdlib | Timestamps in the activity log |
| `os` | stdlib | File path manipulation |



| File | Generated by | Description |
|---|---|---|
| `cleaned_diabetes_data.csv` | Export button | Full cleaned & encoded dataset |
| `chart1_readmission_pie.png` | Generate Charts | Readmission pie chart |
| `chart2_age_hospital_bar.png` | Generate Charts | Age vs hospital stay |
| `chart3_lab_meds_scatter.png` | Generate Charts | Lab vs medications scatter |
| `chart4_gender_boxplot.png` | Generate Charts | Gender boxplot |
| `chart5_top_diagnoses_bar.png` | Generate Charts | Top diagnoses bar chart |
| `chart6_correlation_heatmap.png` | Generate Charts | Correlation heatmap |
| `chart_ml_confusion_matrix.png` | Run ML | Model confusion matrix |


