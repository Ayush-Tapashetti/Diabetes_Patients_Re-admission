import re
import os
import datetime
import warnings
import threading
import traceback

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")                   
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines              import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix)

#GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

warnings.filterwarnings("ignore")

#  SHARED COLOUR PALETTE  (one source of truth for all classes)
PALETTE = {
    "primary"   : "#1A73E8",
    "secondary" : "#E84040",
    "accent"    : "#34A853",
    "warning"   : "#FBBC05",
    "purple"    : "#7B2FBE",
    "bg"        : "#F8F9FA",
    "text"      : "#202124",
    "muted"     : "#5F6368",
    "chart_seq" : "Blues_d",
}

# Ordinal age brackets used in both DataCleaner and Visualizer
AGE_ORDER = (                              # <-- TUPLE: immutable, ordered
    "[0-10)",  "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
)

# Columns we want from the raw UCI file  (set: unordered, unique membership)
REQUIRED_COLS: set = {
    "time_in_hospital", "num_lab_procedures", "num_medications",
    "diag_1", "readmitted",
}

#  CLASS 1 ── DataLoader

class DataLoader:
    """
    Responsible only for reading the raw CSV from disk.
    Replaces '?' with NaN immediately.
    """

    def __init__(self):
        self.df_raw    = None   # pd.DataFrame | None
        self.filepath  = ""

    def load(self, filepath: str) -> dict:
        """
        Load CSV → self.df_raw.
        Returns a summary dict: rows, columns, column names.
        """
        self.filepath = filepath
        self.df_raw   = pd.read_csv(filepath, low_memory=False, na_values=["?"])

        return {
            "rows"    : len(self.df_raw),
            "columns" : len(self.df_raw.columns),
            "cols"    : list(self.df_raw.columns),
        }

#  CLASS 2 ── DataCleaner   

class DataCleaner:

    #Column mapping: raw UCI header to internal name 
    COLUMN_MAP = {
        "gender"           : "patient_gender",
        "patient_gender"   : "patient_gender",
        "age"              : "age",
        "time_in_hospital" : "time_in_hospital",
        "num_lab_procedures": "num_lab_procedures",
        "num_medications"  : "num_medications",
        "number_emergency" : "number_emergency",
        "number_inpatient" : "number_inpatient",
        "diag_1"           : "diag_1",
        "readmitted"       : "readmitted",
    }

    def __init__(self):
        self.df         = None   # cleaned DataFrame
        self.label_enc  = {}     # dict[str, LabelEncoder]
        self.is_cleaned = False

    # EXPERT MODULE: Regex ICD-9 classifier
    @staticmethod
    def _classify_diagnosis(code) -> str:
        if pd.isna(code):
            return "Unknown"

        code = str(code).strip()

        # Remove any leading/trailing whitespace or stray characters with regex
        code = re.sub(r"\s+", "", code)            # collapse all whitespace

        patterns = [
            # (compiled_or_raw_pattern,  human label)
            (r"^250",                             "Diabetes"),
            (r"^(390|39[1-9]|4[0-5]\d)",         "Circulatory"),
            (r"^(460|4[6-9]\d|5[0-1]\d)",        "Respiratory"),
            (r"^(520|5[2-9]\d|6[0-2]\d)",        "Digestive"),
            (r"^(580|5[89]\d|63\d|64[0-9])",     "Genitourinary"),
            (r"^(14[0-9]|1[5-9]\d|2[0-3]\d)",    "Neoplasms"),
            (r"^(800|8[0-9]\d|9[0-4]\d)",        "Injury"),
            (r"^[EV]",                             "External/Supplementary"),
            (r"^\d{3}",                            "Other Numeric"),
        ]

        for pattern, label in patterns:
            if re.match(pattern, code):
                return label

        return "Other"

    # Main cleaning pipeline 
    def clean(self, df_raw: pd.DataFrame, log_fn=None) -> dict:

        def log(msg):
            if log_fn:
                log_fn(msg)

        # Step 1: column selection 
        log("Step 1/6 — Selecting & renaming columns …")
        available = {col: self.COLUMN_MAP[col]
                     for col in df_raw.columns
                     if col in self.COLUMN_MAP}

        if len(available) < 5:
            raise ValueError(
                f"Dataset missing expected columns.\n"
                f"Found:    {list(df_raw.columns[:10])} …\n"
                f"Expected: {list(self.COLUMN_MAP.keys())}"
            )

        df = df_raw[list(available.keys())].rename(columns=available).copy()

        # Step 2: smart null handling
        log("Step 2/6 — Smart null handling …")
        before = len(df)

        thresh = int(0.6 * len(df))             
        df.dropna(axis=1, thresh=thresh, inplace=True)
        log(f"          Columns after null-column drop: {list(df.columns)}")

        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    log(f"          Imputed '{col}' with median={median_val:.2f}")
                else:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    log(f"          Imputed '{col}' with mode='{mode_val}'")

        after = len(df)
        log(f"          Rows before: {before:,} | after: {after:,} (0 dropped by null).")

        # Step 3: regex diagnosis classifier
        log("Step 3/6 — Applying Regex ICD-9 diagnosis classifier …")
        df["diagnosis_category"] = df["diag_1"].apply(self._classify_diagnosis)

        # Step 4: ordinal age encoding 
        log("Step 4/6 — Ordinal age encoding …")
        if "age" in df.columns:
            df["age_group"] = df["age"]          
            df["age_num"]   = df["age"].apply(
 
                lambda x: AGE_ORDER.index(x) if x in AGE_ORDER else -1
            )

        # Step 5: binary readmission flag
        log("Step 5/6 — Deriving binary readmission flag …")
        df["readmitted_flag"] = df["readmitted"].apply(
            lambda x: 1 if str(x).strip() == "<30" else 0
        )

        # Step 6: gender label encoding
        log("Step 6/6 — Encoding categorical columns …")
        if "patient_gender" in df.columns:
            le = LabelEncoder()
            df["gender_enc"] = le.fit_transform(df["patient_gender"].astype(str))
            self.label_enc["patient_gender"] = le


        self.df         = df
        self.is_cleaned = True

        # Summary statistics using NumPy
        stay_arr = df["time_in_hospital"].to_numpy()   # explicit NumPy array
        lab_arr  = df["num_lab_procedures"].to_numpy()

        stats = {
            "total_patients"       : len(df),
            "readmitted_lt30"      : int(df["readmitted_flag"].sum()),
            "readmission_rate_pct" : round(float(np.mean(df["readmitted_flag"])) * 100, 2),
            "avg_time_in_hospital" : round(float(np.mean(stay_arr)),  2),
            "std_time_in_hospital" : round(float(np.std(stay_arr)),   2),
            "avg_lab_procedures"   : round(float(np.mean(lab_arr)),   2),
            "unique_diag_cats"     : df["diagnosis_category"].nunique(),
            "gender_counts"        : df["patient_gender"].value_counts().to_dict()
                                     if "patient_gender" in df.columns else {},
        }
        return stats

    def export_cleaned_csv(self, out_path: str = "cleaned_diabetes_data.csv"):
        if self.df is None:
            raise RuntimeError("No cleaned data to export.")
        self.df.to_csv(out_path, index=False)
        return out_path


#  CLASS 3 ── Visualizer


class Visualizer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

    def _style_ax(self, ax, title: str, xlabel: str = "", ylabel: str = ""):
        """Apply consistent styling to any Axes object."""
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=PALETTE["text"], pad=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    # Chart 1: Pie chart of Readmission distribution
    def chart1_readmission_pie(self) -> plt.Figure:
        """Pie chart showing proportions of NO / <30 / >30 readmission."""
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        counts = self.df["readmitted"].value_counts()
        colors = [PALETTE["secondary"], PALETTE["primary"], PALETTE["accent"]]

        ax.pie(
            counts,
            labels     = counts.index,
            autopct    = "%1.1f%%",
            colors     = colors[:len(counts)],
            wedgeprops = {"linewidth": 2, "edgecolor": "white"},
            startangle = 140,
            pctdistance= 0.80,
        )
        ax.set_title("Patient Readmission Rate Distribution",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["text"], pad=14)
        fig.tight_layout()
        fig.savefig("chart1_readmission_pie.png", dpi=150, bbox_inches="tight")
        return fig

    # Chart 2: Horizontal Bar graph of Avg stay by Age group
    def chart2_age_hospital_bar(self) -> plt.Figure:
        """Mean hospital stay per age group (horizontal bars)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        if "age_group" in self.df.columns:
            grp_col = "age_group"
        else:
            grp_col = "age"

        age_time = (
            self.df.groupby(grp_col, observed=True)["time_in_hospital"]
            .mean()
            .reset_index()
            .sort_values("time_in_hospital", ascending=False)
        )

        bars = ax.barh(age_time[grp_col], age_time["time_in_hospital"],
                       color=PALETTE["primary"], edgecolor="white", linewidth=0.8)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}d", va="center", ha="left",
                    fontsize=8, color=PALETTE["muted"])

        self._style_ax(ax, "Average Hospital Stay by Age Group",
                       "Average Days", "Age Group")
        fig.tight_layout()
        fig.savefig("chart2_age_hospital_bar.png", dpi=150, bbox_inches="tight")
        return fig

    # Chart 3: Scatter plot of Lab procedures vs. Medications
    def chart3_scatter_lab_meds(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        sample = self.df.sample(min(3000, len(self.df)), random_state=42)
        colors = sample["readmitted_flag"].map(
            {0: PALETTE["primary"], 1: PALETTE["secondary"]}
        )

        ax.scatter(sample["num_lab_procedures"], sample["num_medications"],
                   c=colors, alpha=0.45, s=14, linewidths=0)

        # NumPy polynomial trend line (degree 1 = linear regression)
        z = np.polyfit(sample["num_lab_procedures"], sample["num_medications"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample["num_lab_procedures"].min(),
                              sample["num_lab_procedures"].max(), 200)
        ax.plot(x_line, p(x_line),
                color=PALETTE["warning"], linewidth=2, linestyle="--", label="Trend")

        legend_els = [
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=PALETTE["primary"],   markersize=8,
                   label="Not readmitted"),
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=PALETTE["secondary"], markersize=8,
                   label="Readmitted <30d"),
        ]
        ax.legend(handles=legend_els, fontsize=8, framealpha=0.9)

        self._style_ax(ax, "Lab Procedures vs. Medications",
                       "Number of Lab Procedures", "Number of Medications")
        fig.tight_layout()
        fig.savefig("chart3_lab_meds_scatter.png", dpi=150, bbox_inches="tight")
        return fig

    # Chart 4: Boxplot of Hospital stay by Gender
    def chart4_gender_boxplot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        if "patient_gender" not in self.df.columns:
            ax.text(0.5, 0.5, "Gender column not available",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        genders        = self.df["patient_gender"].unique()
        gender_palette = {"Male": PALETTE["primary"],
                          "Female": PALETTE["secondary"],
                          "Unknown": PALETTE["muted"]}
        box_pal = {g: gender_palette.get(g, PALETTE["accent"]) for g in genders}

        sns.boxplot(data=self.df, x="patient_gender", y="time_in_hospital",
                    palette=box_pal, linewidth=1.2,
                    flierprops={"marker":"o","markersize":3,
                                "alpha":0.4,"color":PALETTE["muted"]},
                    ax=ax)

        self._style_ax(ax, "Hospital Stay Duration by Gender",
                       "Patient Gender", "Days in Hospital")
        fig.tight_layout()
        fig.savefig("chart4_gender_boxplot.png", dpi=150, bbox_inches="tight")
        return fig

    # Chart 5: Bar Top 5 diagnosis categories
    def chart5_top_diagnoses_bar(self) -> plt.Figure:
        """Bar chart of the 5 most frequent ICD-9 diagnosis categories."""
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])

        top5   = self.df["diagnosis_category"].value_counts().head(5)
        colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
                  PALETTE["warning"], PALETTE["purple"]]

        bars = ax.bar(top5.index, top5.values,
                      color=colors[:len(top5)],
                      edgecolor="white", linewidth=0.8, width=0.55)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 50,
                    f"{int(h):,}", ha="center", va="bottom",
                    fontsize=8, color=PALETTE["muted"])

        self._style_ax(ax, "Top 5 Diagnosis Categories (ICD-9 via Regex)",
                       "Diagnosis Category", "Number of Patients")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig("chart5_top_diagnoses_bar.png", dpi=150, bbox_inches="tight")
        return fig

    # Chart 6 : Correlation Heatmap 
    def chart6_correlation_heatmap(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(PALETTE["bg"])

        numeric_cols = [c for c in [
            "time_in_hospital", "num_lab_procedures",
            "num_medications",  "number_emergency",
            "age_num",          "readmitted_flag",
        ] if c in self.df.columns]

        corr_matrix = self.df[numeric_cols].corr()   # Pandas .corr() → Pearson

        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title("Feature Correlation Heatmap",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["text"], pad=12)
        fig.tight_layout()
        fig.savefig("chart6_correlation_heatmap.png", dpi=150, bbox_inches="tight")
        return fig

    def all_figures(self) -> list:
        """Return all 6 figures as a list."""
        return [
            self.chart1_readmission_pie(),
            self.chart2_age_hospital_bar(),
            self.chart3_scatter_lab_meds(),
            self.chart4_gender_boxplot(),
            self.chart5_top_diagnoses_bar(),
            self.chart6_correlation_heatmap(),
        ]


#  CLASS 4 ── MLPredictor
class MLPredictor:
    """
    Encapsulates the RandomForest training, cross-validation, and evaluation.
    Separating ML from data-cleaning follows Single Responsibility Principle.
    """

    def __init__(self):
        self.model        = None
        self.model_acc    = None
        self.model_report = ""
        self.cv_scores    = None    # cross-validation scores (NumPy array)

    def train_and_evaluate(self, df: pd.DataFrame) -> dict:
        # Feature selection 
        candidate_features = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_medications",
            "number_emergency",
            "number_inpatient",
            "age_num",
            "gender_enc",
        ]
        features = [f for f in candidate_features if f in df.columns]
        target   = "readmitted_flag"

        X = df[features].fillna(0)   # fill any residual NaN with 0
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators    = 150,
            max_depth       = 8,
            min_samples_leaf= 10,
            random_state    = 42,
            n_jobs          = -1,
        )
        self.model.fit(X_train, y_train)

        self.cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")

        # Evaluate on held-out test set
        y_pred            = self.model.predict(X_test)
        self.model_acc    = accuracy_score(y_test, y_pred)
        self.model_report = classification_report(
            y_test, y_pred, target_names=["Not <30d", "Readmitted <30d"]
        )
        cm = confusion_matrix(y_test, y_pred)

        # Feature importances (pure Python sort)
        importances = list(zip(features, self.model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)

        # Confusion matrix figure
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        fig_cm.patch.set_facecolor(PALETTE["bg"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not <30d", "Readmit <30d"],
                    yticklabels=["Not <30d", "Readmit <30d"],
                    ax=ax_cm, linewidths=0.5)
        ax_cm.set_xlabel("Predicted Label", fontsize=11)
        ax_cm.set_ylabel("True Label",      fontsize=11)
        ax_cm.set_title(
            f"Confusion Matrix  (Test Accuracy = {self.model_acc:.2%})",
            fontsize=12, fontweight="bold", color=PALETTE["text"]
        )
        fig_cm.tight_layout()
        fig_cm.savefig("chart_ml_confusion_matrix.png", dpi=150, bbox_inches="tight")

        return {
            "accuracy"     : self.model_acc,
            "cv_mean"      : float(np.mean(self.cv_scores)),
            "cv_std"       : float(np.std(self.cv_scores)),
            "cv_scores"    : self.cv_scores.tolist(),
            "report"       : self.model_report,
            "importances"  : importances,
            "train_size"   : len(X_train),
            "test_size"    : len(X_test),
            "features_used": features,
            "cm_fig"       : fig_cm,
        }


#  CLASS 5 AppGUI   (Tkinter interface)

class AppGUI:

    # Colour tokens
    C = {
        "bg"        : "#F0F4F8",
        "sidebar"   : "#1E2A3A",
        "header"    : "#132238",
        "card"      : "#FFFFFF",
        "primary"   : "#1A73E8",
        "success"   : "#34A853",
        "warning"   : "#FBBC05",
        "danger"    : "#E84040",
        "text_light": "#FFFFFF",
        "text_dark" : "#202124",
        "muted"     : "#5F6368",
        "border"    : "#DFE1E5",
        "log_bg"    : "#1E2A3A",
        "log_fg"    : "#A8D8A8",
        "accent"    : "#4ECDC4",
    }

    def __init__(self, root: tk.Tk):
        self.root      = root
        self.loader    = DataLoader()
        self.cleaner   = DataCleaner()
        self.predictor = MLPredictor()
        self._setup_window()
        self._build_header()
        self._build_body()
        self._build_statusbar()
        self._log("Application initialised. Load a CSV file to begin.", tag="info")

    # Window setup
    def _setup_window(self):
        self.root.title("Diabetes Readmission Risk Analyser — UCI Dataset")
        self.root.geometry("1200x780")
        self.root.minsize(960, 640)
        self.root.configure(bg=self.C["bg"])
        self.root.resizable(True, True)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame",      background=self.C["bg"])
        style.configure("Card.TFrame", background=self.C["card"], relief="flat")
        style.configure("TLabel",      background=self.C["bg"],
                        foreground=self.C["text_dark"])
        style.configure("TScrollbar",  background=self.C["border"])
        style.configure("TNotebook",   background=self.C["bg"])
        style.configure("TNotebook.Tab",
                        font=("Segoe UI", 9, "bold"), padding=[12, 6])

    # Header
    def _build_header(self):
        header = tk.Frame(self.root, bg=self.C["header"], height=65)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header,
                 text="🏥  Diabetes Patient Readmission Risk Analyser",
                 font=("Segoe UI", 15, "bold"),
                 bg=self.C["header"], fg=self.C["text_light"]
                 ).pack(side="left", padx=22, pady=0)

        tk.Label(header,
                 text="UCI 130-US Hospitals (1999-2008)  |  "
                      "Python · Pandas · NumPy · Sklearn · Tkinter",
                 font=("Segoe UI", 9),
                 bg=self.C["header"], fg="#90A4AE"
                 ).pack(side="left", padx=4)

    # Body: sidebar + notebook
    def _build_body(self):
        body = tk.Frame(self.root, bg=self.C["bg"])
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        self._build_notebook(body)

    # Sidebar
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=self.C["sidebar"], width=220)
        sb.pack(fill="y", side="left")
        sb.pack_propagate(False)

        def section_label(text):
            tk.Label(sb, text=text, font=("Segoe UI", 8, "bold"),
                     bg=self.C["sidebar"], fg="#78909C",
                     anchor="w", padx=18, pady=4
                     ).pack(fill="x", pady=(10, 0))

        def sidebar_btn(text, cmd, color=None, emoji=""):
            clr = color or self.C["primary"]
            btn = tk.Button(sb, text=f"  {emoji}  {text}", command=cmd,
                            font=("Segoe UI", 10),
                            bg=clr, fg="white",
                            activebackground="#0D47A1",
                            activeforeground="white",
                            relief="flat", bd=0,
                            anchor="w", padx=14, pady=10,
                            cursor="hand2")
            btn.pack(fill="x", padx=14, pady=3)
            return btn

        tk.Label(sb, text="⚕", font=("Segoe UI", 36),
                 bg=self.C["sidebar"], fg=self.C["accent"]
                 ).pack(pady=(18, 0))
        tk.Label(sb, text="DATA ANALYSER",
                 font=("Segoe UI", 9, "bold"),
                 bg=self.C["sidebar"], fg=self.C["accent"]
                 ).pack(pady=(0, 8))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=18, pady=4)

        section_label("DATA PIPELINE")
        sidebar_btn("Load CSV File",    self._on_load,    emoji="📂")
        sidebar_btn("Process Data",     self._on_process, emoji="⚙️")
        sidebar_btn("Export Clean CSV", self._on_export,  emoji="💾",
                    color="#455A64")

        section_label("ANALYSIS")
        sidebar_btn("Generate Charts",   self._on_viz, emoji="📊")
        sidebar_btn("Run ML Prediction", self._on_ml,  emoji="🤖",
                    color=self.C["success"])

        section_label("TOOLS")
        sidebar_btn("Clear Log", self._clear_log, emoji="🗑️", color="#455A64")
        sidebar_btn("Exit",      self.root.quit,  emoji="🚪",
                    color=self.C["danger"])

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=18, pady=8)
        self.lbl_file = tk.Label(sb, text="No file loaded",
                                 font=("Segoe UI", 8), wraplength=190,
                                 bg=self.C["sidebar"], fg="#78909C",
                                 padx=14, anchor="w")
        self.lbl_file.pack(fill="x")

    # Notebook (tabbed main area)
    def _build_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)

        self._build_dashboard_tab()
        self._build_charts_tab()
        self._build_log_tab()

    # Tab 1: Dashboard (KPI cards)
    def _build_dashboard_tab(self):
        tab = tk.Frame(self.notebook, bg=self.C["bg"])
        self.notebook.add(tab, text="📋  Dashboard")

        # KPI cards row
        card_row = tk.Frame(tab, bg=self.C["bg"])
        card_row.pack(fill="x", padx=16, pady=14)

        self.kpi_vars = {}
        kpis = [
            ("total_patients",       "Total Patients",    self.C["primary"]),
            ("readmitted_lt30",      "Readmitted <30d",   self.C["danger"]),
            ("readmission_rate_pct", "Readmit Rate (%)",  self.C["warning"]),
            ("avg_time_in_hospital", "Avg Stay (days)",   self.C["success"]),
            ("avg_lab_procedures",   "Avg Lab Procs",     self.C["accent"]),
        ]
        for i, (key, label, color) in enumerate(kpis):
            self._make_kpi_card(card_row, key, label, color, i)

        # Stats panel below KPIs
        stats_frame = tk.LabelFrame(tab, text=" 📊  Dataset Statistics ",
                                    font=("Segoe UI", 9, "bold"),
                                    bg=self.C["bg"], fg=self.C["muted"],
                                    bd=1, relief="groove")
        stats_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self.stats_text = scrolledtext.ScrolledText(
            stats_frame, font=("Consolas", 9),
            bg=self.C["log_bg"], fg="#CE93D8",
            relief="flat", bd=0, state="disabled"
        )
        self.stats_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _make_kpi_card(self, parent, key, label, color, col):
        card = tk.Frame(parent, bg=self.C["card"],
                        highlightbackground=color, highlightthickness=2,
                        relief="flat")
        card.grid(row=0, column=col, padx=5, pady=4, sticky="ew")
        parent.columnconfigure(col, weight=1)

        tk.Label(card, text=label, font=("Segoe UI", 8),
                 bg=self.C["card"], fg=self.C["muted"]
                 ).pack(pady=(8, 2))
        var = tk.StringVar(value="—")
        self.kpi_vars[key] = var
        tk.Label(card, textvariable=var,
                 font=("Segoe UI", 18, "bold"),
                 bg=self.C["card"], fg=color
                 ).pack(pady=(0, 8))

    #Tab 2: Embedded Charts
    def _build_charts_tab(self):
        tab = tk.Frame(self.notebook, bg=self.C["bg"])
        self.notebook.add(tab, text="📈  Charts")

        # Chart selector
        ctrl = tk.Frame(tab, bg=self.C["bg"])
        ctrl.pack(fill="x", padx=12, pady=6)

        tk.Label(ctrl, text="Select chart:", font=("Segoe UI", 9),
                 bg=self.C["bg"]).pack(side="left")

        self.chart_var = tk.StringVar(value="1 — Readmission Pie")
        chart_options  = [
            "1 — Readmission Pie",
            "2 — Age vs Hospital Stay",
            "3 — Lab vs Medications",
            "4 — Gender Boxplot",
            "5 — Top Diagnoses",
            "6 — Correlation Heatmap",
        ]
        self.chart_combo = ttk.Combobox(ctrl, textvariable=self.chart_var,
                                        values=chart_options, state="readonly",
                                        width=28)
        self.chart_combo.pack(side="left", padx=8)
        self.chart_combo.bind("<<ComboboxSelected>>", self._on_chart_select)

        # Canvas container — charts render here
        self.chart_frame = tk.Frame(tab, bg=self.C["bg"])
        self.chart_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self._figures    = []          # list of Figure objects (filled on viz run)
        self._canvas_widget = None     # current TkAgg canvas widget

    def _embed_figure(self, fig: plt.Figure):
        """Replace the current chart canvas with a new figure."""
        if self._canvas_widget:
            self._canvas_widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        self._canvas_widget = canvas.get_tk_widget()
        self._canvas_widget.pack(fill="both", expand=True)

    def _on_chart_select(self, _event=None):
        if not self._figures:
            messagebox.showwarning("No Charts", "Please run Generate Charts first.")
            return
        idx = self.chart_combo.current()        # 0-based index
        if 0 <= idx < len(self._figures):
            self._embed_figure(self._figures[idx])

    #Tab 3: Log
    def _build_log_tab(self):
        tab = tk.Frame(self.notebook, bg=self.C["bg"])
        self.notebook.add(tab, text="📋  Log")

        self.log_box = scrolledtext.ScrolledText(
            tab, font=("Consolas", 9),
            bg=self.C["log_bg"], fg=self.C["log_fg"],
            insertbackground="white", wrap="word",
            relief="flat", bd=0
        )
        self.log_box.pack(fill="both", expand=True, padx=6, pady=6)

        for tag, fg, bold in [
            ("info",    "#90CAF9", False),
            ("success", "#A5D6A7", False),
            ("warning", "#FFE082", False),
            ("error",   "#EF9A9A", False),
            ("header",  "#80DEEA", True),
            ("data",    "#CE93D8", False),
        ]:
            font_spec = ("Consolas", 9, "bold") if bold else ("Consolas", 9)
            self.log_box.tag_configure(tag, foreground=fg, font=font_spec)

    # Status bar
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=self.C["header"], height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")
        self.progress   = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.pack(side="right", padx=12, pady=4)

        tk.Label(bar, textvariable=self.status_var,
                 font=("Segoe UI", 9),
                 bg=self.C["header"], fg="#B0BEC5"
                 ).pack(side="left", padx=14)

    # Logging helpers
    def _log(self, message: str, tag: str = "info"):
        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        txt = f"[{ts}]  {message}\n"
        self.log_box.insert("end", txt, tag)
        self.log_box.see("end")

    def _clear_log(self):
        self.log_box.delete("1.0", "end")
        self._log("Log cleared.", tag="info")

    def _set_status(self, msg: str, busy: bool = False):
        self.status_var.set(msg)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        self.root.update_idletasks()

    def _update_stats_panel(self, stats: dict):
        """Write detailed stats into the Dashboard text panel."""
        lines = [
            "=" * 55,
            "  DATASET STATISTICS",
            "=" * 55,
            f"  Total patients          : {stats['total_patients']:,}",
            f"  Readmitted (<30 days)   : {stats['readmitted_lt30']:,}",
            f"  Readmission rate        : {stats['readmission_rate_pct']}%",
            f"  Avg hospital stay       : {stats['avg_time_in_hospital']} days",
            f"  Std hospital stay (NumPy): {stats['std_time_in_hospital']} days",
            f"  Avg lab procedures      : {stats['avg_lab_procedures']}",
            f"  Unique diagnosis cats   : {stats['unique_diag_cats']}",
            "",
            "  Gender breakdown:",
        ]
        for g, n in stats.get("gender_counts", {}).items():
            lines.append(f"    {g:<12}: {n:,}")
        lines.append("=" * 55)

        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("end", "\n".join(lines))
        self.stats_text.config(state="disabled")

    # Callbacks 
    def _on_load(self):
        path = filedialog.askopenfilename(
            title="Select UCI Diabetes CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        self._set_status("Loading CSV …", busy=True)
        self._log(f"Loading file: {os.path.basename(path)}", tag="header")
        try:
            summary = self.loader.load(path)
            self.lbl_file.config(text=os.path.basename(path))
            self._log(f"✔  Loaded {summary['rows']:,} rows × "
                      f"{summary['columns']} columns.", tag="success")
            self._log(f"   Columns: {', '.join(summary['cols'][:10])} …",
                      tag="data")
            self._set_status(f"Loaded: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"✘  Load error: {exc}", tag="error")
            messagebox.showerror("Load Error", str(exc))
            self._set_status("Load failed.")

    def _on_process(self):
        if self.loader.df_raw is None:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return

        def task():
            self._set_status("Processing data …", busy=True)
            self._log("─" * 55, tag="header")
            self._log("Starting data cleaning pipeline …", tag="header")
            try:
                stats = self.cleaner.clean(
                    self.loader.df_raw,
                    log_fn=lambda m: self._log(m, tag="info")
                )
                for key, var in self.kpi_vars.items():
                    val = stats.get(key, "—")
                    var.set(f"{val:,}" if isinstance(val, int)
                            else (str(val) if val != "—" else "—"))

                self._update_stats_panel(stats)
                self._log("─" * 55, tag="header")
                self._log("✔  Processing complete!", tag="success")
                self._set_status("Data processed successfully.")
            except Exception as exc:
                self._log(f"✘  Processing error: {exc}", tag="error")
                self._log(traceback.format_exc(), tag="error")
                messagebox.showerror("Processing Error", str(exc))
                self._set_status("Processing failed.")

        threading.Thread(target=task, daemon=True).start()

    def _on_export(self):
        """Export the cleaned CSV — satisfies 'reproducible pipeline' mark."""
        if not self.cleaner.is_cleaned:
            messagebox.showwarning("Not Ready", "Please process data first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Cleaned CSV",
            defaultextension=".csv",
            initialfile="cleaned_diabetes_data.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        try:
            saved = self.cleaner.export_cleaned_csv(path)
            self._log(f"✔  Cleaned data exported → {saved}", tag="success")
            messagebox.showinfo("Export Complete", f"Saved to:\n{saved}")
        except Exception as exc:
            self._log(f"✘  Export error: {exc}", tag="error")
            messagebox.showerror("Export Error", str(exc))

    def _on_viz(self):
        if not self.cleaner.is_cleaned:
            messagebox.showwarning("Not Ready", "Please process data first.")
            return

        def task():
            self._set_status("Generating charts …", busy=True)
            self._log("─" * 55, tag="header")
            self._log("Building 6 Matplotlib / Seaborn charts …", tag="header")
            try:
                viz = Visualizer(self.cleaner.df)
                self._figures = viz.all_figures()
                # Embed the first chart immediately
                self.root.after(0, lambda: self._embed_figure(self._figures[0]))
                self.root.after(0, lambda: self.notebook.select(1))  # switch to Charts tab
                chart_names = [
                    "chart1_readmission_pie.png",
                    "chart2_age_hospital_bar.png",
                    "chart3_lab_meds_scatter.png",
                    "chart4_gender_boxplot.png",
                    "chart5_top_diagnoses_bar.png",
                    "chart6_correlation_heatmap.png",
                ]
                for name in chart_names:
                    self._log(f"   ✔  {name}", tag="data")
                self._log("✔  All 6 charts ready (embedded + saved as PNG).",
                          tag="success")
                self._set_status("Charts ready — see Charts tab.")
            except Exception as exc:
                self._log(f"✘  Chart error: {exc}", tag="error")
                self._log(traceback.format_exc(), tag="error")
                messagebox.showerror("Chart Error", str(exc))
                self._set_status("Chart generation failed.")

        threading.Thread(target=task, daemon=True).start()

    def _on_ml(self):
        if not self.cleaner.is_cleaned:
            messagebox.showwarning("Not Ready", "Please process data first.")
            return

        def task():
            self._set_status("Training RandomForest …", busy=True)
            self._log("─" * 55, tag="header")
            self._log("Running RandomForestClassifier + 5-fold CV …",
                      tag="header")
            try:
                results = self.predictor.train_and_evaluate(self.cleaner.df)

                self._log(f"✔  Trained on {results['train_size']:,} | "
                          f"Tested on {results['test_size']:,}", tag="success")
                self._log(f"   Features      : {', '.join(results['features_used'])}",
                          tag="data")
                self._log(f"   Test Accuracy  : {results['accuracy']:.4f} "
                          f"({results['accuracy']:.2%})", tag="success")
                self._log(f"   CV Mean Acc    : {results['cv_mean']:.4f} "
                          f"± {results['cv_std']:.4f}  "
                          f"(5-fold cross-validation)", tag="success")
                self._log(f"   CV Fold Scores : "
                          f"{[round(s,4) for s in results['cv_scores']]}",
                          tag="data")
                self._log("", tag="info")
                self._log("Classification Report:", tag="header")
                for line in results["report"].split("\n"):
                    self._log("   " + line, tag="data")
                self._log("", tag="info")
                self._log("Feature Importances:", tag="header")
                for feat, imp in results["importances"]:
                    bar = "█" * int(imp * 50)
                    self._log(f"   {feat:<25} {imp:.4f}  {bar}", tag="data")

                self._set_status(
                    f"ML done — Acc: {results['accuracy']:.2%} | "
                    f"CV: {results['cv_mean']:.2%}±{results['cv_std']:.2%}"
                )

                # Embed confusion matrix in Charts tab
                if self._figures:
                    self._figures.append(results["cm_fig"])
                else:
                    self._figures = [results["cm_fig"]]
                self.root.after(
                    0, lambda: self._embed_figure(results["cm_fig"])
                )
                self.root.after(0, lambda: self.notebook.select(1))

                messagebox.showinfo(
                    "ML Prediction Results",
                    f"Algorithm   : Random Forest (150 trees)\n"
                    f"Train size  : {results['train_size']:,} patients\n"
                    f"Test size   : {results['test_size']:,} patients\n"
                    f"Test Acc    : {results['accuracy']:.2%}\n"
                    f"5-Fold CV   : {results['cv_mean']:.2%} ± {results['cv_std']:.2%}\n\n"
                    f"Top feature : {results['importances'][0][0]}\n"
                    f"See Log tab for full classification report."
                )

            except Exception as exc:
                self._log(f"✘  ML error: {exc}", tag="error")
                self._log(traceback.format_exc(), tag="error")
                messagebox.showerror("ML Error", str(exc))
                self._set_status("ML failed.")

        threading.Thread(target=task, daemon=True).start()


#  ENTRY POINT
def main():
    root = tk.Tk()
    AppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()