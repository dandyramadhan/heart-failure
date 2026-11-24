"""Utilitas bersama untuk sistem skrining risiko gagal jantung.

Modul ini dipakai oleh ``train_model.py``, ``eda.py``, ``app.py``, dan
pengujian. Semua fungsi bersifat murni (tidak menyimpan state global),
kecuali loader artefak yang membaca berkas dari disk.

Alur prapemrosesan sengaja dibungkus dalam satu fungsi
(``preprocess_and_engineer``) agar dapat dipasang sebagai langkah pertama
di dalam ``Pipeline`` scikit-learn. Dengan begitu, imputasi, penskalaan,
dan *feature engineering* hanya "belajar" dari data latih pada setiap fold
sehingga tidak terjadi *data leakage*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ---------------------------------------------------------------------------
# Lokasi berkas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_heart_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Skema dataset
# ---------------------------------------------------------------------------
TARGET_COLUMN = "HeartDisease"

#: Nilai 0 pada kolom ini tidak mungkin secara medis, jadi diperlakukan
#: sebagai data hilang lalu diimputasi di dalam pipeline.
ZERO_AS_MISSING = ("Cholesterol", "RestingBP")

RAW_NUMERIC_COLUMNS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]

RAW_CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]

ENGINEERED_COLUMNS = [
    "MaxHR_Deficit",
    "HR_Achievement_Ratio",
    "BP_Age_Ratio",
]

NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS
CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS

FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS

#: Label yang dipakai di antarmuka aplikasi.
SEX_LABELS = {"M": "Laki-laki", "F": "Perempuan"}
CHEST_PAIN_LABELS = {
    "ASY": "Asimtomatik (ASY)",
    "NAP": "Nyeri non-angina (NAP)",
    "ATA": "Angina atipikal (ATA)",
    "TA": "Angina tipikal (TA)",
}
RESTING_ECG_LABELS = {
    "Normal": "Normal",
    "ST": "Abnormalitas ST-T",
    "LVH": "Hipertrofi ventrikel kiri",
}
ST_SLOPE_LABELS = {
    "Up": "Menanjak (Up)",
    "Flat": "Mendatar (Flat)",
    "Down": "Menurun (Down)",
}
YES_NO_LABELS = {"Y": "Ya", "N": "Tidak"}
TARGET_LABELS = {0: "Tidak berisiko", 1: "Berisiko"}


# ---------------------------------------------------------------------------
# Pemuatan data
# ---------------------------------------------------------------------------
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Membaca dataset mentah dan memvalidasi kolom wajibnya."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di '{path}'. Letakkan heart.csv pada folder data/."
        )

    df = pd.read_csv(path)

    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {missing}")

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Memisahkan matriks fitur dan vektor target."""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN].astype(int)


# ---------------------------------------------------------------------------
# Pembersihan & feature engineering
# ---------------------------------------------------------------------------
def clean_clinical_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah nilai 0 yang tidak valid secara medis menjadi NaN.

    Kolesterol 0 mg/dl dan tekanan darah istirahat 0 mm Hg secara fisiologis
    tidak mungkin; nilainya menandakan pemeriksaan yang tidak dilakukan.
    Nilai tersebut dibiarkan sebagai NaN agar diimputasi di dalam pipeline
    (median dari data latih saja).
    """
    data = df.copy()
    for column in ZERO_AS_MISSING:
        if column in data.columns:
            data[column] = data[column].astype(float).replace(0.0, np.nan)
    return data


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur turunan klinis.

    - ``MaxHR_Deficit``       : selisih detak jantung maksimal teoretis
                                (220 - usia) dengan MaxHR terukur. Nilai besar
                                menandakan cadangan detak jantung yang rendah.
    - ``HR_Achievement_Ratio``: proporsi MaxHR terhadap batas teoretis.
    - ``BP_Age_Ratio``        : tekanan darah istirahat relatif terhadap usia.
    """
    data = df.copy()

    if {"Age", "MaxHR"}.issubset(data.columns):
        theoretical_max_hr = 220.0 - data["Age"].astype(float)
        data["MaxHR_Deficit"] = theoretical_max_hr - data["MaxHR"].astype(float)
        data["HR_Achievement_Ratio"] = data["MaxHR"].astype(float) / theoretical_max_hr

    if {"RestingBP", "Age"}.issubset(data.columns):
        safe_age = data["Age"].astype(float).replace(0.0, np.nan)
        data["BP_Age_Ratio"] = data["RestingBP"].astype(float) / safe_age

    return data


def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan nilai nol klinis lalu tambahkan fitur turunan.

    Fungsi ini dipasang sebagai ``FunctionTransformer`` di dalam pipeline,
    sehingga selalu dijalankan ulang pada data latih maupun data uji.
    """
    return add_engineered_features(clean_clinical_zeros(df))


# ---------------------------------------------------------------------------
# Preprocessor scikit-learn
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Membuat ``ColumnTransformer`` baru (belum di-fit).

    Selalu mengembalikan objek baru agar setiap pipeline memiliki salinan
    sendiri dan tidak berbagi state hasil fitting.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # RobustScaler memakai median & IQR sehingga tahan terhadap
            # outlier klinis (mis. kolesterol ekstrem).
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier: Any) -> Pipeline:
    """Merangkai feature engineering + preprocessor + classifier."""
    return Pipeline(
        steps=[
            ("feature_eng", _FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


class _FeatureEngineer:
    """Pembungkus picklable untuk ``preprocess_and_engineer``.

    ``FunctionTransformer`` menyimpan referensi fungsi berdasarkan nama
    modul. Kelas ini membuat model tersimpan tetap bisa dimuat walau
    fungsi utilitas dipindah atau diubah namanya.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return preprocess_and_engineer(X)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        return self.transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> "_FeatureEngineer":
        return self


# ---------------------------------------------------------------------------
# Artefak model
# ---------------------------------------------------------------------------
def load_metadata(path: str | Path = METADATA_PATH) -> dict[str, Any]:
    """Membaca metadata hasil pelatihan (metrik, ambang, parameter)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata model tidak ditemukan di '{path}'. "
            "Jalankan `python train_model.py` terlebih dahulu."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_artifacts(
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Memuat pipeline terbaik beserta metadatanya."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{model_path}'. "
            "Jalankan `python train_model.py` terlebih dahulu."
        )
    return joblib.load(model_path), load_metadata(metadata_path)


# ---------------------------------------------------------------------------
# Presentasi hasil
# ---------------------------------------------------------------------------
def risk_level(probability: float, threshold: float) -> str:
    """Mengelompokkan probabilitas menjadi tiga tingkat risiko.

    Ambang keputusan ``threshold`` dipakai agar tingkat risiko tidak pernah
    bertentangan dengan keputusan skrining:

    - ``tinggi`` : probabilitas >= max(threshold, 0,66) — jelas berisiko.
    - ``sedang`` : probabilitas >= min(threshold, 0,33) — melewati ambang
      skrining, atau masih berada di rentang menengah.
    - ``rendah`` : sisanya.

    Dengan aturan ini, keputusan "rujuk" selalu menghasilkan minimal tingkat
    "sedang", dan probabilitas di atas 66% selalu "tinggi" berapa pun ambang
    yang dipilih pengguna.
    """
    if probability >= max(threshold, 0.66):
        return "tinggi"
    if probability >= min(threshold, 0.33):
        return "sedang"
    return "rendah"


def format_probability(probability: float) -> str:
    """Format persentase dengan koma desimal (gaya Indonesia)."""
    return f"{probability * 100:.1f}%".replace(".", ",")
