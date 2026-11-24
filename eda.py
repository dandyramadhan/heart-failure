"""Analisis eksplorasi data (EDA) untuk dataset Heart Failure Prediction.

Menghasilkan gambar di ``reports/figures/`` dan ringkasan naratif di
``reports/eda_summary.md``. Jalankan dengan::

    python eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend tanpa layar, aman untuk skrip/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    CATEGORICAL_COLUMNS,
    CHEST_PAIN_LABELS,
    RAW_NUMERIC_COLUMNS,
    REPORT_DIR,
    TARGET_COLUMN,
    add_engineered_features,
    clean_clinical_zeros,
    load_data,
)

FIGURE_DIR = REPORT_DIR / "figures"
ACCENT = "#b91c1c"
MUTED = "#64748b"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#1e293b",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "figure.facecolor": "white",
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target(df: pd.DataFrame) -> Path:
    counts = df[TARGET_COLUMN].map({0: "Tidak berisiko", 1: "Berisiko"}).value_counts()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(counts.index, counts.values, color=[MUTED, ACCENT], width=0.55)
    ax.bar_label(bars, labels=[f"{v} ({v / len(df):.1%})" for v in counts.values], padding=4)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Jumlah rekam")
    ax.set_title("Distribusi target (HeartDisease)")
    return _save(fig, "01_target_distribution.png")


def plot_numeric_distributions(clean: pd.DataFrame) -> Path:
    columns = RAW_NUMERIC_COLUMNS
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, column in zip(axes.ravel(), columns):
        for label, color in [(0, MUTED), (1, ACCENT)]:
            subset = clean.loc[clean[TARGET_COLUMN] == label, column].dropna()
            ax.hist(subset, bins=22, alpha=0.65, color=color, label=f"Target {label}")
        ax.set_title(column)
        ax.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sebaran fitur numerik menurut status penyakit", fontweight="bold")
    return _save(fig, "02_numeric_distributions.png")


def plot_correlation(clean: pd.DataFrame) -> Path:
    columns = RAW_NUMERIC_COLUMNS + ["MaxHR_Deficit", "HR_Achievement_Ratio", "BP_Age_Ratio"]
    corr = clean[columns + [TARGET_COLUMN]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for row in range(len(corr)):
        for col in range(len(corr)):
            value = corr.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.55 else "#1e293b",
            )
    fig.colorbar(image, ax=ax, shrink=0.82, label="Koefisien korelasi Pearson")
    ax.set_title("Matriks korelasi fitur numerik dan target")
    return _save(fig, "03_correlation_heatmap.png")


def plot_categorical_risk(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4))
    for ax, column in zip(axes, CATEGORICAL_COLUMNS):
        rates = df.groupby(column)[TARGET_COLUMN].mean().sort_values(ascending=False)
        bars = ax.bar(rates.index.astype(str), rates.values, color=ACCENT, width=0.6)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in rates.values], fontsize=7, padding=2)
        ax.set_ylim(0, 1.0)
        ax.set_title(column, fontsize=9)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.suptitle("Proporsi pasien berisiko pada setiap kategori", fontweight="bold")
    return _save(fig, "04_categorical_risk.png")


def plot_zero_values(df: pd.DataFrame) -> Path:
    counts = {
        "Cholesterol = 0": int((df["Cholesterol"] == 0).sum()),
        "RestingBP = 0": int((df["RestingBP"] == 0).sum()),
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bars = ax.bar(
        list(counts), list(counts.values()), color=["#b45309", "#0369a1"], width=0.5
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Jumlah baris")
    ax.set_ylim(0, max(counts.values()) * 1.25)
    ax.set_title("Nilai nol yang tidak valid secara medis")
    return _save(fig, "05_zero_values.png")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    clean = add_engineered_features(clean_clinical_zeros(df))

    figures = [
        plot_target(df),
        plot_numeric_distributions(clean),
        plot_correlation(clean),
        plot_categorical_risk(df),
        plot_zero_values(df),
    ]

    # Statistik pendukung untuk ringkasan naratif.
    positive_rate = df[TARGET_COLUMN].mean()
    zero_chol = int((df["Cholesterol"] == 0).sum())
    zero_bp = int((df["RestingBP"] == 0).sum())
    correlations = (
        clean[RAW_NUMERIC_COLUMNS + [TARGET_COLUMN]]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=np.abs, ascending=False)
    )
    strongest = correlations.head(4)

    lines = [
        "# Ringkasan EDA — Heart Failure Prediction",
        "",
        f"- Jumlah rekam: **{len(df)}**, fitur: **{df.shape[1] - 1}**",
        f"- Prevalensi penyakit jantung: **{positive_rate:.1%}** "
        f"({int(df[TARGET_COLUMN].sum())} positif / {len(df)} rekam)",
        f"- Nilai tidak hilang secara eksplisit: **{int(df.isna().sum().sum())}** sel",
        "",
        "## Anomali nilai nol",
        "",
        f"- `Cholesterol = 0` pada **{zero_chol}** baris ({zero_chol / len(df):.1%}). "
        "Nilai ini tidak mungkin secara fisiologis dan diperlakukan sebagai data hilang.",
        f"- `RestingBP = 0` pada **{zero_bp}** baris.",
        "- Karena proporsi kolesterol yang hilang cukup besar, imputasi median "
        "dilakukan di dalam pipeline (hanya memakai statistik data latih) dan "
        "hasil model perlu dibaca dengan hati-hati.",
        "",
        "## Korelasi terhadap target",
        "",
        "| Fitur | Korelasi dengan HeartDisease |",
        "| --- | ---: |",
    ]
    for feature, value in strongest.items():
        lines.append(f"| {feature} | {value:+.3f} |")

    lines += [
        "",
        "## Temuan utama",
        "",
        "- `ST_Slope` dan `ExerciseAngina` menunjukkan pemisahan kelas yang jelas: "
        "pasien dengan ST menurun/mendatar dan angina saat olahraga jauh lebih "
        "sering berstatus positif.",
        "- `MaxHR` cenderung lebih rendah pada pasien berisiko, sehingga fitur "
        "turunan cadangan detak jantung (`MaxHR_Deficit`, `HR_Achievement_Ratio`) "
        "relevan secara klinis.",
        "- `ChestPainType = ASY` (asimtomatik) justru berkaitan dengan proporsi "
        "penyakit yang tinggi, menegaskan bahwa tidak adanya gejala bukan berarti "
        "risiko rendah.",
        "",
        "## Gambar",
        "",
    ]
    for path in figures:
        lines.append(f"![{path.stem}](figures/{path.name})")
    lines.append("")

    summary_path = REPORT_DIR / "eda_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"EDA selesai. {len(figures)} gambar ditulis ke {FIGURE_DIR}.")
    print(f"Ringkasan naratif: {summary_path}")
    print("\nKorelasi fitur numerik terhadap target:")
    print(correlations.round(3).to_string())


if __name__ == "__main__":
    main()
