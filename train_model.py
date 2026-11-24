"""Pelatihan, optimasi, dan evaluasi model skrining risiko gagal jantung.

Tahapan mengikuti CRISP-DM:

1. Pemuatan data & ringkasan EDA.
2. Pembersihan nilai nol klinis + feature engineering (di dalam pipeline).
3. Pembagian data latih/uji secara stratifikasi *sebelum* preprocessing.
4. Pelatihan 4 algoritma dengan ``GridSearchCV`` (5-fold stratified).
5. Pemilihan model: ROC-AUC cross-validation sebagai kriteria utama,
   Recall cross-validation sebagai penentu seri.
6. Penyetelan ambang keputusan pada prediksi *out-of-fold* dengan
   memaksimalkan F2-score (Recall diberi bobot lebih besar daripada
   Precision) untuk menekan *false negative*.
7. Evaluasi akhir pada data uji (hanya sekali, untuk pelaporan).
8. Penyimpanan artefak: model, metadata, tabel perbandingan, laporan.

Jalankan dengan::

    python train_model.py
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import (
    CATEGORICAL_COLUMNS,
    COMPARISON_PATH,
    ENGINEERED_COLUMNS,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    REPORT_DIR,
    TARGET_COLUMN,
    build_pipeline,
    load_data,
    split_features_target,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
THRESHOLD_GRID = np.round(np.arange(0.20, 0.8001, 0.01), 2)


# ---------------------------------------------------------------------------
# Konfigurasi model
# ---------------------------------------------------------------------------
def build_model_zoo() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Empat algoritma beserta ruang hyperparameter yang diuji."""
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "classifier__max_depth": [3, 5, 7, None],
                "classifier__min_samples_split": [2, 10, 20],
                "classifier__criterion": ["gini", "entropy"],
            },
        ),
        "K-Nearest Neighbors (KNN)": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [5, 9, 13, 17],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
        ),
        "Support Vector Machine (SVM)": (
            # SVC(probability=True) sudah deprecated pada scikit-learn 1.9.
            # CalibratedClassifierCV memberi probabilitas terkalibrasi tanpa
            # memakai parameter yang akan dihapus.
            CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False),
            {
                "classifier__estimator__C": [0.1, 1, 10],
                "classifier__estimator__kernel": ["rbf", "linear"],
                "classifier__estimator__gamma": ["scale", "auto"],
            },
        ),
        "Random Forest (Ensemble)": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 8, 12],
                "classifier__min_samples_leaf": [1, 3, 5],
            },
        ),
    }


# ---------------------------------------------------------------------------
# Utilitas evaluasi
# ---------------------------------------------------------------------------
def evaluate_predictions(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Hitung metrik klasifikasi pada ambang keputusan tertentu."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def tune_threshold(y_true: pd.Series, oof_proba: np.ndarray) -> tuple[float, float]:
    """Pilih ambang yang memaksimalkan F2-score pada prediksi out-of-fold.

    F2 memberi bobot Recall dua kali lebih besar daripada Precision,
    sesuai prioritas skrining: meminimalkan pasien berisiko yang lolos.
    """
    scores = [
        fbeta_score(y_true, (oof_proba >= t).astype(int), beta=2, zero_division=0)
        for t in THRESHOLD_GRID
    ]
    best_index = int(np.argmax(scores))
    return float(THRESHOLD_GRID[best_index]), float(scores[best_index])


def _native(value: Any) -> Any:
    """Ubah tipe numpy menjadi tipe Python agar aman disimpan ke JSON."""
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    return value


# ---------------------------------------------------------------------------
# Tahapan utama
# ---------------------------------------------------------------------------
def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data -----------------------------------------------------------------
    df = load_data()
    X, y = split_features_target(df)

    print("=" * 74)
    print("RINGKASAN DATASET")
    print("=" * 74)
    print(f"Dimensi              : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Fitur prediktor      : {X.shape[1]} kolom mentah")
    print(
        "Distribusi target    : "
        + ", ".join(
            f"{label}={count} ({count / len(df):.1%})"
            for label, count in y.value_counts().sort_index().items()
        )
    )
    print(f"Cholesterol = 0      : {(df['Cholesterol'] == 0).sum()} baris (dianggap hilang)")
    print(f"RestingBP = 0        : {(df['RestingBP'] == 0).sum()} baris (dianggap hilang)")
    print(f"Nilai hilang eksplisit: {int(df.isna().sum().sum())} sel\n")

    # 2. Split ---------------------------------------------------------------
    # Split dilakukan SEBELUM preprocessing agar statistik imputasi dan
    # penskalaan hanya berasal dari data latih (mencegah data leakage).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 3. Pelatihan & tuning --------------------------------------------------
    print("=" * 74)
    print("PELATIHAN & HYPERPARAMETER TUNING (GridSearchCV, 5-fold stratified)")
    print("=" * 74)

    comparison: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for name, (classifier, param_grid) in build_model_zoo().items():
        pipeline = build_pipeline(classifier)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "recall": "recall"},
            refit="roc_auc",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, y_proba, threshold=0.5)

        cv_roc_auc = float(grid.cv_results_["mean_test_roc_auc"][grid.best_index_])
        cv_recall = float(grid.cv_results_["mean_test_recall"][grid.best_index_])

        fitted[name] = best_model
        comparison.append(
            {
                "model": name,
                "cv_roc_auc": cv_roc_auc,
                "cv_recall": cv_recall,
                "test_roc_auc": metrics["roc_auc"],
                "test_recall": metrics["recall"],
                "test_precision": metrics["precision"],
                "test_f1": metrics["f1"],
                "test_accuracy": metrics["accuracy"],
                "best_params": json.dumps(_native(grid.best_params_)),
            }
        )

        print(
            f"  {name:<30} CV ROC-AUC={cv_roc_auc:.4f}  "
            f"CV Recall={cv_recall:.4f}  Test ROC-AUC={metrics['roc_auc']:.4f}"
        )

    # 4. Pemilihan model -----------------------------------------------------
    # Kriteria: ROC-AUC cross-validation tertinggi; bila seri, Recall
    # cross-validation tertinggi. Data uji tidak dipakai untuk memilih model.
    comparison.sort(key=lambda row: (row["cv_roc_auc"], row["cv_recall"]), reverse=True)
    best_row = comparison[0]
    best_name = best_row["model"]
    best_model = fitted[best_name]

    print("\n" + "=" * 74)
    print("PERBANDINGAN MODEL (data uji, ambang 0,50)")
    print("=" * 74)
    print(
        f"{'Model':<30}{'CV AUC':>9}{'CV Recall':>11}"
        f"{'Test AUC':>10}{'Recall':>8}{'Prec.':>8}{'F1':>8}"
    )
    for row in comparison:
        print(
            f"{row['model']:<30}{row['cv_roc_auc']:>9.4f}{row['cv_recall']:>11.4f}"
            f"{row['test_roc_auc']:>10.4f}{row['test_recall']:>8.4f}"
            f"{row['test_precision']:>8.4f}{row['test_f1']:>8.4f}"
        )

    # 5. Penyetelan ambang pada prediksi out-of-fold -------------------------
    from sklearn.model_selection import cross_val_predict

    oof_proba = cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold, f2_oof = tune_threshold(y_train, oof_proba)

    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    metrics_default = evaluate_predictions(y_test, y_proba_test, threshold=0.5)
    metrics_tuned = evaluate_predictions(y_test, y_proba_test, threshold=threshold)

    # 6. Interpretasi model --------------------------------------------------
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    feature_importance = sorted(
        (
            {
                "feature": column,
                "importance": float(mean),
                "std": float(std),
            }
            for column, mean, std in zip(
                X_test.columns, importance.importances_mean, importance.importances_std
            )
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    fpr, tpr, _ = roc_curve(y_test, y_proba_test)

    # 7. Simpan artefak ------------------------------------------------------
    joblib.dump(best_model, MODEL_PATH)

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "path": "data/heart.csv",
            "n_rows": int(len(df)),
            "n_features_raw": int(X.shape[1]),
            "n_features_model": int(len(NUMERIC_COLUMNS) + len(CATEGORICAL_COLUMNS)),
            "positive_rate": float(y.mean()),
            "zero_cholesterol": int((df["Cholesterol"] == 0).sum()),
            "zero_resting_bp": int((df["RestingBP"] == 0).sum()),
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "selection_rule": (
            "ROC-AUC cross-validation tertinggi; Recall cross-validation "
            "sebagai penentu bila seri. Data uji hanya untuk pelaporan."
        ),
        "threshold": {
            "value": threshold,
            "default": 0.5,
            "method": "Maksimal F2-score pada prediksi out-of-fold",
            "oof_f2": f2_oof,
        },
        "best_model": {
            "name": best_name,
            "params": json.loads(best_row["best_params"]),
            "cv_roc_auc": best_row["cv_roc_auc"],
            "cv_recall": best_row["cv_recall"],
            "test_roc_auc": metrics_tuned["roc_auc"],
            "test_recall": metrics_tuned["recall"],
            "test_precision": metrics_tuned["precision"],
            "test_f1": metrics_tuned["f1"],
            "test_accuracy": metrics_tuned["accuracy"],
            "test_recall_default_threshold": metrics_default["recall"],
            "test_precision_default_threshold": metrics_default["precision"],
            "test_f1_default_threshold": metrics_default["f1"],
        },
        "models": [
            {
                key: value
                for key, value in row.items()
                if key not in {"best_params"}
            }
            for row in comparison
        ],
        "model_params": {row["model"]: json.loads(row["best_params"]) for row in comparison},
        "confusion_matrix": {
            "threshold_default": metrics_default["confusion_matrix"],
            "threshold_tuned": metrics_tuned["confusion_matrix"],
        },
        "roc_curve": {
            "fpr": [round(float(value), 4) for value in fpr],
            "tpr": [round(float(value), 4) for value in tpr],
            "auc": metrics_tuned["roc_auc"],
        },
        "feature_importance": feature_importance,
        "columns": {
            "raw_numeric": RAW_NUMERIC_COLUMNS,
            "raw_categorical": CATEGORICAL_COLUMNS,
            "engineered": ENGINEERED_COLUMNS,
            "target": TARGET_COLUMN,
        },
    }

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    # 8. Laporan -------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"MODEL TERPILIH: {best_name}")
    print("=" * 74)
    print(f"Parameter terbaik     : {best_row['best_params']}")
    print(f"Ambang optimal (F2)   : {threshold:.2f}")
    print(f"ROC-AUC uji           : {metrics_tuned['roc_auc']:.4f}")
    print(
        f"Recall uji            : {metrics_tuned['recall']:.4f} "
        f"(ambang 0,50: {metrics_default['recall']:.4f})"
    )
    print(
        f"Precision uji         : {metrics_tuned['precision']:.4f} "
        f"(ambang 0,50: {metrics_default['precision']:.4f})"
    )
    print(f"F1-score uji          : {metrics_tuned['f1']:.4f}")
    print("\nConfusion matrix (ambang optimal):")
    tn, fp, fn, tp = np.array(metrics_tuned["confusion_matrix"]).ravel()
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"\nFitur paling berpengaruh (permutation importance):")
    for item in feature_importance[:5]:
        print(f"  {item['feature']:<24} {item['importance']:.4f}")

    print(f"\nModel disimpan ke    : {MODEL_PATH.relative_to(MODEL_DIR.parent)}")
    print(f"Metadata disimpan ke : {METADATA_PATH.relative_to(MODEL_DIR.parent)}")
    print(f"Perbandingan model   : {COMPARISON_PATH.relative_to(MODEL_DIR.parent)}")

    _write_markdown_report(metadata, comparison)


def _write_markdown_report(
    metadata: dict[str, Any], comparison: list[dict[str, Any]]
) -> None:
    """Tulis laporan ringkas yang siap ditempel ke dokumentasi."""
    best = metadata["best_model"]
    cm = metadata["confusion_matrix"]["threshold_tuned"]
    tn, fp, fn, tp = np.array(cm).ravel()

    lines = [
        "# Laporan model skrining risiko gagal jantung",
        "",
        f"Dibuat: {metadata['generated_at']}  ",
        f"Dataset: {metadata['dataset']['n_rows']} rekam, "
        f"{metadata['dataset']['n_features_raw']} fitur mentah  ",
        f"Split: {metadata['split']['n_train']} latih / "
        f"{metadata['split']['n_test']} uji (stratified)",
        "",
        "## Perbandingan model",
        "",
        "| Model | CV ROC-AUC | CV Recall | Test ROC-AUC | Recall | Precision | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {row['cv_roc_auc']:.4f} | {row['cv_recall']:.4f} "
            f"| {row['test_roc_auc']:.4f} | {row['test_recall']:.4f} "
            f"| {row['test_precision']:.4f} | {row['test_f1']:.4f} |"
        )

    lines += [
        "",
        "## Model terpilih",
        "",
        f"**{best['name']}** dengan ROC-AUC uji {best['test_roc_auc']:.4f} "
        f"dan Recall {best['test_recall']:.4f} pada ambang "
        f"{metadata['threshold']['value']:.2f}.",
        "",
        f"Kriteria pemilihan: {metadata['selection_rule']}",
        "",
        "## Confusion matrix (ambang optimal)",
        "",
        "| | Prediksi negatif | Prediksi positif |",
        "| --- | ---: | ---: |",
        f"| Aktual negatif | {tn} | {fp} |",
        f"| Aktual positif | {fn} | {tp} |",
        "",
        f"- *False negative*: **{fn}** pasien berisiko yang terlewat.",
        f"- *False positive*: **{fp}** pasien tanpa risiko yang dirujuk berlebih.",
        "",
        "## Fitur paling berpengaruh",
        "",
        "| Fitur | Importance (ROC-AUC) |",
        "| --- | ---: |",
    ]
    for item in metadata["feature_importance"][:10]:
        lines.append(f"| {item['feature']} | {item['importance']:.4f} |")

    lines += [
        "",
        "## Catatan metodologis",
        "",
        "- Pembersihan nilai 0 (kolesterol, tekanan darah) dan *feature "
        "engineering* berada di dalam pipeline, sehingga hanya statistik "
        "data latih yang dipakai pada tiap fold.",
        "- Ambang keputusan dipilih dari prediksi *out-of-fold* dengan "
        "memaksimalkan F2-score, bukan dari data uji.",
        "- Data uji hanya dipakai sekali untuk pelaporan akhir.",
        "",
    ]

    path = REPORT_DIR / "model_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Laporan markdown     : {path.relative_to(MODEL_DIR.parent)}")


if __name__ == "__main__":
    main()
