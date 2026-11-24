"""Smoke test untuk pipeline, artefak model, dan logika presentasi.

Jalankan dengan::

    venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (  # noqa: E402  (import setelah sys.path disiapkan)
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    add_engineered_features,
    build_pipeline,
    clean_clinical_zeros,
    format_probability,
    load_data,
    preprocess_and_engineer,
    risk_level,
    split_features_target,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [50.0, 60.0],
            "Sex": ["M", "F"],
            "ChestPainType": ["ASY", "ATA"],
            "RestingBP": [0.0, 140.0],
            "Cholesterol": [0.0, 240.0],
            "FastingBS": [0, 1],
            "RestingECG": ["Normal", "ST"],
            "MaxHR": [140.0, 150.0],
            "ExerciseAngina": ["N", "Y"],
            "Oldpeak": [1.0, 2.0],
            "ST_Slope": ["Up", "Flat"],
            TARGET_COLUMN: [0, 1],
        }
    )


class CleaningTests(unittest.TestCase):
    def test_zero_values_become_missing(self) -> None:
        cleaned = clean_clinical_zeros(sample_frame())

        self.assertTrue(np.isnan(cleaned.loc[0, "Cholesterol"]))
        self.assertTrue(np.isnan(cleaned.loc[0, "RestingBP"]))
        self.assertEqual(cleaned.loc[1, "Cholesterol"], 240.0)
        self.assertEqual(cleaned.loc[1, "RestingBP"], 140.0)

    def test_clean_does_not_mutate_input(self) -> None:
        original = sample_frame()
        clean_clinical_zeros(original)
        self.assertEqual(original.loc[0, "Cholesterol"], 0.0)


class FeatureEngineeringTests(unittest.TestCase):
    def test_heart_rate_reserve(self) -> None:
        engineered = add_engineered_features(clean_clinical_zeros(sample_frame()))

        # 220 - 50 = 170; MaxHR 140 -> defisit 30
        self.assertAlmostEqual(engineered.loc[0, "MaxHR_Deficit"], 30.0)
        self.assertAlmostEqual(engineered.loc[0, "HR_Achievement_Ratio"], 140.0 / 170.0)

    def test_blood_pressure_ratio_uses_measured_value(self) -> None:
        engineered = add_engineered_features(clean_clinical_zeros(sample_frame()))

        self.assertTrue(np.isnan(engineered.loc[0, "BP_Age_Ratio"]))
        self.assertAlmostEqual(engineered.loc[1, "BP_Age_Ratio"], 140.0 / 60.0)

    def test_preprocess_keeps_all_feature_columns(self) -> None:
        processed = preprocess_and_engineer(sample_frame())

        for column in FEATURE_COLUMNS:
            self.assertIn(column, processed.columns)
        self.assertEqual(len(processed), 2)


class PipelineTests(unittest.TestCase):
    def test_pipeline_fits_and_predicts_probabilities(self) -> None:
        pipeline = build_pipeline(DecisionTreeClassifier(random_state=0, max_depth=2))
        frame = sample_frame()
        X, y = split_features_target(frame)

        pipeline.fit(X, y)
        probabilities = pipeline.predict_proba(X)[:, 1]

        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))


class PresentationTests(unittest.TestCase):
    def test_risk_level_thresholds(self) -> None:
        self.assertEqual(risk_level(0.80, threshold=0.50), "tinggi")
        self.assertEqual(risk_level(0.40, threshold=0.50), "sedang")
        self.assertEqual(risk_level(0.10, threshold=0.50), "rendah")

    def test_positive_decision_is_never_labelled_low(self) -> None:
        # Ambang rendah (hasil tuning F2) tetap harus menghasilkan minimal "sedang".
        self.assertEqual(risk_level(0.25, threshold=0.20), "sedang")

    def test_high_probability_stays_high_when_threshold_raised(self) -> None:
        self.assertEqual(risk_level(0.70, threshold=0.50), "tinggi")
        self.assertEqual(risk_level(0.70, threshold=0.90), "sedang")

    def test_probability_format_uses_indonesian_comma(self) -> None:
        self.assertEqual(format_probability(0.7532), "75,3%")


class DatasetTests(unittest.TestCase):
    def test_dataset_shape_and_columns(self) -> None:
        df = load_data()

        self.assertEqual(len(df), 918)
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
            self.assertIn(column, df.columns)
        self.assertTrue(set(df[TARGET_COLUMN].unique()) <= {0, 1})


@unittest.skipUnless(
    MODEL_PATH.exists() and METADATA_PATH.exists(),
    "Model belum dilatih; jalankan `python train_model.py`.",
)
class ArtifactTests(unittest.TestCase):
    def test_saved_model_accepts_raw_patient_row(self) -> None:
        import joblib

        model = joblib.load(MODEL_PATH)
        frame = sample_frame().drop(columns=[TARGET_COLUMN])

        probabilities = model.predict_proba(frame)[:, 1]

        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_metadata_contains_required_sections(self) -> None:
        import json

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        for key in ("best_model", "threshold", "confusion_matrix", "roc_curve"):
            self.assertIn(key, metadata)
        self.assertGreater(metadata["threshold"]["value"], 0.0)
        self.assertLess(metadata["threshold"]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
