# Laporan model skrining risiko gagal jantung

Dibuat: 2026-11-24T10:55:19+00:00  
Dataset: 918 rekam, 11 fitur mentah  
Split: 734 latih / 184 uji (stratified)

## Perbandingan model

| Model | CV ROC-AUC | CV Recall | Test ROC-AUC | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest (Ensemble) | 0.9306 | 0.8819 | 0.9219 | 0.8922 | 0.8667 | 0.8792 |
| Support Vector Machine (SVM) | 0.9195 | 0.8843 | 0.9382 | 0.8725 | 0.8725 | 0.8725 |
| K-Nearest Neighbors (KNN) | 0.9115 | 0.8547 | 0.9333 | 0.8922 | 0.9192 | 0.9055 |
| Decision Tree | 0.9018 | 0.8845 | 0.8828 | 0.7941 | 0.8526 | 0.8223 |

## Model terpilih

**Random Forest (Ensemble)** dengan ROC-AUC uji 0.9219 dan Recall 0.9608 pada ambang 0.23.

Kriteria pemilihan: ROC-AUC cross-validation tertinggi; Recall cross-validation sebagai penentu bila seri. Data uji hanya untuk pelaporan.

## Confusion matrix (ambang optimal)

| | Prediksi negatif | Prediksi positif |
| --- | ---: | ---: |
| Aktual negatif | 49 | 33 |
| Aktual positif | 4 | 98 |

- *False negative*: **4** pasien berisiko yang terlewat.
- *False positive*: **33** pasien tanpa risiko yang dirujuk berlebih.

## Fitur paling berpengaruh

| Fitur | Importance (ROC-AUC) |
| --- | ---: |
| ST_Slope | 0.1155 |
| ChestPainType | 0.0223 |
| Sex | 0.0212 |
| MaxHR | 0.0169 |
| ExerciseAngina | 0.0137 |
| Oldpeak | 0.0103 |
| RestingBP | 0.0060 |
| FastingBS | 0.0049 |
| Cholesterol | 0.0032 |
| RestingECG | 0.0021 |

## Catatan metodologis

- Pembersihan nilai 0 (kolesterol, tekanan darah) dan *feature engineering* berada di dalam pipeline, sehingga hanya statistik data latih yang dipakai pada tiap fold.
- Ambang keputusan dipilih dari prediksi *out-of-fold* dengan memaksimalkan F2-score, bukan dari data uji.
- Data uji hanya dipakai sekali untuk pelaporan akhir.
