# Skrining Dini Risiko Gagal Jantung

Purwarupa sistem pendukung keputusan berbasis *machine learning* untuk
skrining awal risiko penyakit jantung. Dibangun mengikuti alur **CRISP-DM**
dengan dataset publik *Heart Failure Prediction* (918 rekam klinis).

Aplikasi Streamlit memuat model terbaik dan menerjemahkan probabilitas model
menjadi keputusan skrining pada ambang yang dapat disesuaikan.

---

## Ringkasan hasil

| Algoritma | CV ROC-AUC | CV Recall | Test ROC-AUC | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Random Forest (Ensemble)** | **0,9306** | 0,8819 | 0,9219 | 0,8922 | 0,8667 | 0,8792 |
| Support Vector Machine (SVM) | 0,9195 | 0,8843 | 0,9382 | 0,8725 | 0,8725 | 0,8725 |
| K-Nearest Neighbors (KNN) | 0,9115 | 0,8547 | 0,9333 | 0,8922 | 0,9192 | 0,9055 |
| Decision Tree | 0,9018 | 0,8845 | 0,8828 | 0,7941 | 0,8526 | 0,8223 |

Model dipilih berdasarkan **ROC-AUC cross-validation** (bukan data uji).
Random Forest terpilih dengan parameter `max_depth=12`, `min_samples_leaf=5`,
`n_estimators=200`.

Setelah ambang keputusan disetel ke **0,23** (memaksimalkan F2-score pada
prediksi *out-of-fold*), performa pada data uji (184 rekam) menjadi:

| Metrik | Ambang 0,50 | Ambang optimal 0,23 |
| --- | ---: | ---: |
| ROC-AUC | 0,9219 | 0,9219 |
| Recall | 0,8922 | **0,9608** |
| Precision | 0,8667 | 0,7481 |
| F1-score | 0,8792 | 0,8412 |

Confusion matrix pada ambang optimal:

|  | Prediksi negatif | Prediksi positif |
| --- | ---: | ---: |
| **Aktual negatif** | 49 | 33 |
| **Aktual positif** | **4** | 98 |

Menurunkan ambang dari 0,50 ke 0,23 memangkas *false negative* dari 11 menjadi
**4** pasien, dengan konsekuensi 33 *false positive*. Trade-off ini disengaja:
pada skrining, melewatkan pasien berisiko lebih mahal daripada merujuk pasien
yang sehat untuk pemeriksaan lanjutan.

---

## Struktur proyek

```
Heart_Failure_ML/
├── app.py                  # Aplikasi Streamlit (skrining, performa, insight data)
├── train_model.py          # Pelatihan + GridSearchCV + penyimpanan artefak
├── eda.py                  # Analisis eksplorasi data -> reports/
├── utils.py                # Pembersihan, feature engineering, pipeline, artefak
├── requirements.txt
├── .streamlit/config.toml  # Tema aplikasi (terang & gelap)
├── data/heart.csv          # Dataset mentah (918 x 12)
├── models/
│   ├── best_heart_model.joblib
│   ├── metadata.json       # Metrik, ambang, parameter, importance
│   └── model_comparison.csv
├── reports/
│   ├── eda_summary.md
│   ├── model_report.md
│   └── figures/            # 5 gambar EDA (PNG)
└── tests/test_pipeline.py  # 13 smoke test (unittest)
```

---

## Instalasi

Lingkungan virtual sudah tersedia di `venv/`. Bila perlu dibuat ulang:

```powershell
py -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Menjalankan

```powershell
# 1. Analisis eksplorasi data (opsional, menghasilkan reports/)
venv\Scripts\python.exe eda.py

# 2. Latih ulang model (menghasilkan models/)
venv\Scripts\python.exe train_model.py

# 3. Jalankan aplikasi
venv\Scripts\streamlit.exe run app.py

# 4. Uji
venv\Scripts\python.exe -m unittest discover -s tests -v
```

Aplikasi terbuka di <http://localhost:8501>. Model harus dilatih lebih dulu;
bila `models/` kosong, aplikasi menampilkan instruksi pelatihan.

---

## Alur CRISP-DM

| Tahap | Implementasi |
| --- | --- |
| **Business understanding** | Skrining awal untuk meminimalkan pasien berisiko yang tidak terdeteksi (*false negative*). |
| **Data understanding** | `eda.py`: distribusi target, sebaran fitur, korelasi, proporsi risiko per kategori, anomali nilai nol. |
| **Data preparation** | Nilai 0 pada `Cholesterol`/`RestingBP` → `NaN`; imputasi median; `RobustScaler`; *one-hot encoding*; tiga fitur turunan. |
| **Modeling** | Decision Tree, KNN, SVM, Random Forest + `GridSearchCV` 5-fold stratified. |
| **Evaluation** | ROC-AUC, Recall, Precision, F1, confusion matrix; ambang keputusan disetel dengan F2 pada prediksi *out-of-fold*. |
| **Deployment** | Aplikasi Streamlit dengan tiga tab dan tema terang/gelap. |

### Feature engineering

| Fitur | Rumus | Rasional klinis |
| --- | --- | --- |
| `MaxHR_Deficit` | `(220 − Age) − MaxHR` | Cadangan detak jantung yang tidak terpakai; nilai besar menandakan kapasitas jantung rendah. |
| `HR_Achievement_Ratio` | `MaxHR / (220 − Age)` | Proporsi pencapaian batas detak jantung teoretis. |
| `BP_Age_Ratio` | `RestingBP / Age` | Tekanan darah relatif terhadap usia. |

### Mencegah data leakage

- Pembagian data latih/uji (80:20, stratified) dilakukan **sebelum**
  preprocessing apa pun.
- Pembersihan nilai nol, *feature engineering*, imputasi, dan penskalaan
  berada **di dalam** `Pipeline`, sehingga pada setiap fold hanya statistik
  data latih yang dipakai.
- Pemilihan model memakai ROC-AUC cross-validation; data uji hanya dipakai
  sekali untuk pelaporan akhir.
- Ambang keputusan ditentukan dari prediksi *out-of-fold*, bukan data uji.

---

## Interpretasi model

Permutation importance pada data uji (penurunan ROC-AUC saat fitur diacak):

| Fitur | Importance |
| --- | ---: |
| `ST_Slope` | 0,1155 |
| `ChestPainType` | 0,0223 |
| `Sex` | 0,0212 |
| `MaxHR` | 0,0169 |
| `ExerciseAngina` | 0,0137 |

`ST_Slope` mendominasi, sejalan dengan literatur bahwa kelainan segmen ST
merupakan indikator kuat iskemia.

---

## Batasan

- Dataset bersifat *cross-sectional* dan berasal dari satu sumber; hasil belum
  tentu berlaku pada populasi lain.
- **172 dari 918 rekam (18,7%)** memiliki `Cholesterol = 0` sehingga
  diimputasi median. Kesimpulan yang melibatkan kolesterol perlu dibaca hati-hati.
- Probabilitas model belum dikalibrasi secara eksplisit (mis. Platt scaling
  atau isotonic regression); ambang 0,23 adalah hasil optimasi F2 pada data
  ini, bukan angka universal.
- Aplikasi ini **bukan alat diagnosis**. Keputusan klinis tetap berada pada
  tenaga medis.

---

## Referensi

- Dataset: *Heart Failure Prediction Dataset*, fedesoriano (Kaggle), 918 rekam,
  11 fitur klinis + target `HeartDisease`.
- Pedoman alur kerja: CRISP-DM (Cross Industry Standard Process for Data Mining).
