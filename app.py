"""Aplikasi Streamlit: purwarupa skrining dini risiko gagal jantung.

Menampilkan tiga bagian:
1. Formulir skrining pasien beserta probabilitas risiko.
2. Performa dan interpretasi model yang tersimpan.
3. Insight dataset yang dipakai untuk pelatihan.

Jalankan dengan::

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import altair as alt
import streamlit as st

from utils import (
    CATEGORICAL_COLUMNS,
    CHEST_PAIN_LABELS,
    COMPARISON_PATH,
    ENGINEERED_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    RAW_NUMERIC_COLUMNS,
    RESTING_ECG_LABELS,
    SEX_LABELS,
    ST_SLOPE_LABELS,
    TARGET_COLUMN,
    TARGET_LABELS,
    YES_NO_LABELS,
    add_engineered_features,
    clean_clinical_zeros,
    format_probability,
    load_artifacts,
    load_data,
    preprocess_and_engineer,
    risk_level,
)

st.set_page_config(
    page_title="Skrining risiko gagal jantung",
    page_icon="🫀",
    layout="wide",
)

RISK_COLORS = {"tinggi": "#dc2626", "sedang": "#d97706", "rendah": "#15803d"}
RISK_TITLES = {
    "tinggi": "Risiko tinggi terindikasi",
    "sedang": "Risiko menengah",
    "rendah": "Risiko rendah",
}
RECOMMENDATIONS = {
    "tinggi": [
        "Rujuk pasien untuk evaluasi kardiologi dan pemeriksaan lanjutan "
        "(EKG 12 sadapan, ekokardiografi, atau uji latih sesuai indikasi).",
        "Periksa kembali faktor risiko yang dapat dimodifikasi: tekanan darah, "
        "profil lipid, gula darah, dan kebiasaan merokok.",
    ],
    "sedang": [
        "Pertimbangkan pemeriksaan penunjang dan pantau gejala dalam 2–4 minggu.",
        "Diskusikan modifikasi gaya hidup serta kendali faktor risiko "
        "kardiovaskular.",
    ],
    "rendah": [
        "Lanjutkan pemantauan rutin dan edukasi gaya hidup sehat.",
        "Ulangi skrining bila muncul gejala baru atau faktor risiko berubah.",
    ],
}


# ---------------------------------------------------------------------------
# Sumber daya (di-cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_and_metadata():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner=False)
def get_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)


# ---------------------------------------------------------------------------
# Komponen visual
# ---------------------------------------------------------------------------
def probability_gauge(probability: float, level: str) -> dict:
    """Spec ECharts untuk gauge probabilitas (tanpa JavaScript)."""
    return {
        "series": [
            {
                "type": "gauge",
                "min": 0,
                "max": 100,
                "startAngle": 210,
                "endAngle": -30,
                "radius": "95%",
                "progress": {
                    "show": True,
                    "width": 16,
                    "itemStyle": {"color": RISK_COLORS[level]},
                },
                "axisLine": {
                    "lineStyle": {
                        "width": 16,
                        "color": [[1, "rgba(148, 163, 184, 0.28)"]],
                    }
                },
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"show": False},
                "pointer": {"show": False},
                "anchor": {"show": False},
                "title": {
                    "show": True,
                    "offsetCenter": [0, "62%"],
                    "fontSize": 13,
                    "color": "#64748b",
                },
                "detail": {
                    "valueAnimation": True,
                    "offsetCenter": [0, "6%"],
                    "fontSize": 34,
                    "fontWeight": "bold",
                    "color": RISK_COLORS[level],
                    "formatter": "{value}%",
                },
                "data": [
                    {
                        "value": round(probability * 100, 1),
                        "name": "Probabilitas risiko",
                    }
                ],
            }
        ]
    }


def render_result(probability: float, threshold: float, payload: pd.DataFrame) -> None:
    """Menampilkan kartu hasil skrining."""
    level = risk_level(probability, threshold)
    above_threshold = probability >= threshold
    decision_label = "Rujuk" if above_threshold else "Pantau"

    with st.container(border=True):
        gauge_col, text_col = st.columns([1, 2], vertical_alignment="center")

        with gauge_col:
            st.echarts_chart(
                probability_gauge(probability, level),
                height=210,
                alt=(
                    f"Gauge probabilitas risiko gagal jantung "
                    f"{format_probability(probability)}"
                ),
            )

        with text_col:
            st.subheader(RISK_TITLES[level])
            st.markdown(
                f"Probabilitas **{format_probability(probability)}** "
                f"dibandingkan ambang keputusan **{format_probability(threshold)}**."
            )

            if level == "tinggi":
                st.error(
                    f"Probabilitas {format_probability(probability)} melampaui "
                    f"ambang keputusan {format_probability(threshold)}.",
                    title="Perlu evaluasi kardiologi",
                    icon=":material/priority_high:",
                )
            elif above_threshold:
                st.warning(
                    f"Probabilitas {format_probability(probability)} berada di atas "
                    f"ambang skrining {format_probability(threshold)}.",
                    title="Melewati ambang skrining",
                    icon=":material/visibility:",
                )
            elif level == "sedang":
                st.warning(
                    f"Probabilitas {format_probability(probability)} berada di bawah "
                    f"ambang {format_probability(threshold)}, tetapi belum dapat "
                    "disebut rendah.",
                    title="Di bawah ambang, tetap perlu dipantau",
                    icon=":material/monitor_heart:",
                )
            else:
                st.success(
                    f"Probabilitas {format_probability(probability)} berada di bawah "
                    f"ambang skrining {format_probability(threshold)}.",
                    title="Lanjutkan pemantauan rutin",
                    icon=":material/check_circle:",
                )

            for item in RECOMMENDATIONS[level]:
                st.markdown(f"- {item}")

        with st.container(horizontal=True):
            st.metric("Probabilitas risiko", format_probability(probability), border=True)
            st.metric("Ambang keputusan", format_probability(threshold), border=True)
            st.metric("Status skrining", decision_label, border=True)
            st.metric("Tingkat risiko", level.capitalize(), border=True)

        with st.expander("Rincian fitur turunan"):
            engineered = preprocess_and_engineer(payload).iloc[0]
            age = float(payload.iloc[0]["Age"])
            max_hr = float(payload.iloc[0]["MaxHR"])
            resting_bp = payload.iloc[0]["RestingBP"]

            def number(value: float, unit: str, digits: int = 1) -> str:
                if pd.isna(value):
                    return "tidak diukur"
                return f"{value:.{digits}f} {unit}".strip()

            details = pd.DataFrame(
                [
                    {
                        "Fitur": "Detak jantung maksimum teoretis",
                        "Nilai": f"{220 - age:.0f} bpm",
                        "Keterangan": "220 − usia",
                    },
                    {
                        "Fitur": "MaxHR_Deficit",
                        "Nilai": number(engineered["MaxHR_Deficit"], "bpm"),
                        "Keterangan": "Semakin besar, cadangan detak jantung makin rendah",
                    },
                    {
                        "Fitur": "HR_Achievement_Ratio",
                        "Nilai": number(engineered["HR_Achievement_Ratio"], "", 3),
                        "Keterangan": f"MaxHR terukur {max_hr:.0f} bpm / batas teoretis",
                    },
                    {
                        "Fitur": "BP_Age_Ratio",
                        "Nilai": number(engineered["BP_Age_Ratio"], "", 3),
                        "Keterangan": (
                            "Tekanan darah istirahat dibagi usia"
                            if resting_bp != 0
                            else "Tekanan darah tidak diukur"
                        ),
                    },
                ]
            )
            st.dataframe(details, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Bagian 1 — Skrining pasien
# ---------------------------------------------------------------------------
def screening_section(threshold: float) -> None:
    st.markdown(
        "Isi parameter klinis pasien, lalu pilih **Analisis risiko**. "
        "Nilai 0 pada kolesterol atau tekanan darah diartikan sebagai "
        "pemeriksaan yang belum dilakukan dan akan diimputasi otomatis."
    )

    with st.form("screening_form", border=True):
        left, right = st.columns(2)

        with left:
            st.markdown("**Data demografi dan gejala**")
            age = st.number_input(
                "Usia (tahun)", min_value=18, max_value=100, value=54, step=1
            )
            sex = st.segmented_control(
                "Jenis kelamin",
                options=["M", "F"],
                default="M",
                required=True,
                format_func=lambda value: SEX_LABELS.get(value, value),
                width="stretch",
            )
            chest_pain = st.selectbox(
                "Tipe nyeri dada",
                options=list(CHEST_PAIN_LABELS),
                format_func=lambda value: CHEST_PAIN_LABELS.get(value, value),
                help="ASY sering dijumpai pada pasien tanpa keluhan namun berisiko.",
            )
            exercise_angina = st.segmented_control(
                "Angina saat olahraga",
                options=["N", "Y"],
                default="N",
                required=True,
                format_func=lambda value: YES_NO_LABELS.get(value, value),
                width="stretch",
            )
            oldpeak = st.number_input(
                "Oldpeak — depresi segmen ST",
                min_value=-3.0,
                max_value=7.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                help="Selisih depresi ST akibat latihan, dalam mm.",
            )

        with right:
            st.markdown("**Hasil pemeriksaan**")
            resting_bp = st.number_input(
                "Tekanan darah istirahat (mm Hg)",
                min_value=0,
                max_value=250,
                value=130,
                step=1,
                help="Isi 0 bila belum diukur.",
            )
            cholesterol = st.number_input(
                "Kolesterol serum (mg/dl)",
                min_value=0,
                max_value=650,
                value=220,
                step=1,
                help="Isi 0 bila belum diukur.",
            )
            fasting_bs = st.segmented_control(
                "Gula darah puasa > 120 mg/dl",
                options=[0, 1],
                default=0,
                required=True,
                format_func=lambda value: "Ya" if value == 1 else "Tidak",
                width="stretch",
            )
            max_hr = st.number_input(
                "Detak jantung maksimum (bpm)",
                min_value=60,
                max_value=220,
                value=140,
                step=1,
            )
            resting_ecg = st.selectbox(
                "Hasil EKG istirahat",
                options=list(RESTING_ECG_LABELS),
                format_func=lambda value: RESTING_ECG_LABELS.get(value, value),
            )
            st_slope = st.selectbox(
                "Kemiringan segmen ST",
                options=list(ST_SLOPE_LABELS),
                format_func=lambda value: ST_SLOPE_LABELS.get(value, value),
            )

        submitted = st.form_submit_button(
            "Analisis risiko",
            icon=":material/ecg_heart:",
            type="primary",
            width="stretch",
        )

    if submitted:
        payload = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Sex": sex,
                    "ChestPainType": chest_pain,
                    "RestingBP": resting_bp,
                    "Cholesterol": cholesterol,
                    "FastingBS": fasting_bs,
                    "RestingECG": resting_ecg,
                    "MaxHR": max_hr,
                    "ExerciseAngina": exercise_angina,
                    "Oldpeak": oldpeak,
                    "ST_Slope": st_slope,
                }
            ]
        )
        try:
            model, _ = get_model_and_metadata()
            probability = float(model.predict_proba(payload)[0, 1])
        except Exception as error:  # noqa: BLE001 - tampilkan pesan ramah
            st.error(f"Gagal menghitung prediksi: {error}")
        else:
            st.session_state["screening"] = {
                "probability": probability,
                "payload": payload,
            }

    result = st.session_state.get("screening")
    if result is None:
        st.info(
            "Belum ada hasil. Lengkapi formulir di atas untuk memulai skrining.",
            icon=":material/info:",
        )
    else:
        render_result(result["probability"], threshold, result["payload"])
        st.caption(
            "Hasil ini adalah estimasi statistik dari data historis, bukan "
            "diagnosis. Keputusan klinis tetap berada pada tenaga medis."
        )


# ---------------------------------------------------------------------------
# Bagian 2 — Performa model
# ---------------------------------------------------------------------------
def performance_section(metadata: dict, comparison: pd.DataFrame) -> None:
    best = metadata["best_model"]
    st.markdown(
        f"Model terpilih: **{best['name']}**. Metrik di bawah dihitung pada "
        f"{metadata['split']['n_test']} rekam data uji yang tidak dipakai saat "
        "pelatihan maupun pemilihan model."
    )

    with st.container(horizontal=True):
        st.metric("ROC-AUC", f"{best['test_roc_auc']:.3f}", border=True)
        st.metric("Recall", f"{best['test_recall']:.3f}", border=True)
        st.metric("Precision", f"{best['test_precision']:.3f}", border=True)
        st.metric("F1-score", f"{best['test_f1']:.3f}", border=True)

    st.caption(
        f"Recall diukur pada ambang keputusan {metadata['threshold']['value']:.2f} "
        f"(F2-optimal). Pada ambang standar 0,50, recall bernilai "
        f"{best['test_recall_default_threshold']:.3f}."
    )

    chart_col, table_col = st.columns([1.15, 1])

    with chart_col:
        with st.container(border=True):
            st.subheader("Perbandingan empat algoritma")
            labels = {
                "cv_roc_auc": "ROC-AUC (CV)",
                "test_roc_auc": "ROC-AUC (uji)",
                "test_recall": "Recall (uji)",
            }
            order = comparison.sort_values("test_roc_auc")["model"].tolist()
            melted = comparison.melt(
                id_vars="model",
                value_vars=list(labels),
                var_name="metrik",
                value_name="skor",
            )
            melted["metrik"] = melted["metrik"].map(labels)
            chart = (
                alt.Chart(melted)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X(
                        "skor:Q",
                        title="Skor",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y("model:N", title=None, sort=order),
                    yOffset=alt.YOffset("metrik:N", title=None),
                    color=alt.Color("metrik:N", title=None),
                    tooltip=[
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("metrik:N", title="Metrik"),
                        alt.Tooltip("skor:Q", title="Skor", format=".3f"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(
                chart, alt="ROC-AUC dan recall tiap algoritma pada data uji"
            )

    with table_col:
        with st.container(border=True):
            st.subheader("Tabel metrik")
            display = comparison[
                [
                    "model",
                    "cv_roc_auc",
                    "test_roc_auc",
                    "test_recall",
                    "test_precision",
                    "test_f1",
                ]
            ].rename(
                columns={
                    "model": "Model",
                    "cv_roc_auc": "CV AUC",
                    "test_roc_auc": "Test AUC",
                    "test_recall": "Recall",
                    "test_precision": "Precision",
                    "test_f1": "F1",
                }
            )
            st.dataframe(
                display,
                hide_index=True,
                width="stretch",
                column_config={
                    column: st.column_config.NumberColumn(format="%.3f")
                    for column in ["CV AUC", "Test AUC", "Recall", "Precision", "F1"]
                },
            )
            st.caption(
                "Model dipilih berdasarkan ROC-AUC cross-validation; "
                "data uji hanya untuk pelaporan."
            )

    roc_col, cm_col = st.columns(2)

    with roc_col:
        with st.container(border=True):
            st.subheader("Kurva ROC — data uji")
            roc = pd.DataFrame(metadata["roc_curve"])
            diagonal = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
            roc_line = (
                alt.Chart(roc)
                .mark_line(strokeWidth=2.5, color="#b91c1c")
                .encode(
                    x=alt.X(
                        "fpr:Q",
                        title="False positive rate",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y(
                        "tpr:Q",
                        title="True positive rate",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=[
                        alt.Tooltip("fpr:Q", format=".3f"),
                        alt.Tooltip("tpr:Q", format=".3f"),
                    ],
                )
            )
            baseline = (
                alt.Chart(diagonal)
                .mark_line(strokeDash=[5, 5], color="#94a3b8")
                .encode(x="fpr:Q", y="tpr:Q")
            )
            st.altair_chart(
                (baseline + roc_line).properties(height=280),
                alt=f"Kurva ROC data uji dengan AUC {metadata['roc_curve']['auc']:.3f}",
            )

    with cm_col:
        with st.container(border=True):
            st.subheader("Confusion matrix")
            choice = st.segmented_control(
                "Ambang keputusan",
                options=["Optimal (F2)", "Standar 0,50"],
                default="Optimal (F2)",
                key="cm_threshold_choice",
            )
            key = "threshold_tuned" if choice == "Optimal (F2)" else "threshold_default"
            matrix = metadata["confusion_matrix"][key]
            (tn, fp), (fn, tp) = matrix
            cells = pd.DataFrame(
                {
                    "aktual": ["Negatif", "Negatif", "Positif", "Positif"],
                    "prediksi": ["Negatif", "Positif", "Negatif", "Positif"],
                    "jumlah": [tn, fp, fn, tp],
                }
            )
            heatmap = (
                alt.Chart(cells)
                .mark_rect(cornerRadius=4)
                .encode(
                    x=alt.X("prediksi:N", title="Prediksi"),
                    y=alt.Y("aktual:N", title="Aktual"),
                    color=alt.Color(
                        "jumlah:Q", title="Jumlah", scale=alt.Scale(scheme="reds")
                    ),
                    tooltip=["aktual:N", "prediksi:N", "jumlah:Q"],
                )
            )
            text = (
                alt.Chart(cells)
                .mark_text(fontSize=18, fontWeight="bold")
                .encode(
                    x="prediksi:N",
                    y="aktual:N",
                    text="jumlah:Q",
                    color=alt.condition(
                        alt.datum.jumlah > (tn + fp + fn + tp) / 3,
                        alt.value("white"),
                        alt.value("#7f1d1d"),
                    ),
                )
            )
            st.altair_chart(
                (heatmap + text).properties(height=230),
                alt="Confusion matrix model terpilih pada data uji",
            )
            st.caption(f"False negative: **{fn}** pasien berisiko yang terlewat.")

    with st.container(border=True):
        st.subheader("Fitur paling berpengaruh")
        importance = pd.DataFrame(metadata.get("feature_importance", []))
        if importance.empty:
            st.info("Data permutation importance tidak tersedia.")
        else:
            top = importance.sort_values("importance").tail(10)
            chart = (
                alt.Chart(top)
                .mark_bar(cornerRadiusEnd=3, color="#0f766e")
                .encode(
                    x=alt.X("importance:Q", title="Penurunan ROC-AUC saat fitur diacak"),
                    y=alt.Y("feature:N", title=None, sort="-x"),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Fitur"),
                        alt.Tooltip("importance:Q", title="Importance", format=".4f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(
                chart, alt="Sepuluh fitur dengan pengaruh terbesar pada model"
            )
            st.caption(
                "Permutation importance mengukur penurunan ROC-AUC ketika nilai "
                "sebuah fitur diacak pada data uji."
            )


# ---------------------------------------------------------------------------
# Bagian 3 — Insight dataset
# ---------------------------------------------------------------------------
def dataset_section() -> None:
    df = get_dataset()
    clean = add_engineered_features(clean_clinical_zeros(df))

    with st.container(horizontal=True):
        st.metric("Jumlah rekam", f"{len(df):,}".replace(",", "."), border=True)
        st.metric("Fitur klinis", df.shape[1] - 1, border=True)
        st.metric(
            "Prevalensi penyakit", format_probability(float(df[TARGET_COLUMN].mean())),
            border=True,
        )
        st.metric(
            "Kolesterol = 0",
            int((df["Cholesterol"] == 0).sum()),
            border=True,
            help="Nilai tidak valid secara medis; diimputasi di dalam pipeline.",
        )

    balance_col, zero_col = st.columns(2)

    with balance_col:
        with st.container(border=True):
            st.subheader("Keseimbangan kelas target")
            counts = (
                df[TARGET_COLUMN]
                .map(TARGET_LABELS)
                .value_counts()
                .rename_axis("status")
                .reset_index(name="jumlah")
            )
            chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("status:N", title=None),
                    y=alt.Y("jumlah:Q", title="Jumlah rekam"),
                    color=alt.Color(
                        "status:N",
                        title=None,
                        scale=alt.Scale(
                            domain=list(TARGET_LABELS.values()),
                            range=["#64748b", "#b91c1c"],
                        ),
                    ),
                    tooltip=["status:N", "jumlah:Q"],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, alt="Distribusi jumlah pasien berisiko dan tidak")

    with zero_col:
        with st.container(border=True):
            st.subheader("Anomali nilai nol")
            anomalies = pd.DataFrame(
                {
                    "fitur": ["Cholesterol", "RestingBP"],
                    "jumlah": [
                        int((df["Cholesterol"] == 0).sum()),
                        int((df["RestingBP"] == 0).sum()),
                    ],
                }
            )
            chart = (
                alt.Chart(anomalies)
                .mark_bar(cornerRadiusEnd=4, color="#b45309")
                .encode(
                    x=alt.X("jumlah:Q", title="Jumlah baris"),
                    y=alt.Y("fitur:N", title=None),
                    tooltip=["fitur:N", "jumlah:Q"],
                )
                .properties(height=240)
            )
            st.altair_chart(
                chart, alt="Jumlah nilai nol tidak valid pada kolesterol dan tekanan darah"
            )
            st.caption(
                "Nilai 0 diubah menjadi data hilang lalu diimputasi median dari "
                "data latih di dalam pipeline."
            )

    dist_col, cat_col = st.columns(2)

    with dist_col:
        with st.container(border=True):
            st.subheader("Sebaran fitur numerik")
            column = st.selectbox(
                "Pilih fitur",
                options=RAW_NUMERIC_COLUMNS,
                key="distribution_feature",
            )
            source = clean.dropna(subset=[column]).assign(
                status=lambda frame: frame[TARGET_COLUMN].map(TARGET_LABELS)
            )
            chart = (
                alt.Chart(source)
                .mark_bar(opacity=0.65)
                .encode(
                    x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=28), title=column),
                    y=alt.Y("count():Q", title="Jumlah rekam", stack=None),
                    color=alt.Color(
                        "status:N",
                        title=None,
                        scale=alt.Scale(
                            domain=list(TARGET_LABELS.values()),
                            range=["#64748b", "#b91c1c"],
                        ),
                    ),
                    tooltip=[alt.Tooltip("count():Q", title="Jumlah")],
                )
                .properties(height=260)
            )
            st.altair_chart(
                chart, alt=f"Histogram {column} menurut status penyakit"
            )

    with cat_col:
        with st.container(border=True):
            st.subheader("Proporsi risiko per kategori")
            categorical = st.selectbox(
                "Pilih fitur kategorikal",
                options=CATEGORICAL_COLUMNS,
                key="categorical_feature",
            )
            rates = (
                df.groupby(categorical, as_index=False)[TARGET_COLUMN]
                .mean()
                .sort_values(TARGET_COLUMN, ascending=False)
            )
            chart = (
                alt.Chart(rates)
                .mark_bar(cornerRadiusEnd=4, color="#b91c1c")
                .encode(
                    x=alt.X(f"{categorical}:N", title=None, sort="-y"),
                    y=alt.Y(
                        f"{TARGET_COLUMN}:Q",
                        title="Proporsi berisiko",
                        axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=[
                        alt.Tooltip(f"{categorical}:N", title=categorical),
                        alt.Tooltip(
                            f"{TARGET_COLUMN}:Q", title="Proporsi", format=".1%"
                        ),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(
                chart, alt=f"Proporsi pasien berisiko menurut {categorical}"
            )

    with st.container(border=True):
        st.subheader("Korelasi fitur numerik")
        columns = RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS
        corr = clean[columns + [TARGET_COLUMN]].corr(numeric_only=True).reset_index()
        corr = corr.melt(id_vars="index", var_name="kolom", value_name="korelasi")
        chart = (
            alt.Chart(corr)
            .mark_rect()
            .encode(
                x=alt.X("index:N", title=None),
                y=alt.Y("kolom:N", title=None),
                color=alt.Color(
                    "korelasi:Q",
                    title="Korelasi",
                    scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                ),
                tooltip=[
                    alt.Tooltip("index:N", title="Fitur"),
                    alt.Tooltip("kolom:N", title="Dengan"),
                    alt.Tooltip("korelasi:Q", title="Korelasi", format=".2f"),
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, alt="Matriks korelasi fitur numerik dan target")

    with st.expander("Pratinjau data mentah"):
        st.dataframe(df.head(50), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Kerangka aplikasi
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Skrining dini risiko gagal jantung")
    st.caption(
        "Purwarupa pendukung keputusan berbasis Machine Learning · "
        "dataset Heart Failure Prediction (918 rekam)"
    )

    try:
        model, metadata = get_model_and_metadata()
    except FileNotFoundError as error:
        st.error(str(error), title="Model belum tersedia")
        st.code(
            "venv\\Scripts\\python.exe train_model.py\n"
            "venv\\Scripts\\streamlit.exe run app.py",
            language="powershell",
        )
        st.stop()

    best = metadata["best_model"]
    tuned_threshold = float(metadata["threshold"]["value"])

    with st.sidebar:
        st.subheader("Pengaturan skrining")
        threshold = st.slider(
            "Ambang keputusan",
            min_value=0.10,
            max_value=0.90,
            value=min(max(tuned_threshold, 0.10), 0.90),
            step=0.01,
            format="%.2f",
            help=(
                "Ambang optimal dipilih dengan memaksimalkan F2-score pada "
                "prediksi out-of-fold. Menurunkan ambang menaikkan recall "
                "(lebih sedikit kasus terlewat) tetapi menambah false positive."
            ),
        )
        st.caption(
            f"Ambang optimal F2: {tuned_threshold:.2f} · "
            f"Recall uji: {best['test_recall']:.3f}"
        )

        st.divider()
        st.subheader("Model aktif")
        st.markdown(f"**{best['name']}**")
        st.metric("ROC-AUC (uji)", f"{best['test_roc_auc']:.3f}")
        st.metric("Recall (uji)", f"{best['test_recall']:.3f}")
        st.caption(
            f"Dilatih {metadata['generated_at'][:10]} · "
            f"{metadata['split']['n_train']} rekam latih"
        )

        st.divider()
        st.caption(
            "Alat bantu skrining, bukan diagnosis definitif. Selalu "
            "konfirmasi dengan pemeriksaan klinis."
        )

    tab_screen, tab_performance, tab_dataset = st.tabs(
        ["Skrining pasien", "Performa model", "Insight dataset"]
    )

    with tab_screen:
        screening_section(threshold)

    with tab_performance:
        try:
            comparison = get_comparison()
        except FileNotFoundError:
            st.warning(
                "Tabel perbandingan model tidak ditemukan. "
                "Jalankan ulang `python train_model.py`."
            )
        else:
            performance_section(metadata, comparison)

    with tab_dataset:
        dataset_section()

    st.divider()
    st.caption(
        "Sumber data: Heart Failure Prediction Dataset (fedesoriano, Kaggle). "
        "Aplikasi ini dibuat untuk keperluan penelitian dan edukasi."
    )


if __name__ == "__main__":
    main()

