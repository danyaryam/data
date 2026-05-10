

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PsyScreening Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primary"   : "#7C5CBF",
    "secondary" : "#4DA8A1",
    "danger"    : "#E05555",
    "warning"   : "#F5A623",
    "light"     : "#F7F5FF",
    "muted"     : "#888888",
}

st.markdown("""
<style>
    .metric-card {
        background: #f9f7ff;
        border-left: 4px solid #7C5CBF;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-title { font-size: 13px; color: #888; margin: 0; }
    .metric-value { font-size: 28px; font-weight: 600; color: #7C5CBF; margin: 4px 0 0; }
    .insight-box {
        background: #fffbeb;
        border-left: 4px solid #F5A623;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 14px;
        margin: 8px 0;
    }
    .conclusion-box {
        background: #f0fdf4;
        border-left: 4px solid #4DA8A1;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 14px;
        margin: 8px 0;
    }
    .risk-high   { color: #E05555; font-weight: 600; }
    .risk-medium { color: #F5A623; font-weight: 600; }
    .risk-low    { color: #4DA8A1; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD DATA & MODEL
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("survey.csv")
    df.drop(columns=["Timestamp", "state", "comments"], inplace=True)
    df = df[(df["Age"] >= 15) & (df["Age"] <= 75)].copy()

    def ng(g):
        g = str(g).lower().strip()
        if any(k in g for k in ["female","woman","femail","femake","cis female","trans-female"]): return "Female"
        if any(k in g for k in ["male","man","maile","cis male"]): return "Male"
        return "Other"
    df["Gender"] = df["Gender"].apply(ng)
    df["self_employed"].fillna("No", inplace=True)
    df["work_interfere"].fillna("Unknown", inplace=True)
    df["treatment_bin"] = (df["treatment"] == "Yes").astype(int)
    return df

@st.cache_resource
def load_model():
    model_path  = "outputs/psyscreening_model.pkl"
    scaler_path = "outputs/psyscreening_scaler.pkl"
    meta_path   = "outputs/psyscreening_meta.json"
    if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
        return None, None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, meta

df  = load_data()
model, scaler, meta = load_model()


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/280x60/7C5CBF/FFFFFF?text=PsyScreening", use_column_width=True)
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["📊 Overview & EDA",
         "🔍 Analisis Faktor Risiko",
         "📈 Evaluasi Model",
         "🩺 Prediksi Individual"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    **Dataset**
    - OSMI Mental Health in Tech Survey
    - 1.259 responden → 1.251 setelah cleaning
    - 22 fitur, 1 target

    **Model**
    - Logistic Regression (L1, C=0.1)
    - ROC-AUC: **0.8645**
    - F1-Score: **0.775**
    """)


# ──────────────────────────────────────────────
# PAGE 1 — OVERVIEW & EDA
# ──────────────────────────────────────────────
if page == "📊 Overview & EDA":

    st.title("🧠 PsyScreening — Mental Health in Tech")
    st.markdown("""
    > **Pertanyaan Bisnis**: *Faktor apa saja yang paling berpengaruh terhadap keputusan
    seseorang untuk mencari bantuan profesional kesehatan mental di industri teknologi,
    dan seberapa akurat model ML dapat memprediksi hal tersebut?*
    """)

    # KPI Cards
    total   = len(df)
    pct_yes = (df["treatment"] == "Yes").mean()
    pct_fam = (df["family_history"] == "Yes").mean()
    pct_rem = (df["remote_work"] == "Yes").mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">Total Responden</p>
            <p class="metric-value">{total:,}</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">Pernah Cari Treatment</p>
            <p class="metric-value">{pct_yes:.1%}</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">Riwayat Keluarga</p>
            <p class="metric-value">{pct_fam:.1%}</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">Bekerja Remote</p>
            <p class="metric-value">{pct_rem:.1%}</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Target — Treatment")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["treatment"].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=[PALETTE["primary"], PALETTE["secondary"]],
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                    f"{val}\n({val/total:.1%})", ha="center", fontsize=11)
        ax.set_ylabel("Jumlah Responden")
        ax.set_title("Apakah pernah mencari treatment?", fontsize=12)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: Dataset hampir seimbang — 50.6% mencari treatment vs 49.4% tidak.
            Tidak perlu teknik oversampling untuk pemodelan.</div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Distribusi Usia Responden")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df["Age"], bins=25, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
        ax.axvline(df["Age"].median(), color=PALETTE["danger"], linestyle="--",
                   linewidth=1.5, label=f'Median: {df["Age"].median():.0f}')
        ax.set_xlabel("Usia")
        ax.set_ylabel("Frekuensi")
        ax.set_title("Distribusi usia (15–75 tahun)", fontsize=12)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=10)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: Mayoritas responden berusia 25–40 tahun — profil pekerja
            muda di industri teknologi. Median usia: 31 tahun.</div>""", unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribusi Gender")
        g_counts = df["Gender"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors_g = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]]
        wedges, texts, autotexts = ax.pie(g_counts.values, labels=g_counts.index,
                                           autopct="%1.1f%%", colors=colors_g,
                                           startangle=90, pctdistance=0.8)
        for at in autotexts: at.set_fontsize(10)
        ax.set_title("Komposisi Gender Responden", fontsize=12)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: 83% responden adalah laki-laki — mencerminkan
            dominasi gender di industri teknologi (tech gap).</div>""", unsafe_allow_html=True)

    with col4:
        st.subheader("Top 8 Negara Asal Responden")
        top_c = df["Country"].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(top_c.index[::-1], top_c.values[::-1],
                color=PALETTE["primary"], edgecolor="white")
        for i, (val, name) in enumerate(zip(top_c.values[::-1], top_c.index[::-1])):
            ax.text(val + 3, i, f"{val}", va="center", fontsize=9)
        ax.set_xlabel("Jumlah Responden")
        ax.set_title("Distribusi negara", fontsize=12)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: 59% responden berasal dari US, diikuti UK (15%) dan
            Kanada (6%). Kolom Country di-drop dari model karena terlalu banyak
            kardinalitas tinggi.</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE 2 — ANALISIS FAKTOR RISIKO
# ──────────────────────────────────────────────
elif page == "🔍 Analisis Faktor Risiko":

    st.title("🔍 Analisis Faktor Risiko Kesehatan Mental")
    st.markdown("""
    **Pertanyaan Analitik:**
    1. Apakah riwayat keluarga merupakan prediktor terkuat untuk mencari treatment?
    2. Seberapa besar pengaruh gangguan pekerjaan (*work interference*) terhadap treatment?
    3. Apakah gender mempengaruhi keputusan mencari bantuan?
    4. Apakah manfaat (*benefits*) dari perusahaan mendorong karyawan mencari treatment?
    """)

    st.markdown("---")

    # Q1: Family history
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Riwayat Keluarga vs Treatment")
        ct = df.groupby("family_history")["treatment"].value_counts(normalize=True).unstack()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct.plot(kind="bar", ax=ax, color=[PALETTE["secondary"], PALETTE["primary"]],
                edgecolor="white", width=0.5, rot=0)
        ax.set_xlabel("")
        ax.set_ylabel("Proporsi")
        ax.set_title("Treatment rate by family history", fontsize=12)
        ax.legend(["No Treatment", "Seeks Treatment"], fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f%%",
                         labels=[f"{v*100:.0f}%" for v in container.datavalues],
                         fontsize=9, padding=2)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: Responden dengan riwayat keluarga memiliki treatment rate
            <b>74%</b> vs hanya <b>35%</b> tanpa riwayat keluarga.
            Faktor ini adalah prediktor terkuat dalam dataset.</div>""", unsafe_allow_html=True)

    # Q2: Work interference
    with col2:
        st.subheader("2. Gangguan Kerja vs Treatment")
        order = ["Never","Rarely","Sometimes","Often"]
        dw = df[df["work_interfere"].isin(order)].copy()
        ct2 = dw.groupby("work_interfere")["treatment"].value_counts(normalize=True).unstack()
        ct2 = ct2.reindex(order)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct2.plot(kind="bar", ax=ax, color=[PALETTE["secondary"], PALETTE["primary"]],
                 edgecolor="white", width=0.6, rot=0)
        ax.set_xlabel("Tingkat Gangguan Kerja")
        ax.set_ylabel("Proporsi")
        ax.set_title("Treatment rate by work interference", fontsize=12)
        ax.legend(["No Treatment", "Seeks Treatment"], fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: Ada hubungan <b>monoton positif</b> yang jelas — semakin
            sering pekerjaan terganggu, semakin tinggi treatment rate.
            'Often' → 85% mencari treatment. Korelasi ini sangat kuat.</div>""", unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    # Q3: Gender
    with col3:
        st.subheader("3. Gender vs Treatment")
        ct3 = df.groupby("Gender")["treatment"].value_counts(normalize=True).unstack()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct3.plot(kind="bar", ax=ax, color=[PALETTE["secondary"], PALETTE["primary"]],
                 edgecolor="white", width=0.5, rot=0)
        ax.set_xlabel("")
        ax.set_ylabel("Proporsi")
        ax.set_title("Treatment rate by gender", fontsize=12)
        ax.legend(["No Treatment", "Seeks Treatment"], fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: Female memiliki treatment rate <b>70%</b> — lebih tinggi
            dibanding Male (46%). Perempuan di industri teknologi tampaknya lebih terbuka
            mencari bantuan profesional.</div>""", unsafe_allow_html=True)

    # Q4: Benefits
    with col4:
        st.subheader("4. Benefit Perusahaan vs Treatment")
        ct4 = df.groupby("benefits")["treatment"].value_counts(normalize=True).unstack()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct4.plot(kind="bar", ax=ax, color=[PALETTE["secondary"], PALETTE["primary"]],
                 edgecolor="white", width=0.5, rot=0)
        ax.set_xlabel("Benefit Kesehatan Mental Tersedia")
        ax.set_ylabel("Proporsi")
        ax.set_title("Treatment rate by mental health benefits", fontsize=12)
        ax.legend(["No Treatment", "Seeks Treatment"], fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
            📌 <b>Insight</b>: Perusahaan yang menyediakan benefit mental health justru
            memiliki treatment rate lebih tinggi (64%). Ini menunjukkan benefit
            mendorong karyawan untuk <i>aktif mencari bantuan</i>.</div>""", unsafe_allow_html=True)

    # Heatmap korelasi
    st.markdown("---")
    st.subheader("Korelasi Fitur vs Target")

    ordinal_maps = {
        "work_interfere":{"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Unknown":1},
        "leave":{"Very easy":0,"Somewhat easy":1,"Unknown":2,"Somewhat difficult":3,"Very difficult":4},
        "no_employees":{"1-5":0,"6-25":1,"26-100":2,"100-500":3,"500-1000":4,"More than 1000":5},
    }
    X_corr = df.copy()
    X_corr["treatment_bin"] = (X_corr["treatment"] == "Yes").astype(int)
    X_corr.drop(columns=["treatment","Country"], inplace=True)
    for col,mapping in ordinal_maps.items():
        X_corr[col] = X_corr[col].map(mapping).fillna(1)
    le = LabelEncoder()
    for c in list(X_corr.select_dtypes(include="object").columns):
        X_corr[c] = le.fit_transform(X_corr[c].astype(str))

    corr_with_target = X_corr.corr()["treatment_bin"].drop("treatment_bin").sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors_corr = [PALETTE["primary"] if v > 0 else PALETTE["danger"] for v in corr_with_target]
    ax.barh(corr_with_target.index[::-1], corr_with_target.values[::-1],
            color=colors_corr[::-1], edgecolor="white")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Korelasi Pearson dengan target 'treatment'")
    ax.set_title("Korelasi setiap fitur terhadap target", fontsize=13)
    ax.spines[["top","right"]].set_visible(False)
    pos_patch = mpatches.Patch(color=PALETTE["primary"], label="Korelasi positif")
    neg_patch = mpatches.Patch(color=PALETTE["danger"], label="Korelasi negatif")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
    st.pyplot(fig)
    plt.close()
    st.markdown("""<div class="conclusion-box">
        ✅ <b>Kesimpulan</b>: <code>work_interfere</code> dan <code>family_history</code>
        adalah dua fitur dengan korelasi absolut tertinggi terhadap target.
        Fitur benefit dan support perusahaan cenderung berkorelasi positif —
        artinya ketersediaan support mendorong pencarian bantuan.</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE 3 — EVALUASI MODEL
# ──────────────────────────────────────────────
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model Machine Learning")

    if meta is None:
        st.warning("⚠️ Model belum ditemukan. Jalankan `psyscreening_pipeline.py` terlebih dahulu.")
        st.stop()

    st.markdown(f"""
    **Champion Model**: `{meta['champion_model']}`
    | **Best Params**: `{meta['best_params']}`
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">Accuracy</p>
            <p class="metric-value">{meta['test_accuracy']:.1%}</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">F1-Score</p>
            <p class="metric-value">{meta['test_f1']:.3f}</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">ROC-AUC</p>
            <p class="metric-value">{meta['test_auc']:.4f}</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-title">Test Samples</p>
            <p class="metric-value">{meta['test_size']}</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perbandingan Model (ROC-AUC)")
        model_names = list(meta["all_model_aucs"].keys())
        auc_vals    = list(meta["all_model_aucs"].values())
        clrs = [PALETTE["primary"] if n == meta["champion_model"] else PALETTE["muted"] for n in model_names]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(model_names, auc_vals, color=clrs, edgecolor="white")
        ax.set_xlim(0.80, 0.89)
        for bar, val in zip(bars, auc_vals):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=10)
        ax.set_xlabel("ROC-AUC")
        ax.set_title("Model comparison after tuning", fontsize=12)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Perbaikan Baseline vs Tuned")
        phases = ["Baseline RF", "Tuned RF", "Baseline LR", "Tuned LR", "Baseline GB", "Tuned GB"]
        values = [0.7879, meta["all_model_aucs"].get("Random Forest",0.86),
                  0.7755, meta["all_model_aucs"].get("Logistic Regression",0.86),
                  0.7769, meta["all_model_aucs"].get("Gradient Boosting",0.86)]
        clrs2  = [PALETTE["muted"], PALETTE["primary"],
                  PALETTE["muted"], PALETTE["primary"],
                  PALETTE["muted"], PALETTE["primary"]]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(phases, values, color=clrs2, edgecolor="white", width=0.6)
        ax.set_ylim(0.70, 0.90)
        ax.set_ylabel("F1 / AUC Score")
        ax.set_title("Baseline CV F1 vs Tuned AUC", fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("""<div class="conclusion-box">
        ✅ <b>Kesimpulan</b>: Setelah hyperparameter tuning, AUC naik dari baseline F1 ~0.79
        menjadi AUC ~0.86 — peningkatan signifikan. Logistic Regression dengan regularisasi L1
        (C=0.1) memberikan hasil terbaik karena melakukan feature selection otomatis melalui
        sparse weights.</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE 4 — PREDIKSI INDIVIDUAL
# ──────────────────────────────────────────────
elif page == "🩺 Prediksi Individual":
    st.title("🩺 Screening Awal — Prediksi Individual")

    if model is None:
        st.warning("⚠️ Model belum ditemukan. Jalankan `psyscreening_pipeline.py` terlebih dahulu.")
        st.stop()

    st.markdown("""
    Isi form di bawah ini untuk mendapatkan prediksi awal apakah seseorang kemungkinan
    memerlukan bantuan profesional kesehatan mental.

    > ⚠️ *Hasil ini hanya bersifat indikatif / screening awal. Bukan diagnosa klinis.*
    """)

    ordinal_maps = {
        "work_interfere":{"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Unknown":1},
        "leave":{"Very easy":0,"Somewhat easy":1,"Unknown":2,"Somewhat difficult":3,"Very difficult":4},
        "no_employees":{"1-5":0,"6-25":1,"26-100":2,"100-500":3,"500-1000":4,"More than 1000":5},
    }

    feature_cols = meta["feature_columns"]

    with st.form("screening_form"):
        st.subheader("Informasi Demografis")
        col1, col2, col3 = st.columns(3)
        with col1:
            age    = st.number_input("Usia", min_value=15, max_value=75, value=28)
        with col2:
            gender = st.selectbox("Gender", ["Male","Female","Other"])
        with col3:
            self_emp = st.selectbox("Wiraswasta?", ["No","Yes"])

        st.subheader("Kondisi Mental & Pekerjaan")
        col4, col5, col6 = st.columns(3)
        with col4:
            family_hist = st.selectbox("Riwayat keluarga MH?", ["No","Yes"])
            work_int    = st.selectbox("Gangguan kerja?", ["Never","Rarely","Sometimes","Often"])
        with col5:
            no_emp    = st.selectbox("Jumlah karyawan", ["1-5","6-25","26-100","100-500","500-1000","More than 1000"])
            remote    = st.selectbox("Kerja remote?", ["No","Yes"])
        with col6:
            tech_co   = st.selectbox("Perusahaan teknologi?", ["Yes","No"])
            benefits  = st.selectbox("Benefit MH tersedia?", ["Yes","No","Don't know"])

        st.subheader("Kebijakan & Lingkungan Kerja")
        col7, col8, col9 = st.columns(3)
        with col7:
            care_opt  = st.selectbox("Tahu opsi layanan MH?", ["Yes","No","Not sure"])
            wellness  = st.selectbox("Ada program wellness?", ["Yes","No","Don't know"])
            seek      = st.selectbox("Ada sumber cari bantuan?", ["Yes","No","Don't know"])
        with col8:
            anon      = st.selectbox("Anonimitas terlindungi?", ["Yes","No","Don't know"])
            leave_opt = st.selectbox("Kemudahan cuti MH", ["Very easy","Somewhat easy","Don't know","Somewhat difficult","Very difficult"])
            mh_cons   = st.selectbox("Dampak negatif MH di kerja?", ["No","Yes","Maybe"])
        with col9:
            ph_cons   = st.selectbox("Dampak negatif fisik di kerja?", ["No","Yes","Maybe"])
            cowork    = st.selectbox("Nyaman diskusi MH dgn rekan?", ["Yes","No","Some of them"])
            superv    = st.selectbox("Nyaman diskusi MH dgn atasan?", ["Yes","No","Some of them"])

        col10, col11, col12 = st.columns(3)
        with col10:
            mh_int    = st.selectbox("Sebutkan MH saat interview?", ["No","Yes","Maybe"])
        with col11:
            ph_int    = st.selectbox("Sebutkan fisik saat interview?", ["No","Yes","Maybe"])
        with col12:
            mh_vs_ph  = st.selectbox("Perusahaan anggap MH = fisik?", ["Yes","No","Don't know"])
        obs_cons = st.selectbox("Pernah lihat konsekuensi MH?", ["No","Yes"])

        submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

    if submitted:
        input_dict = {
            "Age": age, "Gender": gender, "self_employed": self_emp,
            "family_history": family_hist, "work_interfere": work_int,
            "no_employees": no_emp, "remote_work": remote, "tech_company": tech_co,
            "benefits": benefits, "care_options": care_opt,
            "wellness_program": wellness, "seek_help": seek, "anonymity": anon,
            "leave": leave_opt, "mental_health_consequence": mh_cons,
            "phys_health_consequence": ph_cons, "coworkers": cowork,
            "supervisor": superv, "mental_health_interview": mh_int,
            "phys_health_interview": ph_int, "mental_vs_physical": mh_vs_ph,
            "obs_consequence": obs_cons,
        }

        row = pd.DataFrame([input_dict])
        for col_o, mapping in ordinal_maps.items():
            if col_o in row.columns:
                row[col_o] = row[col_o].map(mapping).fillna(1)
        le = LabelEncoder()
        X_ref = load_data().drop(columns=["treatment","Country","treatment_bin"])
        for col_o in list(row.select_dtypes(include="object").columns):
            if col_o in X_ref.columns:
                le.fit(X_ref[col_o].astype(str).unique())
                try:    row[col_o] = le.transform(row[col_o].astype(str))
                except: row[col_o] = 0
        row = row.reindex(columns=feature_cols, fill_value=0)
        row_input = scaler.transform(row) if meta.get("use_scaled_input") else row.values

        pred  = model.predict(row_input)[0]
        prob  = model.predict_proba(row_input)[0][1]

        st.markdown("---")
        if pred == 1:
            risk_label = "Needs Treatment"
            if prob >= 0.75:
                risk_class = "risk-high"
                risk_txt   = "Risiko Tinggi"
                rec = "Sangat disarankan untuk berkonsultasi dengan psikolog atau psikiater."
            else:
                risk_class = "risk-medium"
                risk_txt   = "Risiko Sedang"
                rec = "Pertimbangkan untuk berbicara dengan konselor atau hotline kesehatan mental."
        else:
            risk_label = "No Treatment Needed"
            risk_class = "risk-low"
            risk_txt   = "Risiko Rendah"
            rec = "Tetap jaga kesehatan mental dengan self-care dan komunitas yang supportif."

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.markdown(f"""
            <div style="text-align:center;padding:24px;background:#f9f7ff;border-radius:12px;">
                <div style="font-size:48px">{"🔴" if pred==1 else "🟢"}</div>
                <div class="{risk_class}" style="font-size:22px;margin:8px 0">{risk_txt}</div>
                <div style="font-size:14px;color:#888">{risk_label}</div>
                <div style="font-size:28px;font-weight:600;color:#7C5CBF;margin-top:12px">{prob:.1%}</div>
                <div style="font-size:12px;color:#888">Probabilitas perlu treatment</div>
            </div>""", unsafe_allow_html=True)

        with col_r2:
            st.subheader("Faktor Risiko yang Terdeteksi")
            risk_factors = []
            if family_hist == "Yes":
                risk_factors.append(("🔴", "Riwayat keluarga dengan gangguan mental", "Sangat kuat"))
            if work_int in ["Often", "Sometimes"]:
                risk_factors.append(("🟠", f"Pekerjaan sering terganggu ({work_int})", "Kuat"))
            if gender == "Female":
                risk_factors.append(("🟡", "Gender Female — secara statistik lebih tinggi treatment rate", "Sedang"))
            if benefits == "Yes":
                risk_factors.append(("🟢", "Benefit MH tersedia — akses lebih mudah", "Protektif"))
            if leave_opt in ["Somewhat difficult","Very difficult"]:
                risk_factors.append(("🟠", f"Cuti MH sulit ({leave_opt})", "Sedang"))
            if mh_cons in ["Yes","Maybe"]:
                risk_factors.append(("🟡", "Khawatir dampak negatif diskusi MH di kerja", "Sedang"))
            if not risk_factors:
                risk_factors.append(("🟢", "Tidak ada faktor risiko mayor terdeteksi", "Rendah"))
            for icon, factor, severity in risk_factors:
                st.markdown(f"{icon} **{factor}** — *{severity}*")

            st.markdown(f"""<div class="conclusion-box" style="margin-top:12px">
                💡 <b>Rekomendasi</b>: {rec}</div>""", unsafe_allow_html=True)
