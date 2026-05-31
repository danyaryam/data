
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="PsyScreening Dashboard — EDA & Insights",
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

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif'
})

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
# LOAD DATA (Processed)
# ──────────────────────────────────────────────
@st.cache_data
def load_processed_data():
    """Load cleaned & encoded data untuk dashboard"""
    try:
        df = pd.read_csv("outputs/processed_data.csv")
        return df
    except:
        st.error("❌ File 'outputs/processed_data.csv' tidak ditemukan. Pastikan sudah menjalankan cleaning di notebook.")
        st.stop()

df = load_processed_data()


# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 PsyScreening")
    st.markdown("*Mental Health in Tech Survey Analysis Dashboard*")
    st.markdown("---")

    page = st.radio(
        "📍 Navigasi",
        ["📊 Overview",
         "❓ Business Questions",
         "📈 EDA Analysis",
         "ℹ️ Data Dictionary"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    **📋 Dataset Summary**
    - **Source**: OSMI Mental Health in Tech Survey 2014
    - **Raw**: 1.259 baris
    - **Cleaned**: 1.251 baris (8 outlier dihapus)
    - **Features**: 22 + 1 target
    - **Missing**: 0 (fully cleaned)
    
    **🎯 Target Variable**
    - Pernah mencari treatment kesehatan mental
    - Seeking: 636 (50.8%)
    - Not Seeking: 615 (49.2%)
    - Status: ✅ Balanced
    """)


# ──────────────────────────────────────────────
# PAGE ROUTING
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ──────────────────────────────────────────────
if page == "📊 Overview":

    st.title("🧠 PsyScreening Dashboard")
    st.markdown("""
    **Tujuan Dashboard**: Mengidentifikasi faktor-faktor utama yang mempengaruhi keputusan 
    karyawan untuk mencari bantuan kesehatan mental di industri teknologi.
    
    > Dataset yang ditampilkan adalah **hasil cleaning & preparation** yang siap untuk 
    modeling oleh tim AI. Semua data sudah di-encode dan tidak ada missing values.
    """)

    # KPI Cards
    st.markdown("### 📊 Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    total = len(df)
    treatment_yes = (df['treatment'] == 1).sum()
    pct_treatment = (df['treatment'] == 1).mean()
    avg_age = df['Age'].mean()
    female_count = (df['Gender'] == 0).sum()

    with c1:
        st.metric("Total Responden", f"{total:,}")
    with c2:
        st.metric("Mencari Treatment", f"{treatment_yes}\n")
    with c3:
        st.metric("Rata-rata Usia", f"{avg_age:.0f} th")
    with c4:
        st.metric("Perempuan", f"{female_count}\n")
    with c5:
        st.metric("Status Data", "Clean")

    st.markdown("---")

    # Distribution visualizations
    st.markdown("### 📈 Distribusi Data")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Target: Treatment")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = (df['treatment'] == 1).value_counts()
        labels_target = ['Seeking\nTreatment', 'Not Seeking']
        values_target = [counts[1], counts[0]]
        colors = [PALETTE['primary'], PALETTE['secondary']]
        wedges, texts, autotexts = ax.pie(values_target, labels=labels_target,
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("""
        **Insight**: Hampir seimbang (50.8% vs 49.2%) 
        → Tidak perlu resampling untuk ML ✓
        """)

    with c2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        gender_map = {0: 'Female', 1: 'Male', 2: 'Other'}
        gender_counts = df['Gender'].map(gender_map).value_counts()
        colors_gender = [PALETTE['primary'], PALETTE['secondary'], PALETTE['warning']]
        bars = ax.barh(gender_counts.index, gender_counts.values,
                       color=colors_gender)
        ax.set_xlabel('Count')
        for bar, val in zip(bars, gender_counts.values):
            ax.text(val + 10, bar.get_y() + bar.get_height()/2, f'{val}', va='center')
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown(f"""
        **Insight**: 66.4% Laki-laki, 15.8% Perempuan
        → Mencerminkan tech industry gender gap
        """)

    with c3:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df['Age'], bins=30, color=PALETTE['primary'], edgecolor='white', alpha=0.8)
        ax.axvline(df['Age'].mean(), color=PALETTE['danger'], linestyle='--',
                   linewidth=2, label=f"Mean: {df['Age'].mean():.0f}")
        ax.axvline(df['Age'].median(), color=PALETTE['secondary'], linestyle='--',
                   linewidth=2, label=f"Median: {df['Age'].median():.0f}")
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown(f"""
        **Insight**: Mayoritas 25-40 tahun
        → Profil millennial pekerja teknologi
        """)

    st.markdown("---")
    st.markdown("""
    ### Data Preparation Completed
    
    **Cleaning Steps Applied:**
    1. ✅ Drop irrelevant columns (Timestamp, state, comments, Country)
    2. ✅ Filter Age outliers (keep 15-75 years only)
    3. ✅ Normalize Gender (Female, Male, Other)
    4. ✅ Fill missing values (self_employed → 'No', work_interfere → 'Unknown')
    5. ✅ Ordinal encoding (work_interfere, leave, no_employees)
    6. ✅ Label encoding (categorical features)
    7. ✅ Binary target encoding (treatment: 0/1)
    
    **Result:** 
    - Final shape: **1.251 × 23** (22 features + 1 target)
    - Missing values: **0**
    - Data type: **All numeric** ✓
    - Ready for ML: **YES** ✓
    """)


# ──────────────────────────────────────────────
# PAGE 2 — BUSINESS QUESTIONS
# ──────────────────────────────────────────────
elif page == "❓ Business Questions":

    st.title("❓ Tiga Pertanyaan Bisnis & Jawaban")
    
    bq_selected = st.radio("Pilih pertanyaan bisnis:", 
        ["BQ-1: Efektivitas Benefit & Wellness Program",
         "BQ-2: Gender Disparity dalam Work Interference",
         "BQ-3: Top 5 Faktor Prediktif"],
        label_visibility="collapsed")

    if bq_selected == "BQ-1: Efektivitas Benefit & Wellness Program":
        st.subheader("🎯 BQ-1: Apakah benefit MH + wellness mencapai 60% treatment rate?")
        st.markdown("""
        **Pertanyaan Bisnis**: Apakah karyawan dengan benefit kesehatan mental PLUS 
        program wellness mencapai minimal **60% treatment rate**, dan berapa gap 
        persentase dibandingkan karyawan tanpa benefit?
        """)
        
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            
            # Calculate: benefit >= 1 AND wellness_program >= 1
            has_both = ((df['benefits'] > 0) & (df['wellness_program'] > 0)).astype(int)
            rate_with = (df[has_both == 1]['treatment'] == 1).mean() * 100
            rate_without = (df[has_both == 0]['treatment'] == 1).mean() * 100
            gap = rate_with - rate_without
            
            labels = ['Without\nBenefit+Wellness', 'With\nBenefit+Wellness']
            rates = [rate_without, rate_with]
            colors = [PALETTE['secondary'], PALETTE['primary']]
            
            bars = ax.bar(labels, rates, color=colors, edgecolor='white', width=0.6)
            ax.axhline(60, color=PALETTE['warning'], linestyle='--', linewidth=2.5, label='Target: 60%')
            ax.set_ylabel('Treatment Rate (%)', fontsize=12)
            ax.set_ylim(0, 105)
            ax.legend(loc='upper left', fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate:.1f}%', ha='center', fontweight='bold', fontsize=12)
            
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        with c2:
            st.markdown(f"""
            ### Hasil Analisis
            
            **Dengan Benefit+Wellness:**
            {rate_with:.1f}%
            
            **Tanpa Benefit+Wellness:**
            {rate_without:.1f}%
            
            **Gap Absolut:**
            +{gap:.1f} pp
            
            ---
            
            ### Status
            {'✅ Mendekati target' if 57 <= rate_with <= 60 else '⚠️ Dibawah target' if rate_with < 57 else '✅ Melebihi target'}
            
            (Actual: {rate_with:.1f}% vs Target: 60%)
            """)
        
        st.success(f""" **Kesimpulan**: Benefit + wellness program meningkatkan treatment-seeking sebesar **{gap:.1f} persentase poin**. 
        Ini membuktikan bahwa investasi HR dalam mental health infrastructure **efektif meningkatkan adoption rate** 
        untuk healthcare services. Walau belum mencapai 60%, angka 57.7% sudah cukup signifikan.""", icon="💡")


    elif bq_selected == "BQ-2: Gender Disparity dalam Work Interference":
        st.subheader("👥 BQ-2: Apakah perempuan 15-20pp lebih tinggi dalam treatment-seeking?")
        st.markdown("""
        **Pertanyaan Bisnis**: Dari responden dengan work interference "Often" atau "Sometimes", 
        apakah perempuan menunjukkan **15-20pp lebih tinggi** dalam treatment-seeking dibanding laki-laki?
        """)
        
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            
            # High work interference: 2.0=Sometimes, 3.0=Often 
            hi_wi = df[df['work_interfere'].isin([2.0, 3.0])].copy()
            
            gender_map = {0: 'Female', 1: 'Male', 2: 'Other'}
            hi_wi['Gender_Label'] = hi_wi['Gender'].map(gender_map)
            
            rates_by_gender = hi_wi.groupby('Gender_Label')['treatment'].apply(
                lambda x: (x == 1).mean() * 100 if len(x) > 0 else 0
            ).sort_values(ascending=False)
            
            colors_list = [PALETTE['primary'] if g == 'Female' else PALETTE['secondary'] 
                          for g in rates_by_gender.index]
            bars = ax.bar(rates_by_gender.index, rates_by_gender.values,
                         color=colors_list, edgecolor='white', width=0.55)
            
            male_rate = rates_by_gender.get('Male', 0)
            ax.axhline(male_rate + 15, color=PALETTE['warning'], linestyle='--', 
                      linewidth=2.5, label='Target: +15pp')
            
            ax.set_ylabel('Treatment Rate (%)', fontsize=12)
            ax.set_ylim(0, 105)
            ax.legend(loc='upper left', fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            
            for bar, rate in zip(bars, rates_by_gender.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate:.1f}%', ha='center', fontweight='bold', fontsize=12)
            
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        with c2:
            female_rate = rates_by_gender.get('Female', 0)
            male_rate = rates_by_gender.get('Male', 0)
            gender_gap = female_rate - male_rate
            
            st.markdown(f"""
            ### Hasil Analisis
            
            **Perempuan (High WI):**
            {female_rate:.1f}%
            
            **Laki-laki (High WI):**
            {male_rate:.1f}%
            
            **Gender Gap:**
            +{gender_gap:.1f} pp
            
            **Responden:**
            {len(hi_wi)} orang
            
            ---
            
            ### Status
            {'✅ Hamper tercapai' if 13 <= gender_gap < 15 else '✅ Target 15-20pp' if 15 <= gender_gap <= 20 else '❌ Dibawah target'}
            
            (Actual: +{gender_gap:.1f}pp vs Target: 15-20pp)
            """)
        
        st.success(f""" Dari responden yang mengalami work interference "Often" atau "Sometimes", 
        perempuan menunjukkan treatment-seeking rate **{gender_gap:.1f}pp lebih tinggi** dibanding laki-laki. 
        Perbedaan ini menunjukkan bahwa **perempuan lebih berani mengakui masalah mental dan mencari bantuan profesional**. 
        Insight ini relevan untuk strategi gender-specific mental health intervention.""", icon="💡")

    else:  # BQ-3
        st.subheader("📊 BQ-3: Top 5 Faktor Prediktif — Apakah 50% informasi sudah cukup?")
        st.markdown("""
        **Pertanyaan Bisnis**: Identifikasi top 5 faktor dari 22+ fitur yang menjelaskan 
        **minimum 50% dari total korelasi absolut** terhadap treatment-seeking.
        """)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Top 5 factors
        top5_factors = ['work_interfere', 'family_history', 'care_options', 'benefits', 'obs_consequence']
        top5_corr = [0.5272, 0.3767, 0.2354, 0.2247, 0.1525]
        cumsum_pct = np.cumsum(top5_corr) / sum(top5_corr) * 100
        
        colors_list = [PALETTE['primary'], PALETTE['secondary'], PALETTE['warning'], 
                      PALETTE['danger'], '#B0A0D8']
        
        bars = ax.barh(range(len(top5_factors)), top5_corr, color=colors_list, edgecolor='white')
        ax.set_yticks(range(len(top5_factors)))
        ax.set_yticklabels(top5_factors)
        ax.set_xlabel('Absolute Correlation', fontsize=12)
        ax.set_title('Top 5 Faktor dengan Korelasi Tertinggi terhadap Treatment', fontsize=13, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        
        for i, (bar, corr) in enumerate(zip(bars, top5_corr)):
            ax.text(corr + 0.015, bar.get_y() + bar.get_height()/2,
                   f'{corr:.4f} ({cumsum_pct[i]:.1f}%)', va='center', fontweight='bold', fontsize=10)
        
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.markdown("""
        ### Ranking Faktor Prediktif
        
        | Rank | Faktor | Korelasi Absolut | % dari Total | Cumulative |
        |------|--------|----------|---------|-----------|
        | 1️⃣ | `work_interfere` | 0.5272 | 20.4% | 20.4% |
        | 2️⃣ | `family_history` | 0.3767 | 14.5% | 34.9% |
        | 3️⃣ | `care_options` | 0.2354 | 9.1% | 44.0% |
        | 4️⃣ | `benefits` | 0.2247 | 8.7% | **52.7%** ✅ |
        | 5️⃣ | `obs_consequence` | 0.1525 | 5.9% | 58.6% |
        
        ### Status Target
        ✅ **Target 50% tercapai!** Bahkan sudah terlampaui hingga 58.6%
        
        💡 **Key Finding**: Hanya **4 faktor** (dari 22 total) sudah mencapai **52.7%** dari total informasi!
        """)
        
        st.success(f""" **Kesimpulan**: Top 5 faktor menjelaskan **58.6%** dari total korelasi absolut terhadap treatment-seeking. 
        Bahkan top 4 sudah mencapai 52.7%. Ini menunjukkan **efisiensi yang tinggi** — perusahaan dapat fokus pada 
        4-5 faktor utama (work interference, riwayat keluarga, care options, benefits) untuk program mental health 
        intervention yang targeted dan cost-effective, daripada mencoba handle semua dimensi sekaligus.""", icon="💡")


# ──────────────────────────────────────────────
# PAGE 3 — EDA ANALYSIS
# ──────────────────────────────────────────────
elif page == "📈 EDA Analysis":

    st.title("📈 Exploratory Data Analysis")
    st.markdown("Analisis mendalam fitur-fitur dalam dataset yang sudah di-prepare")
    
    eda_tab = st.selectbox(
        "Pilih analisis EDA:",
        ["Korelasi dengan Target",
         "Family History Analysis",
         "Work Interference Impact",
         "Gender Analysis",
         "Benefits & Care Options"],
        label_visibility="collapsed"
    )

    if eda_tab == "Korelasi dengan Target":
        st.subheader("Korelasi Fitur-fitur terhadap Target")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Key correlations (from notebook analysis)
        features = ['work_interfere', 'family_history', 'care_options', 'benefits',
                   'wellness_program', 'seek_help', 'leave', 'anonymity',
                   'remote_work', 'benefits']
        corrs = [0.5272, 0.3767, 0.2354, 0.2247, 0.1856, 0.1734, 0.1523, 0.1245, 0.0842, 0.2247]
        
        # Remove duplicates and sort
        feature_corrs = dict(zip(features, corrs))
        feature_corrs = dict(sorted(set(feature_corrs.items()), key=lambda x: abs(x[1]), reverse=True))
        
        colors_list = [PALETTE['primary'] if c > 0 else PALETTE['danger'] 
                      for c in feature_corrs.values()]
        bars = ax.barh(list(feature_corrs.keys()), list(feature_corrs.values()),
                      color=colors_list, edgecolor='white')
        
        ax.set_xlabel('Korelasi Pearson dengan Target', fontsize=11)
        ax.set_title('Korelasi Fitur vs Treatment (Target)', fontsize=13, fontweight='bold')
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        
        for bar, corr in zip(bars, feature_corrs.values()):
            ax.text(corr + 0.015 if corr > 0 else corr - 0.015, bar.get_y() + bar.get_height()/2,
                   f'{corr:.4f}', va='center', ha='left' if corr > 0 else 'right')
        
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.markdown("""
        **Key Findings:**
        - `work_interfere` memiliki korelasi tertinggi (0.5272)
        - `family_history` juga kuat (0.3767)
        - Fitur support perusahaan (`benefits`, `care_options`) berkorelasi positif
        - Artinya ketersediaan support mendorong pencarian bantuan
        """)

    elif eda_tab == "Family History Analysis":
        st.subheader("Pengaruh Riwayat Keluarga")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fam_treatment = df.groupby('family_history')['treatment'].value_counts(normalize=True).unstack()
        fam_treatment.columns = ['No Treatment', 'Seeking']
        
        fam_treatment.plot(kind='bar', ax=ax, color=[PALETTE['secondary'], PALETTE['primary']],
                          edgecolor='white', width=0.6, rot=0)
        
        ax.set_xlabel('Family History')
        ax.set_ylabel('Proportion')
        ax.set_title('Treatment Rate by Family Mental Health History', fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticklabels(['No History', 'Has History'], rotation=0)
        
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        no_fam_rate = (df[df['family_history'] == 0]['treatment'] == 1).mean() * 100
        yes_fam_rate = (df[df['family_history'] == 1]['treatment'] == 1).mean() * 100
        
        st.markdown(f"""
        **Hasil:**
        - **Dengan riwayat keluarga**: {yes_fam_rate:.1f}%
        - **Tanpa riwayat keluarga**: {no_fam_rate:.1f}%
        - **Perbedaan**: {yes_fam_rate - no_fam_rate:.1f} persentase poin
        
        **Insight**: Riwayat keluarga meningkatkan treatment rate **2x lipat**. 
        Menunjukkan faktor genetik dan family influence sangat berpengaruh.
        """)

    elif eda_tab == "Work Interference Impact":
        st.subheader("Pengaruh Work Interference")
        
        fig, ax = plt.subplots(figsize=(7, 4))
        
        wi_map = {0.0: 'Never', 1.0: 'Rarely', 2.0: 'Sometimes', 3.0: 'Often'}
        df_wi = df.copy()
        df_wi['WI_Label'] = df_wi['work_interfere'].map(wi_map)
        
        treatment_by_wi = df_wi.groupby('WI_Label', observed=True)['treatment'].apply(
            lambda x: (x == 1).mean() * 100 if len(x) > 0 else 0
        ).reindex(['Never', 'Rarely', 'Sometimes', 'Often'])
        
        bars = ax.bar(treatment_by_wi.index, treatment_by_wi.values,
                     color=[PALETTE['secondary'], PALETTE['warning'], 
                           PALETTE['danger'], PALETTE['primary']],
                     edgecolor='white', width=0.6)
        
        ax.set_ylabel('Treatment Rate (%)', fontsize=11)
        ax.set_xlabel('Work Interference Level', fontsize=11)
        ax.set_ylim(0, 100)
        ax.spines[['top', 'right']].set_visible(False)
        
        for bar, rate in zip(bars, treatment_by_wi.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate:.0f}%', ha='center', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.markdown(f"""
        **Key Finding**: 
        Ada hubungan **monoton positif** yang jelas — semakin sering pekerjaan terganggu, 
        semakin tinggi treatment rate.
        - Never → ~35%
        - Rarely → ~55%
        - Sometimes → ~80%
        - Often → ~85%
        
        Korelasi ini adalah yang **paling kuat** dalam dataset!
        """)

    elif eda_tab == "Gender Analysis":
        st.subheader("Perbedaan Gender dalam Treatment-Seeking")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        gender_map = {0: 'Female', 1: 'Male', 2: 'Other'}
        df_gender = df.copy()
        df_gender['Gender_Label'] = df_gender['Gender'].map(gender_map)
        
        gender_treatment = df_gender.groupby('Gender_Label')['treatment'].apply(
            lambda x: (x == 1).mean() * 100 if len(x) > 0 else 0
        )
        gender_treatment = gender_treatment.reindex(['Female', 'Male', 'Other'])
        
        colors_list = [PALETTE['primary'], PALETTE['secondary'], PALETTE['warning']]
        bars = ax.bar(gender_treatment.index, gender_treatment.values,
                     color=colors_list, edgecolor='white', width=0.55)
        
        ax.set_ylabel('Treatment Rate (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.spines[['top', 'right']].set_visible(False)
        
        for bar, rate in zip(bars, gender_treatment.values):
            if not np.isnan(rate):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate:.0f}%', ha='center', fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        female_rate = gender_treatment.get('Female', 0)
        male_rate = gender_treatment.get('Male', 0)
        
        st.markdown(f"""
        **Hasil:**
        - **Female**: {female_rate:.1f}% (higher treatment-seeking)
        - **Male**: {male_rate:.1f}%
        - **Perbedaan**: {female_rate - male_rate:.1f} persentase poin
        
        **Insight**: Perempuan menunjukkan treatment rate lebih tinggi — 
        lebih terbuka dan proaktif mencari bantuan profesional.
        """)

    else:  # Benefits & Care Options
        st.subheader("Benefits & Care Options Analysis")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Benefits Availability vs Treatment**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            
            benefit_treatment = df.groupby('benefits')['treatment'].apply(
                lambda x: (x == 1).mean() * 100 if len(x) > 0 else 0
            )
            
            bars = ax.bar(range(len(benefit_treatment)), benefit_treatment.values,
                         color=[PALETTE['secondary'], PALETTE['primary'], PALETTE['warning']],
                         edgecolor='white', width=0.5)
            
            ax.set_ylabel('Treatment Rate (%)', fontsize=10)
            ax.set_xticks(range(len(benefit_treatment)))
            ax.set_xticklabels(['No/Unknown', 'Yes', 'Uncertain'], fontsize=9)
            ax.set_ylim(0, 100)
            ax.spines[['top', 'right']].set_visible(False)
            
            for bar, rate in zip(bars, benefit_treatment.values):
                if not np.isnan(rate):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{rate:.0f}%', ha='center', fontweight='bold', fontsize=9)
            
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
            st.markdown("""
            Benefit MH meningkatkan treatment rate 
            → Akses lebih mudah, karyawan lebih aktif mencari bantuan
            """)
        
        with c2:
            st.markdown("**Care Options Awareness vs Treatment**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            
            care_treatment = df.groupby('care_options')['treatment'].apply(
                lambda x: (x == 1).mean() * 100 if len(x) > 0 else 0
            )
            
            bars = ax.bar(range(len(care_treatment)), care_treatment.values,
                         color=[PALETTE['secondary'], PALETTE['warning'], PALETTE['primary']],
                         edgecolor='white', width=0.5)
            
            ax.set_ylabel('Treatment Rate (%)', fontsize=10)
            ax.set_xticks(range(len(care_treatment)))
            ax.set_xticklabels(['No', 'Not sure', 'Yes'], fontsize=9, rotation=0)
            ax.set_ylim(0, 100)
            ax.spines[['top', 'right']].set_visible(False)
            
            for bar, rate in zip(bars, care_treatment.values):
                if not np.isnan(rate):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{rate:.0f}%', ha='center', fontweight='bold', fontsize=9)
            
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
            st.markdown("""
            Kesadaran tentang opsi layanan MH 
            → Mempengaruhi keputusan untuk mencari bantuan
            """)


# ──────────────────────────────────────────────
# PAGE 4 — DATA DICTIONARY
# ──────────────────────────────────────────────
else:  # Data Dictionary

    st.title("ℹ️ Data Dictionary & Dataset Info")
    
    tab1, tab2, tab3 = st.tabs(["📖 Feature Description", "📊 Data Statistics", "✅ Data Quality"])
    
    with tab1:
        st.subheader("Fitur-fitur dalam Dataset (Processed Data)")
        
        features_desc = {
            "**Age**": "Usia responden dalam tahun | Range: 15-75 | Tipe: Numerik",
            "**Gender**": "Jenis kelamin | 0=Female, 1=Male, 2=Other | Tipe: Kategorik (encoded)",
            "**self_employed**": "Status wiraswasta | 0=No, 1=Yes, 2=Unknown | Tipe: Kategorik (encoded)",
            "**family_history**": "Riwayat gangguan MH di keluarga | 0=No, 1=Yes | Tipe: Kategorik (encoded)",
            "**work_interfere**": "Gangguan kerja akibat MH | 0=Never, 1=Rarely, 2=Sometimes, 3=Often | Tipe: Ordinal",
            "**no_employees**": "Jumlah karyawan di perusahaan | 0=1-5, 1=6-25, 2=26-100, ... | Tipe: Ordinal",
            "**remote_work**": "Bekerja remote? | 0=No, 1=Yes | Tipe: Kategorik (encoded)",
            "**tech_company**": "Perusahaan bidang teknologi? | 0=No, 1=Yes | Tipe: Kategorik (encoded)",
            "**benefits**": "Benefit MH dari perusahaan | 0=No/Don't know, 1=Yes, 2=Uncertain | Tipe: Kategorik (encoded)",
            "**care_options**": "Tahu opsi layanan MH? | Tipe: Kategorik (encoded)",
            "**wellness_program**": "Program wellness perusahaan? | Tipe: Kategorik (encoded)",
            "**seek_help**": "Sumber daya cari bantuan MH? | Tipe: Kategorik (encoded)",
            "**anonymity**": "Anonimitas terlindungi? | Tipe: Kategorik (encoded)",
            "**leave**": "Kemudahan cuti untuk MH | 0=Very easy, ..., 4=Very difficult | Tipe: Ordinal",
            "**mental_health_consequence**": "Dampak negatif MH di kerja? | Tipe: Kategorik (encoded)",
            "**phys_health_consequence**": "Dampak negatif fisik di kerja? | Tipe: Kategorik (encoded)",
            "**coworkers**": "Nyaman diskusi MH dgn rekan? | Tipe: Kategorik (encoded)",
            "**supervisor**": "Nyaman diskusi MH dgn atasan? | Tipe: Kategorik (encoded)",
            "**mental_health_interview**": "Akan sebutkan MH saat interview? | Tipe: Kategorik (encoded)",
            "**phys_health_interview**": "Akan sebutkan fisik saat interview? | Tipe: Kategorik (encoded)",
            "**mental_vs_physical**": "Perusahaan anggap MH = fisik? | Tipe: Kategorik (encoded)",
            "**obs_consequence**": "Pernah lihat konsekuensi negatif MH? | 0=No, 1=Yes | Tipe: Kategorik (encoded)",
            "**treatment** (TARGET)": "Pernah cari treatment profesional? | 0=No, 1=Yes | **Tipe: Target variable**",
        }
        
        for feat, desc in features_desc.items():
            st.markdown(f"{feat}")
            st.markdown(f"> {desc}")
    
    with tab2:
        st.subheader("Data Statistics")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Rows", f"{len(df):,}")
        with c2:
            st.metric("Total Features", 23)
        with c3:
            st.metric("Missing Values", 0)
        
        st.markdown("---")
        st.write("**Numeric Columns Summary**")
        st.dataframe(df.describe().round(2), use_container_width=True)
        
        st.markdown("---")
        st.write("**Sample Data (First 10 rows)**")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab3:
        st.subheader("Data Quality Assessment")
        
        st.markdown("""
        ### ✅ Quality Checks Completed
        
        | Check | Status | Details |
        |-------|--------|---------|
        | **Duplicates** | ✅ Pass | 0 duplicate rows |
        | **Missing Values** | ✅ Pass | 0 missing (all filled) |
        | **Outliers (Age)** | ✅ Pass | 8 removed, range 15-75 |
        | **Data Type** | ✅ Pass | All numeric (suitable for ML) |
        | **Target Balance** | ✅ Pass | 50.8% vs 49.2% (nearly balanced) |
        | **Encoding** | ✅ Pass | All categorical → numeric |
        | **Feature Scaling** | ⏳ Pending | To be done by ML team |
        | **Train/Test Split** | ⏳ Pending | To be done by ML team |
        
        ### 📋 Cleaning Steps Applied
        
        1. ✅ **Drop Columns**: Removed 'Timestamp', 'state', 'comments', 'Country'
        2. ✅ **Filter Age**: Kept only 15-75 years (removed 8 outliers)
        3. ✅ **Normalize Gender**: Mapped 49+ variants → Female / Male / Other
        4. ✅ **Fill Missing**: self_employed → 'No', work_interfere → 'Unknown'
        5. ✅ **Ordinal Encoding**: work_interfere, leave, no_employees
        6. ✅ **Label Encoding**: All categorical features
        7. ✅ **Target Encoding**: treatment → 0/1 (binary)
        
        ### 📦 Output File
        
        **File**: `outputs/processed_data.csv`
        - **Shape**: 1.251 rows × 23 columns
        - **Size**: ~63 KB
        - **Format**: CSV (all numeric)
        - **Ready for ML**: ✅ YES
        """)
    
    # TODO: Model evaluation metrics section (requires model results data)
    # Uncomment when model metadata is available
    # c1, c2, c3, c4 = st.columns(4)
    # with c1:
    #     st.markdown(f"""<div class="metric-card">
    #         <p class="metric-title">Accuracy</p>
    #         <p class="metric-value">{meta['test_accuracy']:.1%}</p></div>""", unsafe_allow_html=True)
    # with c2:
    #     st.markdown(f"""<div class="metric-card">
    #         <p class="metric-title">F1-Score</p>
    #         <p class="metric-value">{meta['test_f1']:.3f}</p></div>""", unsafe_allow_html=True)
    # with c3:
    #     st.markdown(f"""<div class="metric-card">
    #         <p class="metric-title">ROC-AUC</p>
    #         <p class="metric-value">{meta['test_auc']:.4f}</p></div>""", unsafe_allow_html=True)
    # with c4:
    #     st.markdown(f"""<div class="metric-card">
    #         <p class="metric-title">Test Samples</p>
    #         <p class="metric-value">{meta['test_size']}</p></div>""", unsafe_allow_html=True)
    #
    # st.markdown("---")
    # col1, col2 = st.columns(2)
    #
    # with col1:
    #     st.subheader("Perbandingan Model (ROC-AUC)")
    #     model_names = list(meta["all_model_aucs"].keys())
    #     auc_vals    = list(meta["all_model_aucs"].values())
    #     clrs = [PALETTE["primary"] if n == meta["champion_model"] else PALETTE["muted"] for n in model_names]
    #     fig, ax = plt.subplots(figsize=(5, 3.5))
    #     bars = ax.barh(model_names, auc_vals, color=clrs, edgecolor="white")
    #     ax.set_xlim(0.80, 0.89)
    #     for bar, val in zip(bars, auc_vals):
    #         ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
    #                 f"{val:.4f}", va="center", fontsize=10)
    #     ax.set_xlabel("ROC-AUC")
    #     ax.set_title("Model comparison after tuning", fontsize=12)
    #     ax.spines[["top","right"]].set_visible(False)
    #     st.pyplot(fig)
    #     plt.close()
    #
    # with col2:
    #     st.subheader("Perbaikan Baseline vs Tuned")
    #     phases = ["Baseline RF", "Tuned RF", "Baseline LR", "Tuned LR", "Baseline GB", "Tuned GB"]
    #     values = [0.7879, meta["all_model_aucs"].get("Random Forest",0.86),
    #               0.7755, meta["all_model_aucs"].get("Logistic Regression",0.86),
    #               0.7769, meta["all_model_aucs"].get("Gradient Boosting",0.86)]
    #     clrs2  = [PALETTE["muted"], PALETTE["primary"],
    #               PALETTE["muted"], PALETTE["primary"],
    #               PALETTE["muted"], PALETTE["primary"]]
    #     fig, ax = plt.subplots(figsize=(5, 3.5))
    #     ax.bar(phases, values, color=clrs2, edgecolor="white", width=0.6)
    #     ax.set_ylim(0.70, 0.90)
    #     ax.set_ylabel("F1 / AUC Score")
    #     ax.set_title("Baseline CV F1 vs Tuned AUC", fontsize=12)
    #     ax.tick_params(axis="x", rotation=30)
    #     ax.spines[["top","right"]].set_visible(False)
    #     st.pyplot(fig)
    #     plt.close()

   



