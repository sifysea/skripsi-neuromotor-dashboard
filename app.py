import os
import gzip
import glob
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)

# Abaikan warning
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Neuromotor FMA Dashboard",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# KONFIGURASI PIPELINE & STATIK
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
REPO_DIR = os.path.join(SCRIPT_DIR, "UI-PRMD-Analysis")
CORRECT_POS_DIR = os.path.join(REPO_DIR, "data", "Segmented Movements", "Vicon", "Positions")
INCORRECT_POS_DIR = os.path.join(REPO_DIR, "data", "Incorrect Segmented Movements", "Vicon", "Positions")

CSV_PATH = os.path.join(SCRIPT_DIR, "Dataset_Fitur_Neuromotorik.csv")

FS = 100            # Hz
CUTOFF_FREQ = 6.0   # Hz
FILTER_ORDER = 4

VICON_MARKERS = [
    "LFHD", "RFHD", "LBHD", "RBHD",
    "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI",
    "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
]
RWRA_INDEX = VICON_MARKERS.index("RWRA")  # 20
RWRA_COL_START = RWRA_INDEX * 3            # 60
RWRA_COL_END = RWRA_COL_START + 3          # 63

# Fitur model
FEATURES_TO_DROP = [
    "velocity_x_mean", "velocity_y_mean", "velocity_z_mean",
    "peak_jerk_mm_s3", "peak_acceleration_mm_s2", "n_frames",
    "range_x_mm", "std_jerk_mm_s3"
]

FEATURES_TO_KEEP = [
    "peak_velocity_mm_s", "mean_velocity_mm_s", "std_velocity_mm_s",
    "sparc", "dimensionless_jerk", "mean_jerk_mm_s3", "rms_jerk_mm_s3",
    "mean_acceleration_mm_s2", "range_y_mm", "range_z_mm", "duration_s"
]

FMA_LABELS = {
    0: "Near Normal / FMA Tinggi (>50)",
    1: "Gangguan Sedang / FMA Menengah (26-50)",
    2: "Gangguan Berat / FMA Rendah (<25)",
}

FMA_DESCS = {
    0: "Menunjukkan kemampuan motorik ekstremitas atas yang sangat baik, mendekati kondisi normal. Gerakan stabil, halus, dan memiliki sentakan minimum.",
    1: "Menunjukkan adanya gangguan fungsi neuromotorik tingkat sedang. Gerakan masih dapat dilakukan namun kelancarannya berkurang (spastisitas sedang / jerky movement).",
    2: "Menunjukkan gangguan fungsi neuromotorik yang berat. Kecepatan sangat terhambat, koordinasi spasial rendah, durasi gerakan lama, dan tingkat kehalusan gerakan sangat rendah."
}

CLASS_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
CLASS_COLORS_LIGHT = {0: "rgba(46, 204, 113, 0.15)", 1: "rgba(243, 156, 18, 0.15)", 2: "rgba(231, 76, 60, 0.15)"}

MOVEMENT_NAMES = {
    "m07": "Standing shoulder abduction",
    "m08": "Standing shoulder extension",
    "m09": "Standing shoulder internal-external rotation",
    "m10": "Standing shoulder scaption"
}

# CSS Kustom untuk Tampilan Premium
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-card {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid #3498db;
        background-color: #ffffff;
        margin-bottom: 15px;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-card-title {
        font-size: 14px;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    
    .metric-card-value {
        font-size: 28px;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .fma-badge {
        padding: 6px 14px;
        border-radius: 50px;
        font-weight: 700;
        color: white;
        display: inline-block;
        font-size: 14px;
        text-transform: uppercase;
    }
    
    .custom-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(30, 60, 114, 0.15);
    }
    
    .custom-header h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
    }
    
    .custom-header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================================
# FUNGSI UTILITAS DATA & FILTER
# ============================================================================

def butterworth_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered

def compute_velocity(position, dt):
    return np.gradient(position, dt, axis=0)

def compute_jerk(position, dt):
    vel = np.gradient(position, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jrk = np.gradient(acc, dt, axis=0)
    return jrk

def compute_speed(velocity):
    return np.linalg.norm(velocity, axis=1)

def compute_jerk_magnitude(jerk):
    return np.linalg.norm(jerk, axis=1)

def compute_smoothness_sparc(speed, fs, padlevel=4, fc=10.0):
    max_speed = np.max(speed)
    if max_speed <= 0:
        return 0.0
    speed_norm = speed / max_speed

    n = len(speed_norm)
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    freq = np.arange(0, nfft // 2 + 1) * fs / nfft
    speed_fft = np.abs(np.fft.rfft(speed_norm, n=nfft))

    max_fft = np.max(speed_fft)
    if max_fft <= 0:
        return 0.0
    speed_fft = speed_fft / max_fft

    fc_idx = np.argmax(freq > fc) if np.any(freq > fc) else len(freq)
    freq_sel = freq[:fc_idx]
    speed_fft_sel = speed_fft[:fc_idx]

    if len(freq_sel) < 2:
        return 0.0

    d_freq = np.diff(freq_sel)
    d_mag = np.diff(speed_fft_sel)

    freq_range = freq_sel[-1] - freq_sel[0]
    if freq_range == 0:
        return 0.0

    d_freq_norm = d_freq / freq_range
    arc_lengths = np.sqrt(d_freq_norm**2 + d_mag**2)
    sparc = -np.sum(arc_lengths)
    return sparc

def compute_dimensionless_jerk(jerk_mag, speed, duration):
    peak_speed = np.max(speed)
    if duration <= 0 or peak_speed <= 0:
        return 0.0
    mean_jerk_sq = np.mean(jerk_mag ** 2)
    dj = -np.log(duration**3 / (peak_speed**2) * mean_jerk_sq + 1e-10)
    return dj

def extract_features_from_raw_array(position_3d, fs):
    dt = 1.0 / fs
    n_frames = len(position_3d)
    duration = n_frames * dt

    if n_frames < 10:
        return None

    velocity = compute_velocity(position_3d, dt)
    speed = compute_speed(velocity)

    acceleration = np.gradient(velocity, dt, axis=0)
    acc_magnitude = np.linalg.norm(acceleration, axis=1)

    jerk = compute_jerk(position_3d, dt)
    jerk_mag = compute_jerk_magnitude(jerk)

    sparc = compute_smoothness_sparc(speed, fs)
    dim_jerk = compute_dimensionless_jerk(jerk_mag, speed, duration)

    features = {
        "range_x_mm": np.ptp(position_3d[:, 0]),
        "range_y_mm": np.ptp(position_3d[:, 1]),
        "range_z_mm": np.ptp(position_3d[:, 2]),
        "peak_velocity_mm_s": np.max(speed),
        "mean_velocity_mm_s": np.mean(speed),
        "std_velocity_mm_s": np.std(speed),
        "velocity_x_mean": np.mean(velocity[:, 0]),
        "velocity_y_mean": np.mean(velocity[:, 1]),
        "velocity_z_mean": np.mean(velocity[:, 2]),
        "peak_acceleration_mm_s2": np.max(acc_magnitude),
        "mean_acceleration_mm_s2": np.mean(acc_magnitude),
        "peak_jerk_mm_s3": np.max(jerk_mag),
        "mean_jerk_mm_s3": np.mean(jerk_mag),
        "std_jerk_mm_s3": np.std(jerk_mag),
        "rms_jerk_mm_s3": np.sqrt(np.mean(jerk_mag**2)),
        "sparc": sparc,
        "dimensionless_jerk": dim_jerk,
        "duration_s": duration,
        "n_frames": n_frames,
    }
    return features, position_3d, speed, acc_magnitude, jerk_mag

def load_uploaded_raw_file(uploaded_file, filename):
    # Cek kompresi gzip
    if filename.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=uploaded_file, mode='rb') as f:
                data = np.loadtxt(f, delimiter=",")
        except Exception:
            # Fallback jika bytes buffer
            uploaded_file.seek(0)
            with gzip.open(uploaded_file, 'rt') as f:
                data = np.loadtxt(f, delimiter=",")
    else:
        uploaded_file.seek(0)
        data = np.loadtxt(uploaded_file, delimiter=",")
    
    return data

def parse_movement_metadata(filename):
    basename = os.path.basename(filename)
    name_no_ext = basename.split('.')[0]
    
    is_incorrect = "_inc" in name_no_ext
    label = 1 if is_incorrect else 0

    match = re.match(r"(m\d+)_(s\d+)_(e\d+)", name_no_ext)
    if match:
        movement = match.group(1)
        subject = match.group(2)
        episode = match.group(3)
        return {
            "movement": movement,
            "subject": subject,
            "episode": episode,
            "label": label,
            "movement_name": MOVEMENT_NAMES.get(movement, "Unknown Movement"),
            "correctness": "Incorrect/Gangguan" if label == 1 else "Correct/Normal"
        }
    return {
        "movement": "Unknown",
        "subject": "Unknown",
        "episode": "Unknown",
        "label": -1,
        "movement_name": "Gerakan Kustom",
        "correctness": "Tidak Diketahui"
    }

# ============================================================================
# TAHAP TRAINING & EVALUASI DENGAN CACHING
# ============================================================================

@st.cache_resource
def train_and_cache_models():
    """Melatih 6 model secara langsung dan menyimpan hasilnya di memori cache Streamlit."""
    if not os.path.exists(CSV_PATH):
        return None, None, None, None, None
        
    df_raw = pd.read_csv(CSV_PATH)
    
    # 1. Seleksi Fitur Klinis FMA
    df_selected = df_raw.drop(columns=[c for c in FEATURES_TO_DROP if c in df_raw.columns])
    
    # 2. Labeling FMA 3 Kelas
    df_selected.loc[df_selected["label"] == 0, "fma_class"] = 0
    incorrect_mask = df_selected["label"] == 1
    sparc_incorrect = df_selected.loc[incorrect_mask, "sparc"]
    median_sparc = sparc_incorrect.median()
    df_selected.loc[incorrect_mask & (df_selected["sparc"] >= median_sparc), "fma_class"] = 1
    df_selected.loc[incorrect_mask & (df_selected["sparc"] < median_sparc), "fma_class"] = 2
    df_selected["fma_class"] = df_selected["fma_class"].astype(int)
    
    # 3. Pisahkan X & y
    X = df_selected[FEATURES_TO_KEEP].values
    y = df_selected["fma_class"].values
    
    # Stratified Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inisialisasi & Latih Model
    models = {
        "SVM (RBF)": SVC(kernel='rbf', random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                     random_state=42, eval_metric='mlogloss'),
        "MLP Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    eval_results = []
    trained_models = {}
    classes = np.array([0, 1, 2])
    confusion_matrices = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        
        y_test_bin = label_binarize(y_test, classes=classes)
        try:
            auc_roc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        except Exception:
            auc_roc = np.nan
            
        eval_results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "AUC-ROC": auc_roc
        })
        
        confusion_matrices[name] = confusion_matrix(y_test, y_pred, labels=classes)
        
    eval_df = pd.DataFrame(eval_results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # Hitung profil rata-rata per kelas FMA untuk komparasi klinis
    fma_profiles = df_selected.groupby("fma_class")[FEATURES_TO_KEEP].mean()
    
    return trained_models, scaler, eval_df, fma_profiles, confusion_matrices

# Muat model dan database profil
trained_models, scaler, eval_df, fma_profiles, confusion_matrices = train_and_cache_models()

# ============================================================================
# KUMPULKAN BERKAS CONTOH REPOSITORI (JIKA ADA)
# ============================================================================

def scan_prmd_samples():
    samples = []
    
    # 1. Scan folder samples/ terkompresi di repositori GitHub (untuk deployed cloud fallback)
    github_samples_dir = os.path.join(SCRIPT_DIR, "samples")
    if os.path.exists(github_samples_dir):
        for path in glob.glob(os.path.join(github_samples_dir, "m*_*_positions*.txt*")):
            filename = os.path.basename(path)
            is_inc = "_inc" in filename
            samples.append({
                "path": path,
                "filename": filename,
                "type": "Incorrect (Gangguan) [GitHub Sample]" if is_inc else "Correct (Normal) [GitHub Sample]"
            })
            
    # 2. Scan folder Vicon asli jika repositori penuh diclone di lokal
    if os.path.exists(CORRECT_POS_DIR):
        for path in glob.glob(os.path.join(CORRECT_POS_DIR, "m*_*_positions.txt*")):
            filename = os.path.basename(path)
            if not any(s["filename"] == filename for s in samples):
                samples.append({"path": path, "filename": filename, "type": "Correct (Normal) [Vicon Repo]"})
            
    if os.path.exists(INCORRECT_POS_DIR):
        for path in glob.glob(os.path.join(INCORRECT_POS_DIR, "m*_*_positions_inc.txt*")):
            filename = os.path.basename(path)
            if not any(s["filename"] == filename for s in samples):
                samples.append({"path": path, "filename": filename, "type": "Incorrect (Gangguan) [Vicon Repo]"})
            
    samples.sort(key=lambda x: x["filename"])
    return samples

local_samples = scan_prmd_samples()

# ============================================================================
# TAMPILAN ELEMEN APLIKASI
# ============================================================================

# Header utama
st.markdown(
    """
    <div class="custom-header">
        <h1>🦾 Dasbor Visual Interaktif Klasifikasi FMA</h1>
        <p>Analisis Parameter Kinematik Ekstremitas Atas Real-Time Berbasis Sensor Spasial 3D Vicon</p>
    </div>
    """,
    unsafe_allow_html=True
)

if trained_models is None:
    st.error(f"Berkas dataset `{CSV_PATH}` tidak ditemukan! Harap letakkan berkas dataset di folder yang sama dengan skrip ini.")
    st.stop()

# ============================================================================
# SIDEBAR KONFIGURASI DAN UNGGAH
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi Sistem")
    
    # 1. Pilih Model ML
    best_model_name = eval_df.iloc[0]["Model"]
    selected_model_name = st.selectbox(
        "Pilih Model Klasifikasi FMA",
        options=list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name),
        help="Model ML terpilih akan memproses ekstraksi fitur kinematik gerakan untuk mendiagnosis kelas FMA."
    )
    
    st.markdown("---")
    st.markdown("### 📁 Sumber Data Gerakan")
    
    # 2. Pilihan input: Contoh lokal vs Unggah file
    input_source = st.radio(
        "Pilih Sumber Masukan",
        options=["Gunakan Koleksi Contoh Dataset", "Unggah Berkas Spasial Mentah"],
        index=0
    )
    
    uploaded_file = None
    selected_sample = None
    
    if input_source == "Gunakan Koleksi Contoh Dataset":
        if local_samples:
            sample_options = {s["filename"] + f" ({s['type']})": s for s in local_samples}
            selected_option_str = st.selectbox(
                "Pilih Sampel Gerakan Mentah",
                options=list(sample_options.keys())
            )
            selected_sample = sample_options[selected_option_str]
            st.info(f"📂 Memuat berkas dari repositori lokal: \n`{selected_sample['filename']}`")
        else:
            st.warning("Folder repositori data Vicon UI-PRMD tidak terdeteksi. Silakan gunakan metode unggah berkas.")
            input_source = "Unggah Berkas Spasial Mentah"
            
    if input_source == "Unggah Berkas Spasial Mentah":
        uploaded_file = st.file_uploader(
            "Unggah berkas posisi Vicon (.txt / .txt.gz)",
            type=["txt", "gz"],
            help="Berkas harus berupa matriks float comma-separated dengan setidaknya 63 kolom spasial (marker ke-21 / RWRA)."
        )
        if uploaded_file:
            st.success(f"✔️ Berkas berhasil diunggah: `{uploaded_file.name}`")
        else:
            st.info("💡 Unggah berkas data spasial Anda untuk memulai analisis.")

# ============================================================================
# LOAD DATA & EKSTRAKSI FITUR KINEMATIK
# ============================================================================
data_array = None
current_metadata = None
filename_to_display = ""

if input_source == "Gunakan Koleksi Contoh Dataset" and selected_sample:
    path = selected_sample["path"]
    filename_to_display = selected_sample["filename"]
    current_metadata = parse_movement_metadata(filename_to_display)
    
    # Load lokal
    if path.endswith(".gz"):
        with gzip.open(path, 'rt') as f:
            data_array = np.loadtxt(f, delimiter=",")
    else:
        data_array = np.loadtxt(path, delimiter=",")
        
elif input_source == "Unggah Berkas Spasial Mentah" and uploaded_file:
    filename_to_display = uploaded_file.name
    current_metadata = parse_movement_metadata(filename_to_display)
    try:
        data_array = load_uploaded_raw_file(uploaded_file, filename_to_display)
    except Exception as e:
        st.error(f"Gagal memuat file: {e}. Pastikan format file data adalah comma-separated float values.")

# Jalankan ekstraksi jika data termuat
if data_array is not None:
    # 1. Cek jumlah kolom spasial
    n_frames, n_cols = data_array.shape
    if n_cols < RWRA_COL_END:
        st.error(f"Dimensi data tidak valid! File memiliki {n_cols} kolom, tetapi klasifikasi FMA membutuhkan setidaknya {RWRA_COL_END} kolom untuk marker RWRA.")
    else:
        # 2. Ekstrak marker RWRA (X, Y, Z)
        rwra_xyz = data_array[:, RWRA_COL_START:RWRA_COL_END]
        
        # Interpolasi jika ada NaN/Inf
        if np.any(np.isnan(rwra_xyz)) or np.any(np.isinf(rwra_xyz)):
            for col in range(3):
                series = pd.Series(rwra_xyz[:, col])
                series = series.interpolate(method='linear', limit_direction='both')
                rwra_xyz[:, col] = series.values
                
        # 3. Butterworth Filter
        min_len = (FILTER_ORDER * 3) + 1
        if n_frames > min_len:
            rwra_filtered = butterworth_lowpass_filter(
                rwra_xyz, cutoff=CUTOFF_FREQ, fs=FS, order=FILTER_ORDER
            )
        else:
            rwra_filtered = rwra_xyz
            
        # 4. Ekstraksi Fitur Kinematik
        extracted_results = extract_features_from_raw_array(rwra_filtered, FS)
        
        if extracted_results is None:
            st.error("Gagal melakukan ekstraksi parameter kinematik. Durasi gerakan terlalu singkat (<10 frame).")
        else:
            features_dict, pos_filtered, speed, acc_mag, jerk_mag = extracted_results
            
            # Format dataframe fitur individual untuk input prediksi
            input_features = [features_dict[feat] for feat in FEATURES_TO_KEEP]
            input_features_scaled = scaler.transform([input_features])
            
            # 5. Klasifikasi dengan Model Terpilih
            model = trained_models[selected_model_name]
            fma_pred = model.predict(input_features_scaled)[0]
            fma_proba = model.predict_proba(input_features_scaled)[0]
            
            # ====================================================================
            # RENDER TAB-TAB UTAMA
            # ====================================================================
            
            tab_diag, tab_viz, tab_model, tab_insights = st.tabs([
                "📋 Tab 1: Deteksi & Klasifikasi FMA",
                "📈 Tab 2: Visualisasi Kinematik (3D & 2D)",
                "📊 Tab 3: Performa & Analisis Model",
                "🔍 Tab 4: Dataset Insights & Statistik"
            ])
            
            # --------------------------------------------------------------------
            # TAB 1: DETEKSI & KLASIFIKASI FMA
            # --------------------------------------------------------------------
            with tab_diag:
                st.markdown("### 🩺 Hasil Diagnosis Fungsi Neuromotorik")
                
                col_pred, col_details = st.columns([2, 3])
                
                with col_pred:
                    # Kartu hasil prediksi visual
                    color_accent = CLASS_COLORS[fma_pred]
                    bg_accent = CLASS_COLORS_LIGHT[fma_pred]
                    
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 6px solid {color_accent}; background-color: {bg_accent};">
                            <div class="metric-card-title">Prediksi Tingkat Keparahan FMA</div>
                            <div class="metric-card-value" style="color: {color_accent}; font-size: 26px;">FMA KELAS {fma_pred}</div>
                            <div style="font-weight: 700; margin-top: 5px; color: #2c3e50;">{FMA_LABELS[fma_pred]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Tampilkan info berkas yang aktif
                    st.markdown("##### 📄 Metadata Berkas Masukan:")
                    meta_df = pd.DataFrame([
                        {"Parameter": "Nama File", "Value": filename_to_display},
                        {"Parameter": "Nama Gerakan", "Value": current_metadata["movement_name"]},
                        {"Parameter": "Kode Gerakan", "Value": current_metadata["movement"]},
                        {"Parameter": "Subjek / Episode", "Value": f"Subject {current_metadata['subject']} / Episode {current_metadata['episode']}"},
                        {"Parameter": "Ground Truth Biner", "Value": current_metadata["correctness"]}
                    ])
                    st.dataframe(meta_df, hide_index=True, use_container_width=True)
                    
                with col_details:
                    # Deskripsi klinis FMA
                    st.markdown(f"#### Kelas Diagnosis: **Kelas {fma_pred}**")
                    st.markdown(f"*{FMA_DESCS[fma_pred]}*")
                    
                    st.markdown("##### 📊 Probabilitas Keyakinan Prediksi Model:")
                    prob_df = pd.DataFrame({
                        "Kelas FMA": [f"Kelas {i}: {FMA_LABELS[i]}" for i in range(3)],
                        "Probabilitas (%)": [prob * 100 for prob in fma_proba]
                    })
                    
                    fig_prob = px.bar(
                        prob_df,
                        x="Probabilitas (%)",
                        y="Kelas FMA",
                        orientation='h',
                        color="Kelas FMA",
                        color_discrete_map={
                            f"Kelas 0: {FMA_LABELS[0]}": CLASS_COLORS[0],
                            f"Kelas 1: {FMA_LABELS[1]}": CLASS_COLORS[1],
                            f"Kelas 2: {FMA_LABELS[2]}": CLASS_COLORS[2]
                        },
                        text=prob_df["Probabilitas (%)"].map('{:,.2f}%'.format)
                    )
                    fig_prob.update_layout(
                        showlegend=False,
                        height=200,
                        margin=dict(l=0, r=10, t=10, b=10),
                        xaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                st.markdown("---")
                
                # Tabel komparasi parameter kinematik klinis
                st.markdown("### 📊 Analisis Komparatif Kinematik Klinis")
                st.markdown("Perbandingan parameter kinematik gerakan yang diunggah dengan rata-rata populasi tiap kelas FMA dalam database riset:")
                
                comp_rows = []
                for feat in FEATURES_TO_KEEP:
                    val_curr = features_dict[feat]
                    val_c0 = fma_profiles.loc[0, feat]
                    val_c1 = fma_profiles.loc[1, feat]
                    val_c2 = fma_profiles.loc[2, feat]
                    
                    comp_rows.append({
                        "Parameter Kinematik": feat.replace("_", " ").title(),
                        "Gerakan Anda": val_curr,
                        "Rata-rata Kelas 0 (Normal)": val_c0,
                        "Rata-rata Kelas 1 (Sedang)": val_c1,
                        "Rata-rata Kelas 2 (Berat)": val_c2
                    })
                    
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df.style.format({
                    "Gerakan Anda": "{:.4f}",
                    "Rata-rata Kelas 0 (Normal)": "{:.4f}",
                    "Rata-rata Kelas 1 (Sedang)": "{:.4f}",
                    "Rata-rata Kelas 2 (Berat)": "{:.4f}"
                }).highlight_max(axis=1, color="#fcd5d5", subset=["Rata-rata Kelas 0 (Normal)", "Rata-rata Kelas 1 (Sedang)", "Rata-rata Kelas 2 (Berat)"]), use_container_width=True)
                
                # Edukasi klinis singkat mengenai smoothness
                with st.expander("📚 Pelajari Cara Membaca Parameter Klinis Ini"):
                    st.markdown(
                        """
                        - **SPARC (Spectral Arc Length)**: Metrik kehalusan gerakan utama. Nilai SPARC berkisar di rentang negatif. **Semakin mendekati 0, gerakan semakin mulus/normal**. Semakin negatif nilai SPARC, gerakan semakin terputus-putus (*jerky*), yang mengindikasikan spastisitas atau gangguan koordinasi neuromotorik yang berat.
                        - **Dimensionless Jerk (DJ)**: Pengukur sentakan tanpa dimensi. Nilai DJ yang lebih tinggi (ke arah 0 atau positif kecil) menunjukkan sentakan yang lebih sedikit, sedangkan nilai DJ yang lebih negatif mengindikasikan gerakan yang penuh sentakan tidak teratur.
                        - **Peak & Mean Velocity**: Menunjukkan kecepatan absolut gerakan tangan (dalam mm/detik). Pasien dengan gangguan motorik berat (FMA Rendah) cenderung memiliki kecepatan gerak yang lambat guna mempertahankan stabilitas.
                        - **Duration (s)**: Durasi yang dibutuhkan subjek untuk menyelesaikan satu episode gerakan. Durasi yang lama berkorelasi kuat dengan FMA Rendah (Gangguan Berat).
                        """
                    )
            
            # --------------------------------------------------------------------
            # TAB 2: VISUALISASI KINEMATIK INTERAKTIF (3D & 2D)
            # --------------------------------------------------------------------
            with tab_viz:
                st.markdown("### 📈 Visualisasi Ruang & Kecepatan Gerakan")
                
                col_3d, col_2d = st.columns([1, 1])
                
                with col_3d:
                    st.markdown("##### 🦾 Lintasan Spasial 3D Tangan (Marker RWRA)")
                    
                    df_pos = pd.DataFrame(pos_filtered, columns=["X", "Y", "Z"])
                    df_pos["Speed (mm/s)"] = speed
                    df_pos["Frame"] = range(len(df_pos))
                    
                    fig_3d = px.scatter_3d(
                        df_pos, x="X", y="Y", z="Z",
                        color="Speed (mm/s)",
                        color_continuous_scale="Viridis",
                        labels={"X": "Posisi X (mm)", "Y": "Posisi Y (mm)", "Z": "Posisi Z (mm)"},
                        title="Lintasan Posisi 3D Pergelangan Tangan"
                    )
                    
                    # Tambah garis lintasan kontinu
                    fig_3d.add_trace(
                        go.Scatter3d(
                            x=df_pos["X"], y=df_pos["Y"], z=df_pos["Z"],
                            mode="lines",
                            line=dict(color="rgba(44, 62, 80, 0.4)", width=4),
                            showlegend=False
                        )
                    )
                    
                    fig_3d.update_layout(
                        margin=dict(l=0, r=0, b=0, t=40),
                        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                        height=500
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.caption("💡 Putar, seret, dan perbesar koordinat spasial 3D di atas untuk menganalisis variabilitas spasial gerakan pergelangan tangan.")
                    
                with col_2d:
                    st.markdown("##### ⏱️ Profil Kecepatan Skalar (Speed Profile)")
                    
                    time_s = np.arange(len(speed)) / FS
                    df_speed = pd.DataFrame({"Waktu (detik)": time_s, "Kecepatan (mm/s)": speed})
                    
                    fig_speed = px.line(
                        df_speed, x="Waktu (detik)", y="Kecepatan (mm/s)",
                        title="Kurva Kecepatan Terhadap Waktu"
                    )
                    fig_speed.update_traces(line_color="#2980b9", line_width=3)
                    fig_speed.update_layout(height=450, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_speed, use_container_width=True)
                    st.caption("💡 Kurva kecepatan yang halus dan memiliki satu puncak (single-bell curve) menandakan gerakan normal. Banyaknya lembah/puncak kecil menandakan ketidakstabilan motorik.")
                
                st.markdown("---")
                st.markdown("### 🧬 Turunan Kinematik Tingkat Tinggi (Akselerasi & Jerk)")
                
                col_acc, col_jrk = st.columns([1, 1])
                
                with col_acc:
                    df_acc = pd.DataFrame({"Waktu (detik)": time_s, "Akselerasi (mm/s²)": acc_mag})
                    fig_acc = px.line(
                        df_acc, x="Waktu (detik)", y="Akselerasi (mm/s²)",
                        title="Kurva Akselerasi Terhadap Waktu"
                    )
                    fig_acc.update_traces(line_color="#d35400", line_width=2.5)
                    fig_acc.update_layout(height=350, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_acc, use_container_width=True)
                    
                with col_jrk:
                    df_jrk = pd.DataFrame({"Waktu (detik)": time_s, "Sentakan Jerk (mm/s³)": jerk_mag})
                    fig_jrk = px.line(
                        df_jrk, x="Waktu (detik)", y="Sentakan Jerk (mm/s³)",
                        title="Kurva Jerk (Sentakan) Terhadap Waktu"
                    )
                    fig_jrk.update_traces(line_color="#c0392b", line_width=2)
                    fig_jrk.update_layout(height=350, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_jrk, use_container_width=True)
                    
                # Koordinat spasial individual 2D
                with st.expander("🌐 Lihat Grafik Detail Koordinat Spasial 2D (X, Y, Z)"):
                    df_coords = pd.DataFrame(pos_filtered, columns=["Posisi X (mm)", "Posisi Y (mm)", "Posisi Z (mm)"])
                    df_coords["Waktu (detik)"] = time_s
                    df_coords_melt = df_coords.melt(id_vars=["Waktu (detik)"], var_name="Koordinat Spasial", value_name="Posisi (mm)")
                    
                    fig_coords = px.line(
                        df_coords_melt, x="Waktu (detik)", y="Posisi (mm)",
                        color="Koordinat Spasial",
                        title="Perubahan Posisi Spasial 3D Terhadap Waktu"
                    )
                    fig_coords.update_layout(height=400)
                    st.plotly_chart(fig_coords, use_container_width=True)
            
            # --------------------------------------------------------------------
            # TAB 3: PERFORMA & ANALISIS MODEL
            # --------------------------------------------------------------------
            with tab_model:
                st.markdown("### 📊 Ringkasan Performa Model Akademik")
                st.markdown("Evaluasi performa klasifikasi 3-kelas FMA dari ke-6 model Machine Learning yang telah dilatih pada dataset:")
                
                # Tampilkan tabel perbandingan metrik
                st.dataframe(eval_df.style.format({
                    "Accuracy": "{:.2%}",
                    "Macro F1": "{:.4f}",
                    "Weighted F1": "{:.4f}",
                    "AUC-ROC": "{:.4f}"
                }).highlight_max(axis=0, color="#d5fcd5", subset=["Accuracy", "Macro F1", "Weighted F1", "AUC-ROC"]), use_container_width=True)
                
                st.markdown("---")
                
                col_best, col_cm = st.columns([1, 1])
                
                with col_best:
                    st.markdown("##### 🏆 Rekomendasi Model Terbaik:")
                    best_model = eval_df.iloc[0]
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 6px solid #2ecc71; background-color: rgba(46, 204, 113, 0.08); margin-bottom: 20px;">
                            <div class="metric-card-title">Model Pilihan Utama</div>
                            <div class="metric-card-value" style="color: #2ecc71; font-size: 26px;">{best_model['Model']}</div>
                            <div style="font-weight: 600; margin-top: 5px; color: #2c3e50;">Akurasi: {best_model['Accuracy']:.2%} | Macro F1: {best_model['Macro F1']:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Feature Importance dari model terbaik (Random Forest atau XGBoost jika terpilih)
                    st.markdown(f"##### 🎯 Feature Importance - Model Terpilih ({selected_model_name})")
                    current_model = trained_models[selected_model_name]
                    
                    if hasattr(current_model, "feature_importances_"):
                        importances = current_model.feature_importances_
                        sorted_indices = np.argsort(importances)
                        
                        feat_imp_df = pd.DataFrame({
                            "Parameter Kinematik": [FEATURES_TO_KEEP[i].replace("_", " ").title() for i in sorted_indices],
                            "Importance": [importances[i] for i in sorted_indices]
                        })
                        
                        fig_imp = px.bar(
                            feat_imp_df, x="Importance", y="Parameter Kinematik",
                            orientation='h',
                            title=f"Tingkat Relevansi Fitur - {selected_model_name}"
                        )
                        fig_imp.update_traces(marker_color="#3498db")
                        fig_imp.update_layout(height=400, margin=dict(t=40, b=10, l=10, r=10))
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info(f"Model '{selected_model_name}' tidak mendukung kalkulasi nilai Feature Importance secara langsung. Gunakan model Random Forest atau XGBoost untuk melihat pengaruh kontribusi fitur secara spasial.")
                        
                with col_cm:
                    st.markdown(f"##### 🎯 Confusion Matrix - {selected_model_name}")
                    cm = confusion_matrices[selected_model_name]
                    
                    classes_labels = [f"Actual {i}" for i in range(3)]
                    pred_labels = [f"Pred {i}" for i in range(3)]
                    
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted Class", y="Actual Class", color="Jumlah Sampel"),
                        x=pred_labels,
                        y=classes_labels,
                        text_auto=True,
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_layout(
                        title=f"Matriks Kekacauan ({selected_model_name})",
                        height=420,
                        margin=dict(t=50, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            # --------------------------------------------------------------------
            # TAB 4: DATASET INSIGHTS & STATISTIK
            # --------------------------------------------------------------------
            with tab_insights:
                st.markdown("### 📊 Eksplorasi Statistik Dataset Riset FMA")
                
                # Membaca dataset asli
                df_insights = pd.read_csv(CSV_PATH)
                df_insights.loc[df_insights["label"] == 0, "fma_class"] = 0
                incorrect_mask = df_insights["label"] == 1
                sparc_incorrect = df_insights.loc[incorrect_mask, "sparc"]
                median_sparc = sparc_incorrect.median()
                df_insights.loc[incorrect_mask & (df_insights["sparc"] >= median_sparc), "fma_class"] = 1
                df_insights.loc[incorrect_mask & (df_insights["sparc"] < median_sparc), "fma_class"] = 2
                df_insights["fma_class"] = df_insights["fma_class"].astype(int)
                
                col_stat_desc, col_boxplot = st.columns([2, 3])
                
                with col_stat_desc:
                    st.markdown("##### 📈 Statistik Deskriptif Populasi Dataset")
                    feature_selected = st.selectbox(
                        "Pilih Parameter Kinematik untuk Analisis",
                        options=FEATURES_TO_KEEP,
                        help="Pilih metrik kinematik untuk melihat distribusi populasi dan uji signifikansi statistik."
                    )
                    
                    # Deskriptif per fma_class
                    desc_fma = df_insights.groupby("fma_class")[feature_selected].describe()[["mean", "std", "min", "max"]].T
                    desc_fma.columns = [f"Kelas {i}" for i in range(3)]
                    st.markdown(f"**Ringkasan Statistik - {feature_selected.replace('_', ' ').title()}:**")
                    st.dataframe(desc_fma.style.format("{:.4f}"))
                    
                    # Uji signifikansi Mann-Whitney U
                    st.markdown("##### 🔬 Uji Signifikansi Mann-Whitney U (Kelas 0 vs Kelas 2)")
                    class0_vals = df_insights[df_insights["fma_class"] == 0][feature_selected]
                    class2_vals = df_insights[df_insights["fma_class"] == 2][feature_selected]
                    
                    stat, pval = mannwhitneyu(class0_vals, class2_vals, alternative='two-sided')
                    sig = "SIGNIFIKAN (p < 0.05)" if pval < 0.05 else "TIDAK SIGNIFIKAN"
                    
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 5px solid #9b59b6; background-color: rgba(155, 89, 182, 0.05); padding: 15px;">
                            <div class="metric-card-title">Signifikansi Statistik (Normal vs Gangguan Berat)</div>
                            <div style="font-size: 16px; font-weight: 700; color: #8e44ad; margin-bottom: 5px;">{sig}</div>
                            <div style="font-size: 13px; color: #7f8c8d;">U-Stat: <b>{stat:.1f}</b> | P-Value: <b>{pval:.6e}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                with col_boxplot:
                    st.markdown(f"##### 📦 Boxplot Distribusi Fitur: `{feature_selected}`")
                    
                    df_insights_renamed = df_insights.copy()
                    df_insights_renamed["Kelas FMA"] = df_insights_renamed["fma_class"].map(FMA_LABELS)
                    
                    fig_box = px.box(
                        df_insights_renamed,
                        x="Kelas FMA",
                        y=feature_selected,
                        color="Kelas FMA",
                        color_discrete_map={
                            FMA_LABELS[0]: CLASS_COLORS[0],
                            FMA_LABELS[1]: CLASS_COLORS[1],
                            FMA_LABELS[2]: CLASS_COLORS[2]
                        },
                        title=f"Distribusi Spasial Fitur '{feature_selected.replace('_', ' ').title()}'"
                    )
                    fig_box.update_layout(showlegend=False, height=450)
                    st.plotly_chart(fig_box, use_container_width=True)
                    
else:
    # Tampilan awal kosong ketika tidak ada berkas yang dimuat
    st.markdown(
        """
        <div style="text-align: center; padding: 80px 20px; background-color: #f8f9fa; border-radius: 12px; border: 2px dashed #bdc3c7;">
            <span style="font-size: 64px;">🦾</span>
            <h3 style="color: #7f8c8d; margin-top: 15px;">Aplikasi Siap Digunakan</h3>
            <p style="color: #95a5a6; max-width: 500px; margin: 0 auto 20px auto;">
                Harap unggah berkas spasial pergerakan pergelangan tangan (RWRA) dalam format posisi 3D atau pilih salah satu contoh gerakan bawaan dari sidebar untuk memulai analisis kinematik klinis FMA.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
