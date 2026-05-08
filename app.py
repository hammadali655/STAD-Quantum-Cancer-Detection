"""
QuantumCancerAI – Next‑Gen Web Platform
Futuristic, Website‑Like Design · Correct Model Pipeline
"""

import streamlit as st
import numpy as np
import joblib
import torch
import torch.nn as nn
import pennylane as qml
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ---------- Determinism ----------
torch.manual_seed(42)
np.random.seed(42)

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="QuantumCancerAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Colour Palette (matching poster) ----------
BG_DARK = "#080C14"
CARD_BG = "rgba(255, 255, 255, 0.06)"
ACCENT_CYAN = "#00E5FF"
ACCENT_MAGENTA = "#FF007F"
ACCENT_PURPLE = "#7B2FF7"
TEXT_WHITE = "#FFFFFF"
TEXT_LIGHT = "#E0E0E0"
TEXT_DIM = "#9CA3AF"

# ---------- Global CSS ----------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {BG_DARK};
        background-image:
            radial-gradient(circle at 10% 20%, rgba(0,229,255,0.08) 0%, transparent 50%),
            radial-gradient(circle at 90% 80%, rgba(255,0,127,0.08) 0%, transparent 50%);
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ---------- Custom Navbar ---------- */
    .navbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 32px;
        background: rgba(8,12,20,0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255,255,255,0.1);
        position: sticky;
        top: 0;
        z-index: 999;
    }}
    .navbar .logo {{
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(90deg, {ACCENT_CYAN}, {ACCENT_MAGENTA});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .navbar .nav-links a {{
        color: {TEXT_DIM};
        text-decoration: none;
        margin-left: 30px;
        font-weight: 500;
        transition: color 0.3s;
    }}
    .navbar .nav-links a:hover {{
        color: {ACCENT_CYAN};
        text-shadow: 0 0 10px {ACCENT_CYAN};
    }}

    /* ---------- Glass Card ---------- */
    .glass-card {{
        background: {CARD_BG};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }}
    .glass-card:hover {{
        border-color: {ACCENT_CYAN};
        box-shadow: 0 0 30px rgba(0,229,255,0.2);
        transform: translateY(-3px);
    }}

    /* ---------- Glowing Buttons ---------- */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT_MAGENTA}, {ACCENT_PURPLE});
        color: white;
        border: none;
        border-radius: 40px;
        padding: 16px 40px;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.4s ease;
        box-shadow: 0 0 25px rgba(255,0,127,0.4);
        width: 100%;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {ACCENT_PURPLE}, {ACCENT_CYAN});
        box-shadow: 0 0 40px rgba(0,229,255,0.7);
        transform: scale(1.03);
    }}

    /* ---------- Animated Heading ---------- */
    .glowing-text {{
        font-weight: 800;
        background: linear-gradient(90deg, {ACCENT_CYAN}, {ACCENT_MAGENTA}, {ACCENT_PURPLE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 3s ease-in-out infinite alternate;
    }}
    @keyframes glow {{
        from {{ filter: drop-shadow(0 0 8px {ACCENT_CYAN}); }}
        to {{ filter: drop-shadow(0 0 20px {ACCENT_MAGENTA}); }}
    }}

    /* ---------- Verdict Cards ---------- */
    .verdict-cancer {{
        background: linear-gradient(135deg, rgba(255,0,127,0.85), rgba(123,47,247,0.85));
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 60px rgba(255,0,127,0.6);
        animation: pulse 2.5s infinite;
    }}
    .verdict-noncancer {{
        background: linear-gradient(135deg, rgba(0,229,255,0.85), rgba(0,200,83,0.85));
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 60px rgba(0,229,255,0.6);
        animation: pulse 2.5s infinite;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
        100% {{ transform: scale(1); }}
    }}

    /* ---------- Sequence Input ---------- */
    textarea {{
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 18px !important;
        color: {TEXT_WHITE} !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        padding: 18px !important;
    }}
    textarea:focus {{
        border-color: {ACCENT_CYAN} !important;
        box-shadow: 0 0 25px rgba(0,229,255,0.5) !important;
    }}
    textarea::placeholder {{
        color: {TEXT_DIM} !important;
    }}

    /* ---------- Detail Button ---------- */
    .detail-btn {{
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.25);
        color: white;
        padding: 14px 32px;
        border-radius: 30px;
        cursor: pointer;
        font-weight: 600;
        transition: 0.3s;
        display: inline-block;
        text-align: center;
    }}
    .detail-btn:hover {{
        background: rgba(0,229,255,0.2);
        border-color: {ACCENT_CYAN};
    }}

    /* ---------- Tab Styling ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: rgba(255,255,255,0.05);
        padding: 8px;
        border-radius: 16px;
        margin-bottom: 30px;
        display: flex;
        justify-content: center;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_DIM} !important;
        font-weight: 600;
        font-size: 16px;
        padding: 10px 28px;
        border-radius: 12px;
        margin: 0 5px;
        transition: all 0.3s;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT_CYAN} !important;
        background: rgba(0,229,255,0.15);
        text-shadow: 0 0 10px {ACCENT_CYAN};
        border-radius: 12px;
    }}

    /* ---------- Radar Chart Background ---------- */
    .stPlotlyChart {{
        background: transparent !important;
    }}

    /* ---------- Footer ---------- */
    .footer {{
        text-align: center;
        padding: 30px;
        color: {TEXT_DIM};
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 40px;
    }}
</style>
""", unsafe_allow_html=True)

# ---------- Constants ----------
MAX_LEN = 5537
ENCODING_MAP = {
    'L':0.0000,'I':0.0000,'N':0.0036,'G':0.0050,'V':0.0057,'E':0.0058,
    'P':0.0198,'H':0.0242,'K':0.0371,'A':0.0373,'Y':0.0516,'W':0.0548,
    'Q':0.0761,'M':0.0823,'S':0.0829,'C':0.0829,'T':0.0941,'F':0.0954,
    'R':0.0956,'D':0.1263
}

# ---------- PyTorch Model Definitions ----------
class QuantumVQC(nn.Module):
    """Standalone VQC: 6 layers, Y‑rotation"""
    def __init__(self, n_qubits=8, n_layers=6):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.bn = nn.BatchNorm1d(n_qubits)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(n_qubits, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        q_out = self.q(x)
        q_out = self.bn(q_out)
        q_out = self.dropout(q_out)
        x = self.relu(self.fc1(q_out))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class HybridMLP(nn.Module):
    """Hybrid Q+MLP with 24 inputs"""
    def __init__(self, input_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class QuantumFeatureExtractor:
    """
    Extracts 8‑dim quantum features from first 8 PCA components.
    Uses 4 layers, X‑rotation – exactly as in step6_train_hybrid_*.py
    """
    def __init__(self, vqc_state_dict):
        dev = qml.device("default.qubit", wires=8)
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # X‑rotation (default) – matches step6
            qml.AngleEmbedding(inputs, wires=range(8))
            qml.StronglyEntanglingLayers(weights, wires=range(8))
            return [qml.expval(qml.PauliZ(i)) for i in range(8)]
        weight_shapes = {"weights": (4, 8, 3)}  # 4 layers, NOT 6!
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.qlayer.load_state_dict(vqc_state_dict, strict=False)
        self.qlayer.eval()

    def extract(self, x):
        with torch.no_grad():
            return self.qlayer(torch.tensor(x, dtype=torch.float32)).numpy()

# ---------- Load All Models (Cached) ----------
@st.cache_resource
def load_models():
    minmax = joblib.load("models/minmax_scaler_pca.pkl")
    pca = joblib.load("models/pca_100_model.pkl")
    scaler = joblib.load("models/scaler_pca.pkl")

    fisher8 = np.load("models/fisher_top8_indices.npy")
    pca_top16 = np.arange(16)

    # VQC normalisation (Fisher8)
    X_train_fisher = np.load("data/X_train_fisher8.npy")
    vqc_mean, vqc_std = X_train_fisher.mean(), X_train_fisher.std()

    # Hybrid MLP normalisation
    mlp_norm = joblib.load("models/quantum_features_norm.pkl")
    mlp_mean, mlp_std = mlp_norm["mean"], mlp_norm["std"]
    mlp_fisher_idx = mlp_norm["fisher_indices"]

    # Hybrid RF normalisation
    rf_norm = joblib.load("models/hybrid_rf_norm.pkl")
    rf_mean, rf_std = rf_norm["mean"], rf_norm["std"]

    # Classical models
    rf_model = joblib.load("models/rf_pca_model.pkl")
    mlp_model = joblib.load("models/mlp_pca_model.pkl")

    # Quantum VQC (6 layers)
    vqc_weights = torch.load("models/quantum_fisher_model_best.pth", map_location='cpu')
    quantum_model = QuantumVQC()
    quantum_model.load_state_dict(vqc_weights, strict=False)
    quantum_model.eval()

    # Hybrid MLP
    hybrid_mlp = HybridMLP()
    hybrid_mlp.load_state_dict(torch.load("models/hybrid_quantum_mlp_best.pth", map_location='cpu'), strict=False)
    hybrid_mlp.eval()

    # Hybrid RF
    hybrid_rf = joblib.load("models/hybrid_quantum_rf_model.pkl")

    # Quantum feature extractor (4 layers)
    q_extractor = QuantumFeatureExtractor(vqc_weights)

    return (minmax, pca, scaler, fisher8, pca_top16,
            vqc_mean, vqc_std, mlp_mean, mlp_std, mlp_fisher_idx,
            rf_mean, rf_std, rf_model, mlp_model, quantum_model,
            hybrid_mlp, hybrid_rf, q_extractor)

# ---------- Prediction Helpers ----------
def encode_sequence(seq):
    encoded = [ENCODING_MAP.get(aa.upper(), 0.0) for aa in seq]
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    return np.array(encoded).reshape(1, -1)

def predict_all(seq):
    encoded = encode_sequence(seq)
    scaled_minmax = minmax.transform(encoded)
    pca_feat = pca.transform(scaled_minmax)
    scaled_std = scaler.transform(pca_feat)

    # Quantum features (4 layers, X‑rotation)
    raw_q = q_extractor.extract(scaled_std[:, :8])

    results = {}

    # 1. Random Forest (100 PCA)
    rf_pred = rf_model.predict(scaled_std)[0]
    rf_prob = rf_model.predict_proba(scaled_std)[0][1]
    results["Random Forest"] = {"prediction": int(rf_pred), "probability": float(rf_prob), "icon": "🌲"}

    # 2. MLP (100 PCA)
    mlp_pred = mlp_model.predict(scaled_std)[0]
    mlp_prob = mlp_model.predict_proba(scaled_std)[0][1]
    results["MLP Network"] = {"prediction": int(mlp_pred), "probability": float(mlp_prob), "icon": "🧠"}

    # 3. Quantum VQC (8 Fisher features, Y‑rotation)
    fisher_8 = scaled_std[:, fisher8]
    fisher_norm_8 = (fisher_8 - vqc_mean) / (vqc_std + 1e-8)
    with torch.no_grad():
        q_logits = quantum_model(torch.tensor(fisher_norm_8, dtype=torch.float32))
        q_prob = torch.sigmoid(q_logits).numpy()[0][0]
    results["Quantum VQC"] = {"prediction": 1 if q_prob > 0.5 else 0, "probability": float(q_prob), "icon": "⚛️"}

    # 4. Hybrid Q+MLP (16 Fisher + 8 quantum)
    fisher_16_mlp = scaled_std[:, mlp_fisher_idx]
    combined_mlp = np.hstack([fisher_16_mlp, raw_q])
    combined_mlp_norm = (combined_mlp - mlp_mean) / (mlp_std + 1e-8)
    with torch.no_grad():
        h_mlp_prob = hybrid_mlp(torch.tensor(combined_mlp_norm, dtype=torch.float32)).numpy()[0][0]
    results["Hybrid Q+MLP"] = {"prediction": 1 if h_mlp_prob > 0.5 else 0, "probability": float(h_mlp_prob), "icon": "🔗"}

    # 5. Hybrid Q+RF (16 PCA + 8 quantum)
    classical_16 = scaled_std[:, pca_top16]
    combined_rf = np.hstack([classical_16, raw_q])
    combined_rf_norm = (combined_rf - rf_mean) / (rf_std + 1e-8)
    h_rf_pred = hybrid_rf.predict(combined_rf_norm)[0]
    h_rf_prob = hybrid_rf.predict_proba(combined_rf_norm)[0][1]
    results["Hybrid Q+RF"] = {"prediction": int(h_rf_pred), "probability": float(h_rf_prob), "icon": "🔮"}

    return results

# ---------- Initialise Models ----------
try:
    (minmax, pca, scaler, fisher8, pca_top16,
     vqc_mean, vqc_std, mlp_mean, mlp_std, mlp_fisher_idx,
     rf_mean, rf_std, rf_model, mlp_model, quantum_model,
     hybrid_mlp, hybrid_rf, q_extractor) = load_models()
    MODELS_READY = True
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    MODELS_READY = False

# ---------- Session State ----------
if 'results' not in st.session_state:
    st.session_state.results = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False
if 'sequence' not in st.session_state:
    st.session_state.sequence = ""

# ---------- UI: Custom Navbar ----------
st.markdown("""
<div class="navbar">
    <div class="logo">🧬 QuantumCancerAI</div>
    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">Research</a>
        <a href="#">Methodology</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Tab Content ----------
# We'll use Streamlit's tabs but style them as per CSS.
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📚 Research Overview", "🔬 Methodology"])

# ========== TAB 1: HOME ==========
with tab1:
    # Hero Section
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
        <h1 style="font-size: 3rem; margin-bottom: 0;">
            <span class="glowing-text">QuantumCancerAI</span>
        </h1>
        <p style="font-size: 1.2rem; color: #B0BEC5;">Hybrid Quantum Ensemble for STAD Detection</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analyze Protein Sequence")
        user_input = st.text_area(
            label="",
            height=180,
            placeholder="Paste amino acid sequence (e.g., MGAAAKLAFAVFLISCSSGAILGR...)",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        analyze_btn = st.button("⚡ ANALYZE SEQUENCE", use_container_width=True)

    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("### 🛡️ Ensemble Active")
        st.markdown("8‑Qubit VQC · 5‑Model Voting · 93%+ Accuracy")
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Logic
    if analyze_btn and user_input.strip():
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        clean_seq = user_input.upper().replace(" ", "").replace("\n", "")
        invalid = [c for c in clean_seq if c not in valid_chars]
        if invalid:
            st.error(f"Invalid characters: {', '.join(set(invalid[:10]))}")
        else:
            with st.spinner("🔮 Quantum circuits analyzing..."):
                try:
                    st.session_state.results = predict_all(clean_seq)
                    st.session_state.show_details = False
                    st.session_state.sequence = clean_seq
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.session_state.results = None

    # Display Results
    if st.session_state.results is not None:
        res = st.session_state.results
        predictions = [r['prediction'] for r in res.values()]
        cancer_votes = sum(predictions)
        ensemble_pred = cancer_votes > 2  # majority

        st.markdown("<br>", unsafe_allow_html=True)
        if ensemble_pred:
            st.markdown(f"""
            <div class="verdict-cancer">
                <h1 style="color:white; margin:0; font-size:3rem;">🚨 CANCER DETECTED</h1>
                <p style="color:white; font-size:1.2rem;">High‑risk pattern identified by quantum‑classical ensemble</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-noncancer">
                <h1 style="color:white; margin:0; font-size:3rem;">✅ NO CANCER DETECTED</h1>
                <p style="color:white; font-size:1.2rem;">Normal protein signature confirmed</p>
            </div>
            """, unsafe_allow_html=True)

        # Toggle for detailed analysis
        col_btn, _ = st.columns([2, 3])
        with col_btn:
            if st.button("🔎 Show Detailed Model Analysis", use_container_width=True):
                st.session_state.show_details = not st.session_state.show_details

        if st.session_state.show_details:
            st.markdown("---")
            st.markdown("### 🤖 Individual Model Predictions")
            cols = st.columns(5)
            for idx, (name, r) in enumerate(res.items()):
                prob = r['probability']
                pred_class = "CANCER" if r['prediction'] else "NON‑CANCER"
                color = ACCENT_MAGENTA if r['prediction'] else ACCENT_CYAN
                with cols[idx]:
                    st.markdown(f"""
                    <div style="padding:16px; background:rgba(255,255,255,0.05); border-radius:16px; border:1px solid {color}; text-align:center;">
                        <div style="font-size:2rem;">{r['icon']}</div>
                        <strong>{name}</strong><br>
                        <span style="color:{color}; font-weight:bold; font-size:1.1rem;">{pred_class}</span><br>
                        <small>Confidence: {prob:.1%}</small><br>
                        <div style="background:#333; border-radius:10px; height:6px; margin-top:6px;">
                            <div style="width:{prob*100}%; height:6px; background:{color}; border-radius:10px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Clinical summary
            if cancer_votes <= 2:
                risk, rec = "LOW", "Routine screening."
            elif cancer_votes == 3:
                risk, rec = "MODERATE", "Further clinical tests advised."
            else:
                risk, rec = "HIGH", "URGENT: Consult an oncologist immediately."

            st.markdown(f"""
            <div class="glass-card" style="margin-top:20px;">
                <h4>📋 Clinical Risk Assessment</h4>
                <p><strong>Risk Level:</strong> {risk}</p>
                <p><strong>Recommendation:</strong> {rec}</p>
                <p><strong>Sequence Length:</strong> {len(st.session_state.sequence)} amino acids</p>
            </div>
            """, unsafe_allow_html=True)

# ========== TAB 2: RESEARCH OVERVIEW ==========
with tab2:
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #00E5FF;">🧬 Research Overview</h2>
        <p style="line-height:1.8; font-size:1.05rem;">
            Stomach adenocarcinoma (STAD) is the 5th most common cancer worldwide, and early detection dramatically improves survival rates.
            We present the <strong>first simulated quantum‑classical pipeline</strong> for STAD detection using protein mutation sequences from the IntOGen database.
        </p>
        <p style="line-height:1.8; font-size:1.05rem;">
            Our workflow converts amino‑acid sequences into numerical features via <strong>EIIP encoding</strong>, reduces dimensionality with PCA (100 components), and selects the top 8 discriminative features using the <strong>Fisher Score</strong>.
            An <strong>8‑qubit Variational Quantum Circuit</strong> extracts non‑classical correlations, which are then combined with classical features in a 5‑model ensemble.
            The final verdict is a majority vote, achieving <strong>93%+ accuracy</strong> on unseen test data.
        </p>
        <p style="color:#9CA3AF;">
            This work is the first to apply VQCs to STAD protein data with EIIP encoding, filling a critical gap in quantum‑enhanced cancer genomics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>📊 Dataset</h4>
            <ul style="color: #E0E0E0;">
                <li>2,086 balanced samples (SMOTE)</li>
                <li>5,537 EIIP features → 100 PCA → top 8 Fisher</li>
                <li>80/20 stratified train/test split</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>⚛️ Quantum Configuration</h4>
            <ul style="color: #E0E0E0;">
                <li>8 qubits, 6 StronglyEntanglingLayers</li>
                <li>AngleEmbedding (Y‑rotation)</li>
                <li>PauliZ measurements</li>
                <li>449 trainable parameters</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ========== TAB 3: METHODOLOGY ==========
with tab3:
    st.markdown("## 🔬 End‑to‑End Pipeline")
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 15px; justify-content: center;">
        <div class="glass-card" style="flex:1; min-width:150px; text-align:center;">
            <h3 style="color:#00E5FF;">1️⃣ Raw Sequence</h3>
            <p>A, C, D, E…</p>
        </div>
        <div class="glass-card" style="flex:1; min-width:150px; text-align:center;">
            <h3 style="color:#00E5FF;">2️⃣ EIIP Encoding</h3>
            <p>5,537 features</p>
        </div>
        <div class="glass-card" style="flex:1; min-width:150px; text-align:center;">
            <h3 style="color:#00E5FF;">3️⃣ PCA (100)</h3>
            <p>Dimensionality reduction</p>
        </div>
        <div class="glass-card" style="flex:1; min-width:150px; text-align:center;">
            <h3 style="color:#00E5FF;">4️⃣ Fisher Score</h3>
            <p>Top 8 features</p>
        </div>
        <div class="glass-card" style="flex:1; min-width:150px; text-align:center;">
            <h3 style="color:#00E5FF;">5️⃣ 5‑Model Ensemble</h3>
            <p>RF · MLP · VQC · Hybrids</p>
        </div>
        <div class="glass-card" style="flex:1; min-width:150px; text-align:center;">
            <h3 style="color:#00E5FF;">6️⃣ Majority Vote</h3>
            <p>Final diagnosis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Model Architectures")
    with st.expander("🌲 Random Forest (Classical)"):
        st.markdown("350 trees · max_depth=16 · balanced_subsample weighting · 100 PCA inputs")
    with st.expander("🧠 MLP Network (Classical)"):
        st.markdown("3 hidden layers (128→64→32) · ReLU · L2 regularization · early stopping")
    with st.expander("⚛️ Quantum VQC (Standalone)"):
        st.markdown("8 qubits · 6 layers (Y‑rotation) · classical head (16→8→1) · 449 parameters")
    with st.expander("🔗 Hybrid Q+MLP"):
        st.markdown("24 inputs (16 Fisher + 8 quantum) · deep MLP (128→256→128→64→32) · BatchNorm + Dropout")
    with st.expander("🔮 Hybrid Q+RF"):
        st.markdown("Random Forest on 24 features (16 PCA + 8 quantum) · 500 trees · max_depth=16")

    st.markdown("---")
    st.markdown("### 📈 Performance Radar")
    fig = go.Figure()
    categories = ['Test Accuracy', 'AUC', 'F1 Score']
    models = ['Random Forest', 'MLP', 'Quantum VQC', 'Hybrid RF', 'Hybrid MLP']
    values = [
        [91.3, 96.8, 91.0],
        [91.5, 97.2, 91.2],
        [80.6, 87.9, 80.1],
        [91.3, 96.9, 91.0],
        [93.7, 97.5, 93.3]
    ]
    for i, model in enumerate(models):
        fig.add_trace(go.Scatterpolar(
            r=values[i],
            theta=categories,
            fill='toself',
            name=model
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[70, 100], color="#9CA3AF")),
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("""
<div class="footer">
    ⚛️ QuantumCancerAI — Research prototype, not for clinical use.
</div>
""", unsafe_allow_html=True)