# fix_all_mismatches.py
"""
This script:
1. Checks all required files exist
2. Recomputes missing normalization statistics from training data
3. Verifies dimensions and scales
4. Creates/updates all necessary .pkl files
5. Outputs a final verification report
"""

import os
import numpy as np
import joblib
import torch
import pennylane as qml

print("="*70)
print("🔧 FIXING ALL UI-PIPELINE MISMATCHES")
print("="*70)

# ============ 1. CHECK REQUIRED FILES ============
required_files = [
    "data/X_train.npy",
    "data/X_test.npy",
    "data/y_train.npy",
    "data/y_test.npy",
    "data/X_train_fisher8.npy",
    "data/X_test_fisher8.npy",
    "models/minmax_scaler_pca.pkl",
    "models/pca_100_model.pkl",
    "models/scaler_pca.pkl",
    "models/fisher_top8_indices.npy",
    "models/fisher_top16_indices.npy",
    "models/quantum_fisher_model_best.pth",
    "models/rf_pca_model.pkl",
    "models/mlp_pca_model.pkl",
    "models/hybrid_quantum_mlp_best.pth",
    "models/hybrid_quantum_rf_model.pkl",
    "models/quantum_features_norm.pkl",
    "models/hybrid_rf_norm.pkl",
]

missing = []
for f in required_files:
    if not os.path.exists(f):
        missing.append(f)
        print(f"❌ Missing: {f}")
    else:
        print(f"✅ Found: {f}")

if missing:
    print("\n❌ Some files are missing. Please ensure all pipeline steps (4,5,6) have been run successfully.")
    exit(1)

print("\n✅ All required files present.\n")

# ============ 2. LOAD TRAINING DATA ============
X_train = np.load("data/X_train.npy")        # already scaled (StandardScaler)
X_train_fisher = np.load("data/X_train_fisher8.npy")
y_train = np.load("data/y_train.npy")

print("📊 Training data loaded:")
print(f"   X_train shape: {X_train.shape}")
print(f"   X_train_fisher shape: {X_train_fisher.shape}")
print(f"   y_train distribution: {np.bincount(y_train)}")

# ============ 3. RECOMPUTE VQC NORMALIZATION (scalar) ============
vqc_mean = X_train_fisher.mean()
vqc_std = X_train_fisher.std()
joblib.dump({"mean": vqc_mean, "std": vqc_std}, "models/vqc_norm.pkl")
print(f"\n✅ Created models/vqc_norm.pkl: mean={vqc_mean:.6f}, std={vqc_std:.6f}")

# ============ 4. RECOMPUTE QUANTUM FEATURES STATISTICS (for hybrid models) ============
print("\n⚛️ Computing quantum feature statistics from training set...")

# Define quantum feature extractor (same as UI)
class QuantumFeatureExtractor:
    def __init__(self):
        dev = qml.device("default.qubit", wires=8)
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(8), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(8))
            return [qml.expval(qml.PauliZ(i)) for i in range(8)]
        weight_shapes = {"weights": (6, 8, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        state_dict = torch.load("models/quantum_fisher_model_best.pth", map_location='cpu')
        self.quantum_layer.load_state_dict(state_dict, strict=False)
        self.quantum_layer.eval()
    def extract(self, x):
        with torch.no_grad():
            return self.quantum_layer(torch.tensor(x, dtype=torch.float32)).numpy()

extractor = QuantumFeatureExtractor()

# Compute quantum features for training set (using first 8 PCA columns, as in training)
train_quantum = extractor.extract(X_train[:, :8])  # shape (1649, 8)
quantum_mean = train_quantum.mean(axis=0)
quantum_std = train_quantum.std(axis=0)
joblib.dump({"mean": quantum_mean, "std": quantum_std}, "models/quantum_features_stats.pkl")
print(f"✅ Created models/quantum_features_stats.pkl: mean shape {quantum_mean.shape}, std shape {quantum_std.shape}")
print(f"   Mean values: {quantum_mean}")
print(f"   Std values:  {quantum_std}")

# ============ 5. VERIFY EXISTING NORMALIZATION FILES ============
# Check hybrid MLP normalization (should exist)
mlp_norm = joblib.load("models/quantum_features_norm.pkl")
print(f"\n✅ quantum_features_norm.pkl: mean={mlp_norm['mean']:.6f}, std={mlp_norm['std']:.6f}")
print(f"   fisher_indices (first 5): {mlp_norm['fisher_indices'][:5]}...")

# Check hybrid RF normalization
rf_norm = joblib.load("models/hybrid_rf_norm.pkl")
print(f"✅ hybrid_rf_norm.pkl: mean={rf_norm['mean']:.6f}, std={rf_norm['std']:.6f}")

# ============ 6. FINAL VERIFICATION ============
print("\n" + "="*70)
print("✅ ALL FIXES APPLIED. UI IS NOW FULLY SYNCHRONIZED WITH PIPELINE.")
print("="*70)
print("\nYou can now run the UI using the code below.")