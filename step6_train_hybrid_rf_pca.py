# src/step6_train_hybrid_rf_pca.py - PCA VERSION (16 PCA + 8 Quantum)

import numpy as np
import torch
import pennylane as qml
import joblib
import os
import sys
import time
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

# ===============================
# REPORT LOGGING SETUP
# ===============================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

STEP_NAME = os.path.basename(__file__).replace(".py", "")
REPORT_PATH = f"reports/{STEP_NAME}_report.txt"

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

logger = TeeLogger(REPORT_PATH)
sys.stdout = logger

print(f"\nReport File Created: {REPORT_PATH}")
start_time = time.time()

# ===============================
# LOAD DATA
# ===============================
X_train_classical = np.load("data/X_train.npy")
X_test_classical = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("\n" + "="*70)
print("🔥 HYBRID RF: 24 FEATURES (16 PCA + 8 Quantum)")
print("="*70)

# ===============================
# TOP 16 PCA FEATURES (NO FISHER SELECTION)
# ===============================
print("\n" + "="*70)
print("FEATURE SELECTION (Top 16 PCA Components)")
print("="*70)

# Directly take first 16 PCA components (PC1 to PC16)
top16_indices = np.arange(16)  # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
np.save("models/pca_top16_indices.npy", top16_indices)
print(f"Top 16 PCA indices: {top16_indices}")

X_train_classical_16 = X_train_classical[:, top16_indices]
X_test_classical_16 = X_test_classical[:, top16_indices]
print(f"Classical 16 PCA features shape: {X_train_classical_16.shape}")

# ===============================
# QUANTUM FEATURES (Step 5 - NO RETRAINING)
# ===============================
print("\n" + "="*70)
print("QUANTUM FEATURE EXTRACTION (Step 5 VQC)")
print("="*70)

n_qubits = 8
n_layers = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

step5_weights = torch.load("models/quantum_fisher_model_best.pth", map_location=torch.device('cpu'))
quantum_layer.load_state_dict(step5_weights, strict=False)
print("✅ Step 5 Quantum weights LOADED! (No retraining)")

def extract_quantum_features(X):
    with torch.no_grad():
        X_input = torch.tensor(X[:, :n_qubits], dtype=torch.float32)
        return quantum_layer(X_input).numpy()

print("Extracting quantum features...")
quantum_train = extract_quantum_features(X_train_classical)
quantum_test = extract_quantum_features(X_test_classical)
print(f"Quantum features shape: {quantum_train.shape}")

# ===============================
# CREATE HYBRID FEATURES (16 PCA + 8 Quantum = 24)
# ===============================
X_train_hybrid = np.hstack([X_train_classical_16, quantum_train])
X_test_hybrid = np.hstack([X_test_classical_16, quantum_test])

# Normalize
mean_hybrid = X_train_hybrid.mean()
std_hybrid = X_train_hybrid.std()
X_train_hybrid = (X_train_hybrid - mean_hybrid) / (std_hybrid + 1e-8)
X_test_hybrid = (X_test_hybrid - mean_hybrid) / (std_hybrid + 1e-8)

print(f"\n✅ Hybrid features: {X_train_hybrid.shape} (16 PCA + 8 Quantum = 24)")

# ===============================
# OPTIMAL RF PARAMETERS
# ===============================
print("\n" + "="*70)
print("HYBRID RF MODEL")
print("="*70)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\n📊 Model Parameters:")
print(f"   - n_estimators      : 500")
print(f"   - max_depth         : 16")
print(f"   - min_samples_split : 5")
print(f"   - min_samples_leaf  : 2")
print(f"   - max_features      : sqrt")
print(f"   - bootstrap         : True")
print(f"   - class_weight      : balanced_subsample")
print(f"   - random_state      : 42")

print("\n🎯 Generalization Techniques:")
print(f"   ✅ n_estimators (500) - Good balance")
print(f"   ✅ max_depth (16) - Moderate depth")
print(f"   ✅ min_samples_split (5) - Balanced")
print(f"   ✅ min_samples_leaf (2) - Good granularity")
print(f"   ✅ sqrt(max_features) - Reduces overfitting")
print(f"   ✅ balanced_subsample - Handles class imbalance")

# ===============================
# CROSS-VALIDATION
# ===============================
print("\n" + "="*70)
print("CROSS-VALIDATION (5-fold)")
print("="*70)

cv_scores = cross_val_score(rf, X_train_hybrid, y_train, cv=5, scoring='accuracy')
print(f"Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ===============================
# TRAIN FULL MODEL
# ===============================
print("\n" + "="*70)
print("TRAINING HYBRID RF")
print("="*70)

rf.fit(X_train_hybrid, y_train)
print("Training Completed!")

# ===============================
# FINAL EVALUATION
# ===============================
print("\n" + "="*70)
print("HYBRID RF RESULTS")
print("="*70)

train_preds = rf.predict(X_train_hybrid)
test_preds = rf.predict(X_test_hybrid)
train_probs = rf.predict_proba(X_train_hybrid)[:, 1]
test_probs = rf.predict_proba(X_test_hybrid)[:, 1]

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
gap = train_acc - test_acc

cm = confusion_matrix(y_test, test_preds)

print(f"Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Train AUC      : {train_auc:.4f}")
print(f"Test AUC       : {test_auc:.4f}")
print(f"\nPrecision (Test): {precision:.4f}")
print(f"Recall    (Test): {recall:.4f}")
print(f"F1 Score  (Test): {f1:.4f}")
print(f"Overfitting Gap: {gap:.4f} ({gap*100:.2f}%)")

print("\n🔍 Confusion Matrix (Test Data):")
print("                 Predicted")
print("                 NON-CANCER  CANCER")
print(f"Actual NON-CANCER    {cm[0,0]:4d}       {cm[0,1]:4d}")
print(f"Actual CANCER        {cm[1,0]:4d}       {cm[1,1]:4d}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, test_preds))

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(rf, "models/hybrid_quantum_rf_model.pkl")
joblib.dump({"mean": mean_hybrid, "std": std_hybrid, "pca_indices": top16_indices}, "models/hybrid_rf_norm.pkl")

print("\n[SAVED] models/hybrid_quantum_rf_model.pkl")
print("[SAVED] models/hybrid_rf_norm.pkl")

print("\n" + "="*70)
print("STEP 6 COMPLETED")
print("="*70)

logger.close()
sys.stdout = logger.terminal
print("\n✅ Hybrid RF Completed Successfully!")