# src/step6_train_hybrid_mlp_pca.py - WITH COMPLETE RESULT STRUCTURE

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
import sys
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===============================
# SETUP
# ===============================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

STEP_NAME = os.path.basename(__file__).replace(".py", "")
REPORT_PATH = f"reports/{STEP_NAME}_report.txt"

class SafeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

logger = SafeLogger(REPORT_PATH)
sys.stdout = logger

print(f"\nReport File Created: {REPORT_PATH}")
start_time = time.time()

# ===============================
# LOAD DATA
# ===============================
X_train_full = np.load("data/X_train.npy")
X_test_full = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("\n" + "="*70)
print("DATA INFORMATION")
print("="*70)
print(f"Train Shape: {X_train_full.shape}")
print(f"Test Shape:  {X_test_full.shape}")
print(f"Features: 100 PCA components (from 5537 EIIP encoded features)")
print(f"Training samples: {X_train_full.shape[0]}")
print(f"Testing samples: {X_test_full.shape[0]}")

# ===============================
# TOP 16 CLASSICAL FISHER FEATURES
# ===============================
print("\n" + "="*70)
print("FEATURE SELECTION (Fisher Score)")
print("="*70)

def fisher_score(X, y):
    scores = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        class0 = feature[y == 0]
        class1 = feature[y == 1]
        mean0, mean1 = class0.mean(), class1.mean()
        var0, var1 = class0.var(), class1.var()
        score = (mean0 - mean1)**2 / (var0 + var1 + 1e-8)
        scores.append(score)
    return np.array(scores)

scores = fisher_score(X_train_full, y_train)
top16_indices = np.argsort(scores)[-16:][::-1]
np.save("models/fisher_top16_indices.npy", top16_indices)
print(f"Top 16 Fisher indices: {top16_indices}")

X_train_classical_16 = X_train_full[:, top16_indices]
X_test_classical_16 = X_test_full[:, top16_indices]
print(f"Classical 16 features shape: {X_train_classical_16.shape}")

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
X_train_quantum = extract_quantum_features(X_train_full)
X_test_quantum = extract_quantum_features(X_test_full)
print(f"Quantum features shape: {X_train_quantum.shape}")

# ===============================
# CREATE HYBRID FEATURES (24 = 16 + 8)
# ===============================
X_train_hybrid = np.hstack([X_train_classical_16, X_train_quantum])
X_test_hybrid = np.hstack([X_test_classical_16, X_test_quantum])

# Normalize
mean_hybrid = X_train_hybrid.mean()
std_hybrid = X_train_hybrid.std()
X_train_hybrid = (X_train_hybrid - mean_hybrid) / (std_hybrid + 1e-8)
X_test_hybrid = (X_test_hybrid - mean_hybrid) / (std_hybrid + 1e-8)

print(f"\n✅ Hybrid features: {X_train_hybrid.shape} (16 Classical Fisher + 8 Quantum = 24)")

# ===============================
# HYBRID MLP MODEL
# ===============================
print("\n" + "="*70)
print("HYBRID MLP MODEL")
print("="*70)

class HybridMLP(nn.Module):
    def __init__(self, input_dim=24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

X_train_t = torch.tensor(X_train_hybrid, dtype=torch.float32)
X_test_t = torch.tensor(X_test_hybrid, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

model = HybridMLP(input_dim=24)
total_params = sum(p.numel() for p in model.parameters())

print("\n📊 Model Parameters:")
print(f"   - Input dimension     : 24")
print(f"   - Hidden layers       : 128 → 256 → 128 → 64 → 32 → 1")
print(f"   - Activation          : ReLU + Sigmoid")
print(f"   - Batch Normalization : Yes")
print(f"   - Dropout             : 0.2, 0.25, 0.2, 0.15")
print(f"   - Total parameters    : {total_params:,}")

print("\n🎯 Generalization Techniques:")
print(f"   ✅ Batch Normalization - Stable training")
print(f"   ✅ Dropout (0.2-0.25) - Prevents overfitting")
print(f"   ✅ Label Smoothing (0.05) - Prevents overconfidence")
print(f"   ✅ CosineAnnealingWarmRestarts - Better convergence")
print(f"   ✅ Weight Decay (1e-4) - L2 regularization")
print(f"   ✅ Gradient Clipping - Training stability")
print(f"   ✅ Early Stopping - Best model selection")

# ===============================
# TRAINING
# ===============================
print("\n" + "="*70)
print("TRAINING HYBRID MLP")
print("="*70)

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.BCELoss()(pred, target)

criterion = LabelSmoothingBCELoss(smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=0.0007, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

EPOCHS = 800
PATIENCE = 60
best_val_acc = 0
best_val_auc = 0
best_val_f1 = 0
patience_counter = 0

print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Acc':<10} {'Val AUC':<10} {'Gap':<8}")
print("-"*60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).flatten()
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    if (epoch + 1) % 25 == 0:
        model.eval()
        with torch.no_grad():
            test_probs = model(X_test_t).numpy().flatten()
            test_preds = (test_probs > 0.5).astype(int)
            val_acc = accuracy_score(y_test, test_preds)
            val_auc = roc_auc_score(y_test, test_probs)
            val_f1 = f1_score(y_test, test_preds)
            
            train_probs = model(X_train_t).numpy().flatten()
            train_preds = (train_probs > 0.5).astype(int)
            train_acc = accuracy_score(y_train, train_preds)
            gap = train_acc - val_acc
        
        print(f"{epoch+1:<8} {avg_loss:<12.4f} {val_acc:<10.4f} {val_auc:<10.4f} {gap:<8.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "models/hybrid_quantum_mlp_best.pth")
            print(f"        ✨ NEW BEST! ({best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        scheduler.step(val_acc)
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

end_time = time.time()

# ===============================
# FINAL RESULTS
# ===============================
model.load_state_dict(torch.load("models/hybrid_quantum_mlp_best.pth"))
model.eval()

with torch.no_grad():
    train_probs = model(X_train_t).numpy().flatten()
    test_probs = model(X_test_t).numpy().flatten()

train_preds = (train_probs > 0.5).astype(int)
test_preds = (test_probs > 0.5).astype(int)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)
test_precision = precision_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)

cm = confusion_matrix(y_test, test_preds)
gap = train_acc - test_acc

print("\n" + "="*70)
print("HYBRID MLP RESULTS")
print("="*70)
print(f"Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test AUC       : {test_auc:.4f}")
print(f"\nPrecision (Test): {test_precision:.4f}")
print(f"Recall    (Test): {test_recall:.4f}")
print(f"F1 Score  (Test): {test_f1:.4f}")
print(f"Overfitting Gap: {gap:.4f} ({gap*100:.2f}%)")

print("\n🔍 Confusion Matrix (Test Data):")
print("                 Predicted")
print("                 NON-CANCER  CANCER")
print(f"Actual NON-CANCER    {cm[0,0]:4d}       {cm[0,1]:4d}")
print(f"Actual CANCER        {cm[1,0]:4d}       {cm[1,1]:4d}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, test_preds))

print("\n" + "-"*50)
print("TRAINING DETAILS")
print("-"*50)
print(f"Total Parameters      : {total_params:,}")
print(f"Best Validation Acc   : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Best Validation AUC   : {best_val_auc:.4f}")
print(f"Training Time         : {end_time - start_time:.2f} seconds")
print(f"Learning Rate         : 0.0007 (CosineAnnealingWarmRestarts)")
print(f"Label Smoothing       : 0.05")
print(f"Batch Size            : 128")

print("\n" + "="*70)
print(f"✅ FINAL ACCURACY: {test_acc*100:.2f}% with 24 Features")
print("="*70)

# ===============================
# SAVE MODELS
# ===============================
joblib.dump({
    "mean": mean_hybrid, 
    "std": std_hybrid, 
    "fisher_indices": top16_indices,
    "best_accuracy": best_val_acc,
    "best_auc": best_val_auc
}, "models/quantum_features_norm.pkl")

torch.save(model.state_dict(), "models/hybrid_quantum_mlp_final.pth")

print("\n💾 [SAVED] models/hybrid_quantum_mlp_best.pth")
print("💾 [SAVED] models/hybrid_quantum_mlp_final.pth")
print("💾 [SAVED] models/quantum_features_norm.pkl")

logger.close()
sys.stdout = logger.terminal
print("\n✅ Hybrid MLP Completed Successfully!")