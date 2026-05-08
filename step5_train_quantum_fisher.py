# src/step5_train_quantum_fisher.py - WITH IMPROVED OUTPUT STRUCTURE (CODE SAME)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import time

# ===============================
# REPORT LOGGING SETUP
# ===============================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

STEP_NAME = os.path.basename(__file__).replace(".py", "")
REPORT_PATH = f"reports/{STEP_NAME}_report.txt"
report_file = open(REPORT_PATH, "w", encoding="utf-8")

class Logger(object):
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        pass

sys.stdout = Logger(sys.stdout, report_file)
print(f"\nReport File Created: {REPORT_PATH}")

# ===============================
# LOAD FISHER SELECTED DATA
# ===============================
X_train = np.load("data/X_train_fisher8.npy")
X_test  = np.load("data/X_test_fisher8.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")

print("\n" + "="*70)
print("DATA INFORMATION")
print("="*70)
print(f"Train Shape: {X_train.shape}")
print(f"Test Shape:  {X_test.shape}")
print(f"Features: {X_train.shape[1]} Fisher-selected features (from 100 PCA components)")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Class distribution - Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)}")

# ===============================
# IMPROVED NORMALIZATION
# ===============================
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / (std + 1e-8)
X_test  = (X_test - mean) / (std + 1e-8)

# ===============================
# TORCH TENSORS
# ===============================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32)

# ===============================
# IMPROVED QUANTUM MODEL (MORE QUBITS + MORE LAYERS)
# ===============================
n_qubits = 8
n_layers = 6  # Increased from 4 to 6 for better learning

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Improved encoding
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')  # Changed to Y rotation
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class ImprovedQuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = quantum_layer
        # Added batch norm and dropout for stability
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
        out = self.fc3(x)
        return out

model = ImprovedQuantumClassifier()
total_params = sum(p.numel() for p in model.parameters())

print("\n" + "="*70)
print("QUANTUM VQC MODEL")
print("="*70)
print("\n📊 Model Parameters:")
print(f"   - Qubits            : {n_qubits}")
print(f"   - Layers            : {n_layers}")
print(f"   - Encoding          : AngleEmbedding (Y-rotation)")
print(f"   - Entanglement      : StronglyEntanglingLayers")
print(f"   - Post-processing   : BatchNorm + Dropout + 3 FC layers")
print(f"   - Total parameters  : {total_params:,}")

print("\n🎯 Generalization Techniques:")
print(f"   ✅ Y-rotation encoding - Better feature mapping")
print(f"   ✅ Batch Normalization - Stable training")
print(f"   ✅ Dropout (0.2) - Prevents overfitting")
print(f"   ✅ Gradient Clipping (1.0) - Training stability")
print(f"   ✅ Weight Decay (0.001) - L2 regularization")
print(f"   ✅ Learning Rate Scheduling - Better convergence")
print(f"   ✅ Early Stopping (patience=20) - Best model selection")

# ===============================
# IMPROVED LOSS & OPTIMIZER
# ===============================
# Calculate class weights for imbalance
class_counts = np.bincount(y_train)
class_weights = torch.tensor([1.0, class_counts[0] / class_counts[1]])
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

# Better optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Larger batch size for stability
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,  # Increased from 16
    shuffle=True
)

EPOCHS = 300
PATIENCE = 20  # Increased patience
best_loss = float("inf")
best_acc = 0
epochs_no_improve = 0

print("\n" + "="*70)
print("TRAINING QUANTUM VQC")
print("="*70)
print(f"{'Epoch':<8} {'Loss':<10} {'Val Acc':<10}")
print("-"*35)

start_time = time.time()

# ===============================
# TRAINING LOOP WITH VALIDATION
# ===============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).flatten()
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    # Validation every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_test_t).flatten()
            val_probs = torch.sigmoid(val_logits).numpy()
            val_preds = (val_probs > 0.5).astype(int)
            val_acc = accuracy_score(y_test, val_preds)
        
        print(f"{epoch+1:<8} {avg_loss:<10.4f} {val_acc:<10.4f}")
        
        # Save best model by accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/quantum_fisher_model_best.pth")
            print(f"        ✨ NEW BEST! ({best_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1
    else:
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:<8} {avg_loss:<10.4f} {'-':<10}")
    
    # Learning rate scheduling
    scheduler.step(avg_loss)
    
    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

end_time = time.time()
print(f"\n⏱️ Training Complete in {end_time - start_time:.2f} seconds")
print(f"✨ Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

# ===============================
# LOAD BEST MODEL & EVALUATE
# ===============================
model.load_state_dict(torch.load("models/quantum_fisher_model_best.pth"))
model.eval()

with torch.no_grad():
    train_logits = model(X_train_t).flatten()
    test_logits  = model(X_test_t).flatten()
    
    train_probs = torch.sigmoid(train_logits).numpy()
    test_probs  = torch.sigmoid(test_logits).numpy()

train_preds = (train_probs > 0.5).astype(int)
test_preds  = (test_probs > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, test_preds)

print("\n" + "="*70)
print("QUANTUM VQC RESULTS")
print("="*70)
print(f"Train Accuracy : {accuracy_score(y_train, train_preds):.4f} ({accuracy_score(y_train, train_preds)*100:.2f}%)")
print(f"Test Accuracy  : {accuracy_score(y_test, test_preds):.4f} ({accuracy_score(y_test, test_preds)*100:.2f}%)")
print(f"Train AUC      : {roc_auc_score(y_train, train_probs):.4f}")
print(f"Test AUC       : {roc_auc_score(y_test, test_probs):.4f}")

precision = precision_score(y_test, test_preds)
recall    = recall_score(y_test, test_preds)
f1        = f1_score(y_test, test_preds)

print(f"\nPrecision (Test): {precision:.4f}")
print(f"Recall    (Test): {recall:.4f}")
print(f"F1 Score  (Test): {f1:.4f}")

print("\n🔍 Confusion Matrix (Test Data):")
print("                 Predicted")
print("                 NON-CANCER  CANCER")
print(f"Actual NON-CANCER    {cm[0,0]:4d}       {cm[0,1]:4d}")
print(f"Actual CANCER        {cm[1,0]:4d}       {cm[1,1]:4d}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, test_preds))

# Save final model
torch.save(model.state_dict(), "models/quantum_fisher_model_final.pth")
print("\n[SAVED] models/quantum_fisher_model_final.pth")
print("[SAVED] models/quantum_fisher_model_best.pth")

print("\n" + "="*70)
print("STEP 5 COMPLETED")
print("="*70)

report_file.close()