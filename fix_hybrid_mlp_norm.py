# fix_hybrid_mlp_norm.py
import numpy as np
import joblib
import torch
import pennylane as qml

# ----- Load training data -----
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

# ----- Load VQC weights for extractor -----
vqc_state = torch.load("models/quantum_fisher_model_best.pth", map_location='cpu')

# ----- Define 6‑layer X‑rotation extractor (matching step6) -----
class QuantumFeatureExtractor:
    def __init__(self, vqc_state_dict):
        dev = qml.device("default.qubit", wires=8)
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(8))
            qml.StronglyEntanglingLayers(weights, wires=range(8))
            return [qml.expval(qml.PauliZ(i)) for i in range(8)]
        weight_shapes = {"weights": (6, 8, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.quantum_layer.load_state_dict(vqc_state_dict, strict=False)
        self.quantum_layer.eval()
    def extract(self, x):
        with torch.no_grad():
            return self.quantum_layer(torch.tensor(x, dtype=torch.float32)).numpy()

extractor = QuantumFeatureExtractor(vqc_state)
raw_q_train = extractor.extract(X_train[:, :8])

# ----- Compute Fisher scores on current X_train -----
def fisher_score(X, y):
    scores = []
    for i in range(X.shape[1]):
        feat = X[:, i]
        c0 = feat[y == 0]
        c1 = feat[y == 1]
        score = (c0.mean() - c1.mean())**2 / (c0.var() + c1.var() + 1e-8)
        scores.append(score)
    return np.array(scores)

scores = fisher_score(X_train, y_train)
top16 = np.argsort(scores)[-16:][::-1]
np.save("models/fisher_top16_indices.npy", top16)
print(f"Updated fisher_top16_indices.npy: {top16}")

# ----- Create combined features and compute normalization -----
X_classical_16 = X_train[:, top16]
X_hybrid = np.hstack([X_classical_16, raw_q_train])
mlp_mean = X_hybrid.mean(axis=0)
mlp_std = X_hybrid.std(axis=0) + 1e-8

joblib.dump({
    "mean": mlp_mean,
    "std": mlp_std,
    "fisher_indices": top16
}, "models/quantum_features_norm.pkl")
print("Updated models/quantum_features_norm.pkl")