# src/step4_fisher_feature_selection.py

import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# ===============================
# REPORT LOGGING SETUP
# ===============================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

STEP_NAME = os.path.basename(__file__).replace(".py", "")
REPORT_PATH = f"reports/{STEP_NAME}_report.txt"
report_file = open(REPORT_PATH, "w")

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
# LOAD ORIGINAL DATA
# ===============================
X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")

print("\nOriginal Feature Shape:", X_train.shape)

# ===============================
# FISHER SCORE FUNCTION
# ===============================
def fisher_score(X, y):
    scores = []

    for i in range(X.shape[1]):
        feature = X[:, i]

        class0 = feature[y == 0]
        class1 = feature[y == 1]

        mean0 = np.mean(class0)
        mean1 = np.mean(class1)

        var0 = np.var(class0)
        var1 = np.var(class1)

        score = (mean0 - mean1) ** 2 / (var0 + var1 + 1e-8)
        scores.append(score)

    return np.array(scores)

# ===============================
# COMPUTE FISHER SCORES
# ===============================
scores = fisher_score(X_train, y_train)

# Get top 8 feature indices
top_k = 8
top_indices = np.argsort(scores)[-top_k:][::-1]

print("\nTop 8 Selected Feature Indices:")
print(top_indices)

# ===============================
# REDUCE FEATURES
# ===============================
X_train_selected = X_train[:, top_indices]
X_test_selected  = X_test[:, top_indices]

print("\nReduced Feature Shape:", X_train_selected.shape)

# ===============================
# SAVE FILES
# ===============================
np.save("data/X_train_fisher8.npy", X_train_selected)
np.save("data/X_test_fisher8.npy", X_test_selected)
np.save("models/fisher_top8_indices.npy", top_indices)

print("\nSaved:")
print("data/X_train_fisher8.npy")
print("data/X_test_fisher8.npy")
print("models/fisher_top8_indices.npy")

print("\nStep 4 (Fisher Feature Selection) Completed Successfully.")