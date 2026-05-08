# src/step3_preprocessing_split_scale.py

import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===============================
# REPORT LOGGING SETUP
# ===============================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

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
# USE SMOTE BALANCED DATASET
# ===============================
DATA_PATH = "data/smote_balanced_pca_dataset.csv"

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
print("\nLoaded Dataset Shape:", df.shape)

# ===============================
# REMOVE DUPLICATES (if any)
# ===============================
df = df.drop_duplicates().reset_index(drop=True)
print("Dataset Shape after removing duplicates:", df.shape)

# ===============================
# SEPARATE FEATURES & LABEL
# ===============================
y = df["output"].astype(int).values
X = df.iloc[:, 2:].astype(float).values   # skip output & Gene Name

# ===============================
# TRAIN / TEST SPLIT (STRATIFIED)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain Distribution:", np.bincount(y_train))
print("Test Distribution:", np.bincount(y_test))

# ===============================
# SCALING (FIT ONLY ON TRAIN)
# ===============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# SAVE SCALER
# ===============================
joblib.dump(scaler, "models/scaler_pca.pkl")

# ===============================
# SAVE SPLIT DATA
# ===============================
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("\nScaler saved: models/scaler_pca.pkl")
print("Train/Test data saved as .npy files")

print("\nSaved Files:")
print("models/scaler_pca.pkl")
print("data/X_train.npy")
print("data/X_test.npy")
print("data/y_train.npy")
print("data/y_test.npy")

print("\nStep 3 Completed Successfully.")
