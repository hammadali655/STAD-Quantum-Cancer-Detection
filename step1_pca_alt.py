# src/step1_pca_alt.py

import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import joblib

# ===============================
# SETUP
# ===============================
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

report = open("reports/step1_pca_alt_report.txt", "w")
sys.stdout = report

print("Loading dataset...")

# ===============================
# LOAD DATA (UPDATED PATH)
# ===============================
dataset = pd.read_csv('data/EIIPSTDADsequencesencoding.csv', header=None)

dataset = dataset.fillna(0)

print("Original Shape:", dataset.shape)

# ===============================
# SPLIT (IMPORTANT FIX)
# ===============================
labels = dataset.iloc[:, 0]
genes = dataset.iloc[:, 1]
X = dataset.iloc[:, 2:]

# ===============================
# SCALING
# ===============================
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()
X_scaled = sc_X.fit_transform(X)

print("Scaling Done")

# ===============================
# PCA (100 ONLY)
# ===============================
from sklearn.decomposition import PCA

print("Applying PCA (100 components)...")

pca = PCA(n_components=100)
Y = pca.fit_transform(X_scaled)

print("PCA Shape:", Y.shape)

# ===============================
# SAVE MODELS (FOR FRONTEND)
# ===============================
joblib.dump(sc_X, "models/minmax_scaler_pca.pkl")
joblib.dump(pca, "models/pca_100_model.pkl")

print("Saved Models:")
print("models/minmax_scaler_pca.pkl")
print("models/pca_100_model.pkl")

# ===============================
# SAVE DATASET
# ===============================
df = pd.DataFrame(Y)

df.insert(0, "Gene Name", genes)
df.insert(0, "output", labels)

df.to_csv('data/PCA_Mutatedgenesfeatures_Minmaxscaled_100.csv', index=False)

print("Saved PCA Dataset")

print("Step 1 (Alt) Completed Successfully")

report.close()