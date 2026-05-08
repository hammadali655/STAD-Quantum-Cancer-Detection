# src/step2_smote_augmentation.py

import pandas as pd
import numpy as np
import os
import sys
from imblearn.over_sampling import SMOTE

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
# LOAD DATA
# ===============================
DATA_PATH = "data/PCA_Mutatedgenesfeatures_Minmaxscaled_100.csv"
OUTPUT_PATH = "data/smote_balanced_pca_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset Shape:", df.shape)
print("\nOriginal Class Distribution:")
print(df["output"].value_counts())

# ===============================
# PREPARE FEATURES & LABELS
# ===============================
gene_names = df["Gene Name"].values
X = df.drop(columns=["output", "Gene Name"])
y = df["output"]

# ===============================
# APPLY SMOTE
# ===============================
print("\nApplying SMOTE for dataset balancing...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

num_original = len(df)
num_total = len(X_resampled)
num_synthetic = num_total - num_original

print(f"\nSynthetic Samples Created: {num_synthetic}")

# ===============================
# BUILD FINAL DATAFRAME (CLEAN METHOD)
# ===============================

# Create dataframe directly with all columns at once
final_df = pd.DataFrame(
    np.column_stack([y_resampled,
                     np.array(list(gene_names) + ["Synthetic_Gene"] * num_synthetic),
                     X_resampled]),
    columns=["output", "Gene Name"] + list(X.columns)
)

# Ensure correct data types
final_df["output"] = final_df["output"].astype(int)

# Shuffle
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# SAVE DATA
# ===============================
final_df.to_csv(OUTPUT_PATH, index=False)

print("\nAfter SMOTE Balancing:")
print(final_df["output"].value_counts())

print("\nSaved at:", OUTPUT_PATH)
print("\nFinal Dataset Shape:", final_df.shape)
print("\nStep 2 (SMOTE Augmentation) Completed Successfully.")
