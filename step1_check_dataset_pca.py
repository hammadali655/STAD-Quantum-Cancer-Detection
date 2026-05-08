# src/step1_check_dataset_pca.py

import pandas as pd
# ===============================
# REPORT LOGGING SETUP
# ===============================
import os
import sys

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

DATA_PATH = "data/PCA_Mutatedgenesfeatures_Minmaxscaled_100.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== DATASET INFO =====")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:", df.isnull().sum().sum())

# ===============================
# CORRECT LABEL COLUMN
# ===============================
labels = df["output"]

print("\n===== CORRECT CLASS DISTRIBUTION =====")
print(labels.value_counts())

print("\nClass Ratio:")
print(labels.value_counts(normalize=True))
print("\nClass Ratio:")
print(labels.value_counts(normalize=True))

print("\nStep 1 Completed Successfully.")
