import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("\n===== FINAL MODEL EVALUATION =====")

# ===============================
# REPORT FILES
# ===============================
reports = {
    "Random Forest": "reports/step4_train_rf_pca_report.txt",
    "Quantum VQC": "reports/step5_train_quantum_fisher_report.txt",
    "Hybrid Q + RF": "reports/step6_train_hybrid_rf_pca_report.txt"
}

metrics = {}

# ===============================
# FUNCTION TO EXTRACT NUMBER
# ===============================
def extract_value(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


# ===============================
# READ REPORTS
# ===============================
for model, path in reports.items():

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    train_acc = extract_value(r"Train Accuracy\s*:\s*([0-9.]+)", text)
    test_acc  = extract_value(r"Test Accuracy\s*:\s*([0-9.]+)", text)
    test_auc  = extract_value(r"Test AUC\s*:\s*([0-9.]+)", text)
    precision = extract_value(r"Precision.*:\s*([0-9.]+)", text)
    recall    = extract_value(r"Recall.*:\s*([0-9.]+)", text)
    f1        = extract_value(r"F1.*:\s*([0-9.]+)", text)

    metrics[model] = {
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "AUC": test_auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

df = pd.DataFrame(metrics).T

print("\nExtracted Metrics\n")
print(df)


# ==================================================
# SINGLE FINAL GRAPH (ALL METRICS)
# ==================================================

metrics_list = ["Train Accuracy", "Test Accuracy", "AUC", "Precision", "Recall", "F1 Score"]

x = np.arange(len(df.index))
width = 0.12

plt.figure(figsize=(12,7))

for i, metric in enumerate(metrics_list):
    plt.bar(x + (i * width), df[metric], width, label=metric)

plt.xticks(x + width*2.5, df.index)

plt.title("RF Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)

# Grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Legend outside for clarity
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")

# Add value labels
for i, metric in enumerate(metrics_list):
    for j, v in enumerate(df[metric]):
        plt.text(j + (i * width), v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

plt.tight_layout()

plt.savefig("reports/RF_model_comparison.png", dpi=300)

plt.close()

print("\nSTEP 8 COMPLETED SUCCESSFULLY")