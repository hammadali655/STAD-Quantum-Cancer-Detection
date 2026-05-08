# test_rf_8.py
# Train RF and MLP using only 8 features (first 8 columns after the target)

import numpy as np
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# ===============================
# REPORT LOGGING SETUP
# ===============================
import sys

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
# LOAD DATA AND SELECT FIRST 8 FEATURES
# ===============================

if os.path.exists("data/X_train.npy") and os.path.exists("data/X_test.npy"):
    X_train_full = np.load("data/X_train.npy")
    X_test_full = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    print("\nLoaded data from .npy files (SMOTE based).")
    
    # Keep only first 8 features
    X_train = X_train_full[:, :8]
    X_test = X_test_full[:, :8]
    print(f"Reduced from {X_train_full.shape[1]} to 8 features")

else:
    print("\nNPY files not found. Rebuilding from SMOTE dataset...")
    df = pd.read_csv("data/smote_balanced_pca_dataset.csv")
    y = df["output"].astype(int).values
    X_full = df.iloc[:, 2:].astype(float).values   # all features
    
    # Take only first 8 features
    X = X_full[:, :8]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    joblib.dump(scaler, "models/scaler_8feat.pkl")
    
    np.save("data/X_train_8feat.npy", X_train)
    np.save("data/X_test_8feat.npy", X_test)
    np.save("data/y_train_8feat.npy", y_train)
    np.save("data/y_test_8feat.npy", y_test)
    
    print("Rebuilt and saved splits with 8 features.")

print("\n" + "="*70)
print("DATA INFORMATION (FIRST 8 FEATURES ONLY)")
print("="*70)
print(f"Train Shape: {X_train.shape}")
print(f"Test Shape:  {X_test.shape}")
print(f"Number of features used: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ===============================
# RANDOM FOREST MODEL (8 FEATURES)
# ===============================
print("\n" + "="*70)
print("RANDOM FOREST MODEL (8 features)")
print("="*70)

print("\n📊 Model Parameters:")
print(f"   - n_estimators      : 350")
print(f"   - max_depth         : 16")
print(f"   - min_samples_split : 10")
print(f"   - min_samples_leaf  : 5")
print(f"   - max_features      : sqrt")
print(f"   - bootstrap         : True")
print(f"   - class_weight      : balanced_subsample")
print(f"   - random_state      : 42")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("Training Completed")

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n📊 Cross-validation (5-fold):")
print(f"   Scores: {cv_scores}")
print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Predictions
train_preds = rf.predict(X_train)
test_preds = rf.predict(X_test)
train_probs = rf.predict_proba(X_train)
test_probs = rf.predict_proba(X_test)

# Metrics
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
train_loss = log_loss(y_train, train_probs)
test_loss = log_loss(y_test, test_probs)
train_auc = roc_auc_score(y_train, train_probs[:, 1])
test_auc = roc_auc_score(y_test, test_probs[:, 1])
overfit_gap = train_acc - test_acc

# Confusion Matrix
cm_rf = confusion_matrix(y_test, test_preds)

print("\n" + "="*70)
print("RANDOM FOREST RESULTS")
print("="*70)
print(f"Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Train Log Loss : {train_loss:.4f}")
print(f"Test Log Loss  : {test_loss:.4f}")
print(f"Train AUC      : {train_auc:.4f}")
print(f"Test AUC       : {test_auc:.4f}")
print(f"Overfitting Gap: {overfit_gap:.4f} ({overfit_gap*100:.2f}%)")

print("\n🔍 Confusion Matrix (Test Data):")
print("                 Predicted")
print("                 NON-CANCER  CANCER")
print(f"Actual NON-CANCER    {cm_rf[0,0]:4d}       {cm_rf[0,1]:4d}")
print(f"Actual CANCER        {cm_rf[1,0]:4d}       {cm_rf[1,1]:4d}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, test_preds))

# Save model (optional)
joblib.dump(rf, "models/rf_8feat_model.pkl")
print("\n[SAVED] models/rf_8feat_model.pkl")

# ===============================
# MLP MODEL (8 FEATURES)
# ===============================
print("\n" + "="*70)
print("MLP MODEL (8 features)")
print("="*70)

print("\n📊 Model Parameters:")
print(f"   - hidden_layer_sizes : (128,64,32)")
print(f"   - activation         : relu")
print(f"   - solver             : adam")
print(f"   - alpha              : 0.0005 (L2 regularization)")
print(f"   - batch_size         : 32")
print(f"   - learning_rate      : adaptive")
print(f"   - learning_rate_init : 0.0005")
print(f"   - max_iter           : 300")
print(f"   - early_stopping     : True")
print(f"   - validation_fraction: 0.2")
print(f"   - n_iter_no_change   : 15")

print("\nTraining MLP Model...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128,64,32),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.0005,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    random_state=42,
    verbose=False
)

mlp.fit(X_train, y_train)
print(f"Training Completed in {mlp.n_iter_} iterations")

# Predictions
mlp_train_preds = mlp.predict(X_train)
mlp_test_preds = mlp.predict(X_test)
mlp_train_probs = mlp.predict_proba(X_train)
mlp_test_probs = mlp.predict_proba(X_test)

# Metrics
mlp_train_acc = accuracy_score(y_train, mlp_train_preds)
mlp_test_acc = accuracy_score(y_test, mlp_test_preds)
mlp_train_auc = roc_auc_score(y_train, mlp_train_probs[:,1])
mlp_test_auc = roc_auc_score(y_test, mlp_test_probs[:,1])
mlp_precision = precision_score(y_test, mlp_test_preds)
mlp_recall = recall_score(y_test, mlp_test_preds)
mlp_f1 = f1_score(y_test, mlp_test_preds)
mlp_overfit_gap = mlp_train_acc - mlp_test_acc

# Confusion Matrix
cm_mlp = confusion_matrix(y_test, mlp_test_preds)

print("\n" + "="*70)
print("MLP RESULTS")
print("="*70)
print(f"Train Accuracy : {mlp_train_acc:.4f} ({mlp_train_acc*100:.2f}%)")
print(f"Test Accuracy  : {mlp_test_acc:.4f} ({mlp_test_acc*100:.2f}%)")
print(f"Train AUC      : {mlp_train_auc:.4f}")
print(f"Test AUC       : {mlp_test_auc:.4f}")
print(f"Precision (Test): {mlp_precision:.4f}")
print(f"Recall    (Test): {mlp_recall:.4f}")
print(f"F1 Score  (Test): {mlp_f1:.4f}")
print(f"Overfitting Gap: {mlp_overfit_gap:.4f} ({mlp_overfit_gap*100:.2f}%)")

print("\n🔍 Confusion Matrix (Test Data - MLP):")
print("                 Predicted")
print("                 NON-CANCER  CANCER")
print(f"Actual NON-CANCER    {cm_mlp[0,0]:4d}       {cm_mlp[0,1]:4d}")
print(f"Actual CANCER        {cm_mlp[1,0]:4d}       {cm_mlp[1,1]:4d}")

print("\nClassification Report (Test Data - MLP):")
print(classification_report(y_test, mlp_test_preds))

# Save model
joblib.dump(mlp, "models/mlp_8feat_model.pkl")
print("\n[SAVED] models/mlp_8feat_model.pkl")

print("\n" + "="*70)
print("COMPLETED - Both models trained on 8 features")
print("="*70)

report_file.close()