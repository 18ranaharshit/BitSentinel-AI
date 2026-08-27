"""
==============================================================================
BABD-13 Reduced-Feature Multi-Class Classifier (Raw-Block Compatible)
==============================================================================
Trains a lightweight RandomForestClassifier using ONLY the 5 honest features
extractable from raw block transaction outputs/inputs:
  1. total_received      (mapped from BABD-13 PAIa11-1: Total BTC Received)
  2. tx_count            (mapped from BABD-13 PDIa1-1: Total In-Degree / Tx Count)
  3. avg_value_per_tx    (total_received / max(tx_count, 1))
  4. active_duration_sec (mapped from BABD-13 PTIa1: Lifetime in days * 86400)
  5. tx_frequency        (tx_count / max(active_duration_sec, 1))

Evaluates on babd13_val.csv and babd13_test.csv.
Saves model artifact to models/babd13_reduced_model.pkl.
==============================================================================
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 80

print(f"\n{SEP}")
print("  TRAINING BABD-13 REDUCED 5-FEATURE MODEL (RAW-BLOCK COMPATIBLE)")
print(SEP)

print("\n[Feature Mapping & Semantic Justification]")
print("  - 'total_received'      <-- PAIa11-1 (Total received Bitcoin volume)")
print("  - 'tx_count'            <-- PDIa1-1  (Total in-degree / receiving transactions)")
print("  - 'avg_value_per_tx'    <-- PAIa11-1 / max(PDIa1-1, 1)")
print("  - 'active_duration_sec' <-- PTIa1 * 86400 (Account active span from days to seconds)")
print("  - 'tx_frequency'        <-- PDIa1-1 / max(active_duration_sec, 1)")

FEATURE_NAMES = [
    "total_received",
    "tx_count",
    "avg_value_per_tx",
    "active_duration_sec",
    "tx_frequency"
]

def extract_reduced_features_from_babd13(df):
    total_received = df["PAIa11-1"].fillna(0).values.astype(np.float32)
    tx_count = np.maximum(df["PDIa1-1"].fillna(1).values.astype(np.float32), 1.0)
    avg_value_per_tx = total_received / tx_count
    # PTIa1 is lifetime in days. Single transaction addresses have PTIa1 == 1
    # Convert days to seconds to match raw block timestamp delta
    lifetime_days = np.maximum(df["PTIa1"].fillna(1).values.astype(np.float32) - 1.0, 0.0)
    active_duration_sec = lifetime_days * 86400.0
    tx_frequency = tx_count / np.maximum(active_duration_sec, 1.0)

    X = np.column_stack([
        total_received,
        tx_count,
        avg_value_per_tx,
        active_duration_sec,
        tx_frequency
    ]).astype(np.float32)
    
    # Replace any infinities or NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=0.0)
    return X

# Load datasets
print("\n[1] Loading processed BABD-13 splits ...")
df_train = pd.read_csv(PROCESSED_DIR / "babd13_train.csv")
df_val   = pd.read_csv(PROCESSED_DIR / "babd13_val.csv")
df_test  = pd.read_csv(PROCESSED_DIR / "babd13_test.csv")

X_train = extract_reduced_features_from_babd13(df_train)
X_val   = extract_reduced_features_from_babd13(df_val)
X_test  = extract_reduced_features_from_babd13(df_test)

unique_labels = sorted(df_train["label"].unique())
label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
idx_to_label = {idx: lbl for idx, lbl in enumerate(unique_labels)}

y_train = df_train["label"].map(label_to_idx).values.astype(int)
y_val   = df_val["label"].map(label_to_idx).values.astype(int)
y_test  = df_test["label"].map(label_to_idx).values.astype(int)

BABD_LABEL_NAMES = {
    0: "Blackmail", 1: "Cyber-security service", 2: "Darknet market",
    3: "Centralized exchange", 5: "P2P financial service", 6: "Gambling",
    10: "Mining pool", 11: "Tumbler", 12: "Individual wallet",
    13: "other_illicit"
}

# 2. Train Model
print("\n[2] Training Reduced-Feature Random Forest Classifier (100 trees) ...")
rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_reduced.fit(X_train, y_train)

# 3. Evaluate
print("\n[3] Evaluating Reduced-Feature Model Performance:")
print("=" * 80)
print("  REDUCED-FEATURE MODEL (raw-block-compatible) EVALUATION METRICS")
print("  Note: Operating on 5 derived honest features vs. full 150-feature schema.")
print("=" * 80)

def evaluate(y_true, y_pred, split_name):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    print(f"\n  [{split_name} Split]")
    print(f"    - Accuracy        : {acc * 100:.2f}%")
    print(f"    - Macro F1-Score  : {macro_f1:.4f}")
    print(f"    - Weighted F1     : {weighted_f1:.4f}")
    print(f"    - Macro Precision : {macro_prec:.4f}")
    print(f"    - Macro Recall    : {macro_rec:.4f}")
    return {"Split": split_name, "Accuracy": acc, "Macro F1": macro_f1, "Weighted F1": weighted_f1}

y_pred_val = rf_reduced.predict(X_val)
val_metrics = evaluate(y_val, y_pred_val, "Validation")

y_pred_test = rf_reduced.predict(X_test)
test_metrics = evaluate(y_test, y_pred_test, "Test")

# Save reduced model
save_path = MODELS_DIR / "babd13_reduced_model.pkl"
with open(save_path, "wb") as f:
    pickle.dump({
        "model": rf_reduced,
        "feature_names": FEATURE_NAMES,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "label_names": BABD_LABEL_NAMES
    }, f)

print(f"\n[4] Saved Reduced-Feature Model -> {save_path}")
