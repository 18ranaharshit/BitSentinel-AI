"""
==============================================================================
BABD-13 Multi-Class Address Fraud Classification & Benchmarking
==============================================================================
Trains multi-class classifiers (Random Forest, LightGBM, XGBoost) on 10 address categories.
Evaluates on BABD-13 Val and Test splits measuring Macro F1, Weighted F1, and Accuracy.
==============================================================================
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 80

print(f"\n{SEP}")
print("  BABD-13 MULTI-CLASS FRAUD CLASSIFICATION")
print(SEP)

# 1. Load splits
print("\n[1] Loading processed BABD-13 train/val/test splits ...")
df_train = pd.read_csv(PROCESSED_DIR / "babd13_train.csv")
df_val   = pd.read_csv(PROCESSED_DIR / "babd13_val.csv")
df_test  = pd.read_csv(PROCESSED_DIR / "babd13_test.csv")

ignore_cols = ["account", "label"]
feature_cols = [c for c in df_train.columns if c not in ignore_cols]

# Identify categorical (string) columns in features
cat_cols = df_train[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
if cat_cols:
    print(f"    Categorical feature columns detected: {cat_cols}")
    # One-hot encode categorical features across all splits
    df_combined = pd.concat([df_train[feature_cols], df_val[feature_cols], df_test[feature_cols]], keys=["train", "val", "test"])
    df_combined_encoded = pd.get_dummies(df_combined, columns=cat_cols, drop_first=True)
    
    df_train_encoded = df_combined_encoded.xs("train")
    df_val_encoded   = df_combined_encoded.xs("val")
    df_test_encoded  = df_combined_encoded.xs("test")
    
    encoded_feature_cols = df_train_encoded.columns.tolist()
    print(f"    Total Features after One-Hot Encoding: {len(encoded_feature_cols)}")
    
    X_train = df_train_encoded.values.astype(np.float32)
    X_val   = df_val_encoded.values.astype(np.float32)
    X_test  = df_test_encoded.values.astype(np.float32)
else:
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val   = df_val[feature_cols].values.astype(np.float32)
    X_test  = df_test[feature_cols].values.astype(np.float32)
    encoded_feature_cols = feature_cols

# Map original class labels to contiguous 0..N-1
unique_labels = sorted(df_train["label"].unique())
label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
idx_to_label = {idx: lbl for idx, lbl in enumerate(unique_labels)}

y_train = df_train["label"].map(label_to_idx).values.astype(int)
y_val   = df_val["label"].map(label_to_idx).values.astype(int)
y_test  = df_test["label"].map(label_to_idx).values.astype(int)

def evaluate_multiclass(y_true, y_pred, model_name, split_name):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": acc,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Macro Precision": macro_prec,
        "Macro Recall": macro_rec
    }

results = []
test_preds = {}

# 2. Train Random Forest Classifier
print("\n[2] Training Random Forest Multi-Class Classifier ...")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_val = rf.predict(X_val)
results.append(evaluate_multiclass(y_val, y_pred_val, "Random Forest", "Val"))

y_pred_test_rf = rf.predict(X_test)
results.append(evaluate_multiclass(y_test, y_pred_test_rf, "Random Forest", "Test"))
test_preds["Random Forest"] = y_pred_test_rf

best_model = rf
best_name = "Random Forest"
best_val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)

# 3. Train LightGBM (if available)
if HAS_LGB:
    print("\n[3] Training LightGBM Multi-Class Classifier ...")
    lgb_cls = lgb.LGBMClassifier(
        n_estimators=150, max_depth=8, learning_rate=0.05,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_cls.fit(X_train, y_train)
    
    y_pred_val = lgb_cls.predict(X_val)
    val_res = evaluate_multiclass(y_val, y_pred_val, "LightGBM", "Val")
    results.append(val_res)
    
    y_pred_test_lgb = lgb_cls.predict(X_test)
    results.append(evaluate_multiclass(y_test, y_pred_test_lgb, "LightGBM", "Test"))
    test_preds["LightGBM"] = y_pred_test_lgb
    
    if val_res["Macro F1"] > best_val_f1:
        best_val_f1 = val_res["Macro F1"]
        best_model = lgb_cls
        best_name = "LightGBM"

# 4. Train XGBoost (if available)
if HAS_XGB:
    print("\n[4] Training XGBoost Multi-Class Classifier ...")
    xgb_cls = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        objective="multi:softprob", random_state=42, n_jobs=-1
    )
    xgb_cls.fit(X_train, y_train)
    
    y_pred_val = xgb_cls.predict(X_val)
    val_res = evaluate_multiclass(y_val, y_pred_val, "XGBoost", "Val")
    results.append(val_res)
    
    y_pred_test_xgb = xgb_cls.predict(X_test)
    results.append(evaluate_multiclass(y_test, y_pred_test_xgb, "XGBoost", "Test"))
    test_preds["XGBoost"] = y_pred_test_xgb
    
    if val_res["Macro F1"] > best_val_f1:
        best_val_f1 = val_res["Macro F1"]
        best_model = xgb_cls
        best_name = "XGBoost"

# 5. Save best model and test evaluation results
best_model_path = MODELS_DIR / "babd13_best_model.pkl"
with open(best_model_path, "wb") as f:
    pickle.dump({
        "model": best_model,
        "feature_cols": encoded_feature_cols,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label
    }, f)

print(f"\nSaved Best Model ({best_name}) -> {best_model_path}")

df_results = pd.DataFrame(results)
df_results.to_csv(MODELS_DIR / "babd13_benchmark_results.csv", index=False)
with open(MODELS_DIR / "babd13_test_preds.pkl", "wb") as f:
    pickle.dump({
        "y_test": y_test,
        "test_preds": test_preds,
        "best_name": best_name,
        "unique_labels": unique_labels
    }, f)

# Summary Table
print(f"\n{SEP}")
print("  BABD-13 FRAUD CLASSIFICATION BENCHMARK SUMMARY")
print(SEP)
print("\n" + df_results.to_string(index=False))
print(f"\n{SEP}\n")
