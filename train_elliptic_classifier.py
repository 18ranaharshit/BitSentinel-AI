"""
==============================================================================
Elliptic Downstream Fraud Classification & Benchmarking
==============================================================================
Compares:
  1. Raw Feature Baselines (Random Forest, XGBoost, LightGBM)
  2. Pretrained Hybrid Models (Raw Features + 128-dim GraphSAGE Embeddings)
  3. End-to-End Fine-Tuned GraphSAGE Classifier

Evaluated on Elliptic Val and Test sets using Precision, Recall, F1, ROC-AUC, PR-AUC.
==============================================================================
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
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

from torch_geometric.nn import SAGEConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
RAW_DIR = Path("raw data")

SEP = "=" * 80

print(f"\n{SEP}")
print("  ELLIPTIC DOWNSTREAM FRAUD CLASSIFICATION & BENCHMARKING")
print(SEP)
print(f"  Using PyTorch Device: {device}")

# 1. Load processed train/val/test split CSVs
print("\n[1] Loading processed Elliptic splits and pretrained embeddings ...")
df_train = pd.read_csv(PROCESSED_DIR / "elliptic_train.csv")
df_val   = pd.read_csv(PROCESSED_DIR / "elliptic_val.csv")
df_test  = pd.read_csv(PROCESSED_DIR / "elliptic_test.csv")

feat_cols = [f"feat_{i}" for i in range(1, 166)]

X_train_raw = df_train[feat_cols].values.astype(np.float32)
y_train = df_train["class"].values.astype(int)

X_val_raw = df_val[feat_cols].values.astype(np.float32)
y_val = df_val["class"].values.astype(int)

X_test_raw = df_test[feat_cols].values.astype(np.float32)
y_test = df_test["class"].values.astype(int)

print(f"    Train: {X_train_raw.shape} | Illicit: {y_train.sum():,}")
print(f"    Val  : {X_val_raw.shape}   | Illicit: {y_val.sum():,}")
print(f"    Test : {X_test_raw.shape}  | Illicit: {y_test.sum():,}")

# Load node embeddings and txid mapping
embeddings = torch.load(MODELS_DIR / "elliptic_node_embeddings.pt").numpy()
with open(MODELS_DIR / "txid_to_idx.pkl", "rb") as f:
    txid_to_idx = pickle.load(f)

# Extract node embeddings for each split using txId
train_emb = np.array([embeddings[txid_to_idx[tx]] for tx in df_train["txId"]])
val_emb   = np.array([embeddings[txid_to_idx[tx]] for tx in df_val["txId"]])
test_emb  = np.array([embeddings[txid_to_idx[tx]] for tx in df_test["txId"]])

# Concatenate Raw Features + GraphSAGE Embeddings (165 + 128 = 293 features)
X_train_hybrid = np.hstack([X_train_raw, train_emb])
X_val_hybrid   = np.hstack([X_val_raw, val_emb])
X_test_hybrid  = np.hstack([X_test_raw, test_emb])

print(f"    Hybrid Feature Matrix Shape: {X_train_hybrid.shape}")

def evaluate_predictions(y_true, y_pred, y_prob, model_name, split_name):
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    roc  = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    return {
        "Model": model_name,
        "Split": split_name,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc,
        "PR-AUC": pr_auc
    }

results = []
probs_dict = {}

# 2. Train and Evaluate Baselines on Raw Features
print("\n[2] Training Supervised Models on Raw Features (165 features) ...")

# 2a. Random Forest
print("    Training Random Forest (Raw) ...")
rf_raw = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf_raw.fit(X_train_raw, y_train)

y_prob_rf_val = rf_raw.predict_proba(X_val_raw)[:, 1]
results.append(evaluate_predictions(y_val, (y_prob_rf_val >= 0.5).astype(int), y_prob_rf_val, "Random Forest (Raw)", "Val"))

y_prob_rf_test = rf_raw.predict_proba(X_test_raw)[:, 1]
results.append(evaluate_predictions(y_test, (y_prob_rf_test >= 0.5).astype(int), y_prob_rf_test, "Random Forest (Raw)", "Test"))
probs_dict["Random Forest (Raw)"] = y_prob_rf_test

# 2b. XGBoost (if available)
if HAS_XGB:
    print("    Training XGBoost (Raw) ...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb_raw = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    xgb_raw.fit(X_train_raw, y_train)
    
    y_prob_val = xgb_raw.predict_proba(X_val_raw)[:, 1]
    results.append(evaluate_predictions(y_val, (y_prob_val >= 0.5).astype(int), y_prob_val, "XGBoost (Raw)", "Val"))
    
    y_prob_test = xgb_raw.predict_proba(X_test_raw)[:, 1]
    results.append(evaluate_predictions(y_test, (y_prob_test >= 0.5).astype(int), y_prob_test, "XGBoost (Raw)", "Test"))
    probs_dict["XGBoost (Raw)"] = y_prob_test

# 2c. LightGBM (if available)
if HAS_LGB:
    print("    Training LightGBM (Raw) ...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    lgb_raw = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1)
    lgb_raw.fit(X_train_raw, y_train)
    
    y_prob_val = lgb_raw.predict_proba(X_val_raw)[:, 1]
    results.append(evaluate_predictions(y_val, (y_prob_val >= 0.5).astype(int), y_prob_val, "LightGBM (Raw)", "Val"))
    
    y_prob_test = lgb_raw.predict_proba(X_test_raw)[:, 1]
    results.append(evaluate_predictions(y_test, (y_prob_test >= 0.5).astype(int), y_prob_test, "LightGBM (Raw)", "Test"))
    probs_dict["LightGBM (Raw)"] = y_prob_test

# 3. Train and Evaluate Hybrid Models (Raw Features + GraphSAGE Embeddings)
print("\n[3] Training Supervised Models on Hybrid Features (Raw + 128D Embeddings) ...")

# 3a. Random Forest (Hybrid)
print("    Training Random Forest (Hybrid) ...")
rf_hyb = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf_hyb.fit(X_train_hybrid, y_train)

y_prob_val = rf_hyb.predict_proba(X_val_hybrid)[:, 1]
results.append(evaluate_predictions(y_val, (y_prob_val >= 0.5).astype(int), y_prob_val, "Random Forest (Hybrid)", "Val"))

y_prob_test = rf_hyb.predict_proba(X_test_hybrid)[:, 1]
results.append(evaluate_predictions(y_test, (y_prob_test >= 0.5).astype(int), y_prob_test, "Random Forest (Hybrid)", "Test"))
probs_dict["Random Forest (Hybrid)"] = y_prob_test

# 3b. XGBoost (Hybrid)
if HAS_XGB:
    print("    Training XGBoost (Hybrid) ...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb_hyb = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    xgb_hyb.fit(X_train_hybrid, y_train)
    
    y_prob_val = xgb_hyb.predict_proba(X_val_hybrid)[:, 1]
    results.append(evaluate_predictions(y_val, (y_prob_val >= 0.5).astype(int), y_prob_val, "XGBoost (Hybrid)", "Val"))
    
    y_prob_test = xgb_hyb.predict_proba(X_test_hybrid)[:, 1]
    results.append(evaluate_predictions(y_test, (y_prob_test >= 0.5).astype(int), y_prob_test, "XGBoost (Hybrid)", "Test"))
    probs_dict["XGBoost (Hybrid)"] = y_prob_test

# 3c. LightGBM (Hybrid)
if HAS_LGB:
    print("    Training LightGBM (Hybrid) ...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    lgb_hyb = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1)
    lgb_hyb.fit(X_train_hybrid, y_train)
    
    y_prob_val = lgb_hyb.predict_proba(X_val_hybrid)[:, 1]
    results.append(evaluate_predictions(y_val, (y_prob_val >= 0.5).astype(int), y_prob_val, "LightGBM (Hybrid)", "Val"))
    
    y_prob_test = lgb_hyb.predict_proba(X_test_hybrid)[:, 1]
    results.append(evaluate_predictions(y_test, (y_prob_test >= 0.5).astype(int), y_prob_test, "LightGBM (Hybrid)", "Test"))
    probs_dict["LightGBM (Hybrid)"] = y_prob_test

# 4. End-to-End Fine-Tuning of GraphSAGE Classifier
print("\n[4] End-to-End Fine-Tuning of GraphSAGE Classifier ...")

# Build full PyG graph structure
df_raw_feat = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None)
tx_ids_all = df_raw_feat[0].values
x_all = df_raw_feat.iloc[:, 2:].values.astype(np.float32)

N_all = len(tx_ids_all)
df_edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
src = df_edges["txId1"].map(txid_to_idx)
dst = df_edges["txId2"].map(txid_to_idx)
valid = src.notnull() & dst.notnull()
edge_index = torch.tensor(np.array([src[valid].astype(np.int64).values, dst[valid].astype(np.int64).values]), dtype=torch.long).to(device)
x_tensor = torch.tensor(x_all, dtype=torch.float).to(device)

masks = torch.load(MODELS_DIR / "node_split_mask.pt")
train_mask = masks["train_mask"].to(device)
val_mask = masks["val_mask"].to(device)
test_mask = masks["test_mask"].to(device)

# Map labels to node indices
y_all = torch.zeros(N_all, dtype=torch.float).to(device)
for tx, cls in zip(df_train["txId"], df_train["class"]):
    y_all[txid_to_idx[tx]] = float(cls)
for tx, cls in zip(df_val["txId"], df_val["class"]):
    y_all[txid_to_idx[tx]] = float(cls)
for tx, cls in zip(df_test["txId"], df_test["class"]):
    y_all[txid_to_idx[tx]] = float(cls)

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        logits = self.classifier(x).squeeze(-1)
        return logits

gcn_model = GraphSAGEClassifier(in_channels=165, hidden_channels=128).to(device)
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.005, weight_decay=1e-4)

# Pos weight calculation
n_neg = (y_all[train_mask] == 0).sum().item()
n_pos = (y_all[train_mask] == 1).sum().item()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"    Fine-Tuning for 50 Epochs (Pos Weight: {pos_weight.item():.2f}) ...")
gcn_model.train()
for epoch in range(1, 51):
    optimizer.zero_grad()
    out = gcn_model(x_tensor, edge_index)
    loss = criterion(out[train_mask], y_all[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        gcn_model.eval()
        with torch.no_grad():
            val_logits = gcn_model(x_tensor, edge_index)[val_mask]
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
        print(f"      Epoch {epoch:>2d}/50 | Train Loss: {loss.item():.4f} | Val F1 (Illicit): {val_f1:.4f}")
        gcn_model.train()

# Final Fine-Tuned GCN Evaluation
gcn_model.eval()
with torch.no_grad():
    all_logits = gcn_model(x_tensor, edge_index)
    val_probs = torch.sigmoid(all_logits[val_mask]).cpu().numpy()
    test_probs = torch.sigmoid(all_logits[test_mask]).cpu().numpy()

results.append(evaluate_predictions(y_val, (val_probs >= 0.5).astype(int), val_probs, "GraphSAGE (Fine-Tuned)", "Val"))
results.append(evaluate_predictions(y_test, (test_probs >= 0.5).astype(int), test_probs, "GraphSAGE (Fine-Tuned)", "Test"))
probs_dict["GraphSAGE (Fine-Tuned)"] = test_probs

# Save GCN Model and results
torch.save(gcn_model.state_dict(), MODELS_DIR / "elliptic_gcn_classifier.pt")
print(f"\n    Saved Fine-Tuned GCN Classifier -> {MODELS_DIR / 'elliptic_gcn_classifier.pt'}")

if HAS_XGB:
    with open(MODELS_DIR / "elliptic_xgb_hybrid.pkl", "wb") as f:
        pickle.dump(xgb_hyb, f)
    print(f"    Saved XGBoost Hybrid Model -> {MODELS_DIR / 'elliptic_xgb_hybrid.pkl'}")

# Save results for visualization report
df_results = pd.DataFrame(results)
df_results.to_csv(MODELS_DIR / "elliptic_benchmark_results.csv", index=False)
with open(MODELS_DIR / "elliptic_test_probs.pkl", "wb") as f:
    pickle.dump({"y_test": y_test, "probs_dict": probs_dict}, f)

# 5. Print Comparison Benchmark Table
print(f"\n{SEP}")
print("  ELLIPTIC FRAUD CLASSIFICATION BENCHMARK SUMMARY")
print(SEP)
print("\n" + df_results.to_string(index=False))
print(f"\n{SEP}\n")
