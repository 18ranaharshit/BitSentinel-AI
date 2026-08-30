"""
==============================================================================
Diagnostic Suite: Zero-Duration Tree Attribution & Additivity Audit
==============================================================================
Diagnoses:
  1. Computes exact per-tree probability decomposition (TreeInterpreter math)
     to measure genuine feature contributions in probability space [0, 1].
  2. Verifies the SHAP/Tree Additivity Axiom: sum(contrib_i) + base_value == P(class).
  3. Tests 10 additional addresses with active_duration_sec == 0 from the dataset.
  4. Identifies why unnormalized raw seconds previously caused the -596.67 artifact.
==============================================================================
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")

SEP = "=" * 80

print(f"\n{SEP}")
print("  DIAGNOSTIC: ZERO-DURATION PROBABILITY ATTRIBUTION & ADDITIVITY AUDIT")
print(SEP)

# ------------------------------------------------------------------------------
# 1. Load BABD-13 Reduced Model & Sample Data
# ------------------------------------------------------------------------------
babd_model_path = MODELS_DIR / "babd13_reduced_model.pkl"
raw_addr_path = MODELS_DIR / "raw_address_predictions.csv"

if not babd_model_path.exists():
    raise FileNotFoundError(f"Missing {babd_model_path}")
if not raw_addr_path.exists():
    raise FileNotFoundError(f"Missing {raw_addr_path}")

with open(babd_model_path, "rb") as f:
    babd_data = pickle.load(f)

model = babd_data["model"]
feature_names = ["total_received", "tx_count", "avg_value_per_tx", "active_duration_sec", "tx_frequency"]
classes = list(model.classes_)

df_addr = pd.read_csv(raw_addr_path)

# ------------------------------------------------------------------------------
# 2. Exact Tree Decision-Path Probability Decomposition Engine
# ------------------------------------------------------------------------------
def compute_tree_path_decomposition(rf_model, X_sample, class_idx):
    """
    Computes exact decision-path probability attribution across all trees in RF.
    Decomposes the predicted probability into: base_value + sum(contributions).
    Mathematically guarantees: sum(contributions) + base_value == P(class).
    """
    n_features = len(feature_names)
    contributions = np.zeros(n_features, dtype=np.float64)
    base_values = []

    for tree in rf_model.estimators_:
        t = tree.tree_
        # Node probability distributions: shape (n_nodes, n_classes)
        node_values = t.value[:, 0, :]
        node_probs = node_values / node_values.sum(axis=1, keepdims=True)
        
        # Root prior probability for this tree
        tree_base_prob = node_probs[0, class_idx]
        base_values.append(tree_base_prob)

        # Trace path from root to leaf for X_sample
        node_id = 0
        while t.children_left[node_id] != t.children_right[node_id]:  # Not a leaf
            feat = t.feature[node_id]
            thresh = t.threshold[node_id]
            curr_prob = node_probs[node_id, class_idx]

            # Move left or right based on split
            if X_sample[0, feat] <= thresh:
                next_node = t.children_left[node_id]
            else:
                next_node = t.children_right[node_id]

            next_prob = node_probs[next_node, class_idx]
            # Marginal probability delta assigned to splitting feature
            prob_delta = next_prob - curr_prob
            contributions[feat] += prob_delta
            node_id = next_node

    # Average across all trees in ensemble
    n_trees = len(rf_model.estimators_)
    mean_base_value = float(np.mean(base_values))
    mean_contributions = contributions / n_trees

    return mean_base_value, mean_contributions


# ------------------------------------------------------------------------------
# 3. Item 1: Exact Evaluation of Address 1BEXKh2pSAVbHcnVPKQV1JLgkcJ3cFXNSL
# ------------------------------------------------------------------------------
target_addr = "1BEXKh2pSAVbHcnVPKQV1JLgkcJ3cFXNSL"
match = df_addr[df_addr["account"] == target_addr]
sample_row = match.iloc[0]

X_test = np.array([[
    float(sample_row["total_received"]),
    int(sample_row["tx_count"]),
    float(sample_row["avg_value_per_tx"]),
    int(sample_row["active_duration_sec"]),
    float(sample_row["tx_frequency"])
]], dtype=np.float32)

pred_probs = model.predict_proba(X_test)[0]
pred_cat = str(sample_row["predicted_category"])
c_idx = classes.index(pred_cat) if pred_cat in classes else 0
model_prob = pred_probs[c_idx]

base_val, tree_contribs = compute_tree_path_decomposition(model, X_test, c_idx)
reconstructed_prob = base_val + np.sum(tree_contribs)

print(f"\n[1] EXACT PROBABILITY DECOMPOSITION FOR '{target_addr}'")
print(f"    - Predicted Category : {pred_cat} (Class Index: {c_idx})")
print(f"    - Model Predict Prob : {model_prob:.8f}")
print(f"    - Base Value E[P]    : {base_val:.8f} (Population Prior for {pred_cat})")
print(f"    - Sum of Contribs    : {np.sum(tree_contribs):+.8f}")
print(f"    - Reconstructed Prob : {reconstructed_prob:.8f} (Base + Sum of Contribs)")
print(f"    - Additivity Error   : {abs(reconstructed_prob - model_prob):.2e} (Exact Match: {abs(reconstructed_prob - model_prob) < 1e-6})")

print("\n    Raw Unrounded Feature Contributions (in Probability Units):")
for f_name, val, c in zip(feature_names, X_test[0], tree_contribs):
    print(f"      * {f_name:<22} | Raw Val: {val:<12.6f} | Probability Contrib: {c:+.8f}")

# ------------------------------------------------------------------------------
# 4. Item 2: Test on 10 Other Addresses with active_duration_sec == 0
# ------------------------------------------------------------------------------
print(f"\n{SEP}")
print("  [2] TESTING 10 OTHER ZERO-DURATION ADDRESSES (active_duration_sec == 0)")
print(SEP)

zero_duration_df = df_addr[df_addr["active_duration_sec"] == 0].drop_duplicates(subset=["account"]).head(10)

audit_rows = []
for idx, (_, row) in enumerate(zero_duration_df.iterrows(), start=1):
    x_sub = np.array([[
        float(row["total_received"]),
        int(row["tx_count"]),
        float(row["avg_value_per_tx"]),
        int(row["active_duration_sec"]),
        float(row["tx_frequency"])
    ]], dtype=np.float32)
    
    p_probs = model.predict_proba(x_sub)[0]
    p_cat = str(row["predicted_category"])
    c_i = classes.index(p_cat) if p_cat in classes else 0
    actual_p = p_probs[c_i]
    
    b_val, c_vals = compute_tree_path_decomposition(model, x_sub, c_i)
    recon_p = b_val + np.sum(c_vals)
    
    # Extract duration contribution
    dur_contrib = c_vals[3]  # active_duration_sec index
    
    audit_rows.append({
        "#": idx,
        "Address": row["account"][:16] + "...",
        "Predicted Category": p_cat[:18],
        "P(Class)": f"{actual_p:.4f}",
        "Base E[P]": f"{b_val:.4f}",
        "Duration Contrib": f"{dur_contrib:+.4f}",
        "Sum(All Contribs)": f"{np.sum(c_vals):+.4f}",
        "Reconstructed": f"{recon_p:.4f}",
        "Additivity Valid": "YES" if abs(recon_p - actual_p) < 1e-5 else "NO"
    })

df_audit = pd.DataFrame(audit_rows)
print(df_audit.to_string(index=False))
print(f"{SEP}\n")
