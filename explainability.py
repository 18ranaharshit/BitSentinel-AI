"""
==============================================================================
SHAP & Tree Explainability Engine (BitSentinel-AI - Fix 6)
==============================================================================
Provides transparent, mathematically rigorous reasoning for every risk decision:
  1. Primary: SHAP TreeExplainer for Tree-based models (Combined Risk RF, BABD-13 RF).
  2. Fallback: Exact Decision-Tree Path Probability Decomposition (TreeInterpreter math).
     Guarantees the Additivity Axiom: sum(contributions) + base_value == P(class).
  3. Plain-English feature translation for non-technical intelligence analysts.
  4. Rule-based input factor extraction for heuristic proxy scores (Zero ML faking).
==============================================================================
"""

import numpy as np
import pandas as pd

# Global explainer cache to avoid recomputing TreeExplainer overhead per request
_EXPLAINER_CACHE = {}

# BABD-13 categories that are inherently benign — should NOT be framed as "High Risk"
BENIGN_CATEGORIES = {
    "Centralized exchange", "Decentralized exchange", "Mining pool",
    "Individual wallet", "Light service", "Payment processor",
    "Exchange wallet", "Hosted wallet", "Other"
}

# Dictionary mapping raw feature column names to human-readable intelligence labels
FEATURE_NAME_MAP = {
    # Multimodal Combined Risk Features
    "blockchain_risk_score": "On-Chain Graph Fraud Suspicion",
    "src_subnet24_peer_count": "Shared Subnet with Multiple Peer Wallets",
    "time_cluster_peer_count": "Dense Temporal Transaction Burst (6h Window)",
    "src_asn_peer_count": "Autonomous System (ASN) Peer Density",
    "is_correlated_cluster": "Coordinated Botnet Subnet Pattern",

    # BABD-13 Address Behavioral Features
    "total_received": "Total BTC Lifetime Inflow Volume",
    "tx_count": "Lifetime Transaction Count",
    "avg_value_per_tx": "Average Value per Transaction (BTC)",
    "active_duration_sec": "Wallet Lifespan & Activity Duration",
    "tx_frequency": "Transaction Frequency & Velocity",

    # Local Elliptic Graph Features
    "feat_1": "Normalized Transaction Fee / Value Ratio",
    "feat_2": "Incoming vs. Outgoing BTC Volume Imbalance",
    "feat_3": "Multi-Hop Peeling/Branching Factor (In/Out Degree)",
    "feat_4": "Transaction Graph Neighbor Density",
    "feat_5": "Graph Local Clustering Coefficient"
}

def get_plain_english_name(feat_name):
    """Translates raw model feature names into human-readable descriptions."""
    if feat_name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[feat_name]
    if feat_name.startswith("feat_"):
        return f"Graph Topological Pattern #{feat_name.replace('feat_', '')}"
    return feat_name.replace("_", " ").title()


def get_tree_explainer(model):
    """Retrieves or creates a cached SHAP TreeExplainer instance."""
    model_id = id(model)
    if model_id not in _EXPLAINER_CACHE:
        try:
            import shap
            _EXPLAINER_CACHE[model_id] = shap.TreeExplainer(model)
        except Exception:
            return None
    return _EXPLAINER_CACHE[model_id]


def _format_factor_clause(factor):
    """Formats a single factor into a readable string with its raw value and direction."""
    val = factor["value"]
    contrib = factor["contribution"]
    label = factor["label"]
    dir_str = "pushed toward risk" if contrib > 0 else "pushed toward safe"
    
    if isinstance(val, float) and val < 1.0:
        return f"{label} ({val:.4f}, {dir_str}: {contrib:+.3f})"
    elif isinstance(val, float):
        return f"{label} ({val:.2f}, {dir_str}: {contrib:+.3f})"
    else:
        return f"{label} ({val}, {dir_str}: {contrib:+.3f})"


def _exact_tree_path_decomposition(rf_model, X_sample, feature_names, class_idx=1, top_n=3, category_name=None):
    """
    Computes exact decision-path probability attribution across all trees in RF.
    Decomposes the predicted probability into: base_value + sum(contributions).
    Guarantees mathematically that: sum(contributions) + base_value == P(class).
    All contributions are in bounded probability units [-1.0, +1.0].
    """
    n_features = len(feature_names)
    contributions = np.zeros(n_features, dtype=np.float64)
    base_values = []

    if hasattr(rf_model, "estimators_"):
        trees = rf_model.estimators_
    else:
        trees = [rf_model]

    for tree in trees:
        t = tree.tree_
        node_values = t.value[:, 0, :]
        node_probs = node_values / np.maximum(node_values.sum(axis=1, keepdims=True), 1e-9)
        
        c_i = min(class_idx, node_probs.shape[1] - 1)
        tree_base_prob = node_probs[0, c_i]
        base_values.append(tree_base_prob)

        node_id = 0
        while t.children_left[node_id] != t.children_right[node_id]:
            feat = t.feature[node_id]
            thresh = t.threshold[node_id]
            curr_prob = node_probs[node_id, c_i]

            if X_sample[0, feat] <= thresh:
                next_node = t.children_left[node_id]
            else:
                next_node = t.children_right[node_id]

            next_prob = node_probs[next_node, c_i]
            prob_delta = next_prob - curr_prob
            if feat < n_features:
                contributions[feat] += prob_delta
            node_id = next_node

    n_trees = max(len(trees), 1)
    mean_base_value = float(np.mean(base_values))
    mean_contributions = contributions / n_trees

    # Build factor records
    factors = []
    for name, val, contrib in zip(feature_names, X_sample[0], mean_contributions):
        factors.append({
            "feature": name,
            "label": get_plain_english_name(name),
            "value": float(val),
            "contribution": float(contrib),
            "direction": "risk" if contrib > 0 else "safe"
        })

    # Sort by magnitude of contribution
    factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top_factors = factors[:top_n]

    # Predicted probability
    predicted_p = mean_base_value + float(np.sum(mean_contributions))
    is_high_risk = bool(predicted_p >= 0.50)

    # Category-aware framing: benign BABD-13 categories get attribution language, not "High Risk"
    use_benign_framing = category_name is not None and category_name in BENIGN_CATEGORIES

    if use_benign_framing:
        # Attribution mode: explain what drove the classification without risk language
        clauses = [_format_factor_clause(f) for f in top_factors]
        summary = f"Classified as '{category_name}' primarily due to: " + "; ".join(clauses) + "."
    elif is_high_risk:
        clauses = [_format_factor_clause(f) for f in top_factors if f["contribution"] > 0]
        if not clauses:
            clauses = [_format_factor_clause(f) for f in top_factors]
        summary = "Flagged as High Risk primarily due to: " + "; ".join(clauses) + "."
    else:
        clauses = [_format_factor_clause(f) for f in top_factors if f["contribution"] < 0]
        opposing = [_format_factor_clause(f) for f in top_factors if f["contribution"] > 0]
        if clauses and opposing:
            summary = "Assessed as Clean / Low Risk primarily due to: " + "; ".join(clauses) + " (mitigating risk despite opposing network factors: " + "; ".join(opposing) + ")."
        elif clauses:
            summary = "Assessed as Clean / Low Risk primarily due to: " + "; ".join(clauses) + "."
        else:
            summary = "All features within normal baseline bounds: " + "; ".join([_format_factor_clause(f) for f in top_factors]) + "."

    return {
        "summary": summary,
        "top_factors": top_factors,
        "base_value": mean_base_value,
        "engine": "Exact Decision-Tree Path Decomposition"
    }


def explain_tree_prediction(model, feature_vector, feature_names, top_n=3, target_class_idx=1, category_name=None):
    """
    Computes SHAP or exact decision-tree path probability decomposition.
    Returns top_n features pushing toward or away from the target classification.
    category_name: if provided, benign BABD-13 categories get attribution framing
    instead of "High Risk" language.
    """
    if isinstance(feature_vector, (pd.DataFrame, pd.Series)):
        X = feature_vector.values.reshape(1, -1)
    elif isinstance(feature_vector, list):
        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    else:
        X = feature_vector.reshape(1, -1) if feature_vector.ndim == 1 else feature_vector

    explainer = get_tree_explainer(model)
    if explainer is None:
        return _exact_tree_path_decomposition(model, X, feature_names, class_idx=target_class_idx, top_n=top_n, category_name=category_name)

    try:
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            c_idx = min(target_class_idx, len(shap_values) - 1)
            sv = shap_values[c_idx][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                c_idx = min(target_class_idx, shap_values.shape[2] - 1)
                sv = shap_values[0, :, c_idx]
            else:
                sv = shap_values[0]
        else:
            sv = np.array(shap_values)[0]

        factors = []
        for name, val, contrib in zip(feature_names, X[0], sv):
            factors.append({
                "feature": name,
                "label": get_plain_english_name(name),
                "value": float(val),
                "contribution": float(contrib),
                "direction": "risk" if contrib > 0 else "safe"
            })

        factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_factors = factors[:top_n]

        # Use predict_proba for proper risk threshold (>= 0.50) instead of raw SHAP sum
        try:
            proba = model.predict_proba(X)
            c_idx = min(target_class_idx, proba.shape[1] - 1)
            predicted_p = float(proba[0, c_idx])
        except Exception:
            predicted_p = float(np.sum(sv))  # fallback
        is_high_risk = bool(predicted_p >= 0.50)

        # Category-aware framing: benign BABD-13 categories get attribution language
        use_benign_framing = category_name is not None and category_name in BENIGN_CATEGORIES

        if use_benign_framing:
            clauses = [_format_factor_clause(f) for f in top_factors]
            summary = f"Classified as '{category_name}' primarily due to: " + "; ".join(clauses) + "."
        elif is_high_risk:
            clauses = [_format_factor_clause(f) for f in top_factors if f["contribution"] > 0]
            if not clauses:
                clauses = [_format_factor_clause(f) for f in top_factors]
            summary = "Flagged as High Risk primarily due to: " + "; ".join(clauses) + "."
        else:
            clauses = [_format_factor_clause(f) for f in top_factors if f["contribution"] < 0]
            opposing = [_format_factor_clause(f) for f in top_factors if f["contribution"] > 0]
            if clauses and opposing:
                summary = "Assessed as Clean / Low Risk primarily due to: " + "; ".join(clauses) + " (mitigating risk despite opposing network factors: " + "; ".join(opposing) + ")."
            elif clauses:
                summary = "Assessed as Clean / Low Risk primarily due to: " + "; ".join(clauses) + "."
            else:
                summary = "All features within normal baseline bounds: " + "; ".join([_format_factor_clause(f) for f in top_factors]) + "."

        return {
            "summary": summary,
            "top_factors": top_factors,
            "engine": "SHAP TreeExplainer"
        }

    except Exception:
        return _exact_tree_path_decomposition(model, X, feature_names, class_idx=target_class_idx, top_n=top_n, category_name=category_name)


def explain_heuristic_prediction(tx_dict, score):
    """
    Constructs an honest, transparent explanation for rule-based heuristic scores.
    Strictly documents the rule inputs without fabricating ML/SHAP weights.
    """
    fee_btc = float(tx_dict.get("fee_btc", 0) or (tx_dict.get("fee", 0) / 1e8 if "fee" in tx_dict else 0))
    val_btc = float(tx_dict.get("value_btc", 0) or (tx_dict.get("outputs_value", 0) / 1e8 if "outputs_value" in tx_dict else 0))
    in_count = int(tx_dict.get("inputs_count", 0))
    out_count = int(tx_dict.get("outputs_count", 0))

    fee_ratio = fee_btc / max(val_btc, 0.0001)
    reasons = []

    if fee_ratio > 0.05:
        reasons.append(f"Unusually high transaction fee-to-value ratio ({fee_ratio:.3f})")
    elif fee_btc > 0.01:
        reasons.append(f"Elevated absolute miner fee ({fee_btc:.4f} BTC)")

    if val_btc > 50.0:
        reasons.append(f"Large capital transfer volume ({val_btc:.2f} BTC)")

    if in_count >= 10 and out_count <= 2:
        reasons.append(f"Multi-input wallet sweep/consolidation pattern ({in_count} inputs -> {out_count} outputs)")
    elif in_count <= 2 and out_count >= 10:
        reasons.append(f"High fan-out peeling/dispersion pattern ({in_count} inputs -> {out_count} outputs)")

    if not reasons:
        if score >= 0.70:
            reasons.append("Elevated composite fee-degree heuristic threshold")
        else:
            reasons.append("Standard peer-to-peer transfer structure within normal baseline limits")

    summary = "Rule-Based Heuristic Evaluation: " + "; ".join(reasons) + "."

    top_factors = [
        {"feature": "fee_ratio", "label": "Fee-to-Value Ratio", "value": round(fee_ratio, 4), "contribution": round(min(1.0, fee_ratio * 2.0), 3)},
        {"feature": "value_btc", "label": "Transfer Value (BTC)", "value": round(val_btc, 4), "contribution": round(min(1.0, val_btc / 100.0), 3)},
        {"feature": "io_ratio", "label": "Input/Output Ratio", "value": f"{in_count} in / {out_count} out", "contribution": round(min(1.0, abs(in_count - out_count) / 10.0), 3)}
    ]

    return {
        "summary": summary,
        "top_factors": top_factors,
        "is_heuristic": True,
        "engine": "Rule-Based Heuristic Parser"
    }
