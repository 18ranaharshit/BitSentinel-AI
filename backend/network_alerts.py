"""
==============================================================================
Network-Blockchain Correlation Alerts API Router (BitSentinel-AI)
==============================================================================
Provides REST API endpoints for fused cross-layer risk intelligence:
  - GET /api/network-alerts          : Ranked list of multimodal correlated alerts
  - GET /api/network-alerts/clusters : Coordinated subnet & ASN cluster aggregations
  - GET /api/network-alerts/{txid}   : Detailed forensic drilldown with XAI tree attribution
==============================================================================
"""

import sys
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from explainability import explain_tree_prediction

MODELS_DIR = BASE_DIR / "models"

router = APIRouter(prefix="/api", tags=["network-correlation"])

# Load Combined Risk Model once at module import
combined_model_dict = None
combined_model_path = MODELS_DIR / "combined_risk_model.pkl"
if combined_model_path.exists():
    try:
        with open(combined_model_path, "rb") as f:
            combined_model_dict = pickle.load(f)
            print("  [+] [network_alerts] Loaded Combined Risk model for multimodal explanations.")
    except Exception as e:
        print(f"  [!] [network_alerts] Could not load Combined Risk model: {e}")


def get_risk_tier(fused_prob: float) -> str:
    """Classifies risk probability into standardized security tier."""
    if fused_prob >= 0.85:
        return "critical"
    elif fused_prob >= 0.70:
        return "high"
    elif fused_prob >= 0.50:
        return "medium"
    return "low"


@router.get("/network-alerts")
def get_network_alerts(
    min_prob: float = Query(0.5, ge=0.0, le=1.0, description="Minimum fused risk probability threshold"),
    limit: int = Query(50, ge=1, le=500, description="Maximum alerts to return"),
    correlated_only: bool = Query(False, description="Filter strictly to coordinated network clusters")
):
    """Returns top ranked multimodal fused alerts sorted by risk probability."""
    alerts_path = MODELS_DIR / "network_correlated_alerts.csv"
    if not alerts_path.exists():
        return {
            "status": "not_generated_yet",
            "message": "Network correlation alerts not generated yet. Please run 'python train_combined_risk_model.py' first.",
            "total_matching": 0,
            "returned": 0,
            "alerts": []
        }

    df = pd.read_csv(alerts_path)
    df["txid"] = df["txid"].astype(str)

    # Filter
    filtered = df[df["fused_prob"] >= min_prob]
    if correlated_only:
        filtered = filtered[filtered["is_correlated_cluster"] == True]

    # Sort defensively descending by fused_prob
    filtered = filtered.sort_values(by="fused_prob", ascending=False)
    total_matching = int(len(filtered))

    # Slice top rows
    top_df = filtered.head(limit).copy()
    top_df["risk_tier"] = top_df["fused_prob"].apply(get_risk_tier)

    # Clean NaNs for valid JSON serialization
    alerts_list = top_df.replace({np.nan: None}).to_dict(orient="records")

    return {
        "status": "ready",
        "total_matching": total_matching,
        "returned": len(alerts_list),
        "alerts": alerts_list
    }


@router.get("/network-alerts/clusters")
def get_network_clusters():
    """Aggregates correlated transactions by /24 subnet to surface coordinated network clusters."""
    alerts_path = MODELS_DIR / "network_correlated_alerts.csv"
    if not alerts_path.exists():
        return {
            "status": "not_generated_yet",
            "message": "Network correlation alerts not generated yet. Please run 'python train_combined_risk_model.py' first.",
            "total_clusters": 0,
            "clusters": []
        }

    df = pd.read_csv(alerts_path)
    df["txid"] = df["txid"].astype(str)

    # Filter strictly to correlated cluster rows
    corr_df = df[df["is_correlated_cluster"] == True]
    if corr_df.empty:
        return {
            "status": "empty",
            "message": "No coordinated subnet clusters found.",
            "total_clusters": 0,
            "clusters": []
        }

    clusters_list = []
    for subnet, grp in corr_df.groupby("src_subnet24"):
        countries = grp["src_country"].dropna().tolist()
        country_mode = countries[0] if countries else "Unknown"
        
        asns = grp["src_asn"].dropna().tolist()
        asn_val = asns[0] if asns else "Unknown"

        asn_names = grp["src_asn_name"].dropna().tolist()
        asn_name_val = asn_names[0] if asn_names else "Unknown"

        ts_min = float(grp["timestamp"].min())
        ts_max = float(grp["timestamp"].max())
        time_span_h = round((ts_max - ts_min) / 3600.0, 2)

        clusters_list.append({
            "subnet": str(subnet),
            "country": str(country_mode),
            "asn": str(asn_val),
            "asn_name": str(asn_name_val),
            "tx_count": int(len(grp)),
            "avg_fused_prob": round(float(grp["fused_prob"].mean()), 4),
            "max_fused_prob": round(float(grp["fused_prob"].max()), 4),
            "time_span_hours": time_span_h,
            "linked_txids": grp["txid"].head(10).tolist()
        })

    # Sort descending by max_fused_prob then tx_count
    clusters_list.sort(key=lambda x: (x["max_fused_prob"], x["tx_count"]), reverse=True)

    return {
        "status": "ready",
        "total_clusters": len(clusters_list),
        "clusters": clusters_list
    }


@router.get("/network-alerts/{txid}")
def get_single_network_alert(txid: str):
    """Fetches detailed forensic evidence & SHAP tree explanation for a single transaction."""
    alerts_path = MODELS_DIR / "network_correlated_alerts.csv"
    if not alerts_path.exists():
        return {
            "found": False,
            "txid": txid,
            "status": "not_generated_yet",
            "message": "Network correlation alerts not generated yet. Please run 'python train_combined_risk_model.py' first."
        }

    df = pd.read_csv(alerts_path)
    target_tx = str(txid).strip()
    match = df[df["txid"].astype(str) == target_tx]

    if match.empty:
        return {
            "found": False,
            "txid": target_tx,
            "message": f"No network correlation record found for TXID '{target_tx}'."
        }

    row = match.iloc[0]
    fused_prob = float(row["fused_prob"])
    risk_tier = get_risk_tier(fused_prob)

    # Compute explanation using existing explain_tree_prediction
    explanation_str = "Attributed to cross-layer correlation and graph risk score."
    top_factors = []

    if combined_model_dict and "model" in combined_model_dict:
        feat_names = combined_model_dict.get(
            "feature_cols",
            ["blockchain_risk_score", "src_subnet24_peer_count", "time_cluster_peer_count", "src_asn_peer_count", "is_correlated_cluster"]
        )
        feat_vals = [float(row.get(c, 0.0)) for c in feat_names]
        exp_res = explain_tree_prediction(
            combined_model_dict["model"],
            feat_vals,
            feat_names,
            top_n=5,
            target_class_idx=1
        )
        explanation_str = exp_res.get("summary", explanation_str)
        top_factors = exp_res.get("top_factors", [])

    return {
        "found": True,
        "txid": target_tx,
        "fused_prob": round(fused_prob, 4),
        "fused_pred": int(row["fused_pred"]),
        "risk_tier": risk_tier,
        "network_evidence": {
            "src_ip": str(row.get("src_ip", "Unknown")),
            "src_subnet24": str(row.get("src_subnet24", "Unknown")),
            "src_asn": str(row.get("src_asn", "Unknown")),
            "src_asn_name": str(row.get("src_asn_name", "Unknown")),
            "src_country": str(row.get("src_country", "Unknown")),
            "src_subnet24_peer_count": int(row.get("src_subnet24_peer_count", 0)),
            "time_cluster_peer_count": int(row.get("time_cluster_peer_count", 0)),
            "src_asn_peer_count": int(row.get("src_asn_peer_count", 0)),
            "is_correlated_cluster": bool(row.get("is_correlated_cluster", False))
        },
        "blockchain_risk_score": round(float(row.get("blockchain_risk_score", 0.0)), 4),
        "explanation": explanation_str,
        "top_factors": top_factors,
        "scoring_engine": "Multimodal Cross-Layer Fusion (Network Correlation + Elliptic GNN/RF Risk)"
    }
