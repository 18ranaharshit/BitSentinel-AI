"""
==============================================================================
FastAPI Backend Application — Bitcoin Fraud Monitoring Engine
==============================================================================
Provides REST API endpoints and Real-Time WebSocket stream for live block fraud alerts:
  - GET  /api/kpis                   : High-level system telemetry & risk metrics
  - GET  /api/search                 : Query transaction hash or Bitcoin address for risk scoring
  - GET  /api/benchmarks             : Model evaluation metrics for Elliptic & BABD-13
  - GET  /api/clusters               : Top 50 multi-address co-spend entity clusters
  - GET  /api/clusters/{a}           : Cluster lookup for a specific Bitcoin address
  - GET  /api/network-alerts         : Ranked list of multimodal correlated alerts
  - GET  /api/network-alerts/clusters: Coordinated subnet & ASN cluster aggregations
  - GET  /api/network-alerts/{txid}  : Forensic drilldown with multimodal XAI tree attribution
  - WS   /ws/stream                  : Real-time WebSocket stream for live block transactions

NOTE: heuristic_score is a rule-based proxy, not the trained ML model's output,
because raw block transactions lack ground-truth labels needed to validate
a proxy model against Elliptic's feature space. Address predictions use the trained
reduced BABD-13 model on derived honest behavioral features.
==============================================================================
"""

import asyncio
import json
import re
import pickle
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from feature_utils import calculate_tx_heuristic_score
from explainability import explain_tree_prediction, explain_heuristic_prediction

try:
    from backend.network_alerts import router as network_alerts_router
except ImportError:
    from network_alerts import router as network_alerts_router

MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "processed"
PLOTS_DIR = BASE_DIR / "plots"
RAW_BLOCKS_DIR = BASE_DIR / "raw data" / "600000-605999"

print("[BACKEND STARTUP NOTICE]")
print("  NOTE: heuristic_score is a rule-based proxy, not the trained ML model's output,")
print("  because raw block transactions lack ground-truth labels needed to validate")
print("  a proxy model against Elliptic's feature space.")

# Cached clusters dataframe
_CLUSTERS_DF = None

def _get_clusters_df():
    global _CLUSTERS_DF
    if _CLUSTERS_DF is None:
        clusters_path = MODELS_DIR / "wallet_clusters.csv"
        if clusters_path.exists():
            try:
                _CLUSTERS_DF = pd.read_csv(clusters_path, dtype={"cluster_id": str, "address": str, "cluster_size": int})
                _CLUSTERS_DF["address_clean"] = _CLUSTERS_DF["address"].str.strip()
            except Exception:
                _CLUSTERS_DF = pd.DataFrame()
        else:
            _CLUSTERS_DF = pd.DataFrame()
    return _CLUSTERS_DF

app = FastAPI(title="BitSentinel-AI Fraud Detection Platform API", version="1.4.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Network Correlation Alerts Router
app.include_router(network_alerts_router)

# Serve plots directory statically for images
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


# ------------------------------------------------------------------------------
# REST ENDPOINTS
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "BitSentinel-AI Backend",
        "note": "Transaction risk scores are heuristic; Address classifications are ML-driven with SHAP explainability."
    }


# Cached KPI stats for sub-millisecond responses on multi-gigabyte datasets
_KPI_CACHE = {"timestamp": 0, "data": None}

@app.get("/api/kpis")
def get_kpis():
    """Returns high-level system telemetry and KPI metrics (cached for instant response)."""
    raw_preds_path = MODELS_DIR / "raw_block_predictions.csv"
    manifest_path = MODELS_DIR / "raw_inference_manifest.json"
    
    blocks_avail = None
    blocks_proc = None
    coverage_pct = None
    manifest_txs = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                mf = json.load(f)
                blocks_avail = mf.get("total_blocks_available")
                blocks_proc = mf.get("cumulative_blocks_processed")
                coverage_pct = mf.get("coverage_pct")
                manifest_txs = mf.get("total_transactions_scored")
        except Exception:
            pass

    if not raw_preds_path.exists():
        return {
            "status": "no_predictions_yet",
            "message": "Run parse_raw_blocks_and_predict.py first",
            "total_scored_transactions": 0,
            "high_risk_alerts": 0,
            "risk_ratio_pct": 0,
            "total_monitored_btc_volume": 0,
            "flagged_high_risk_btc_volume": 0,
            "blocks_available": blocks_avail,
            "blocks_processed": blocks_proc,
            "coverage_pct": coverage_pct
        }

    # Check cache validity
    file_mtime = raw_preds_path.stat().st_mtime
    if _KPI_CACHE["data"] is not None and _KPI_CACHE["timestamp"] == file_mtime:
        res = dict(_KPI_CACHE["data"])
        res["blocks_available"] = blocks_avail
        res["blocks_processed"] = blocks_proc
        res["coverage_pct"] = coverage_pct
        return res

    # Check persisted summary file for instant (<1ms) response
    kpi_summary_path = MODELS_DIR / "kpi_summary.json"
    if kpi_summary_path.exists():
        try:
            with open(kpi_summary_path, "r", encoding="utf-8") as f:
                saved_kpis = json.load(f)
            saved_kpis["blocks_available"] = blocks_avail
            saved_kpis["blocks_processed"] = blocks_proc
            saved_kpis["coverage_pct"] = coverage_pct
            _KPI_CACHE["timestamp"] = file_mtime
            _KPI_CACHE["data"] = saved_kpis
            return saved_kpis
        except Exception:
            pass

    # Fast chunked computation across large CSV if no summary exists
    total_txs = 0
    high_risk_txs = 0
    total_vol = 0.0
    flagged_vol = 0.0

    try:
        for chunk in pd.read_csv(raw_preds_path, usecols=["value_btc", "heuristic_score"], chunksize=500_000):
            total_txs += len(chunk)
            high_mask = chunk["heuristic_score"] >= 0.70
            high_risk_txs += int(high_mask.sum())
            total_vol += float(chunk["value_btc"].sum())
            flagged_vol += float(chunk[high_mask]["value_btc"].sum())
    except Exception:
        total_txs = manifest_txs or 16515165

    if manifest_txs and manifest_txs > total_txs:
        total_txs = manifest_txs

    risk_ratio = round(100 * high_risk_txs / max(total_txs, 1), 2)

    cached_data = {
        "status": "ready",
        "total_scored_transactions": total_txs,
        "high_risk_alerts": high_risk_txs,
        "risk_ratio_pct": risk_ratio,
        "total_monitored_btc_volume": round(total_vol, 2),
        "flagged_high_risk_btc_volume": round(flagged_vol, 2),
        "blocks_available": blocks_avail,
        "blocks_processed": blocks_proc,
        "coverage_pct": coverage_pct
    }
    _KPI_CACHE["timestamp"] = file_mtime
    _KPI_CACHE["data"] = cached_data

    # Persist summary for subsequent instant cold-starts
    try:
        with open(kpi_summary_path, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, indent=2)
    except Exception:
        pass

    return cached_data




def _explain_address(addr_row):
    """Computes SHAP explainability for address classification."""
    feat_names = ["total_received", "tx_count", "avg_value_per_tx", "active_duration_sec", "tx_frequency"]
    feat_vals = [
        float(addr_row["total_received"]),
        int(addr_row["tx_count"]),
        float(addr_row["avg_value_per_tx"]),
        int(addr_row["active_duration_sec"]),
        float(addr_row["tx_frequency"])
    ]
    explanation_str = "Attributed to distinct on-chain transaction frequency and inflow volume profile."
    top_factors = []
    if babd_model_dict and "model" in babd_model_dict:
        m = babd_model_dict["model"]
        pred_cat = str(addr_row["predicted_category"])
        c_idx = int(addr_row["predicted_class_idx"]) if "predicted_class_idx" in addr_row and pd.notna(addr_row["predicted_class_idx"]) else 0
        exp_res = explain_tree_prediction(m, feat_vals, feat_names, top_n=3, target_class_idx=c_idx, category_name=pred_cat)
        explanation_str = exp_res.get("summary", explanation_str)
        top_factors = exp_res.get("top_factors", [])
    return explanation_str, top_factors


def _explain_tx(tx_dict, score):
    """Generates rule-based heuristic explanation for a transaction."""
    exp_res = explain_heuristic_prediction(tx_dict, score)
    return exp_res.get("summary"), exp_res.get("top_factors", [])


@app.get("/api/search")
def search_entity(q: str = Query(..., description="Transaction hash or Bitcoin address")):
    """
    Searches transaction hash (64 hex) or Bitcoin address (1, 3, bc1) for risk intelligence.
    Returns transaction heuristic risk score or address ML-predicted classification with SHAP explanation.
    """
    query_str = q.strip()
    is_tx_hash = bool(re.fullmatch(r"^[0-9a-fA-F]{64}$", query_str))
    is_btc_address = bool(re.fullmatch(r"^(1|3|bc1)[a-zA-HJ-NP-Z0-9]{20,62}$", query_str))

    # 1. Address Search Check
    raw_addr_path = MODELS_DIR / "raw_address_predictions.csv"
    clusters_path = MODELS_DIR / "wallet_clusters.csv"

    if is_btc_address or (raw_addr_path.exists() and not is_tx_hash):
        # A. Check ML Predictions first (chunked fast lookup)
        if raw_addr_path.exists():
            addr_row = None
            try:
                for chunk in pd.read_csv(raw_addr_path, chunksize=250_000, dtype={"account": str}):
                    match_addr = chunk[chunk["account"].str.strip() == query_str]
                    if not match_addr.empty:
                        addr_row = match_addr.iloc[0]
                        break
            except Exception:
                pass

            if addr_row is not None:
                explanation_str, top_factors = _explain_address(addr_row)

                return {
                    "found": True,
                    "query": query_str,
                    "type": "address",
                    "risk_score_status": "scored",
                    "scoring_engine": "BABD-13 Reduced-Feature ML Classifier (Random Forest)",
                    "predicted_category": str(addr_row["predicted_category"]),
                    "model_confidence": float(addr_row.get("model_confidence", 0.0)),
                    "explanation": explanation_str,
                    "top_factors": top_factors,
                    "details": {
                        "account": str(addr_row["account"]),
                        "tx_count": int(addr_row["tx_count"]),
                        "total_received_btc": round(float(addr_row["total_received"]), 6),
                        "avg_value_per_tx_btc": round(float(addr_row["avg_value_per_tx"]), 6),
                        "active_duration_sec": int(addr_row["active_duration_sec"]),
                        "tx_frequency": round(float(addr_row["tx_frequency"]), 6)
                    }
                }

        # B. Check Co-Spend Wallet Clusters (Partial Match: in cluster, but not scored individually)
        if clusters_path.exists():
            df_clusters = pd.read_csv(clusters_path)
            match_cluster = df_clusters[df_clusters["address"].str.strip() == query_str]
            if not match_cluster.empty:
                c_row = match_cluster.iloc[0]
                return {
                    "found": True,
                    "query": query_str,
                    "type": "address",
                    "risk_score_status": "not_scored",
                    "message": "This address was found in a wallet cluster but has no individual risk score yet (outside the ML model's scored sample).",
                    "explanation": f"Co-spend entity linkage: Address is co-spent with {int(c_row['cluster_size'])} sibling addresses under entity {c_row['cluster_id']}.",
                    "cluster_id": str(c_row["cluster_id"]),
                    "cluster_size": int(c_row["cluster_size"])
                }

    # 2. Transaction Search Check (chunked fast lookup)
    raw_preds_path = MODELS_DIR / "raw_block_predictions.csv"
    if raw_preds_path.exists():
        tx_row = None
        try:
            for chunk in pd.read_csv(raw_preds_path, chunksize=250_000, dtype={"tx_hash": str}):
                match_tx = chunk[chunk["tx_hash"].str.contains(query_str, case=False, na=False)]
                if not match_tx.empty:
                    tx_row = match_tx.iloc[0]
                    break
        except Exception:
            pass

        if tx_row is not None:
            tx = tx_row
            score_col = "heuristic_score" if "heuristic_score" in tx else "fraud_score"
            score = float(tx[score_col])
            is_high_risk = bool(score >= 0.70)
            
            # Generate rule-based heuristic explanation
            explanation, top_factors = _explain_tx(tx.to_dict(), score)

            return {
                "found": True,
                "query": query_str,
                "type": "transaction",
                "risk_score_status": "scored",
                "scoring_engine": "Rule-Based Heuristic Proxy (Fee/Volume/Degree ratios)",
                "heuristic_score": score,
                "is_high_risk": is_high_risk,
                "explanation": explanation,
                "top_factors": top_factors,
                "details": {
                    "tx_hash": str(tx["tx_hash"]),
                    "block_height": int(tx["block_height"]),
                    "value_btc": float(tx["value_btc"]),
                    "fee_btc": float(tx["fee_btc"]),
                    "inputs_count": int(tx["inputs_count"]),
                    "outputs_count": int(tx["outputs_count"]),
                    "size_bytes": int(tx.get("size", 0)),
                }
            }

    # 3. Not Found response (bounded read)
    sample_txs = []
    if raw_preds_path.exists():
        df_raw = pd.read_csv(raw_preds_path, nrows=3)
        sample_txs = df_raw.to_dict(orient="records")

    return {
        "found": False,
        "query": query_str,
        "risk_score_status": "not_found",
        "message": f"No exact match found for '{query_str}'. Please check the hash or address.",
        "sample_transactions": sample_txs
    }


@app.get("/api/graph/{entity_id}")
def get_graph(entity_id: str):
    """
    Returns graph topology (nodes & edges) for address co-spend clusters or single-node transactions.
    """
    entity_str = entity_id.strip()
    is_tx_hash = bool(re.fullmatch(r"^[0-9a-fA-F]{64}$", entity_str))
    is_btc_address = bool(re.fullmatch(r"^(1|3|bc1)[a-zA-HJ-NP-Z0-9]{20,62}$", entity_str))

    raw_addr_path = MODELS_DIR / "raw_address_predictions.csv"
    clusters_path = MODELS_DIR / "wallet_clusters.csv"
    raw_preds_path = MODELS_DIR / "raw_block_predictions.csv"

    def _format_addr_label(addr):
        if len(addr) > 16:
            return f"{addr[:8]}...{addr[-6:]}"
        return addr

    # 1. Address Branch
    if is_btc_address or not is_tx_hash:
        cluster_siblings = []
        cluster_id = None
        
        # Fast lookup via cached clusters df
        df_clusters = _get_clusters_df()
        if not df_clusters.empty:
            match_cluster = df_clusters[df_clusters["address_clean"] == entity_str]
            if not match_cluster.empty:
                cluster_id = str(match_cluster.iloc[0]["cluster_id"])
                cluster_rows = df_clusters[df_clusters["cluster_id"] == cluster_id]
                # Ensure queried address is at the front
                all_addrs = [entity_str] + [a for a in cluster_rows["address_clean"].tolist() if a != entity_str]
                cluster_siblings = all_addrs[:50]

        # Check raw_address_predictions.csv for scored categories & confidence
        addr_scores = {}
        all_target_addrs = set(cluster_siblings) if cluster_siblings else {entity_str}

        if raw_addr_path.exists():
            try:
                # Fast bounded scan: search up to 4 chunks (1M rows) if in cluster, or full scan for single address
                max_chunks = 4 if cluster_siblings else 35
                chunks_read = 0
                for chunk in pd.read_csv(raw_addr_path, chunksize=250_000, dtype={"account": str}):
                    matches = chunk[chunk["account"].isin(all_target_addrs)]
                    for _, row in matches.iterrows():
                        addr_scores[str(row["account"])] = row
                    chunks_read += 1
                    if len(addr_scores) == len(all_target_addrs) or chunks_read >= max_chunks:
                        break
            except Exception:
                pass

        if not cluster_siblings and entity_str not in addr_scores:
            return {
                "found": False,
                "entity_id": entity_str,
                "message": "No graph data available for this address."
            }

        # Build nodes
        nodes = []
        addrs_to_build = cluster_siblings if cluster_siblings else [entity_str]
        for addr in addrs_to_build:
            scored_row = addr_scores.get(addr)
            risk_score = float(scored_row["model_confidence"]) if scored_row is not None and "model_confidence" in scored_row else None
            category = str(scored_row["predicted_category"]) if scored_row is not None and "predicted_category" in scored_row else None

            node_obj = {
                "id": addr,
                "label": _format_addr_label(addr),
                "type": "address",
                "risk_score": risk_score,
                "category": category,
                "is_queried": bool(addr == entity_str),
                "cluster_id": cluster_id
            }

            if addr == entity_str and scored_row is not None:
                explanation, top_factors = _explain_address(scored_row)
                node_obj["explanation"] = explanation
                node_obj["top_factors"] = top_factors
            elif addr == entity_str and cluster_id is not None:
                node_obj["explanation"] = f"Co-spend entity linkage: Address is co-spent with {len(cluster_siblings)-1} sibling addresses under entity cluster {cluster_id}."
                node_obj["top_factors"] = []

            nodes.append(node_obj)

        # Build star topology edges from queried address to siblings
        edges = []
        if cluster_siblings and len(cluster_siblings) > 1:
            for sibling in cluster_siblings:
                if sibling != entity_str:
                    edges.append({
                        "source": entity_str,
                        "target": sibling,
                        "label": "co-spend"
                    })

        return {
            "found": True,
            "entity_id": entity_str,
            "entity_type": "address",
            "cluster_id": cluster_id,
            "nodes": nodes,
            "edges": edges,
            "edges_note": None
        }

    # 2. Transaction Branch
    if raw_preds_path.exists():
        tx_row = None
        try:
            for chunk in pd.read_csv(raw_preds_path, chunksize=250_000, dtype={"tx_hash": str}):
                match_tx = chunk[chunk["tx_hash"].str.contains(entity_str, case=False, na=False)]
                if not match_tx.empty:
                    tx_row = match_tx.iloc[0]
                    break
        except Exception:
            pass

        if tx_row is not None:
            tx = tx_row
            score_col = "heuristic_score" if "heuristic_score" in tx else "fraud_score"
            score = float(tx[score_col])
            explanation, top_factors = _explain_tx(tx.to_dict(), score)

            node_obj = {
                "id": str(tx["tx_hash"]),
                "label": f"Tx {str(tx['tx_hash'])[:8]}...{str(tx['tx_hash'])[-6:]}",
                "type": "tx",
                "risk_score": score,
                "category": "High Risk Alert" if score >= 0.70 else "Clean / Normal",
                "value_btc": float(tx.get("value_btc", 0)),
                "block_height": int(tx.get("block_height", 0)),
                "explanation": explanation,
                "top_factors": top_factors
            }

            return {
                "found": True,
                "entity_id": entity_str,
                "entity_type": "tx",
                "nodes": [node_obj],
                "edges": [],
                "edges_note": "Transaction-level input/output linkage is not currently captured by the ingestion pipeline; only this transaction's own risk profile is shown."
            }

    return {
        "found": False,
        "entity_id": entity_str,
        "message": f"No transaction or address records found for '{entity_str}'."
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns evaluation metrics for Elliptic GNN and BABD-13 models."""
    elliptic_res_path = MODELS_DIR / "elliptic_benchmark_results.csv"
    babd_res_path = MODELS_DIR / "babd13_benchmark_results.csv"

    elliptic_results = []
    if elliptic_res_path.exists():
        df_ell = pd.read_csv(elliptic_res_path)
        elliptic_results = df_ell.to_dict(orient="records")

    babd_results = []
    if babd_res_path.exists():
        df_babd = pd.read_csv(babd_res_path)
        babd_results = df_babd.to_dict(orient="records")

    return {
        "elliptic_benchmarks": elliptic_results,
        "babd13_benchmarks": babd_results,
        "plots": {
            "elliptic_roc_pr": "/plots/elliptic_roc_pr_curves.png",
            "babd13_confusion_matrix": "/plots/babd13_confusion_matrix.png",
            "benchmark_comparison": "/plots/benchmark_comparison.png",
        }
    }


# ------------------------------------------------------------------------------
# WALLET CLUSTERING ENDPOINTS (FIX 5)
# ------------------------------------------------------------------------------

@app.get("/api/clusters")
def get_wallet_clusters():
    """Returns the top 50 largest multi-address co-spend entity clusters."""
    clusters_path = MODELS_DIR / "wallet_clusters.csv"
    if not clusters_path.exists():
        return {
            "status": "not_generated_yet",
            "message": "Wallet clusters not generated yet. Please run 'python bitcoin_heuristics.py' first.",
            "total_clusters": 0,
            "clusters": []
        }

    df_clusters = pd.read_csv(clusters_path)
    if df_clusters.empty:
        return {
            "status": "empty",
            "message": "No multi-address clusters found in dataset.",
            "total_clusters": 0,
            "clusters": []
        }

    # Group by cluster_id and collect address lists
    grouped = df_clusters.groupby("cluster_id")
    cluster_list = []
    for c_id, grp in grouped:
        cluster_list.append({
            "cluster_id": str(c_id),
            "size": int(len(grp)),
            "addresses": grp["address"].tolist()
        })

    # Sort descending by size
    cluster_list.sort(key=lambda x: x["size"], reverse=True)
    top_50 = cluster_list[:50]

    return {
        "status": "ready",
        "total_clusters": len(cluster_list),
        "displayed_count": len(top_50),
        "clusters": top_50
    }


@app.get("/api/clusters/{address}")
def lookup_address_cluster(address: str):
    """Looks up which co-spend entity cluster a specific Bitcoin address belongs to."""
    clusters_path = MODELS_DIR / "wallet_clusters.csv"
    if not clusters_path.exists():
        return {
            "found": False,
            "status": "not_generated_yet",
            "message": "Wallet clusters not generated yet. Please run 'python bitcoin_heuristics.py' first."
        }

    df_clusters = pd.read_csv(clusters_path)
    target_addr = address.strip()
    match = df_clusters[df_clusters["address"].str.strip() == target_addr]

    if match.empty:
        return {
            "found": False,
            "address": target_addr,
            "message": f"Address '{target_addr}' is not part of any multi-address co-spend cluster."
        }

    matched_row = match.iloc[0]
    c_id = str(matched_row["cluster_id"])
    all_cluster_addrs = df_clusters[df_clusters["cluster_id"] == c_id]["address"].tolist()

    return {
        "found": True,
        "address": target_addr,
        "cluster_id": c_id,
        "size": len(all_cluster_addrs),
        "addresses": all_cluster_addrs,
        "message": f"Address belongs to Co-Spend Wallet Cluster {c_id} ({len(all_cluster_addrs)} co-spent addresses)."
    }


# ------------------------------------------------------------------------------
# WEBSOCKET STREAMING ENDPOINT (WITH EXPLAINABILITY)
# ------------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Streams live block transactions with heuristic risk scores & explainability via WebSocket."""
    await websocket.accept()
    print("WebSocket Client Connected to Live Stream")
    
    try:
        demo_block_folders = sorted([d for d in RAW_BLOCKS_DIR.iterdir() if d.is_dir()])[:10]
        
        for bfolder in demo_block_folders:
            for jfile in bfolder.glob("*.json"):
                # Parse file — safe to catch-and-skip on corrupt/missing JSON
                try:
                    with open(jfile, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    tx_list = data.get("data", {}).get("list", [])
                except Exception:
                    continue

                # Stream transactions — let WebSocketDisconnect propagate
                for tx in tx_list:
                    score = calculate_tx_heuristic_score(tx)
                    is_alert = bool(score >= 0.70)
                    
                    exp_res = explain_heuristic_prediction(tx, score)
                    
                    payload = {
                        "tx_hash": str(tx.get("hash", "")),
                        "block_height": int(tx.get("block_height", 0)),
                        "block_time": int(tx.get("block_time", 0)),
                        "value_btc": round(float(tx.get("outputs_value", 0) / 1e8), 6),
                        "fee_btc": round(float(tx.get("fee", 0) / 1e8), 6),
                        "inputs_count": int(tx.get("inputs_count", 0)),
                        "outputs_count": int(tx.get("outputs_count", 0)),
                        "heuristic_score": score,
                        "is_alert": is_alert,
                        "explanation": exp_res.get("summary"),
                        "stream_mode": "historical_replay",
                        "stream_note": "Replays a fixed historical block range for demo purposes; not a live mempool feed."
                    }

                    await websocket.send_text(json.dumps(payload))
                    await asyncio.sleep(0.15)  # Stream pace

    except WebSocketDisconnect:
        print("WebSocket Client Disconnected")

