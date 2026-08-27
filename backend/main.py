"""
==============================================================================
FastAPI Backend Application — Bitcoin Fraud Monitoring Engine
==============================================================================
Provides REST API endpoints and Real-Time WebSocket stream for live block fraud alerts:
  - GET  /api/kpis         : High-level system telemetry & risk metrics
  - GET  /api/search       : Query transaction hash or Bitcoin address for risk scoring
  - GET  /api/benchmarks   : Model evaluation metrics for Elliptic & BABD-13
  - WS   /ws/stream        : Real-time WebSocket stream for live block transactions

NOTE: heuristic_score is a rule-based proxy, not the trained ML model's output,
because raw block transactions lack ground-truth labels needed to validate
a proxy model against Elliptic's feature space. Address predictions use the trained
reduced BABD-13 model on derived honest behavioral features.
==============================================================================
"""

import asyncio
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from feature_utils import calculate_tx_heuristic_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "processed"
PLOTS_DIR = BASE_DIR / "plots"
RAW_BLOCKS_DIR = BASE_DIR / "raw data" / "600000-605999"

print("[BACKEND STARTUP NOTICE]")
print("  NOTE: heuristic_score is a rule-based proxy, not the trained ML model's output,")
print("  because raw block transactions lack ground-truth labels needed to validate")
print("  a proxy model against Elliptic's feature space.")

app = FastAPI(title="BitSentinel-AI Fraud Detection Platform API", version="1.1.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "note": "Transaction risk scores are heuristic; Address classifications are ML-driven."
    }


@app.get("/api/kpis")
def get_kpis():
    """Returns high-level system telemetry and KPI metrics."""
    raw_preds_path = MODELS_DIR / "raw_block_predictions.csv"
    if not raw_preds_path.exists():
        return {
            "status": "no_predictions_yet",
            "message": "Run parse_raw_blocks_and_predict.py first",
            "total_scored_transactions": 0,
            "high_risk_alerts": 0,
            "risk_ratio_pct": 0,
            "total_monitored_btc_volume": 0,
            "flagged_high_risk_btc_volume": 0
        }

    df_raw = pd.read_csv(raw_preds_path)
    score_col = "heuristic_score" if "heuristic_score" in df_raw.columns else "fraud_score"
    total_txs = int(len(df_raw))
    high_risk_txs = int((df_raw[score_col] >= 0.70).sum())
    total_vol = float(df_raw["value_btc"].sum())
    flagged_vol = float(df_raw[df_raw[score_col] >= 0.70]["value_btc"].sum())
    risk_ratio = round(100 * high_risk_txs / max(total_txs, 1), 2)

    return {
        "status": "ready",
        "total_scored_transactions": total_txs,
        "high_risk_alerts": high_risk_txs,
        "risk_ratio_pct": risk_ratio,
        "total_monitored_btc_volume": round(total_vol, 2),
        "flagged_high_risk_btc_volume": round(flagged_vol, 2),
    }


@app.get("/api/search")
def search_entity(q: str = Query(..., description="Transaction hash or Bitcoin address")):
    """
    Searches transaction hash (64 hex) or Bitcoin address (1, 3, bc1) for risk intelligence.
    Returns transaction heuristic risk score or address ML-predicted classification.
    """
    query_str = q.strip()
    is_tx_hash = bool(re.fullmatch(r"^[0-9a-fA-F]{64}$", query_str))
    is_btc_address = bool(re.fullmatch(r"^(1|3|bc1)[a-zA-HJ-NP-Z0-9]{25,62}$", query_str))

    # 1. Address Search Check
    raw_addr_path = MODELS_DIR / "raw_address_predictions.csv"
    if is_btc_address or (raw_addr_path.exists() and not is_tx_hash):
        if raw_addr_path.exists():
            df_addr = pd.read_csv(raw_addr_path)
            match_addr = df_addr[df_addr["account"].str.lower() == query_str.lower()]
            if not match_addr.empty:
                addr_row = match_addr.iloc[0]
                return {
                    "found": True,
                    "query": query_str,
                    "type": "address",
                    "scoring_engine": "BABD-13 Reduced-Feature ML Classifier (Random Forest)",
                    "predicted_category": str(addr_row["predicted_category"]),
                    "model_confidence": float(addr_row.get("model_confidence", 0.0)),
                    "details": {
                        "account": str(addr_row["account"]),
                        "tx_count": int(addr_row["tx_count"]),
                        "total_received_btc": round(float(addr_row["total_received"]), 6),
                        "avg_value_per_tx_btc": round(float(addr_row["avg_value_per_tx"]), 6),
                        "active_duration_sec": int(addr_row["active_duration_sec"]),
                        "tx_frequency": round(float(addr_row["tx_frequency"]), 6)
                    }
                }

    # 2. Transaction Search Check
    raw_preds_path = MODELS_DIR / "raw_block_predictions.csv"
    if raw_preds_path.exists():
        df_raw = pd.read_csv(raw_preds_path)
        match_tx = df_raw[df_raw["tx_hash"].str.contains(query_str, case=False, na=False)]

        if not match_tx.empty:
            tx = match_tx.iloc[0]
            score_col = "heuristic_score" if "heuristic_score" in tx else "fraud_score"
            score = float(tx[score_col])
            is_high_risk = bool(score >= 0.70)
            
            return {
                "found": True,
                "query": query_str,
                "type": "transaction",
                "scoring_engine": "Rule-Based Heuristic Proxy (Fee/Volume/Degree ratios)",
                "heuristic_score": score,
                "is_high_risk": is_high_risk,
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

    # 3. Not Found response
    sample_txs = []
    if raw_preds_path.exists():
        df_raw = pd.read_csv(raw_preds_path)
        sample_txs = df_raw.head(3).to_dict(orient="records")

    return {
        "found": False,
        "query": query_str,
        "message": f"No exact match found for '{query_str}'. Please check the hash or address.",
        "sample_transactions": sample_txs
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
# WEBSOCKET STREAMING ENDPOINT
# ------------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Streams live block transactions with heuristic risk scores via WebSocket."""
    await websocket.accept()
    print("WebSocket Client Connected to Live Stream")
    
    try:
        block_folders = sorted([d for d in RAW_BLOCKS_DIR.iterdir() if d.is_dir()])[:10]
        
        for bfolder in block_folders:
            for jfile in bfolder.glob("*.json"):
                try:
                    with open(jfile, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    tx_list = data.get("data", {}).get("list", [])

                    for tx in tx_list:
                        score = calculate_tx_heuristic_score(tx)
                        payload = {
                            "tx_hash": str(tx.get("hash", "")),
                            "block_height": int(tx.get("block_height", 0)),
                            "block_time": int(tx.get("block_time", 0)),
                            "value_btc": round(float(tx.get("outputs_value", 0) / 1e8), 6),
                            "fee_btc": round(float(tx.get("fee", 0) / 1e8), 6),
                            "inputs_count": int(tx.get("inputs_count", 0)),
                            "outputs_count": int(tx.get("outputs_count", 0)),
                            "heuristic_score": score,
                            "is_alert": bool(score >= 0.70)
                        }

                        await websocket.send_text(json.dumps(payload))
                        await asyncio.sleep(0.15)  # Stream pace

                except Exception:
                    continue

    except WebSocketDisconnect:
        print("WebSocket Client Disconnected")
