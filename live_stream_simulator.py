"""
==============================================================================
Live Bitcoin Blockchain & Mempool Stream Simulator
==============================================================================
Simulates a real-time blockchain feed reading raw block JSON files sequentially,
scoring incoming transactions for fraud risk, and triggering live alerts.
==============================================================================
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

RAW_BLOCKS_DIR = Path("raw data/600000-605999")


def calculate_tx_fraud_score(tx):
    """Calculates heuristic fraud score for a live raw block transaction."""
    fee_btc = tx.get("fee", 0) / 1e8
    out_val_btc = tx.get("outputs_value", 0) / 1e8
    inp_val_btc = tx.get("inputs_value", 0) / 1e8
    inp_cnt = tx.get("inputs_count", 0)
    out_cnt = tx.get("outputs_count", 0)

    fee_ratio = fee_btc / (out_val_btc + 1e-8)
    in_out_ratio = inp_cnt / (out_cnt + 1e-8)

    score = (
        0.35 * np.clip(fee_ratio * 10, 0, 1) +
        0.35 * np.clip(in_out_ratio / 5, 0, 1) +
        0.30 * (out_val_btc > 10.0 or inp_val_btc > 10.0)
    )
    return float(np.round(score, 4))


def stream_live_blocks(raw_dir=RAW_BLOCKS_DIR, max_blocks=10, delay_sec=0.1):
    """Generator function yielding live scored transactions."""
    block_folders = sorted([d for d in Path(raw_dir).iterdir() if d.is_dir()])[:max_blocks]

    for bfolder in block_folders:
        b_height = bfolder.name
        for jfile in bfolder.glob("*.json"):
            try:
                with open(jfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tx_list = data.get("data", {}).get("list", [])

                for tx in tx_list:
                    tx_hash = tx.get("hash", "")
                    score = calculate_tx_fraud_score(tx)
                    tx["fraud_score"] = score
                    tx["value_btc"] = tx.get("outputs_value", 0) / 1e8
                    tx["fee_btc"] = tx.get("fee", 0) / 1e8
                    
                    yield tx
                    if delay_sec > 0:
                        time.sleep(delay_sec)

            except Exception:
                continue


if __name__ == "__main__":
    print("=" * 80)
    print("  SIMULATING LIVE BITCOIN BLOCKCHAIN STREAM & FRAUD ALERTS")
    print("=" * 80)

    alert_count = 0
    total_streamed = 0

    for tx in stream_live_blocks(max_blocks=5, delay_sec=0.01):
        total_streamed += 1
        score = tx["fraud_score"]
        if score >= 0.70:
            alert_count += 1
            print(f"  🚨 [HIGH-RISK ALERT] Tx: {tx['hash'][:16]}... | Block: {tx['block_height']} | Value: {tx['value_btc']:.4f} BTC | Score: {score:.4f}")

    print("=" * 80)
    print(f"Stream Completed: {total_streamed:,} transactions processed, {alert_count:,} high-risk alerts triggered.")
    print("=" * 80)
