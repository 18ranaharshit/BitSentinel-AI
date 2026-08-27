"""
==============================================================================
Live Bitcoin Blockchain & Mempool Stream Simulator
==============================================================================
Simulates a real-time blockchain feed reading raw block JSON files sequentially,
scoring incoming transactions for fraud risk using shared feature utilities,
and triggering live alerts.

NOTE: heuristic_score is a rule-based proxy, not the trained ML model's output,
because raw block transactions lack ground-truth labels needed to validate
a proxy model against Elliptic's feature space.
==============================================================================
"""

import json
import time
from pathlib import Path
import pandas as pd

from feature_utils import calculate_tx_heuristic_score

RAW_BLOCKS_DIR = Path("raw data/600000-605999")


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
                    score = calculate_tx_heuristic_score(tx)
                    tx["heuristic_score"] = score
                    tx["value_btc"] = tx.get("outputs_value", 0) / 1e8
                    tx["fee_btc"] = tx.get("fee", 0) / 1e8
                    tx["is_alert"] = bool(score >= 0.70)
                    
                    yield tx
                    if delay_sec > 0:
                        time.sleep(delay_sec)

            except Exception:
                continue


if __name__ == "__main__":
    print("=" * 80)
    print("  SIMULATING LIVE BITCOIN BLOCKCHAIN STREAM & HEURISTIC RISK ALERTS")
    print("  NOTE: heuristic_score is a rule-based proxy, not trained ML model output.")
    print("=" * 80)

    alert_count = 0
    total_streamed = 0

    for tx in stream_live_blocks(max_blocks=5, delay_sec=0.01):
        total_streamed += 1
        score = tx["heuristic_score"]
        if score >= 0.70:
            alert_count += 1
            print(f"  🚨 [HIGH-RISK ALERT] Tx: {tx.get('hash', '')[:16]}... | Block: {tx.get('block_height')} | Value: {tx['value_btc']:.4f} BTC | Heuristic Score: {score:.4f}")

    print("=" * 80)
    print(f"Stream Completed: {total_streamed:,} transactions processed, {alert_count:,} high-risk alerts triggered.")
    print("=" * 80)
