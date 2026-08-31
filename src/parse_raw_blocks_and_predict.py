"""
==============================================================================
Raw Block Feature Extraction & Real-Time Fraud Inference Engine (Ultra-Low RAM)
==============================================================================
1. Scans raw block JSON files from raw data/600000-605999 (configurable via CLI).
2. Direct disk streaming: Writes transactions immediately to disk per block,
   consuming 0 MB of RAM for transaction accumulation.
3. Chunked address classification: Evaluates millions of addresses in 250k-row
   numpy batches, streaming directly to CSV.
4. Resumable checkpointing via raw_inference_manifest.json.
==============================================================================
"""

import os
import gc
import csv
import json
import pickle
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from feature_utils import calculate_tx_heuristic_score
except ImportError:
    from src.feature_utils import calculate_tx_heuristic_score

RAW_BLOCKS_DIR = Path("raw data/600000-605999")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 80

# ------------------------------------------------------------------------------
# CLI Arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Raw Bitcoin Block Inference & Heuristic Scoring Engine")
parser.add_argument(
    "--max-blocks",
    type=int,
    default=20,
    help="Number of new block folders to process (default: 20, use -1 for all available)."
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Process all available block folders on disk."
)
args = parser.parse_args()

max_blocks = -1 if args.all else args.max_blocks

print(f"\n{SEP}")
print("  RAW BLOCK FEATURE EXTRACTION & FRAUD INFERENCE ENGINE (STREAMING)")
print(SEP)

# Startup Disclaimer & Transparency Note
print("\n[TRANSPARENCY NOTE]")
print("  NOTE: heuristic_score is a rule-based proxy, not the trained ML model's output,")
print("  because raw block transactions lack ground-truth labels needed to validate")
print("  a proxy model against Elliptic's PCA-anonymized feature space.")
print("  Address classifications are ML-driven using the trained reduced BABD-13 model.")
print("  KPIs reflect only the blocks processed so far (tracked in models/raw_inference_manifest.json).")

# ------------------------------------------------------------------------------
# 1. Load trained models
# ------------------------------------------------------------------------------
print("\n[1] Loading trained model artifacts ...")

babd_reduced_path = MODELS_DIR / "babd13_reduced_model.pkl"
if not babd_reduced_path.exists():
    raise FileNotFoundError(
        f"Missing {babd_reduced_path}! Please run 'python train_babd13_reduced_model.py' first."
    )

with open(babd_reduced_path, "rb") as f:
    babd_artifact = pickle.load(f)

babd_model = babd_artifact["model"]
feature_names = babd_artifact["feature_names"]
idx_to_label = babd_artifact["idx_to_label"]
label_names = babd_artifact["label_names"]

print(f"    Loaded Reduced BABD-13 Address Classifier: {type(babd_model).__name__}")
print(f"    Features Expected: {feature_names}")

# ------------------------------------------------------------------------------
# 2. Checkpoint Manifest & Resumable Folder Discovery
# ------------------------------------------------------------------------------
manifest_path = MODELS_DIR / "raw_inference_manifest.json"
manifest = {
    "total_blocks_available": 0,
    "blocks_processed_this_run": 0,
    "cumulative_blocks_processed": 0,
    "processed_folder_names": [],
    "total_transactions_scored": 0,
    "total_unique_addresses": 0,
    "last_run_timestamp_utc": None,
    "coverage_pct": 0.0
}

if manifest_path.exists():
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"    [!] Warning: could not parse existing manifest: {e}")

if not RAW_BLOCKS_DIR.exists():
    raise FileNotFoundError(f"Missing raw blocks directory: {RAW_BLOCKS_DIR}")

all_block_folders = sorted([d for d in RAW_BLOCKS_DIR.iterdir() if d.is_dir()])
total_available = len(all_block_folders)
manifest["total_blocks_available"] = total_available

processed_set = set(manifest.get("processed_folder_names", []))
new_folders = [d for d in all_block_folders if d.name not in processed_set]

if max_blocks == -1:
    folders_to_process = new_folders
else:
    folders_to_process = new_folders[:max_blocks]

print(f"\n[2] Resumable Block Batch Configuration:")
print(f"    - Total block folders on disk     : {total_available:,}")
print(f"    - Previously processed folders    : {len(processed_set):,}")
print(f"    - Unprocessed block folders       : {len(new_folders):,}")
print(f"    - Folders to process in this run  : {len(folders_to_process):,} (Requested max: {'ALL' if max_blocks == -1 else max_blocks})")

# ------------------------------------------------------------------------------
# 3. Direct-to-Disk Transaction Streaming & Address Tracking
# ------------------------------------------------------------------------------
output_tx_path = MODELS_DIR / "raw_block_predictions.csv"
output_addr_path = MODELS_DIR / "raw_address_predictions.csv"

tx_headers = [
    "tx_hash", "block_height", "block_time", "fee",
    "inputs_count", "outputs_count", "inputs_value", "outputs_value",
    "size", "value_btc", "fee_btc", "heuristic_score", "is_high_risk"
]

# Check existing files
tx_file_exists = output_tx_path.exists() and output_tx_path.stat().st_size > 0
addr_file_exists = output_addr_path.exists() and output_addr_path.stat().st_size > 0

# If starting fresh on all blocks (from folder 40), open append mode
write_tx_mode = "a" if tx_file_exists and len(processed_set) > 0 else "w"

# Open transaction CSV file stream
tx_file = open(output_tx_path, mode=write_tx_mode, newline="", encoding="utf-8")
tx_writer = csv.writer(tx_file)

if write_tx_mode == "w":
    tx_writer.writerow(tx_headers)

address_stats = {}
tx_count_this_run = 0

# Pre-load address stats from existing CSV on resumed runs to accumulate across runs
if len(processed_set) > 0 and addr_file_exists:
    print("    [RESUME] Pre-loading existing address stats from raw_address_predictions.csv ...")
    try:
        for chunk in pd.read_csv(output_addr_path, chunksize=250_000, dtype={"account": str}):
            for _, row in chunk.iterrows():
                addr = str(row["account"])
                tx_cnt = int(row.get("tx_count", 1))
                tot_recv_sat = int(float(row.get("total_received", 0)) * 1e8)
                # Use first/last_seen_block_time if available (backward compat)
                first_bt = int(row["first_seen_block_time"]) if "first_seen_block_time" in row and pd.notna(row.get("first_seen_block_time")) else 0
                last_bt = int(row["last_seen_block_time"]) if "last_seen_block_time" in row and pd.notna(row.get("last_seen_block_time")) else 0
                if first_bt == 0 and last_bt == 0:
                    dur_sec = int(row.get("active_duration_sec", 0))
                    last_bt = dur_sec  # approximate
                    first_bt = 0
                address_stats[addr] = [tx_cnt, tot_recv_sat, first_bt, last_bt]
        print(f"    [RESUME] Loaded {len(address_stats):,} previously-scored addresses into memory.")
    except Exception as e:
        print(f"    [!] Warning: could not pre-load address stats: {e}. Starting fresh.")
        address_stats = {}


if folders_to_process:
    print(f"\n[3] Streaming {len(folders_to_process):,} block folders directly to disk ...")
    
    for idx, bfolder in enumerate(folders_to_process, 1):
        if idx % 200 == 0 or idx == len(folders_to_process):
            print(f"    Processing folder {idx:>5,d}/{len(folders_to_process):,} ({bfolder.name}) | Txs: {tx_count_this_run:>9,d} | Addrs: {len(address_stats):>9,d}")

        for jfile in bfolder.glob("*.json"):
            try:
                with open(jfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tx_list = data.get("data", {}).get("list", [])
                
                for tx in tx_list:
                    tx_hash = tx.get("hash", "")
                    b_height = tx.get("block_height", 0)
                    b_time = tx.get("block_time", 0)
                    fee = tx.get("fee", 0)
                    inp_cnt = tx.get("inputs_count", 0)
                    out_cnt = tx.get("outputs_count", 0)
                    inp_val = tx.get("inputs_value", 0)
                    out_val = tx.get("outputs_value", 0)
                    size = tx.get("size", 0)
                    
                    heuristic_score = calculate_tx_heuristic_score(tx)
                    val_btc = round(out_val / 1e8, 6)
                    fee_btc = round(fee / 1e8, 6)
                    is_high = heuristic_score >= 0.70

                    # Write row directly to disk (0 RAM consumption)
                    tx_writer.writerow([
                        tx_hash, b_height, b_time, fee,
                        inp_cnt, out_cnt, inp_val, out_val,
                        size, val_btc, fee_btc, round(heuristic_score, 4), is_high
                    ])
                    tx_count_this_run += 1
                    
                    # Track address behavioral profile in compact memory
                    for out in tx.get("outputs", []):
                        addrs = out.get("addresses", [])
                        val_sat = out.get("value", 0)
                        for addr in addrs:
                            if addr:
                                if addr not in address_stats:
                                    address_stats[addr] = [1, val_sat, b_time, b_time]
                                else:
                                    rec = address_stats[addr]
                                    rec[0] += 1
                                    rec[1] += val_sat
                                    if b_time > rec[3]:
                                        rec[3] = b_time
                                
            except Exception:
                continue

    tx_file.flush()
    tx_file.close()

print(f"\n    Finished streaming transactions to {output_tx_path}")

# Count total scored transactions from disk
try:
    total_txs_final = sum(1 for _ in open(output_tx_path, "r", encoding="utf-8", errors="ignore")) - 1
except Exception:
    total_txs_final = tx_count_this_run

print(f"    Total Scored Transactions on Disk: {total_txs_final:,}")

# ------------------------------------------------------------------------------
# 4. Chunked Streaming ML Inference on Address Features (Low Memory)
# ------------------------------------------------------------------------------
total_addrs = len(address_stats)
print(f"\n[4] Classifying {total_addrs:,} unique addresses in 250,000-row chunks ...")

CHUNK_SIZE = 250_000
addr_keys = list(address_stats.keys())
n_chunks = (total_addrs + CHUNK_SIZE - 1) // CHUNK_SIZE if total_addrs > 0 else 0

# Always rewrite full address file to avoid duplicates from partial overlap
addr_headers = [
    "account", "total_received", "tx_count", "avg_value_per_tx",
    "active_duration_sec", "tx_frequency", "predicted_class_idx",
    "model_confidence", "predicted_category",
    "first_seen_block_time", "last_seen_block_time"
]

addr_file = open(output_addr_path, mode="w", newline="", encoding="utf-8")
addr_writer = csv.writer(addr_file)
addr_writer.writerow(addr_headers)

for chunk_idx in range(n_chunks):
    start_i = chunk_idx * CHUNK_SIZE
    end_i = min(start_i + CHUNK_SIZE, total_addrs)
    chunk_keys = addr_keys[start_i:end_i]
    chunk_len = len(chunk_keys)

    print(f"    Running ML inference on chunk {chunk_idx + 1:>3,d}/{n_chunks:,} ({chunk_len:>7,d} addresses) ...")

    X_chunk = np.empty((chunk_len, 5), dtype=np.float32)

    for row_i, addr in enumerate(chunk_keys):
        s = address_stats[addr]
        tx_cnt = max(s[0], 1)
        tot_rec = s[1] / 1e8
        avg_val = tot_rec / tx_cnt
        dur_sec = max(s[3] - s[2], 0)
        freq = tx_cnt / max(dur_sec, 1)

        X_chunk[row_i, 0] = tot_rec
        X_chunk[row_i, 1] = tx_cnt
        X_chunk[row_i, 2] = avg_val
        X_chunk[row_i, 3] = dur_sec
        X_chunk[row_i, 4] = freq

    # ML Inference
    preds = babd_model.predict(X_chunk)
    probs = babd_model.predict_proba(X_chunk)
    confs = np.round(np.max(probs, axis=1), 4)

    for row_i, addr in enumerate(chunk_keys):
        p_idx = int(preds[row_i])
        cat_name = label_names.get(idx_to_label.get(p_idx, p_idx), f"Class {p_idx}")
        
        s = address_stats[addr]
        addr_writer.writerow([
            addr,
            round(float(X_chunk[row_i, 0]), 6),
            int(X_chunk[row_i, 1]),
            round(float(X_chunk[row_i, 2]), 6),
            int(X_chunk[row_i, 3]),
            round(float(X_chunk[row_i, 4]), 6),
            p_idx,
            confs[row_i],
            cat_name,
            int(s[2]),
            int(s[3])
        ])

    del X_chunk, preds, probs, confs
    gc.collect()

addr_file.flush()
addr_file.close()

# Release dictionary
del address_stats, addr_keys
gc.collect()

# Count total unique addresses on disk
try:
    final_addr_count = sum(1 for _ in open(output_addr_path, "r", encoding="utf-8", errors="ignore")) - 1
except Exception:
    final_addr_count = total_addrs

# ------------------------------------------------------------------------------
# 5. Update Manifest & Coverage Report
# ------------------------------------------------------------------------------
newly_processed_names = [d.name for d in folders_to_process]
all_processed_names = sorted(list(processed_set.union(set(newly_processed_names))))

manifest["blocks_processed_this_run"] = len(folders_to_process)
manifest["processed_folder_names"] = all_processed_names
manifest["cumulative_blocks_processed"] = len(all_processed_names)
manifest["total_transactions_scored"] = int(total_txs_final)
manifest["total_unique_addresses"] = int(total_addrs)  # use merged address_stats count
manifest["last_run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
manifest["coverage_pct"] = round((len(all_processed_names) / max(total_available, 1)) * 100.0, 2)

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{SEP}")
print("  [COVERAGE & INFERENCE SUMMARY]")
print(f"    Blocks available on disk        : {manifest['total_blocks_available']:,}")
print(f"    Blocks processed (this run)     : {manifest['blocks_processed_this_run']:,}")
print(f"    Blocks processed (cumulative)   : {manifest['cumulative_blocks_processed']:,}")
print(f"    Dataset coverage                : {manifest['coverage_pct']:.2f}%")
print(f"    Total transactions scored       : {manifest['total_transactions_scored']:,}")
print(f"    Total unique addresses scored   : {manifest['total_unique_addresses']:,}")
print(f"    Manifest checkpoint saved       : {manifest_path}")
print(SEP)

print(f"\n{SEP}")
print("  RAW BLOCK FRAUD INFERENCE ENGINE COMPLETE (STREAMING FINISHED)")
print(f"{SEP}\n")
