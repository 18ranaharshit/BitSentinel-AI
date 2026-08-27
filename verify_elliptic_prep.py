"""
==============================================================================
Elliptic Dataset Split Verification & Graph Diagnostics Script
==============================================================================
Checks:
  1. Load processed/ elliptic_train.csv, elliptic_val.csv, elliptic_test.csv, elliptic_unlabeled.csv
  2. Confirm zero txId overlap between train/val/test
  3. Confirm min/max time_step ranges
  4. Print class balance (counts and %) in train, val, and test
  5. Load edgelist and count intra-split vs cross-split edges
  6. Confirm unlabeled txId count and zero overlap with labeled splits
==============================================================================
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("processed")
RAW_DIR = Path("raw data")

SEP = "=" * 80

print(f"\n{SEP}")
print("  ELLIPTIC DATASET SPLIT VERIFICATION & GRAPH DIAGNOSTICS")
print(SEP)

# 1. Load split files
print("\n[1] Loading processed Elliptic files ...")
df_train = pd.read_csv(PROCESSED_DIR / "elliptic_train.csv")
df_val   = pd.read_csv(PROCESSED_DIR / "elliptic_val.csv")
df_test  = pd.read_csv(PROCESSED_DIR / "elliptic_test.csv")
df_unlabeled = pd.read_csv(PROCESSED_DIR / "elliptic_unlabeled.csv")

print(f"    Train shape    : {df_train.shape}")
print(f"    Val shape      : {df_val.shape}")
print(f"    Test shape     : {df_test.shape}")
print(f"    Unlabeled shape: {df_unlabeled.shape}")

# 2. Check txId overlap between train/val/test
print("\n[2] Checking txId overlap between train, val, and test splits ...")
tx_train = set(df_train["txId"])
tx_val   = set(df_val["txId"])
tx_test  = set(df_test["txId"])

ov_tr_val = len(tx_train & tx_val)
ov_tr_test = len(tx_train & tx_test)
ov_val_test = len(tx_val & tx_test)

print(f"    Train & Val txId overlap : {ov_tr_val}")
print(f"    Train & Test txId overlap: {ov_tr_test}")
print(f"    Val & Test txId overlap  : {ov_val_test}")
if ov_tr_val + ov_tr_test + ov_val_test == 0:
    print("    ✓ VERIFIED: Zero txId overlap across train/val/test splits.")

# 3. Check time_step ranges
print("\n[3] Time step ranges per split:")
print(f"    Train     : Min step = {df_train['time_step'].min():>2d}, Max step = {df_train['time_step'].max():>2d}")
print(f"    Val       : Min step = {df_val['time_step'].min():>2d}, Max step = {df_val['time_step'].max():>2d}")
print(f"    Test      : Min step = {df_test['time_step'].min():>2d}, Max step = {df_test['time_step'].max():>2d}")
print(f"    Unlabeled : Min step = {df_unlabeled['time_step'].min():>2d}, Max step = {df_unlabeled['time_step'].max():>2d}")

# 4. Class balance in train, val, test
print("\n[4] Class balance (illicit=1 vs licit=0) per split:")
for name, df_split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    total = len(df_split)
    vc = df_split["class"].value_counts().to_dict()
    licit = vc.get(0, 0)
    illicit = vc.get(1, 0)
    print(f"    {name:<5s} | Total: {total:>6,d} | Licit (0): {licit:>6,d} ({100*licit/total:>5.2f}%) | Illicit (1): {illicit:>5,d} ({100*illicit/total:>5.2f}%) | Ratio: {licit/illicit:.2f}:1")

# 5. Edgelist graph connection diagnostics
print("\n[5] Loading edgelist and analyzing split-boundary graph edges ...")
df_edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
total_edges = len(df_edges)
print(f"    Total directed edges in edgelist: {total_edges:,}")

# Build mapping txId -> split_name
split_map = {}
for tx in tx_train:
    split_map[tx] = "Train"
for tx in tx_val:
    split_map[tx] = "Val"
for tx in tx_test:
    split_map[tx] = "Test"
for tx in df_unlabeled["txId"]:
    split_map[tx] = "Unlabeled"

# Map edge endpoints
edge_s1 = df_edges["txId1"].map(split_map).fillna("Unknown")
edge_s2 = df_edges["txId2"].map(split_map).fillna("Unknown")

edge_types = edge_s1 + " -> " + edge_s2
edge_counts = edge_types.value_counts()

print("\n    Edge distribution by split endpoints:")
intra_counts = {}
cross_counts = 0

for pair, cnt in edge_counts.items():
    s1, s2 = pair.split(" -> ")
    pct = 100 * cnt / total_edges
    if s1 == s2:
        intra_counts[s1] = cnt
        print(f"      Intra-{s1:<9s} ({s1} -> {s2}): {cnt:>7,d} edges ({pct:>5.2f}%)")
    else:
        cross_counts += cnt
        print(f"      Cross-{s1} -> {s2:<10s}: {cnt:>7,d} edges ({pct:>5.2f}%)")

print(f"\n    Summary of Graph Connections:")
print(f"      Intra-Train edges : {intra_counts.get('Train', 0):>7,d} ({100*intra_counts.get('Train', 0)/total_edges:.2f}%)")
print(f"      Intra-Val edges   : {intra_counts.get('Val', 0):>7,d} ({100*intra_counts.get('Val', 0)/total_edges:.2f}%)")
print(f"      Intra-Test edges  : {intra_counts.get('Test', 0):>7,d} ({100*intra_counts.get('Test', 0)/total_edges:.2f}%)")
print(f"      Total Cross-Edges : {cross_counts:>7,d} ({100*cross_counts/total_edges:.2f}%)")

# 6. Unlabeled txId check
print("\n[6] Unlabeled set verification:")
tx_unlabeled = set(df_unlabeled["txId"])
print(f"    Total unlabeled transactions: {len(tx_unlabeled):,}")

ov_unlabeled_labeled = tx_unlabeled & (tx_train | tx_val | tx_test)
print(f"    Overlap between Unlabeled and (Train/Val/Test): {len(ov_unlabeled_labeled)}")
if len(ov_unlabeled_labeled) == 0:
    print("    ✓ VERIFIED: Zero overlap between unlabeled set and labeled splits.")

print(f"\n{SEP}")
print("  ELLIPTIC PREP VERIFICATION COMPLETE")
print(SEP)
