"""
==============================================================================
Bitcoin Fraud-Detection Project — Exploratory Data Analysis
==============================================================================
Three-section exploration script for:
  A) Elliptic dataset  (transaction-level, graph-based)
  B) BABD-13 dataset   (address-level, behavioral features)
  C) Raw Bitcoin block JSON data (schema discovery only)

Libraries: pandas, matplotlib, seaborn, json, os, pathlib, collections
==============================================================================
"""

import os
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (safe on servers)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(r"raw data")
ELLIPTIC_FEATURES = BASE / "elliptic_txs_features.csv"
ELLIPTIC_CLASSES  = BASE / "elliptic_txs_classes.csv"
ELLIPTIC_EDGES    = BASE / "elliptic_txs_edgelist.csv"
BABD_CSV          = BASE / "BABD-13.csv"
RAW_BLOCKS_DIR    = BASE / "600000-605999"

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

SEP = "=" * 80


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 :  ELLIPTIC  DATASET  EXPLORATION                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print(f"\n{SEP}")
print("  SECTION 1: ELLIPTIC DATASET EXPLORATION")
print(SEP)

# ---------- 1a. Load all three Elliptic files ----------
print("\n[1a] Loading Elliptic files …")

# Features file has NO header row
feat_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, 166)]
df_features = pd.read_csv(ELLIPTIC_FEATURES, header=None, names=feat_cols)

df_classes = pd.read_csv(ELLIPTIC_CLASSES)
df_edges   = pd.read_csv(ELLIPTIC_EDGES)

# ---------- 1b. Shapes ----------
print(f"\n[1b] Shapes:")
print(f"     Features : {df_features.shape}  (rows × cols)")
print(f"     Classes  : {df_classes.shape}")
print(f"     Edges    : {df_edges.shape}")

# ---------- 1c. Merge features + classes ----------
print("\n[1c] Merging features with classes on txId …")
df_elliptic = df_features.merge(df_classes, on="txId", how="left")
print(f"     Merged shape: {df_elliptic.shape}")

# ---------- 1d. Class distribution ----------
print("\n[1d] Class distribution (value_counts):")
class_counts = df_elliptic["class"].value_counts()
print(class_counts)

total = len(df_elliptic)
print("\n     Percentage breakdown:")
for cls, cnt in class_counts.items():
    label_map = {"1": "illicit", "2": "licit", "unknown": "unknown"}
    nice = label_map.get(str(cls), str(cls))
    print(f"       {nice:>8s}  →  {cnt:>7,d}  ({100*cnt/total:.2f}%)")

# ---------- 1e. Time steps ----------
print("\n[1e] Time-step statistics:")
n_steps = df_elliptic["time_step"].nunique()
print(f"     Number of distinct time steps: {n_steps}")

txns_per_step = df_elliptic.groupby("time_step").size()
print(f"     Transactions per time step  — min: {txns_per_step.min()}, "
      f"max: {txns_per_step.max()}, mean: {txns_per_step.mean():.1f}")
print(f"\n     First 5 time steps:\n{txns_per_step.head()}")

# ---------- 1f. Missing values & duplicates ----------
print("\n[1f] Missing values & duplicates:")
missing = df_elliptic.isnull().sum()
cols_with_missing = missing[missing > 0]
if len(cols_with_missing) == 0:
    print("     No missing values in merged DataFrame.")
else:
    print(f"     Columns with missing values:\n{cols_with_missing}")

dup_txids = df_elliptic["txId"].duplicated().sum()
print(f"     Duplicate txIds in features: {dup_txids}")

# ---------- 1g. Descriptive statistics for a handful of features ----------
print("\n[1g] describe() on selected feature columns:")
sample_feats = ["feat_1", "feat_50", "feat_100", "feat_150", "feat_165"]
sample_feats = [c for c in sample_feats if c in df_elliptic.columns]
print(df_elliptic[sample_feats].describe().to_string())

# ---------- 1h. Bar chart — class distribution ----------
print("\n[1h] Plotting Elliptic class distribution bar chart …")
fig, ax = plt.subplots(figsize=(7, 5))
label_order = ["1", "2", "unknown"]
plot_labels = ["Illicit (1)", "Licit (2)", "Unknown"]
counts = [class_counts.get(c, 0) for c in label_order]
colors = ["#e74c3c", "#2ecc71", "#95a5a6"]
bars = ax.bar(plot_labels, counts, color=colors, edgecolor="black", linewidth=0.6)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
            f"{cnt:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Elliptic Dataset — Class Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Transactions")
ax.set_xlabel("Class")
sns.despine()
plt.tight_layout()
fig.savefig(PLOT_DIR / "elliptic_class_distribution.png", dpi=150)
plt.close(fig)
print(f"     Saved → {PLOT_DIR / 'elliptic_class_distribution.png'}")

# Free memory from huge features DataFrame
del df_features


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 :  BABD-13  DATASET  EXPLORATION                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print(f"\n\n{SEP}")
print("  SECTION 2: BABD-13 DATASET EXPLORATION")
print(SEP)

# ---------- 2a. Load & print shape, column names ----------
print("\n[2a] Loading BABD-13.csv …")
df_babd = pd.read_csv(BABD_CSV)
print(f"     Shape: {df_babd.shape}")
print(f"\n     All column names ({len(df_babd.columns)}):")
for i, col in enumerate(df_babd.columns):
    print(f"       [{i:>3d}] {col}")

# ---------- 2b. Identify label column ----------
print("\n[2b] Label / category column identification:")
# The last column is 'label'
label_col = "label"
if label_col not in df_babd.columns:
    # Fallback: find a column that looks like a label
    candidates = [c for c in df_babd.columns if "label" in c.lower() or "class" in c.lower() or "category" in c.lower()]
    label_col = candidates[0] if candidates else df_babd.columns[-1]
    print(f"     (Auto-detected label column: '{label_col}')")
else:
    print(f"     Label column confirmed: '{label_col}'")

BABD_LABEL_MAP = {
    0: "Blackmail",
    1: "Cyber-security service",
    2: "Darknet market",
    3: "Centralized exchange",
    4: "P2P financial infrastructure",
    5: "P2P financial service",
    6: "Gambling",
    7: "Govt. criminal blacklist",
    8: "Money laundering",
    9: "Ponzi scheme",
    10: "Mining pool",
    11: "Tumbler",
    12: "Individual wallet",
}

print(f"\n     value_counts() for '{label_col}':")
babd_vc = df_babd[label_col].value_counts().sort_index()
print(babd_vc)
print(f"\n     With label names:")
for lbl, cnt in babd_vc.items():
    name = BABD_LABEL_MAP.get(int(lbl), f"unknown-{lbl}")
    print(f"       {lbl:>2d}  {name:<32s}  {cnt:>8,d}  ({100*cnt/len(df_babd):.2f}%)")

# ---------- 2c. Missing values & duplicate rows ----------
print("\n[2c] Missing values & duplicates:")
missing_babd = df_babd.isnull().sum()
cols_missing_babd = missing_babd[missing_babd > 0]
if len(cols_missing_babd) == 0:
    print("     No missing values.")
else:
    print(f"     Columns with missing values ({len(cols_missing_babd)}):")
    print(cols_missing_babd.to_string())

dup_rows = df_babd.duplicated().sum()
dup_accounts = df_babd["account"].duplicated().sum() if "account" in df_babd.columns else "N/A"
print(f"     Duplicate rows   : {dup_rows}")
print(f"     Duplicate accounts: {dup_accounts}")

# ---------- 2d. Descriptive stats on numeric columns ----------
print("\n[2d] describe() on numeric columns (first 10):")
numeric_cols = df_babd.select_dtypes(include="number").columns.tolist()
show_cols = numeric_cols[:10]
print(df_babd[show_cols].describe().to_string())

# ---------- 2e. Bar chart — label distribution ----------
print("\n[2e] Plotting BABD-13 label distribution bar chart …")
fig, ax = plt.subplots(figsize=(14, 6))
x_labels = [f"{k}: {BABD_LABEL_MAP.get(int(k), '?')}" for k in babd_vc.index]
bars = ax.bar(x_labels, babd_vc.values, color=sns.color_palette("viridis", len(babd_vc)),
              edgecolor="black", linewidth=0.5)
for bar, cnt in zip(bars, babd_vc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(df_babd)*0.002,
            f"{cnt:,}", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=45)
ax.set_title("BABD-13 Dataset — Label Distribution (13 classes)", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Addresses")
ax.set_xlabel("Label")
plt.xticks(rotation=45, ha="right", fontsize=9)
sns.despine()
plt.tight_layout()
fig.savefig(PLOT_DIR / "babd13_label_distribution.png", dpi=150)
plt.close(fig)
print(f"     Saved → {PLOT_DIR / 'babd13_label_distribution.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 :  RAW BLOCK JSON  EXPLORATION  (Schema Discovery Only)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print(f"\n\n{SEP}")
print("  SECTION 3: RAW BLOCK JSON EXPLORATION  (schema discovery)")
print(SEP)


def discover_schema(obj, prefix=""):
    """Recursively extract all keys from a nested dict/list, returning
    a set of dot-separated key paths."""
    paths = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            paths.add(p)
            paths |= discover_schema(v, p)
    elif isinstance(obj, list) and len(obj) > 0:
        # Inspect first element to get representative schema
        paths |= discover_schema(obj[0], prefix + "[]")
    return paths


# ---------- 3a. Recursively list all JSON files ----------
print("\n[3a] Scanning JSON files under raw data/600000-605999 …")
json_files = sorted(RAW_BLOCKS_DIR.rglob("*.json"))
n_json = len(json_files)
print(f"     Total JSON files found: {n_json:,}")

block_folders = sorted([d for d in RAW_BLOCKS_DIR.iterdir() if d.is_dir()])
print(f"     Total block-height folders: {len(block_folders):,}")

# ---------- 3b. Load ONE sample file and pretty-print schema ----------
print("\n[3b] Loading sample file: 605999/27_1.json …")
sample_path = RAW_BLOCKS_DIR / "605999" / "27_1.json"
with open(sample_path, "r", encoding="utf-8") as f:
    sample_data = json.load(f)

schema = discover_schema(sample_data)
print(f"\n     Full nested schema ({len(schema)} key paths):")
for s in sorted(schema):
    print(f"       {s}")

# Check if data.list exists
has_data_list = "data" in sample_data and "list" in sample_data.get("data", {})
print(f"\n     Has 'data.list' array? {has_data_list}")

if has_data_list:
    tx_list = sample_data["data"]["list"]
    print(f"     Number of transactions in this file: {len(tx_list)}")
    first_tx = tx_list[0]
    print(f"\n     Top-level keys of first transaction:")
    for k in sorted(first_tx.keys()):
        val = first_tx[k]
        vtype = type(val).__name__
        if isinstance(val, list):
            vtype = f"list[{len(val)}]"
        elif isinstance(val, dict):
            vtype = f"dict[{len(val)} keys]"
        preview = str(val)[:80]
        print(f"       {k:<25s}  ({vtype:>12s})  →  {preview}")

    # Show input/output sub-schema
    if "inputs" in first_tx and len(first_tx["inputs"]) > 0:
        print(f"\n     Input sub-schema (keys of inputs[0]):")
        for k, v in sorted(first_tx["inputs"][0].items()):
            print(f"       {k:<25s}  →  {str(v)[:60]}")

    if "outputs" in first_tx and len(first_tx["outputs"]) > 0:
        print(f"\n     Output sub-schema (keys of outputs[0]):")
        for k, v in sorted(first_tx["outputs"][0].items()):
            print(f"       {k:<25s}  →  {str(v)[:60]}")


# ---------- 3c. Load 5 more samples from different blocks, check schema consistency ----------
print("\n[3c] Loading 5 more samples from different block-height folders …")
import random
random.seed(42)

# Pick 5 evenly-spaced block folders
step = max(1, len(block_folders) // 6)
check_folders = [block_folders[i] for i in range(0, len(block_folders), step)][:5]

all_schemas = {}
for folder in check_folders:
    files_in_folder = sorted(folder.glob("*.json"))
    if not files_in_folder:
        continue
    fpath = files_in_folder[0]
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    sch = discover_schema(data)
    all_schemas[folder.name] = sch
    print(f"     {folder.name}/{fpath.name}  →  {len(sch)} key paths, ", end="")
    if has_data_list:
        n_tx = len(data.get("data", {}).get("list", []))
        print(f"{n_tx} transactions")
    else:
        print()

# Compare schemas
ref_schema = schema
consistent = True
for name, sch in all_schemas.items():
    if sch != ref_schema:
        diff = sch.symmetric_difference(ref_schema)
        print(f"     ⚠ Schema difference in {name}: {diff}")
        consistent = False
if consistent:
    print("     ✓ All sampled files share the SAME schema.")
else:
    print("     ⚠ Schema varies across some files (see diffs above).")


# ---------- 3d. File naming pattern investigation ----------
print("\n[3d] File naming pattern investigation …")
# Pattern: <prefix>_<chunk_number>.json
# Question: does prefix correspond to something? Does chunk # = page?

# Check block 605999: all files are 27_*.json → prefix=27
folder_605999 = RAW_BLOCKS_DIR / "605999"
files_605999 = sorted(folder_605999.glob("*.json"))
prefixes_605999 = set()
for f in files_605999:
    parts = f.stem.split("_")
    prefixes_605999.add(parts[0])
print(f"     Block 605999: {len(files_605999)} files, prefix(es): {prefixes_605999}")

# Check block 600000
folder_600000 = RAW_BLOCKS_DIR / "600000"
files_600000 = sorted(folder_600000.glob("*.json"))
prefixes_600000 = set()
for f in files_600000:
    parts = f.stem.split("_")
    prefixes_600000.add(parts[0])
print(f"     Block 600000: {len(files_600000)} files, prefix(es): {prefixes_600000}")

# Verify block_height inside the JSON matches folder name
print("\n     Verifying block_height consistency …")
for folder_name, block_height_expected in [("605999", 605999), ("600000", 600000)]:
    folder = RAW_BLOCKS_DIR / folder_name
    test_files = sorted(folder.glob("*.json"))[:3]
    for tf in test_files:
        with open(tf, "r", encoding="utf-8") as f:
            d = json.load(f)
        txs = d.get("data", {}).get("list", [])
        if txs:
            bh = txs[0].get("block_height", "N/A")
            match = "✓" if bh == block_height_expected else "✗"
            print(f"       {folder_name}/{tf.name}  block_height={bh}  {match}")

# Check if prefix varies across blocks
print("\n     Prefix survey across a sample of blocks:")
sample_blocks = block_folders[::500]  # every 500th block
for bfolder in sample_blocks[:10]:
    bfiles = sorted(bfolder.glob("*.json"))
    prefixes = set(f.stem.split("_")[0] for f in bfiles)
    chunk_nums = sorted(int(f.stem.split("_")[1]) for f in bfiles if "_" in f.stem)
    print(f"       {bfolder.name}: {len(bfiles)} files, prefix={prefixes}, "
          f"chunks={chunk_nums[0]}..{chunk_nums[-1]}" if chunk_nums else "")

# Same-prefix files = chunks of the same block (paginated API response)
print("\n     Conclusion: Each block-height folder contains multiple JSON files")
print("     (chunks/pages) from a paginated API. The prefix is likely a batch ID,")
print("     and _1, _2, … _N are successive pages of transactions for that block.")


# ---------- 3e. Total transactions and total size on disk ----------
print("\n[3e] Counting total transactions and disk size (scanning all files) …")
total_tx = 0
total_bytes = 0
files_scanned = 0

for jf in json_files:
    total_bytes += jf.stat().st_size
    try:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)
        txs = d.get("data", {}).get("list", [])
        total_tx += len(txs)
    except Exception as e:
        print(f"     ⚠ Error reading {jf}: {e}")
    files_scanned += 1
    if files_scanned % 5000 == 0:
        print(f"       … scanned {files_scanned:,}/{n_json:,} files …")

print(f"     Total transactions across all files: {total_tx:,}")
print(f"     Total size on disk: {total_bytes / (1024**3):.2f} GB  ({total_bytes / (1024**2):.0f} MB)")


# ---------- 3f. Summary of raw block data ----------
print(f"\n[3f] RAW BLOCK JSON — SUMMARY")
print(f"     Total block-height folders : {len(block_folders):,}")
print(f"     Total JSON files           : {n_json:,}")
print(f"     Total transactions         : {total_tx:,}")
print(f"     Total disk size            : {total_bytes / (1024**3):.2f} GB")
print(f"\n     Full field schema per transaction:")
if has_data_list:
    tx_keys = sorted(tx_list[0].keys())
    for k in tx_keys:
        v = tx_list[0][k]
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            sub_keys = sorted(v[0].keys())
            print(f"       {k}  →  list of dicts with keys: {sub_keys}")
        else:
            print(f"       {k}  →  {type(v).__name__}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY :  CROSS-DATASET COMPARISON                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print(f"\n\n{SEP}")
print("  FINAL SUMMARY — CROSS-DATASET COMPARISON")
print(SEP)

# Labeled sample counts
n_elliptic_labeled = class_counts.get("1", 0) + class_counts.get("2", 0)
n_elliptic_total = total
n_babd_labeled = len(df_babd)
print(f"\n  1) LABELED SAMPLE COUNTS")
print(f"     Elliptic : {n_elliptic_labeled:>10,d} labeled  /  {n_elliptic_total:>10,d} total")
print(f"     BABD-13  : {n_babd_labeled:>10,d} labeled  /  {n_babd_labeled:>10,d} total")

# Class imbalance
illicit = class_counts.get("1", 0)
licit   = class_counts.get("2", 0)
if illicit > 0:
    ratio_elliptic = licit / illicit
    print(f"\n  2) CLASS IMBALANCE")
    print(f"     Elliptic  (binary: licit vs illicit):")
    print(f"       Illicit: {illicit:,}  |  Licit: {licit:,}  |  Ratio (licit:illicit) = {ratio_elliptic:.2f}:1")
    print(f"       Plus {class_counts.get('unknown', 0):,} unlabeled ('unknown')")

babd_max = babd_vc.max()
babd_min = babd_vc.min()
print(f"\n     BABD-13  (13-class):")
print(f"       Largest class : {babd_max:,}  |  Smallest class: {babd_min:,}")
print(f"       Imbalance ratio (max/min): {babd_max/babd_min:.1f}:1")

# Conceptual feature overlap
print(f"\n  3) CONCEPTUALLY OVERLAPPING FEATURES")
print("""
     Although named differently, the following concepts likely overlap:

     ┌──────────────────────────────────┬─────────────────────────────────────┐
     │  Elliptic (anonymized)           │  BABD-13 (named)                   │
     ├──────────────────────────────────┼─────────────────────────────────────┤
     │  feat_1 … feat_94               │  PAIa* / PDIa* / PTIa*             │
     │  (94 local tx features:         │  (Payment/Price/Degree/Time         │
     │   in-degree, out-degree,        │   indicators — e.g. PAIa11-1 =     │
     │   tx value, fee, #inputs,       │   total BTC received, PDIa1-1 =    │
     │   #outputs, etc.)               │   number of txns, etc.)            │
     ├──────────────────────────────────┼─────────────────────────────────────┤
     │  feat_95 … feat_165             │  CI* (Correlation indicators),     │
     │  (71 aggregated neighbor        │  S* (Statistical features),        │
     │   features — means/medians      │  account-level aggregates          │
     │   of neighbor tx properties)    │                                     │
     ├──────────────────────────────────┼─────────────────────────────────────┤
     │  time_step (temporal bin)        │  PTIa1, PTIa2 (time features)     │
     ├──────────────────────────────────┼─────────────────────────────────────┤
     │  txId (transaction hash int)     │  account (Bitcoin address string)  │
     └──────────────────────────────────┴─────────────────────────────────────┘

     Key differences:
       • Elliptic = transaction-level, features are anonymized/PCA'd
       • BABD-13  = address-level, features have semantic meaning
       • Elliptic has graph structure (edgelist); BABD-13 does not
       • BABD-13 has 13 fine-grained fraud categories; Elliptic has binary (illicit/licit)
""")

# Raw JSON data complement
print(f"  4) RAW BLOCK JSON DATA — COMPLEMENTARY VALUE")
print(f"""
     The raw block JSON contains FULL on-chain transaction data for
     blocks {block_folders[0].name} – {block_folders[-1].name}, including:

       • Transaction hashes, fees, input/output counts & values
       • Input/output addresses with previous tx references
       • Block timestamps, confirmations, SegWit flags
       • Spent-by-tx references (forward linkage)

     How it complements the labeled datasets:
       a) Can compute Elliptic-style graph features for NEW transactions
          (in-degree, out-degree, neighbor aggregates, fee stats)
       b) Can compute BABD-style address-level behavioral features
          (total received/sent, tx frequency, lifetime, counterparty stats)
       c) Provides ground-truth graph structure to validate/extend
          the Elliptic edgelist for overlapping time ranges
       d) Enables address clustering (co-spend / common-input heuristic)
          using the detailed input arrays

     ⚠  The raw JSON is UNLABELED — it needs to be linked to Elliptic
        txIds or BABD-13 addresses to inherit their labels.
""")

print(f"{SEP}")
print("  EDA COMPLETE — check 'plots/' for saved charts.")
print(SEP)
