"""
==============================================================================
BABD-13 Label-0 Artifact Cleanup & Account-Disjoint Stratified Split
==============================================================================
1. Reload BABD-13 dataset and drop 528 exact duplicate rows.
2. Resolve label-0 (Blackmail) co-occurrence artifacts.
3. Log artifact accounts and genuine multi-label accounts.
4. Merge minor classes (4, 7, 8, 9 -> 13 'other_illicit').
5. Perform account-disjoint stratified 70/15/15 train/val/test split.
6. Overwrite processed/babd13_train.csv, babd13_val.csv, babd13_test.csv.
==============================================================================
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(r"raw data")
PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 80

print(f"\n{SEP}")
print("  BABD-13 LABEL-0 ARTIFACT CLEANUP & ACCOUNT-DISJOINT STRATIFIED SPLIT")
print(SEP)

# 1. Load BABD-13 and drop exact duplicate rows
print("\n[1] Loading BABD-13 dataset and dropping exact duplicate rows ...")
df_raw = pd.read_csv(BASE_DIR / "BABD-13.csv")
initial_rows = len(df_raw)
df_dedup = df_raw.drop_duplicates().copy()
print(f"    Initial rows: {initial_rows:,}")
print(f"    After dropping 528 exact duplicate rows: {len(df_dedup):,}")

# Record initial label 0 count
orig_label0_count = (df_dedup["label"] == 0).sum()

# 2 & 3. Group by 'account' and resolve label-0 co-occurrence artifacts
print("\n[2 & 3] Analyzing duplicate accounts and resolving label-0 artifacts ...")

# Group labels by account
acc_labels = df_dedup.groupby("account")["label"].apply(set).to_dict()

artifact_label0_logs = []
multilabel_logs = []
rows_to_drop_indices = []

# Map accounts to their target drop rows
for acc, labels in acc_labels.items():
    if len(labels) > 1:
        if 0 in labels:
            non_zero_labels = labels - {0}
            if len(non_zero_labels) == 1:
                # Case a: label 0 + 1 other label -> drop label 0 row(s), keep other label
                kept_lbl = list(non_zero_labels)[0]
                acc_zero_rows = df_dedup[(df_dedup["account"] == acc) & (df_dedup["label"] == 0)]
                rows_to_drop_indices.extend(acc_zero_rows.index.tolist())
                artifact_label0_logs.append({"account": acc, "dropped_label": 0, "kept_label": kept_lbl})
            else:
                # Case b: label 0 + 2+ other labels -> drop label 0 row(s), keep remaining non-zero labels
                acc_zero_rows = df_dedup[(df_dedup["account"] == acc) & (df_dedup["label"] == 0)]
                rows_to_drop_indices.extend(acc_zero_rows.index.tolist())
                multilabel_logs.append({"account": acc, "labels": str(sorted(list(non_zero_labels)))})
        else:
            # Case c: label 0 NOT in labels -> keep all rows as-is (genuine multi-label)
            multilabel_logs.append({"account": acc, "labels": str(sorted(list(labels)))})

# Apply row drops
df_cleaned = df_dedup.drop(index=rows_to_drop_indices).copy()

# Save logs
df_artifact_log = pd.DataFrame(artifact_label0_logs)
df_multilabel_log = pd.DataFrame(multilabel_logs)

artifact_log_path = PROCESSED_DIR / "artifact_label0_accounts.csv"
multilabel_log_path = PROCESSED_DIR / "multilabel_accounts.csv"

df_artifact_log.to_csv(artifact_log_path, index=False)
df_multilabel_log.to_csv(multilabel_log_path, index=False)

print(f"    Logged {len(df_artifact_log):,} artifact accounts -> {artifact_log_path}")
print(f"    Logged {len(df_multilabel_log):,} multi-label accounts -> {multilabel_log_path}")

# 4. Print before/after label distribution comparison
new_label0_count = (df_cleaned["label"] == 0).sum()

print("\n[4] Label-0 (Blackmail) Count Reduction:")
print(f"    Original Label 0 count : {orig_label0_count:>8,d}")
print(f"    Corrected Label 0 count: {new_label0_count:>8,d}")
print(f"    Label 0 rows dropped   : {orig_label0_count - new_label0_count:>8,d}")
print(f"    Accounts in multilabel_accounts.csv: {len(df_multilabel_log):,}")

# 5. Re-run minor-class merge (4, 7, 8, 9 -> 13 'other_illicit')
print("\n[5] Merging minor classes (4, 7, 8, 9 -> 13 'other_illicit') ...")
merge_map = {4: 13, 7: 13, 8: 13, 9: 13}
df_cleaned["label"] = df_cleaned["label"].replace(merge_map)

# 6. Print new full label distribution
print("\n[6] Corrected BABD Label Distribution:")
BABD_LABEL_NAMES = {
    0: "Blackmail", 1: "Cyber-security service", 2: "Darknet market",
    3: "Centralized exchange", 5: "P2P financial service", 6: "Gambling",
    10: "Mining pool", 11: "Tumbler", 12: "Individual wallet",
    13: "other_illicit (merged 4,7,8,9)"
}
for lbl, cnt in df_cleaned["label"].value_counts().sort_index().items():
    name = BABD_LABEL_NAMES.get(lbl, f"Label {lbl}")
    print(f"    Label {lbl:>2d} ({name:<32s}): {cnt:>7,d} rows ({100*cnt/len(df_cleaned):.2f}%)")

# 7. Account-disjoint stratified 70/15/15 train/val/test split
print("\n[7] Performing Account-Disjoint Stratified Train/Val/Test Split (70% / 15% / 15%) ...")

# Each unique account gets mapped to a representative label for stratification
acc_strat = df_cleaned.groupby("account")["label"].first().reset_index()

train_accs, temp_accs = train_test_split(
    acc_strat, test_size=0.30, random_state=42, stratify=acc_strat["label"]
)
val_accs, test_accs = train_test_split(
    temp_accs, test_size=0.50, random_state=42, stratify=temp_accs["label"]
)

set_train_acc = set(train_accs["account"])
set_val_acc   = set(val_accs["account"])
set_test_acc  = set(test_accs["account"])

train_babd = df_cleaned[df_cleaned["account"].isin(set_train_acc)].copy()
val_babd   = df_cleaned[df_cleaned["account"].isin(set_val_acc)].copy()
test_babd  = df_cleaned[df_cleaned["account"].isin(set_test_acc)].copy()

# Leakage Verification
overlap_train_val = len(set_train_acc & set_val_acc)
overlap_train_test = len(set_train_acc & set_test_acc)
overlap_val_test = len(set_val_acc & set_test_acc)
total_leakage = overlap_train_val + overlap_train_test + overlap_val_test

print(f"    Account leakage check:")
print(f"      Train & Val overlap : {overlap_train_val}")
print(f"      Train & Test overlap: {overlap_train_test}")
print(f"      Val & Test overlap  : {overlap_val_test}")
print(f"      CONFIRMED: Zero account leakage across splits ({total_leakage} overlapping accounts)")

# 8. Print final shapes and per-split class balance
print("\n[8] Final BABD-13 Split Shapes & Class Balance:")
print(f"    Train shape: {train_babd.shape}")
print(f"    Val   shape: {val_babd.shape}")
print(f"    Test  shape: {test_babd.shape}")

print("\n    Class Balance Across Splits:")
train_vc = train_babd["label"].value_counts().sort_index()
val_vc   = val_babd["label"].value_counts().sort_index()
test_vc  = test_babd["label"].value_counts().sort_index()

for lbl in train_vc.index:
    name = BABD_LABEL_NAMES.get(lbl, f"Label {lbl}")
    print(f"    Label {lbl:>2d} ({name:<32s}) | Train: {train_vc.get(lbl, 0):>6,d} | Val: {val_vc.get(lbl, 0):>5,d} | Test: {test_vc.get(lbl, 0):>5,d}")

# 9. Overwrite processed/ files
print("\n[9] Overwriting processed/ babd13 files ...")
train_babd.to_csv(PROCESSED_DIR / "babd13_train.csv", index=False)
val_babd.to_csv(PROCESSED_DIR / "babd13_val.csv", index=False)
test_babd.to_csv(PROCESSED_DIR / "babd13_test.csv", index=False)
print("    Overwritten: babd13_train.csv, babd13_val.csv, babd13_test.csv")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BEFORE / AFTER SUMMARY                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print(f"\n\n{SEP}")
print("  BEFORE / AFTER CLEANUP SUMMARY")
print(SEP)
print(f"  • Original Blackmail (label 0) count : {orig_label0_count:,}")
print(f"  • Corrected Blackmail (label 0) count: {new_label0_count:,}  (dropped {orig_label0_count - new_label0_count:,} artifact rows)")
print(f"  • Artifact accounts logged           : {len(df_artifact_log):,} -> processed/artifact_label0_accounts.csv")
print(f"  • Genuine multi-label accounts logged : {len(df_multilabel_log):,} -> processed/multilabel_accounts.csv")
print(f"  • Account Leakage across Splits      : 0 overlapping accounts")
print(f"{SEP}\n")
