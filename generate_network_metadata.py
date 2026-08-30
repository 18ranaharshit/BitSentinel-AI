"""
==============================================================================
Synthetic Bitcoin Network-Metadata Dataset Generator (BitSentinel-AI - Fix 1/3b)
==============================================================================
Generates plausible P2P network telemetry with overlapping cluster sizes:
  1. Illicit Botnet Clusters (6 clusters, sizes drawn independently from U(20, 150))
     -> is_injected_pattern = True, is_legit_bursty_cluster = False
  2. Legitimate High-Volume Exchange/Pool Clusters (3 clusters, sizes drawn from U(20, 150))
     -> is_injected_pattern = False, is_legit_bursty_cluster = True
  3. Background Random Traffic
     -> is_injected_pattern = False, is_legit_bursty_cluster = False

Both cluster types draw from the IDENTICAL U(20, 150) size distribution to ensure
cluster size carries zero synthetic information/leakage.
Exports to processed/network_metadata.csv.
==============================================================================
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = PROCESSED_DIR / "network_metadata.csv"

SEP = "=" * 80

print(f"\n{SEP}")
print("  SYNTHETIC BITCOIN NETWORK-METADATA GENERATOR (OVERLAPPING CLUSTER SIZES)")
print(SEP)

# ------------------------------------------------------------------------------
# 1. Load labeled TXIDs from processed Elliptic splits
# ------------------------------------------------------------------------------
print("\n[1] Loading labeled Elliptic splits ...")

train_file = PROCESSED_DIR / "elliptic_train.csv"
val_file   = PROCESSED_DIR / "elliptic_val.csv"
test_file  = PROCESSED_DIR / "elliptic_test.csv"

cols_to_load = ["txId", "class"]
sample_head = pd.read_csv(train_file, nrows=1)
has_timestep = "time_step" in sample_head.columns
if has_timestep:
    cols_to_load.append("time_step")

df_train = pd.read_csv(train_file, usecols=cols_to_load)
df_val   = pd.read_csv(val_file,   usecols=cols_to_load)
df_test  = pd.read_csv(test_file,  usecols=cols_to_load)

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_all.rename(columns={"txId": "txid", "class": "is_illicit"}, inplace=True)

df_all["txid"] = df_all["txid"].astype(str)
df_all["is_illicit"] = df_all["is_illicit"].astype(int)

total_txs = len(df_all)
total_illicit = int((df_all["is_illicit"] == 1).sum())
total_licit = int((df_all["is_illicit"] == 0).sum())

print(f"    Loaded Total Transactions : {total_txs:,}")
print(f"      - Illicit (class 1)     : {total_illicit:,}")
print(f"      - Licit   (class 0)     : {total_licit:,}")

# ------------------------------------------------------------------------------
# 2. IP & Port Helpers
# ------------------------------------------------------------------------------
VALID_FIRST_OCTETS = [
    i for i in range(1, 224)
    if i not in (10, 100, 127, 169, 172, 192, 0)
]

def generate_random_public_ip():
    o1 = random.choice(VALID_FIRST_OCTETS)
    o2 = random.randint(0, 255)
    o3 = random.randint(0, 255)
    o4 = random.randint(1, 254)
    return f"{o1}.{o2}.{o3}.{o4}"

def generate_random_subnet():
    o1 = random.choice(VALID_FIRST_OCTETS)
    o2 = random.randint(0, 255)
    o3 = random.randint(0, 255)
    return f"{o1}.{o2}.{o3}"

T_START_2019 = 1546300800
T_END_2019   = 1569888000
TOTAL_SPAN_SEC = T_END_2019 - T_START_2019

def generate_timestamp(time_step=None):
    if time_step is not None and not pd.isna(time_step):
        step = max(1, min(49, int(time_step)))
        step_duration = TOTAL_SPAN_SEC / 49.0
        base = T_START_2019 + (step - 1) * step_duration
        offset = random.uniform(0, step_duration)
        return int(base + offset)
    return random.randint(T_START_2019, T_END_2019)

# ------------------------------------------------------------------------------
# 3. Inject Illicit Botnet Clusters (Sizes randomly drawn from U(20, 150))
# ------------------------------------------------------------------------------
print("\n[2] Injecting 6 illicit botnet clusters with randomized sizes (20 to 150 txs) ...")

illicit_indices = df_all[df_all["is_illicit"] == 1].index.tolist()
random.shuffle(illicit_indices)

NUM_ILLICIT_CLUSTERS = 6
injected_info = {}  # index -> {src_ip, timestamp, is_injected_pattern, is_legit_bursty_cluster}
cluster_metadata = []

illicit_offset = 0
for c_id in range(1, NUM_ILLICIT_CLUSTERS + 1):
    c_size = random.randint(20, 150)
    c_indices = illicit_indices[illicit_offset : illicit_offset + c_size]
    illicit_offset += c_size

    subnet = generate_random_subnet()
    shared_src_ip = f"{subnet}.{random.randint(1, 254)}"
    
    window_duration_sec = random.randint(2 * 3600, 6 * 3600)
    c_start_time = random.randint(T_START_2019, T_END_2019 - window_duration_sec)
    c_end_time = c_start_time + window_duration_sec

    for idx in c_indices:
        ip = f"{subnet}.{random.randint(1, 254)}" if random.random() < 0.3 else shared_src_ip
        ts = random.randint(c_start_time, c_end_time)
        injected_info[idx] = {
            "src_ip": ip,
            "timestamp": ts,
            "is_injected_pattern": True,
            "is_legit_bursty_cluster": False
        }

    cluster_metadata.append({
        "Cluster Type": "🚨 Illicit Botnet",
        "Cluster ID": f"Botnet-{c_id}",
        "Subnet (/24)": f"{subnet}.0/24",
        "Tx Count": len(c_indices),
        "Window Start": pd.to_datetime(c_start_time, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "Span (Hours)": round(window_duration_sec / 3600.0, 2)
    })

# ------------------------------------------------------------------------------
# 4. Inject 3 Legitimate Exchange Clusters (IDENTICAL U(20, 150) SIZE DISTRIBUTION)
# ------------------------------------------------------------------------------
print("\n[3] Injecting 3 legitimate exchange clusters with randomized sizes (20 to 150 txs) ...")

licit_indices = df_all[df_all["is_illicit"] == 0].index.tolist()
random.shuffle(licit_indices)

num_legit_clusters = 3
licit_offset = 0

for c_id in range(1, num_legit_clusters + 1):
    c_size = random.randint(20, 150)
    c_indices = licit_indices[licit_offset : licit_offset + c_size]
    licit_offset += c_size

    subnet = generate_random_subnet()
    shared_src_ip = f"{subnet}.{random.randint(1, 254)}"
    
    window_duration_sec = random.randint(2 * 3600, 6 * 3600)
    c_start_time = random.randint(T_START_2019, T_END_2019 - window_duration_sec)
    c_end_time = c_start_time + window_duration_sec

    for idx in c_indices:
        ip = f"{subnet}.{random.randint(1, 254)}" if random.random() < 0.3 else shared_src_ip
        ts = random.randint(c_start_time, c_end_time)
        injected_info[idx] = {
            "src_ip": ip,
            "timestamp": ts,
            "is_injected_pattern": False,
            "is_legit_bursty_cluster": True
        }

    cluster_metadata.append({
        "Cluster Type": "🏦 Legit Exchange/Pool",
        "Cluster ID": f"Exchange-{c_id}",
        "Subnet (/24)": f"{subnet}.0/24",
        "Tx Count": len(c_indices),
        "Window Start": pd.to_datetime(c_start_time, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "Span (Hours)": round(window_duration_sec / 3600.0, 2)
    })

df_cluster_meta = pd.DataFrame(cluster_metadata)
print("\n" + "-" * 80)
print("  INJECTED CLUSTERS SUMMARY (VISUAL PROOF OF OVERLAPPING SIZE DISTRIBUTIONS)")
print("-" * 80)
print(df_cluster_meta.to_string(index=False))
print("-" * 80)

# ------------------------------------------------------------------------------
# 5. Synthesize Background Traffic
# ------------------------------------------------------------------------------
print("\n[4] Synthesizing remaining background traffic ...")

timestamps = []
src_ips = []
dst_ips = []
src_ports = []
dst_ports = []
is_injected = []
is_legit_burst = []

for idx, row in df_all.iterrows():
    if idx in injected_info:
        info = injected_info[idx]
        ts = info["timestamp"]
        s_ip = info["src_ip"]
        inj = info["is_injected_pattern"]
        legit_b = info["is_legit_bursty_cluster"]
    else:
        t_step = row["time_step"] if has_timestep else None
        ts = generate_timestamp(t_step)
        s_ip = generate_random_public_ip()
        inj = False
        legit_b = False

    d_ip = generate_random_public_ip()
    s_port = random.randint(1024, 65535)
    d_port = 8333 if random.random() < 0.80 else random.randint(1024, 65535)

    timestamps.append(ts)
    src_ips.append(s_ip)
    dst_ips.append(d_ip)
    src_ports.append(s_port)
    dst_ports.append(d_port)
    is_injected.append(inj)
    is_legit_burst.append(legit_b)

df_all["timestamp"] = timestamps
df_all["src_ip"] = src_ips
df_all["dst_ip"] = dst_ips
df_all["src_port"] = src_ports
df_all["dst_port"] = dst_ports
df_all["is_injected_pattern"] = is_injected
df_all["is_legit_bursty_cluster"] = is_legit_burst

final_cols = [
    "txid",
    "is_illicit",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "is_injected_pattern",
    "is_legit_bursty_cluster"
]
df_final = df_all[final_cols]

# ------------------------------------------------------------------------------
# 6. Save Dataset
# ------------------------------------------------------------------------------
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\n[5] Saved Synthetic Network Metadata -> {OUTPUT_CSV}")

total_rows = len(df_final)
injected_count = int(df_final["is_injected_pattern"].sum())
legit_burst_count = int(df_final["is_legit_bursty_cluster"].sum())
bg_count = total_rows - (injected_count + legit_burst_count)

print(f"\n  Total Dataset Rows                      : {total_rows:,}")
print(f"  - Injected Illicit Botnet Rows (True)   : {injected_count:,} ({100 * injected_count / total_rows:.2f}%)")
print(f"  - Legitimate Exchange Bursty Rows (HN)  : {legit_burst_count:,} ({100 * legit_burst_count / total_rows:.2f}%)")
print(f"  - Standard Independent Background Rows  : {bg_count:,} ({100 * bg_count / total_rows:.2f}%)")

print(f"\n{SEP}")
print("  GENERATION COMPLETE: processed/network_metadata.csv is ready.")
print(f"{SEP}\n")
