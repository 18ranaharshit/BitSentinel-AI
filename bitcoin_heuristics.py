"""
==============================================================================
Bitcoin Address Heuristics & Wallet Entity Clustering Engine (BitSentinel-AI)
==============================================================================
Implements:
  1. Common-Input (Co-Spend) Heuristic: Addresses co-spent as inputs in the
     same transaction belong to the same entity (wallet).
  2. Disjoint-Set (Union-Find) clustering data structure with path compression.
  3. Cluster persistence to models/wallet_clusters.csv for REST API and Frontend.
==============================================================================
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


class UnionFind:
    """Disjoint-Set data structure with path compression and union by rank."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 != root2:
            # Union by rank
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def cluster_raw_blocks(raw_blocks_dir, max_blocks=30):
    """Parses raw block JSON files and clusters addresses using common-input heuristic."""
    uf = UnionFind()
    raw_dir = Path(raw_blocks_dir)
    block_folders = sorted([d for d in raw_dir.iterdir() if d.is_dir()])[:max_blocks]

    tx_count = 0
    co_spend_events = 0

    print(f"Parsing {len(block_folders)} block folders for co-spend address clustering ...")

    for bfolder in block_folders:
        for jfile in bfolder.glob("*.json"):
            try:
                with open(jfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tx_list = data.get("data", {}).get("list", [])

                for tx in tx_list:
                    tx_count += 1
                    input_addrs = set()
                    for inp in tx.get("inputs", []):
                        for addr in inp.get("prev_addresses", []):
                            if addr and len(addr) > 5:
                                input_addrs.add(addr)

                    # Apply Co-Spend Heuristic: Union all input addresses in this transaction
                    input_addrs_list = list(input_addrs)
                    if len(input_addrs_list) > 1:
                        first_addr = input_addrs_list[0]
                        for other_addr in input_addrs_list[1:]:
                            uf.union(first_addr, other_addr)
                            co_spend_events += 1

            except Exception:
                continue

    # Build cluster membership map
    cluster_groups = defaultdict(list)
    for addr in list(uf.parent.keys()):
        root = uf.find(addr)
        cluster_groups[root].append(addr)

    print(f"\nClustering Summary:")
    print(f"  Total Transactions Analyzed: {tx_count:,}")
    print(f"  Co-Spend Linkage Events    : {co_spend_events:,}")
    print(f"  Total Unique Addresses     : {len(uf.parent):,}")
    print(f"  Total Entity Clusters      : {len(cluster_groups):,}")

    multi_addr_clusters = {k: v for k, v in cluster_groups.items() if len(v) > 1}
    print(f"  Multi-Address Entity Wallet Clusters (>1 address): {len(multi_addr_clusters):,}")

    return uf, cluster_groups


def save_clusters_to_csv(cluster_groups, output_path="models/wallet_clusters.csv"):
    """
    Filters to multi-address clusters (len > 1), formats into a DataFrame,
    sorts by cluster_size descending, and exports to CSV.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Filter and sort clusters by size descending
    sorted_clusters = sorted(
        [(k, v) for k, v in cluster_groups.items() if len(v) > 1],
        key=lambda item: len(item[1]),
        reverse=True
    )

    rows = []
    for cluster_idx, (root_addr, addrs) in enumerate(sorted_clusters, start=1):
        c_id = f"CL-{cluster_idx:05d}"
        c_size = len(addrs)
        for addr in addrs:
            rows.append({
                "cluster_id": c_id,
                "address": addr,
                "cluster_size": c_size
            })

    df_clusters = pd.DataFrame(rows)
    df_clusters.to_csv(output_file, index=False)

    print(f"\n[+] Saved {len(df_clusters):,} clustered address records to -> {output_file}")
    print(f"    Total Distinct Multi-Address Wallet Entities: {len(sorted_clusters):,}")

    # Print Top 10 largest clusters as a sanity check
    print("\n" + "=" * 70)
    print("  TOP 10 LARGEST DETECTED WALLET CLUSTERS (CO-SPEND ENTITIES)")
    print("=" * 70)
    top_10 = sorted_clusters[:10]
    summary_rows = []
    for idx, (root, addrs) in enumerate(top_10, start=1):
        summary_rows.append({
            "Cluster ID": f"CL-{idx:05d}",
            "Address Count": len(addrs),
            "Lead Address": root[:18] + "..."
        })
    df_top10 = pd.DataFrame(summary_rows)
    print(df_top10.to_string(index=False))
    print("=" * 70 + "\n")

    return df_clusters


if __name__ == "__main__":
    RAW_DIR = Path("raw data/600000-605999")
    if RAW_DIR.exists():
        uf, clusters = cluster_raw_blocks(RAW_DIR, max_blocks=30)
        save_clusters_to_csv(clusters, output_path="models/wallet_clusters.csv")
    else:
        print(f"[!] Raw data directory '{RAW_DIR}' not found.")
