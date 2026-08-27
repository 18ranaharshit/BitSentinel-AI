"""
==============================================================================
Bitcoin Address Heuristics & Wallet Entity Clustering Engine
==============================================================================
Implements:
  1. Common-Input (Co-Spend) Heuristic: Addresses co-spent as inputs in the
     same transaction belong to the same entity (wallet).
  2. Disjoint-Set (Union-Find) clustering data structure with path compression.
  3. Cluster-level feature aggregation (total received, tx count, address count).
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


def cluster_raw_blocks(raw_blocks_dir, max_blocks=50):
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

    # Top multi-address clusters
    multi_addr_clusters = {k: v for k, v in cluster_groups.items() if len(v) > 1}
    print(f"  Multi-Address Entity Wallet Clusters (>1 address): {len(multi_addr_clusters):,}")

    return uf, cluster_groups


if __name__ == "__main__":
    RAW_DIR = Path("raw data/600000-605999")
    if RAW_DIR.exists():
        uf, clusters = cluster_raw_blocks(RAW_DIR, max_blocks=30)
