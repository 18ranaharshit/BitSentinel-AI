"""
==============================================================================
Self-Supervised Pretraining on Elliptic Transaction Graph via GraphSAGE
==============================================================================
1. Build PyG Data graph using ALL Elliptic transactions (features + edgelist).
2. Save txid_to_idx mapping and node split masks (train/val/test/unlabeled).
3. 2-layer GraphSAGE encoder (torch_geometric.nn.SAGEConv) -> 128-dim embeddings.
4. Pretraining via link prediction (dot product + negative sampling + BCE loss).
5. Train for 100 epochs with Adam optimizer.
6. Save node embeddings (elliptic_node_embeddings.pt) to 'models/'.
7. Perform positive vs negative link scoring sanity check.
==============================================================================
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_DIR = Path("raw data")
PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 80

print(f"\n{SEP}")
print("  SELF-SUPERVISED GRAPH PRETRAINING (GRAPHSAGE LINK PREDICTION)")
print(SEP)
print(f"  Using PyTorch Device: {device}")

# 1. Load full features and edgelist
print("\n[1] Loading Elliptic features and building node mapping ...")
df_feat = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None)

tx_ids = df_feat[0].values
x_features = df_feat.iloc[:, 2:].values.astype(np.float32)

N = len(tx_ids)
txid_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
print(f"    Total nodes (transactions): {N:,}")
print(f"    Features per node          : {x_features.shape[1]}")

print("\n    Loading edgelist and constructing edge_index ...")
df_edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")

src = df_edges["txId1"].map(txid_to_idx)
dst = df_edges["txId2"].map(txid_to_idx)

# Filter valid edges where both endpoints exist in features
valid_mask = src.notnull() & dst.notnull()
src_valid = src[valid_mask].astype(np.int64).values
dst_valid = dst[valid_mask].astype(np.int64).values

edge_index = torch.tensor(np.array([src_valid, dst_valid]), dtype=torch.long)
x = torch.tensor(x_features, dtype=torch.float)

print(f"    Total directed edges: {edge_index.size(1):,}")

# Save txid_to_idx mapping
txid_map_path = MODELS_DIR / "txid_to_idx.pkl"
with open(txid_map_path, "wb") as f:
    pickle.dump(txid_to_idx, f)
print(f"    Saved txid_to_idx mapping -> {txid_map_path}")

# 2. Build and save node split boolean masks
print("\n[2] Creating node split masks (train / val / test / unlabeled) ...")
df_train = pd.read_csv(PROCESSED_DIR / "elliptic_train.csv")
df_val   = pd.read_csv(PROCESSED_DIR / "elliptic_val.csv")
df_test  = pd.read_csv(PROCESSED_DIR / "elliptic_test.csv")
df_unlabeled = pd.read_csv(PROCESSED_DIR / "elliptic_unlabeled.csv")

train_mask = torch.zeros(N, dtype=torch.bool)
val_mask   = torch.zeros(N, dtype=torch.bool)
test_mask  = torch.zeros(N, dtype=torch.bool)
unlabeled_mask = torch.zeros(N, dtype=torch.bool)

train_mask[[txid_to_idx[t] for t in df_train["txId"] if t in txid_to_idx]] = True
val_mask[[txid_to_idx[t] for t in df_val["txId"] if t in txid_to_idx]] = True
test_mask[[txid_to_idx[t] for t in df_test["txId"] if t in txid_to_idx]] = True
unlabeled_mask[[txid_to_idx[t] for t in df_unlabeled["txId"] if t in txid_to_idx]] = True

split_masks = {
    "train_mask": train_mask,
    "val_mask": val_mask,
    "test_mask": test_mask,
    "unlabeled_mask": unlabeled_mask
}
mask_path = MODELS_DIR / "node_split_mask.pt"
torch.save(split_masks, mask_path)
print(f"    Saved split masks -> {mask_path}")
print(f"      Train nodes    : {train_mask.sum().item():>7,d}")
print(f"      Val nodes      : {val_mask.sum().item():>7,d}")
print(f"      Test nodes     : {test_mask.sum().item():>7,d}")
print(f"      Unlabeled nodes: {unlabeled_mask.sum().item():>7,d}")


# 3. 2-layer GraphSAGE Encoder definition
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 4 & 5. Self-supervised Pretraining Loop (Link Prediction)
print("\n[3, 4 & 5] Initializing GraphSAGE Encoder & starting link prediction pretraining ...")
model = GraphSAGEEncoder(in_channels=165, hidden_channels=128, out_channels=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

x = x.to(device)
edge_index = edge_index.to(device)
num_edges = edge_index.size(1)

model.train()
print("\n    Training Progress (100 Epochs):")

for epoch in range(1, 101):
    optimizer.zero_grad()
    
    # Compute embeddings
    z = model(x, edge_index)
    
    # Sample negative edges
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=N,
        num_neg_samples=num_edges,
        method="sparse"
    )
    
    # Score positive and negative pairs via dot product
    pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    
    logits = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    
    if epoch == 1 or epoch % 10 == 0:
        with torch.no_grad():
            pos_prob = torch.sigmoid(pos_score).mean().item()
            neg_prob = torch.sigmoid(neg_score).mean().item()
        print(f"      Epoch {epoch:>3d}/100 | Loss: {loss.item():.4f} | Avg Pos Prob: {pos_prob:.4f} | Avg Neg Prob: {neg_prob:.4f}")

# 6. Save final embeddings
print("\n[6] Computing final embeddings for all nodes ...")
model.eval()
with torch.no_grad():
    final_embeddings = model(x, edge_index)

embeddings_path = MODELS_DIR / "elliptic_node_embeddings.pt"
torch.save(final_embeddings.cpu(), embeddings_path)
print(f"    Saved final node embeddings shape {list(final_embeddings.shape)} -> {embeddings_path}")

# 7. Sanity Check: 5 random positive vs 5 random negative pairs
print(f"\n{SEP}")
print("  SANITY CHECK: POSITIVE VS NEGATIVE PAIR LINK PREDICTION SCORES")
print(SEP)

torch.manual_seed(42)
perm = torch.randperm(num_edges)[:5]
pos_sample_src = edge_index[0, perm]
pos_sample_dst = edge_index[1, perm]

neg_sample = negative_sampling(edge_index, num_nodes=N, num_neg_samples=5)
neg_sample_src = neg_sample[0]
neg_sample_dst = neg_sample[1]

with torch.no_grad():
    pos_dots = (final_embeddings[pos_sample_src] * final_embeddings[pos_sample_dst]).sum(dim=-1)
    pos_probs = torch.sigmoid(pos_dots)
    
    neg_dots = (final_embeddings[neg_sample_src] * final_embeddings[neg_sample_dst]).sum(dim=-1)
    neg_probs = torch.sigmoid(neg_dots)

print("\n  5 Random POSITIVE (Connected) Node Pairs:")
for i in range(5):
    u = pos_sample_src[i].item()
    v = pos_sample_dst[i].item()
    print(f"    Pair {i+1}: Node {u:>6d} <-> Node {v:>6d} | Dot Product: {pos_dots[i].item():>7.4f} | Prob: {pos_probs[i].item():.4f}")

print("\n  5 Random NEGATIVE (Unconnected) Node Pairs:")
for i in range(5):
    u = neg_sample_src[i].item()
    v = neg_sample_dst[i].item()
    print(f"    Pair {i+1}: Node {u:>6d} <-> Node {v:>6d} | Dot Product: {neg_dots[i].item():>7.4f} | Prob: {neg_probs[i].item():.4f}")

print(f"\n  Average Connected Pair Score   : Dot Product = {pos_dots.mean().item():.4f} (Prob = {pos_probs.mean().item():.4f})")
print(f"  Average Unconnected Pair Score : Dot Product = {neg_dots.mean().item():.4f} (Prob = {neg_probs.mean().item():.4f})")
print(f"{SEP}\n")
