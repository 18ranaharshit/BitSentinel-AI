"""
==============================================================================
Feature Extraction & Heuristic Risk Utilities for Bitcoin Transactions
==============================================================================
Provides shared feature extraction and transparent heuristic risk scoring
for raw Bitcoin block / live stream transactions.

NOTE: 'heuristic_score' is a rule-based proxy, NOT the trained ML model's
output, because raw block transactions lack ground-truth labels needed to
validate a proxy model against Elliptic's PCA-anonymized feature space.
Address-level predictions use the trained reduced-feature BABD-13 model.
==============================================================================
"""

import numpy as np


def build_elliptic_proxy_features(tx):
    """
    Extracts raw transaction-level features available in raw Bitcoin blocks:
      - fee_btc, inputs_value_btc, outputs_value_btc
      - inputs_count, outputs_count, size_bytes
      - fee_ratio, val_per_out, val_per_inp, in_out_ratio
      - is_coinbase, is_large_tx
    Returns a 1D numpy array of engineered features.
    """
    fee_btc = float(tx.get("fee", 0)) / 1e8
    out_val_btc = float(tx.get("outputs_value", 0)) / 1e8
    inp_val_btc = float(tx.get("inputs_value", 0)) / 1e8
    inp_cnt = max(int(tx.get("inputs_count", 0)), 1)
    out_cnt = max(int(tx.get("outputs_count", 0)), 1)
    size_bytes = float(tx.get("size", 0))
    
    fee_ratio = fee_btc / (out_val_btc + 1e-8)
    val_per_out = out_val_btc / out_cnt
    val_per_inp = inp_val_btc / inp_cnt
    in_out_ratio = inp_cnt / out_cnt
    is_coinbase = 1.0 if tx.get("is_coinbase", False) or inp_cnt == 0 else 0.0
    is_large_tx = 1.0 if (out_val_btc > 10.0 or inp_val_btc > 10.0) else 0.0

    features = np.array([
        fee_btc,
        out_val_btc,
        inp_val_btc,
        float(inp_cnt),
        float(out_cnt),
        size_bytes,
        fee_ratio,
        val_per_out,
        val_per_inp,
        in_out_ratio,
        is_coinbase,
        is_large_tx
    ], dtype=np.float32)

    return features


def calculate_tx_heuristic_score(tx):
    """
    Computes a transparent rule-based heuristic risk score [0.0 - 1.0] for a raw transaction.
    
    NOTE: heuristic_score is a rule-based proxy, not the trained ML model's output,
    because raw block transactions lack ground-truth labels needed to validate
    a proxy model against Elliptic's feature space.
    """
    fee_btc = float(tx.get("fee", 0)) / 1e8
    out_val_btc = float(tx.get("outputs_value", 0)) / 1e8
    inp_val_btc = float(tx.get("inputs_value", 0)) / 1e8
    inp_cnt = float(tx.get("inputs_count", 0))
    out_cnt = float(tx.get("outputs_count", 0))

    fee_ratio = fee_btc / (out_val_btc + 1e-8)
    in_out_ratio = inp_cnt / (out_cnt + 1e-8)

    score = (
        0.35 * np.clip(fee_ratio * 10, 0, 1) +
        0.35 * np.clip(in_out_ratio / 5, 0, 1) +
        0.30 * float(out_val_btc > 10.0 or inp_val_btc > 10.0)
    )
    return float(np.round(score, 4))
