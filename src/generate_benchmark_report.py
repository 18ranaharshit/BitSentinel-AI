"""
==============================================================================
Benchmark Report & Visualization Generator
==============================================================================
Generates evaluation charts in plots/:
  1. elliptic_roc_pr_curves.png: ROC and Precision-Recall curves.
  2. babd13_confusion_matrix.png: Multi-class Confusion Matrix heatmap.
  3. benchmark_comparison.png: Overall model performance comparison bar chart.
==============================================================================
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 80

print(f"\n{SEP}")
print("  BENCHMARK REPORT & VISUALIZATION GENERATOR")
print(SEP)

# 1. Load Elliptic probability predictions and plot ROC / PR Curves
elliptic_probs_path = MODELS_DIR / "elliptic_test_probs.pkl"
if elliptic_probs_path.exists():
    print("\n[1] Generating Elliptic ROC & Precision-Recall Curves ...")
    with open(elliptic_probs_path, "rb") as f:
        ell_data = pickle.load(f)
    
    y_test = ell_data["y_test"]
    probs_dict = ell_data["probs_dict"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curves
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax1.plot(fpr, tpr, label=name, linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    ax1.set_title("Elliptic Test Set — ROC Curves", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # PR Curves
    for name, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ax2.plot(recall, precision, label=name, linewidth=2)
    ax2.set_title("Elliptic Test Set — Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recall (Illicit Class)")
    ax2.set_ylabel("Precision (Illicit Class)")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    sns.despine()
    plt.tight_layout()
    roc_pr_file = PLOTS_DIR / "elliptic_roc_pr_curves.png"
    fig.savefig(roc_pr_file, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {roc_pr_file}")

# 2. Load BABD-13 predictions and plot Confusion Matrix
babd_preds_path = MODELS_DIR / "babd13_test_preds.pkl"
if babd_preds_path.exists():
    print("\n[2] Generating BABD-13 Multi-Class Confusion Matrix ...")
    with open(babd_preds_path, "rb") as f:
        babd_data = pickle.load(f)
    
    y_test_babd = babd_data["y_test"]
    test_preds_babd = babd_data["test_preds"]
    best_name = babd_data["best_name"]
    unique_labels = babd_data["unique_labels"]
    
    best_preds = test_preds_babd[best_name]
    cm = confusion_matrix(y_test_babd, best_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    BABD_SHORT_NAMES = [
        "Blackmail", "Cybersec", "Darknet", "Exchange",
        "P2P Serv", "Gambling", "Mining", "Tumbler", "Wallet", "Other Illicit"
    ]
    
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=BABD_SHORT_NAMES, yticklabels=BABD_SHORT_NAMES, ax=ax)
    ax.set_title(f"BABD-13 Normalized Confusion Matrix ({best_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Category", fontsize=11)
    ax.set_ylabel("True Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    cm_file = PLOTS_DIR / "babd13_confusion_matrix.png"
    fig.savefig(cm_file, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {cm_file}")

# 3. Overall Benchmark Comparison Bar Chart
print("\n[3] Generating Overall Benchmark Performance Comparison Bar Chart ...")
elliptic_csv = MODELS_DIR / "elliptic_benchmark_results.csv"
babd_csv = MODELS_DIR / "babd13_benchmark_results.csv"

if elliptic_csv.exists():
    df_ell = pd.read_csv(elliptic_csv)
    df_ell_test = df_ell[df_ell["Split"] == "Test"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = df_ell_test["Model"]
    f1_scores = df_ell_test["F1-Score"]
    pr_aucs = df_ell_test["PR-AUC"]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, f1_scores, width, label="F1-Score (Illicit)", color="#e74c3c", edgecolor="black")
    ax.bar(x + width/2, pr_aucs, width, label="PR-AUC", color="#3498db", edgecolor="black")
    
    ax.set_title("Elliptic Test Performance: Baselines vs GraphSAGE Hybrid vs Fine-Tuned", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    sns.despine()
    plt.tight_layout()
    comp_file = PLOTS_DIR / "benchmark_comparison.png"
    fig.savefig(comp_file, dpi=150)
    plt.close(fig)
    print(f"    Saved -> {comp_file}")

print(f"\n{SEP}")
print("  ALL BENCHMARK PLOTS GENERATED IN 'plots/'")
print(SEP)
