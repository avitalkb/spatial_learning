# ============================================================================
# FILE 5: visualize.py
# ============================================================================

"""Visualization functions for mobility attention project."""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot_attention_correct_vs_incorrect(correct_results, incorrect_results, save_path=None):
    """Compare attention patterns: correct vs incorrect predictions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Attention on target (boxplot)
    ax1 = axes[0]
    correct_target_attn = [r["attention"].get(r["target"], 0) for r in correct_results 
                          if r["target"] in r["attention"]]
    incorrect_target_attn = [r["attention"].get(r["target"], 0) for r in incorrect_results 
                            if r["target"] in r["attention"]]
    
    bp = ax1.boxplot([correct_target_attn, incorrect_target_attn], 
                     positions=[1, 2], widths=0.6, patch_artist=True)
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([f"Correct\\n(n={len(correct_target_attn)})", 
                         f"Incorrect\\n(n={len(incorrect_target_attn)})"])
    ax1.set_ylabel("Attention on Target Category")
    ax1.set_title("When Target in History:\\nCorrect Predictions Attend More to Target")
    
    # Add means
    ax1.axhline(y=np.mean(correct_target_attn), color="#27ae60", linestyle="--", alpha=0.7)
    ax1.axhline(y=np.mean(incorrect_target_attn), color="#c0392b", linestyle="--", alpha=0.7)
    
    # Right: High attention → accuracy
    ax2 = axes[1]
    all_results = correct_results + incorrect_results
    max_attns = [(np.max(r["attn_values"]), r["correct"]) for r in all_results]
    
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    bin_labels = ["0-0.3\\n(diffuse)", "0.3-0.5", "0.5-0.7", "0.7-1.0\\n(focused)"]
    
    accs = []
    for low, high in bins:
        in_bin = [(a, c) for a, c in max_attns if low <= a < high]
        accs.append(sum(c for a, c in in_bin) / len(in_bin) * 100 if in_bin else 0)
    
    bars = ax2.bar(range(len(bins)), accs, color=["#f39c12", "#e67e22", "#d35400", "#c0392b"])
    ax2.set_xticks(range(len(bins)))
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel("Max Attention")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Higher Attention = Higher Accuracy")
    
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{acc:.1f}%", ha="center", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_heatmap(results, cat_to_group, top_n_targets=10, top_n_history=12, 
                           title="Attention Heatmap", save_path=None):
    """Heatmap: when predicting [Target], how much attention on [History]?"""
    
    # Get top targets and history categories
    all_targets = Counter(r["target"] for r in results)
    all_history = Counter(cat for r in results for cat in r["history"])
    
    top_targets = [t for t, _ in all_targets.most_common(top_n_targets)]
    top_hist = [h for h, _ in all_history.most_common(top_n_history)]
    
    # Build attention matrix
    attention_by_target = {t: {} for t in top_targets}
    for r in results:
        if r["target"] not in top_targets:
            continue
        for hist_cat, att in r["attention"].items():
            if hist_cat not in attention_by_target[r["target"]]:
                attention_by_target[r["target"]][hist_cat] = []
            attention_by_target[r["target"]][hist_cat].append(att)
    
    matrix = np.zeros((len(top_targets), len(top_hist)))
    for i, target in enumerate(top_targets):
        for j, hist in enumerate(top_hist):
            if hist in attention_by_target[target] and len(attention_by_target[target][hist]) >= 2:
                matrix[i, j] = np.mean(attention_by_target[target][hist])
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    
    short_targets = [t[:12] + ".." if len(t) > 14 else t for t in top_targets]
    short_hist = [h[:10] + ".." if len(h) > 12 else h for h in top_hist]
    
    ax.set_xticks(range(len(top_hist)))
    ax.set_yticks(range(len(top_targets)))
    ax.set_xticklabels(short_hist, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_targets, fontsize=9)
    ax.set_xlabel("History Category (what model looks at)")
    ax.set_ylabel("Target Category (what model predicts)")
    ax.set_title(title)
    
    # Add values
    for i in range(len(top_targets)):
        for j in range(len(top_hist)):
            if matrix[i, j] > 0.1:
                color = "white" if matrix[i, j] > 0.3 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", 
                       fontsize=8, color=color)
    
    plt.colorbar(im, label="Average Attention")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_lookback(results, n_examples=6, save_path=None):
    """Visualize how model looks back at history for predictions."""
    
    # Get mix of correct/incorrect with long history
    long_correct = [r for r in results if len(r["history"]) >= 7 and r["correct"]][:3]
    long_incorrect = [r for r in results if len(r["history"]) >= 7 and not r["correct"]][:3]
    examples = long_correct + long_incorrect
    
    fig, axes = plt.subplots(len(examples), 1, figsize=(16, 3 * len(examples)))
    
    for ax, r in zip(axes, examples):
        history = r["history"]
        attn = r["attn_values"]
        n = len(history)
        
        # Bar colors based on attention
        colors = plt.cm.Reds(np.array(attn) / max(attn) * 0.8 + 0.2)
        bars = ax.bar(range(n), attn, color=colors, edgecolor="black", linewidth=1)
        
        # Highlight target matches
        for i, (bar, cat) in enumerate(zip(bars, history)):
            if cat == r["target"]:
                bar.set_edgecolor("#2ecc71")
                bar.set_linewidth(3)
        
        # Labels
        short_names = [c[:10] + ".." if len(c) > 12 else c for c in history]
        ax.set_xticks(range(n))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
        
        # Arrow showing lookback
        ax.annotate("", xy=(0, max(attn) + 0.15), xytext=(n - 0.5, max(attn) + 0.15),
                   arrowprops=dict(arrowstyle="<-", color="blue", lw=2))
        ax.text(n/2, max(attn) + 0.2, "Model looks back", ha="center", fontsize=10, color="blue")
        
        # Prediction info
        status = "✓" if r["correct"] else "✗"
        color = "#2ecc71" if r["correct"] else "#e74c3c"
        ax.text(n + 0.5, max(attn)/2, f"Predict: {r['predicted'][:12]}\\nActual: {r['target'][:12]}\\n{status}", 
               fontsize=10, ha="left", va="center",
               bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))
        
        ax.set_ylabel("Attention")
        ax.set_xlim(-0.5, n + 2)
        ax.set_ylim(0, max(attn) + 0.3)
    
    plt.suptitle("Attention = Looking Back\\n(Taller bar = more attention, Green border = target in history)", 
                fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_dict, cat_to_group, save_path=None):
    """Compare multiple models on exact and group accuracy."""
    
    models = list(results_dict.keys())
    exact_accs = []
    group_accs = []
    
    for name, results in results_dict.items():
        exact = np.mean([r["correct"] for r in results])
        group = np.mean([
            cat_to_group.get(r["predicted"], "Other") == cat_to_group.get(r["target"], "Other")
            for r in results
        ])
        exact_accs.append(exact * 100)
        group_accs.append(group * 100)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, exact_accs, width, label="Exact Accuracy", color="#3498db")
    bars2 = ax.bar(x + width/2, group_accs, width, label="Same-Group Accuracy", color="#2ecc71")
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    for bar, val in zip(bars1, exact_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f"{val:.1f}%", ha="center", fontsize=10)
    for bar, val in zip(bars2, group_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f"{val:.1f}%", ha="center", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
