# ============================================================================
# FILE 4: attention.py
# ============================================================================

"""Attention extraction and analysis."""

import torch
import numpy as np
from collections import Counter


def get_attention_weights(model, sequence, pad_idx):
    """Extract attention weights from transformer."""
    model.eval()
    with torch.no_grad():
        token_emb = model.embedding(sequence.unsqueeze(0))
        positions = torch.arange(sequence.shape[0])
        pos_emb = model.pos_embedding(positions)
        x = token_emb + pos_emb
        
        attn_layer = model.transformer.layers[0].self_attn
        pad_mask = (sequence == pad_idx).unsqueeze(0)
        _, attn_weights = attn_layer(x, x, x, key_padding_mask=pad_mask, average_attn_weights=True)
        
    # Return attention FROM last position TO all positions
    return attn_weights.squeeze(0)[-1].numpy()


def analyze_attention(model, dataset, vocab, idx_to_cat, n_samples=500):
    """Analyze attention patterns across test samples."""
    results = []
    
    for i in range(min(n_samples, len(dataset))):
        seq, target_idx = dataset[i]
        target = idx_to_cat[target_idx]
        
        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(seq.unsqueeze(0), pad_idx=vocab["[PAD]"])
            pred_idx = logits.argmax(dim=1).item()
            pred = idx_to_cat[pred_idx]
            confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()
        
        # Get attention
        attn = get_attention_weights(model, seq, vocab["[PAD]"])
        
        # Map to categories
        history = [idx_to_cat[idx.item()] for idx in seq if idx.item() != vocab["[PAD]"]]
        attn_no_pad = attn[-len(history):]
        
        results.append({
            "history": history,
            "target": target,
            "predicted": pred,
            "correct": pred == target,
            "confidence": confidence,
            "attention": dict(zip(history, attn_no_pad)),
            "attn_values": attn_no_pad
        })
    
    return results


def get_attention_stats(results):
    """Calculate attention statistics for results."""
    max_attns = []
    attn_spreads = []
    top2_gaps = []
    
    for r in results:
        attn = r["attn_values"]
        if len(attn) > 1:
            sorted_attn = sorted(attn, reverse=True)
            max_attns.append(sorted_attn[0])
            attn_spreads.append(np.std(attn))
            top2_gaps.append(sorted_attn[0] - sorted_attn[1])
    
    return {
        "max_attn": np.mean(max_attns),
        "attn_spread": np.mean(attn_spreads),
        "top2_gap": np.mean(top2_gaps)
    }


def analyze_by_transition_type(results, cat_to_group=None):
    """Split analysis by return visits vs true transitions."""
    
    return_visits = [r for r in results if r["target"] in r["history"]]
    true_transitions = [r for r in results if r["target"] not in r["history"]]
    
    analysis = {
        "return_visits": {
            "count": len(return_visits),
            "exact_acc": np.mean([r["correct"] for r in return_visits]) if return_visits else 0,
        },
        "true_transitions": {
            "count": len(true_transitions),
            "exact_acc": np.mean([r["correct"] for r in true_transitions]) if true_transitions else 0,
        }
    }
    
    # Add group accuracy if groups provided
    if cat_to_group:
        for key, subset in [("return_visits", return_visits), ("true_transitions", true_transitions)]:
            group_correct = sum(
                1 for r in subset 
                if cat_to_group.get(r["predicted"], "Other") == cat_to_group.get(r["target"], "Other")
            )
            analysis[key]["group_acc"] = group_correct / len(subset) if subset else 0
    
    return analysis


def analyze_diagonal_attention(results, cat_to_group):
    """Analyze attention on same-category vs different-category."""
    
    diagonal = []  # Same category or same group
    off_diagonal = []  # Different group
    
    for r in results:
        target = r["target"]
        target_group = cat_to_group.get(target, "Other")
        
        for hist_cat, att in r["attention"].items():
            hist_group = cat_to_group.get(hist_cat, "Other")
            
            if hist_cat == target:
                diagonal.append({"attn": att, "correct": r["correct"], "type": "exact"})
            elif hist_group == target_group:
                diagonal.append({"attn": att, "correct": r["correct"], "type": "group"})
            else:
                off_diagonal.append({"attn": att, "correct": r["correct"]})
    
    return {
        "diagonal": {
            "count": len(diagonal),
            "avg_attn": np.mean([d["attn"] for d in diagonal]) if diagonal else 0,
            "acc_when_high": np.mean([d["correct"] for d in diagonal if d["attn"] > 0.3]) if diagonal else 0
        },
        "off_diagonal": {
            "count": len(off_diagonal),
            "avg_attn": np.mean([d["attn"] for d in off_diagonal]) if off_diagonal else 0,
            "acc_when_high": np.mean([d["correct"] for d in off_diagonal if d["attn"] > 0.3]) if off_diagonal else 0
        }
    }
