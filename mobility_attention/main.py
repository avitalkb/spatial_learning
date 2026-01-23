

# ============================================================================
# FILE 7: main.py
# ============================================================================

"""Main script to run mobility attention experiments."""

import argparse
from data import prepare_data
from model import TrajectoryTransformer, build_similarity_matrix, SoftCrossEntropyLoss
from train import train_model, evaluate_model, split_data
from attention import analyze_attention, get_attention_stats, analyze_by_transition_type
from visualize import (plot_attention_correct_vs_incorrect, 
                       plot_attention_heatmap, 
                       plot_attention_lookback)


def main(dataset="gowalla", epochs=30, batch_size=64):
    """Run full experiment pipeline."""
    
    print("=" * 60)
    print(f"MOBILITY ATTENTION EXPERIMENT")
    print(f"Dataset: {dataset}")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\\n[1/5] Loading data...")
    data = prepare_data(dataset=dataset, collapse=True)
    
    print(f"  Users: {data['df']['userid'].nunique()}")
    print(f"  Categories: {len(data['categories'])}")
    print(f"  Trajectories: {len(data['trajectories'])}")
    print(f"  Training samples: {len(data['dataset'])}")
    
    # 2. Split data
    print("\\n[2/5] Splitting data...")
    train_loader, test_loader, train_dataset, test_dataset = split_data(
        data["trajectories"], data["user_ids"], data["vocab"], batch_size=batch_size
    )
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # 3. Train model
    print("\\n[3/5] Training model...")
    model = SoftCrossEntropyLoss(vocab_size=len(data["vocab"]))
    
    random_baseline = 1 / len(data["categories"])
    print(f"  Random baseline: {random_baseline:.2%}")
    
    history = train_model(model, train_loader, data["vocab"], epochs=epochs,print_every=1)
    
    # 4. Evaluate
    print("\\n[4/5] Evaluating...")
    results = evaluate_model(
        model, test_loader, data["vocab"], 
        data["idx_to_cat"], data["cat_to_group"]
    )
    
    print(f"  Exact accuracy: {results['exact_accuracy']:.2%}")
    print(f"  Group accuracy: {results.get('group_accuracy', 'N/A'):.2%}")
    print(f"  Lift over random: {results['exact_accuracy'] / random_baseline:.1f}x")
    
    # 5. Analyze attention
    print("\\n[5/5] Analyzing attention...")
    attention_results = analyze_attention(
        model, test_dataset, data["vocab"], data["idx_to_cat"], n_samples=1000
    )
    
    correct = [r for r in attention_results if r["correct"]]
    incorrect = [r for r in attention_results if not r["correct"]]
    
    print(f"  Analyzed: {len(attention_results)} samples")
    print(f"  Correct: {len(correct)}, Incorrect: {len(incorrect)}")
    
    # Attention stats
    correct_stats = get_attention_stats(correct)
    incorrect_stats = get_attention_stats(incorrect)
    
    print(f"\\n  Attention on target (when in history):")
    correct_target = [r["attention"].get(r["target"], 0) for r in correct if r["target"] in r["attention"]]
    incorrect_target = [r["attention"].get(r["target"], 0) for r in incorrect if r["target"] in r["attention"]]
    
    if correct_target:
        print(f"    Correct predictions: {sum(correct_target)/len(correct_target):.3f}")
    if incorrect_target:
        print(f"    Incorrect predictions: {sum(incorrect_target)/len(incorrect_target):.3f}")
    
    # Transition analysis
    transition_analysis = analyze_by_transition_type(attention_results, data["cat_to_group"])
    print(f"\\n  Return visits accuracy: {transition_analysis['return_visits']['exact_acc']:.1%}")
    print(f"  True transitions accuracy: {transition_analysis['true_transitions']['exact_acc']:.1%}")
    
    # 6. Generate visualizations
    print("\\n[6/6] Generating visualizations...")
    
    plot_attention_correct_vs_incorrect(
        correct, incorrect, 
        save_path="attention_correct_vs_incorrect.png"
    )
    
    plot_attention_heatmap(
        attention_results, data["cat_to_group"],
        save_path="attention_heatmap.png"
    )
    
    plot_attention_lookback(
        attention_results,
        save_path="attention_lookback.png"
    )
    
    print("\\n" + "=" * 60)
    print("DONE! Visualizations saved.")
    print("=" * 60)
    
    return {
        "model": model,
        "data": data,
        "results": results,
        "attention_results": attention_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentPars