# %%Option 1: Run full pipeline
from main import main
results = main(dataset='foresquare', epochs=30)
# %%
# Option 2: Use individual components
from data import prepare_data
from model import TrajectoryTransformer
from train import train_model, evaluate_model, split_data
from attention import analyze_attention
from visualize import plot_attention_lookback

# Load data
data = prepare_data(dataset="gowalla", collapse=True)

# Train
train_loader, test_loader, _, test_dataset = split_data(data["dataset"])
model = TrajectoryTransformer(vocab_size=len(data["vocab"]))
train_model(model, train_loader, data["vocab"], epochs=30)

# Analyze
results = analyze_attention(model, test_dataset, data["vocab"], data["idx_to_cat"])

# Visualize
plot_attention_lookback(results, save_path="my_figure.png")
# %%
# Reload
import importlib
import train
importlib.reload(train)
from train import train_model, split_data
from data import prepare_data

# Get data
from data import TrajectoryDataset
data = prepare_data(dataset="forsquare", collapse=True)

trajectories = data['trajectories']
vocab = data['vocab']
cat_to_group = data['cat_to_group']

# Create dataset and split
dataset = TrajectoryDataset(trajectories, vocab)
train_loader, test_loader, _, _ = split_data(dataset, batch_size=64)

# Train with soft loss
from model import TrajectoryTransformer
soft_model = TrajectoryTransformer(len(vocab))
soft_history = train_model(soft_model, train_loader, vocab, epochs=30, 
                           loss_type="soft", cat_to_group=cat_to_group, temperature=0.3)
# %%
from attention import analyze_attention

# Get idx_to_cat
idx_to_cat = {v: k for k, v in vocab.items() if k != "[PAD]"}

# Use the trajectories directly
trajectories = data['trajectories']
split_idx = int(len(trajectories) * 0.8)
test_trajs = trajectories[split_idx:]

test_dataset = TrajectoryDataset(test_trajs, vocab)

# Analyze attention on soft model
soft_attention_results = analyze_attention(soft_model, test_dataset, vocab, idx_to_cat, n_samples=1000)

# Check the pattern
print(f"Total: {len(soft_attention_results)}")
correct = [r for r in soft_attention_results if r['correct']]
print(f"Accuracy: {len(correct)/len(soft_attention_results)*100:.1f}%")

# High attention → accuracy?
import numpy as np
max_attns = [(np.max(r['attn_values']), r['correct']) for r in soft_attention_results]

bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
print("\nMax Attention → Accuracy:")
for low, high in bins:
    in_bin = [(a, c) for a, c in max_attns if low <= a < high]
    if in_bin:
        acc = sum(c for a, c in in_bin) / len(in_bin) * 100
        print(f"  {low}-{high}: {acc:.1f}% (n={len(in_bin)})")
# %%
import importlib
import data, train, model, attention
importlib.reload(data)
importlib.reload(train)

from data import prepare_data
from model import TrajectoryTransformer
from train import train_model, split_data
from attention import analyze_attention
import numpy as np
import pickle

results_all = {}

for dataset_name in ["foursquare", "gowalla"]:
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    data_dict = prepare_data(dataset=dataset_name, collapse=True)
    print(f"Trajectories: {len(data_dict['trajectories'])}, Users: {len(set(data_dict['user_ids']))}")
    
    # Split by user (same split for both loss types)
    train_loader, test_loader, train_ds, test_ds = split_data(
        data_dict['trajectories'], 
        data_dict['user_ids'], 
        data_dict['vocab'],
        batch_size=64
    )
    
    idx_to_cat = {v: k for k, v in data_dict['vocab'].items() if k != "[PAD]"}
    
    for loss_type in ["hard", "soft"]:
        print(f"\n--- {loss_type.upper()} LOSS ---")
        
        # Train
        mdl = TrajectoryTransformer(len(data_dict['vocab']))
        history = train_model(
            mdl, train_loader, data_dict['vocab'], 
            epochs=30, loss_type=loss_type, 
            cat_to_group=data_dict['cat_to_group']
        )
        
        # Analyze
        attn_results = analyze_attention(mdl, test_ds, data_dict['vocab'], idx_to_cat, n_samples=1000)
        
        # Quick summary
        correct = sum(r['correct'] for r in attn_results)
        return_visits = [r for r in attn_results if r['target'] in r['history']]
        true_trans = [r for r in attn_results if r['target'] not in r['history']]
        
        rv_acc = sum(r['correct'] for r in return_visits) / len(return_visits) * 100
        tt_acc = sum(r['correct'] for r in true_trans) / len(true_trans) * 100
        
        print(f"Accuracy: {correct/len(attn_results)*100:.1f}%")
        print(f"Return visits: {rv_acc:.1f}% | True transitions: {tt_acc:.1f}%")
        
        # Save
        key = f"{dataset_name}_{loss_type}"
        results_all[key] = {
            'attention_results': attn_results,
            'history': history,
            'cat_to_group': data_dict['cat_to_group'],
            'vocab': data_dict['vocab']
        }

# Save to file
with open('/Users/avitalkleinbrill/Documents/vscode_projects/spatiel_learning/mobility_attention/results_all.pkl', 'wb') as f:
    pickle.dump(results_all, f)

print("\n\nSaved to results_all.pkl!")
print(f"Keys: {list(results_all.keys())}")

# %%
# Reload modules
import importlib
import data, train, model, attention
importlib.reload(data)
importlib.reload(train)
importlib.reload(model)
importlib.reload(attention)

from data import prepare_data
from model import TrajectoryTransformer
from train import train_model, split_data
from attention import analyze_attention

# Load foursquare with user ids
data_fs = prepare_data(dataset="foursquare", collapse=True)
print(f"Keys: {data_fs.keys()}")
print(f"Trajectories: {len(data_fs['trajectories'])}")
print(f"User IDs: {len(data_fs['user_ids'])}")

# Split by user
train_loader, test_loader, train_ds, test_ds = split_data(
    data_fs['trajectories'], 
    data_fs['user_ids'], 
    data_fs['vocab'],
    batch_size=64
)

# Train
model_fs = TrajectoryTransformer(len(data_fs['vocab']))
history = train_model(model_fs, train_loader, data_fs['vocab'], epochs=30)

# Analyze attention
idx_to_cat = {v: k for k, v in data_fs['vocab'].items() if k != "[PAD]"}
attn_results = analyze_attention(model_fs, test_ds, data_fs['vocab'], idx_to_cat, n_samples=1000)

# Quick check
correct = sum(r['correct'] for r in attn_results)
print(f"\nTest Accuracy: {correct/len(attn_results)*100:.1f}%")
# %%
import numpy as np

# Return visits vs true transitions
return_visits = [r for r in attn_results if r['target'] in r['history']]
true_transitions = [r for r in attn_results if r['target'] not in r['history']]

rv_acc = sum(r['correct'] for r in return_visits) / len(return_visits) * 100 if return_visits else 0
tt_acc = sum(r['correct'] for r in true_transitions) / len(true_transitions) * 100 if true_transitions else 0

print(f"Return visits: {len(return_visits)} - Acc: {rv_acc:.1f}%")
print(f"True transitions: {len(true_transitions)} - Acc: {tt_acc:.1f}%")

# Attention on target for return visits
print("\n--- Attention on target ---")
for threshold in [0.3, 0.5]:
    high = [r for r in return_visits if r['target'] in r['attention'] and r['attention'][r['target']] > threshold]
    if high:
        acc = sum(r['correct'] for r in high) / len(high) * 100
        print(f"  Attention > {threshold}: {acc:.1f}% (n={len(high)})")
# %%
# %%


print("Keys:", list(results_all.keys()))

# Analyze each
for key in results_all.keys():
    print(f"\n{'='*50}")
    print(f"{key.upper()}")
    print(f"{'='*50}")
    
    attn_results = results_all[key]['attention_results']
    
    # 1. Overall accuracy
    acc = sum(r['correct'] for r in attn_results) / len(attn_results) * 100
    
    # 2. Return visits vs true transitions
    return_visits = [r for r in attn_results if r['target'] in r['history']]
    true_trans = [r for r in attn_results if r['target'] not in r['history']]
    rv_acc = sum(r['correct'] for r in return_visits) / len(return_visits) * 100
    tt_acc = sum(r['correct'] for r in true_trans) / len(true_trans) * 100
    
    # 3. High attention on target → accuracy (return visits only)
    high_attn = [r for r in return_visits if r['target'] in r['attention'] and r['attention'][r['target']] > 0.3]
    low_attn = [r for r in return_visits if r['target'] in r['attention'] and r['attention'][r['target']] <= 0.3]
    
    high_acc = sum(r['correct'] for r in high_attn) / len(high_attn) * 100 if high_attn else 0
    low_acc = sum(r['correct'] for r in low_attn) / len(low_attn) * 100 if low_attn else 0
    
    # 4. Max attention → accuracy (all samples)
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    
    print(f"Overall: {acc:.1f}%")
    print(f"Return visits: {rv_acc:.1f}% (n={len(return_visits)}) | True trans: {tt_acc:.1f}% (n={len(true_trans)})")
    print(f"Attention on target >0.3: {high_acc:.1f}% (n={len(high_attn)}) | ≤0.3: {low_acc:.1f}% (n={len(low_attn)})")
    
    print("\nMax attention → accuracy:")
    for low, high in bins:
        in_bin = [(np.max(r['attn_values']), r['correct']) for r in attn_results if low <= np.max(r['attn_values']) < high]
        if in_bin:
            bin_acc = sum(c for a, c in in_bin) / len(in_bin) * 100
            print(f"  {low}-{high}: {bin_acc:.1f}% (n={len(in_bin)})")
# %%
