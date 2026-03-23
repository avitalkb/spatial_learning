"""
Generate all figures from saved results and save to results/ folder.
No training required — loads results_clean.pkl.
Usage: python run_analysis.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── Setup ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import config, data, model, attention, visualize
from config import FOURSQUARE_URL, GOWALLA_URL, CATEGORY_KEYWORDS
from data import prepare_data, load_gowalla_data, load_foursquare_data, auto_categorize

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def save(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  saved: {name}')


# ── Load saved results ────────────────────────────────────────────────────────
print('Loading saved results...')
with open(os.path.join(SCRIPT_DIR, 'results_clean.pkl'), 'rb') as f:
    saved = pickle.load(f)

history_std_gw  = saved['gowalla_standard']['history']
history_feat_gw = saved['gowalla_features']['history']
history_std_fs  = saved['foursquare_standard']['history']
history_feat_fs = saved['foursquare_features']['history']

attn_std_gw  = saved['gowalla_standard'].get('attention_results')
attn_feat_gw = saved['gowalla_features'].get('attention_results')
attn_std_fs  = saved['foursquare_standard'].get('attention_results')
attn_feat_fs = saved['foursquare_features'].get('attention_results')


# ── Load raw data (for EDA) ───────────────────────────────────────────────────
print('Loading raw data for EDA...')
df_foursquare = load_foursquare_data(FOURSQUARE_URL)
df_gowalla    = load_gowalla_data(GOWALLA_URL)

print('Preparing trajectories...')
data_gw = prepare_data(dataset='gowalla',     group_level=True, include_features=False)
data_fs = prepare_data(dataset='foursquare',  group_level=True, include_features=False)


# ═════════════════════════════════════════════════════════════════════════════
# 1. EDA
# ═════════════════════════════════════════════════════════════════════════════
print('\n[1] EDA figures...')

# Top 15 raw categories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, df) in zip(axes, [('Foursquare', df_foursquare), ('Gowalla', df_gowalla)]):
    top_cats = df['spot_categ'].value_counts().head(15)
    top_cats.plot(kind='barh', ax=ax)
    ax.set_title(f'{name} — Top 15 Categories')
    ax.set_xlabel('Count')
plt.tight_layout()
save('eda_top_categories.png')

# Check-ins per user distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, df) in zip(axes, [('Foursquare', df_foursquare), ('Gowalla', df_gowalla)]):
    checkins = df.groupby('userid').size()
    ax.hist(checkins, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(checkins.median(), color='red', linestyle='--', label=f'Median: {checkins.median():.0f}')
    ax.set_title(f'{name} — Check-ins per User')
    ax.set_xlabel('Check-ins')
    ax.set_ylabel('Users')
    ax.legend()
plt.tight_layout()
save('eda_checkins_per_user.png')

# Trajectory length + unique categories
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for col, (name, d) in enumerate([('Gowalla', data_gw), ('Foursquare', data_fs)]):
    trajs   = d['trajectories']
    lengths = [len(t) for t in trajs]
    uniques = [len(set(t)) for t in trajs]
    axes[0, col].hist(lengths, bins=20, edgecolor='black', alpha=0.7)
    axes[0, col].set_title(f'{name} — Trajectory Lengths')
    axes[0, col].set_xlabel('Length'); axes[0, col].set_ylabel('Count')
    axes[1, col].hist(uniques, bins=range(1, max(uniques) + 2), edgecolor='black', alpha=0.7)
    axes[1, col].set_title(f'{name} — Unique Categories per Trajectory')
    axes[1, col].set_xlabel('Unique categories'); axes[1, col].set_ylabel('Count')
plt.tight_layout()
save('eda_trajectory_stats.png')


# ═════════════════════════════════════════════════════════════════════════════
# 2. Training curves
# ═════════════════════════════════════════════════════════════════════════════
print('[2] Training curve figures...')

epochs_range = range(1, len(history_std_gw['loss']) + 1)

# Standard models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_range, history_std_gw['loss'], 'o-', label='Gowalla', color='#2196F3')
axes[0].plot(epochs_range, history_std_fs['loss'], 's-', label='Foursquare', color='#FF5722')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Standard Model — Training Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(epochs_range, [a*100 for a in history_std_gw['accuracy']], 'o--', label='Gowalla Train', color='#2196F3')
axes[1].plot(epochs_range, [a*100 for a in history_std_gw['test_accuracy']], 'o-', label='Gowalla Test', color='#2196F3')
axes[1].plot(epochs_range, [a*100 for a in history_std_fs['accuracy']], 's--', label='Foursquare Train', color='#FF5722')
axes[1].plot(epochs_range, [a*100 for a in history_std_fs['test_accuracy']], 's-', label='Foursquare Test', color='#FF5722')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Standard Model — Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
save('training_standard.png')

# Standard vs feature model comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for row, (name, h_std, h_feat) in enumerate([
    ('Gowalla', history_std_gw, history_feat_gw),
    ('Foursquare', history_std_fs, history_feat_fs)
]):
    axes[row, 0].plot(epochs_range, h_std['loss'], 'o-', label='Standard', color='#2196F3')
    axes[row, 0].plot(epochs_range, h_feat['loss'], 's-', label='With Features', color='#FF5722')
    axes[row, 0].set_xlabel('Epoch'); axes[row, 0].set_ylabel('Loss')
    axes[row, 0].set_title(f'{name} — Training Loss')
    axes[row, 0].legend(); axes[row, 0].grid(True, alpha=0.3)
    axes[row, 1].plot(epochs_range, [a*100 for a in h_std['accuracy']], 'o--', label='Std Train', color='#2196F3')
    axes[row, 1].plot(epochs_range, [a*100 for a in h_std['test_accuracy']], 'o-', label='Std Test', color='#2196F3')
    axes[row, 1].plot(epochs_range, [a*100 for a in h_feat['accuracy']], 's--', label='Feat Train', color='#FF5722')
    axes[row, 1].plot(epochs_range, [a*100 for a in h_feat['test_accuracy']], 's-', label='Feat Test', color='#FF5722')
    axes[row, 1].set_xlabel('Epoch'); axes[row, 1].set_ylabel('Accuracy (%)')
    axes[row, 1].set_title(f'{name} — Accuracy')
    axes[row, 1].legend(); axes[row, 1].grid(True, alpha=0.3)
plt.tight_layout()
save('training_comparison.png')


# ═════════════════════════════════════════════════════════════════════════════
# 3. Attention analysis
# ═════════════════════════════════════════════════════════════════════════════
print('[3] Attention figures...')

# Single-example attention bar charts
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for col, (name, results) in enumerate([('Gowalla Std', attn_std_gw), ('Gowalla Feat', attn_feat_gw)]):
    ex = [r for r in results if len(r['history']) >= 5][10]
    axes[0, col].barh(range(len(ex['history'])), ex['attn_values'])
    axes[0, col].set_yticks(range(len(ex['history'])))
    axes[0, col].set_yticklabels(ex['history'], fontsize=8)
    axes[0, col].set_xlabel('Attention Weight')
    axes[0, col].set_title(f"{name}\nTarget: {ex['target']} | Pred: {ex['predicted']} ({'Correct' if ex['correct'] else 'Wrong'})")
for col, (name, results) in enumerate([('Foursquare Std', attn_std_fs), ('Foursquare Feat', attn_feat_fs)]):
    ex = [r for r in results if len(r['history']) >= 5][0]
    axes[1, col].barh(range(len(ex['history'])), ex['attn_values'])
    axes[1, col].set_yticks(range(len(ex['history'])))
    axes[1, col].set_yticklabels(ex['history'], fontsize=8)
    axes[1, col].set_xlabel('Attention Weight')
    axes[1, col].set_title(f"{name}\nTarget: {ex['target']} | Pred: {ex['predicted']} ({'Correct' if ex['correct'] else 'Wrong'})")
plt.tight_layout()
save('attention_examples.png')

# Recency bias
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, results_list) in zip(axes, [
    ('Gowalla',    [('Standard', attn_std_gw), ('Features', attn_feat_gw)]),
    ('Foursquare', [('Standard', attn_std_fs), ('Features', attn_feat_fs)])
]):
    for label, results in results_list:
        pos_attn = {i: [] for i in range(10)}
        for r in results:
            for j, a in enumerate(r['attn_values']):
                rel_pos = len(r['attn_values']) - 1 - j
                if rel_pos < 10:
                    pos_attn[rel_pos].append(a)
        means = [np.mean(pos_attn[p]) if pos_attn[p] else 0 for p in range(10)]
        ax.plot(range(10), means, 'o-', label=label)
    ax.set_xlabel('Relative Position (0 = most recent)')
    ax.set_ylabel('Mean Attention')
    ax.set_title(f'{name} — Attention by Position')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
save('attention_recency.png')

# Prediction source breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (dataset_name, models) in zip(axes, [
    ('Gowalla',    [('Standard', attn_std_gw), ('Features', attn_feat_gw)]),
    ('Foursquare', [('Standard', attn_std_fs), ('Features', attn_feat_fs)]),
]):
    labels, correct_pcts, from_history_pcts, novel_pcts = [], [], [], []
    for model_name, results in models:
        for visit_type in ['Return', 'Transition']:
            subset = [r for r in results if (r['target'] in r['history']) == (visit_type == 'Return')]
            if not subset: continue
            n = len(subset)
            labels.append(f'{model_name}\n{visit_type}\n(n={n})')
            correct_pcts.append(sum(r['correct'] for r in subset) / n * 100)
            from_history_pcts.append(sum(not r['correct'] and r['predicted'] in r['history'] for r in subset) / n * 100)
            novel_pcts.append(sum(not r['correct'] and r['predicted'] not in r['history'] for r in subset) / n * 100)
    x = np.arange(len(labels))
    ax.bar(x, correct_pcts, 0.6, label='Correct', color='#4CAF50')
    ax.bar(x, from_history_pcts, 0.6, bottom=correct_pcts, label='Wrong (from history)', color='#FF9800')
    ax.bar(x, novel_pcts, 0.6, bottom=[c+h for c, h in zip(correct_pcts, from_history_pcts)], label='Wrong (novel)', color='#F44336')
    ax.set_ylabel('% of predictions'); ax.set_title(dataset_name)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8); ax.set_ylim(0, 105)
plt.suptitle('Where Do Predictions Come From?', fontsize=14, y=1.02)
plt.tight_layout()
save('attention_prediction_source.png')

# Confidence distribution: correct vs wrong
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for row, (name, results) in enumerate([('Gowalla Standard', attn_std_gw), ('Gowalla Features', attn_feat_gw)]):
    conf_c = [r['confidence'] for r in results if r['correct']]
    conf_w = [r['confidence'] for r in results if not r['correct']]
    axes[0, row].hist(conf_c, bins=25, alpha=0.6, density=True, label=f'Correct (n={len(conf_c)})', color='#4CAF50')
    axes[0, row].hist(conf_w, bins=25, alpha=0.6, density=True, label=f'Wrong (n={len(conf_w)})', color='#F44336')
    axes[0, row].axvline(np.mean(conf_c), color='#4CAF50', linestyle='--', linewidth=2)
    axes[0, row].axvline(np.mean(conf_w), color='#F44336', linestyle='--', linewidth=2)
    axes[0, row].set_xlabel('Confidence'); axes[0, row].set_ylabel('Density')
    axes[0, row].set_xlim(0, 1); axes[0, row].set_title(name); axes[0, row].legend()
for row, (name, results) in enumerate([('Foursquare Standard', attn_std_fs), ('Foursquare Features', attn_feat_fs)]):
    conf_c = [r['confidence'] for r in results if r['correct']]
    conf_w = [r['confidence'] for r in results if not r['correct']]
    axes[1, row].hist(conf_c, bins=25, alpha=0.6, density=True, label=f'Correct (n={len(conf_c)})', color='#4CAF50')
    axes[1, row].hist(conf_w, bins=25, alpha=0.6, density=True, label=f'Wrong (n={len(conf_w)})', color='#F44336')
    axes[1, row].axvline(np.mean(conf_c), color='#4CAF50', linestyle='--', linewidth=2)
    axes[1, row].axvline(np.mean(conf_w), color='#F44336', linestyle='--', linewidth=2)
    axes[1, row].set_xlabel('Confidence'); axes[1, row].set_ylabel('Density')
    axes[1, row].set_title(name); axes[1, row].legend()
plt.suptitle('Prediction Confidence: Correct vs Wrong', fontsize=14, y=1.02)
plt.tight_layout()
save('attention_confidence.png')

# Attention on target: correct vs wrong (return visits only)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for row, (name, results) in enumerate([('Gowalla Standard', attn_std_gw), ('Gowalla Features', attn_feat_gw)]):
    attn_c, attn_w = [], []
    for r in results:
        if r['target'] not in r['history']: continue
        total = sum(a for cat, a in r['attention'].items() if cat == r['target'])
        (attn_c if r['correct'] else attn_w).append(total)
    if attn_c and attn_w:
        _, bins = np.histogram(attn_c + attn_w, bins='auto')
        axes[0, row].hist(attn_c, bins=bins, alpha=0.6, density=True, label=f'Correct (n={len(attn_c)})', color='#4CAF50')
        axes[0, row].hist(attn_w, bins=bins, alpha=0.6, density=True, label=f'Wrong (n={len(attn_w)})', color='#F44336')
        axes[0, row].axvline(np.mean(attn_c), color='#4CAF50', linestyle='--', linewidth=2)
        axes[0, row].axvline(np.mean(attn_w), color='#F44336', linestyle='--', linewidth=2)
    axes[0, row].set_xlabel('Attention on Target'); axes[0, row].set_ylabel('Density')
    axes[0, row].set_title(name); axes[0, row].legend()
for row, (name, results) in enumerate([('Foursquare Standard', attn_std_fs), ('Foursquare Features', attn_feat_fs)]):
    attn_c, attn_w = [], []
    for r in results:
        if r['target'] not in r['history']: continue
        total = sum(a for cat, a in r['attention'].items() if cat == r['target'])
        (attn_c if r['correct'] else attn_w).append(total)
    if attn_c and attn_w:
        _, bins = np.histogram(attn_c + attn_w, bins='auto')
        axes[1, row].hist(attn_c, bins=bins, alpha=0.6, density=True, label=f'Correct (n={len(attn_c)})', color='#4CAF50')
        axes[1, row].hist(attn_w, bins=bins, alpha=0.6, density=True, label=f'Wrong (n={len(attn_w)})', color='#F44336')
        axes[1, row].axvline(np.mean(attn_c), color='#4CAF50', linestyle='--', linewidth=2)
        axes[1, row].axvline(np.mean(attn_w), color='#F44336', linestyle='--', linewidth=2)
    axes[1, row].set_xlabel('Attention on Target'); axes[1, row].set_ylabel('Density')
    axes[1, row].set_title(name); axes[1, row].legend()
plt.suptitle('Attention on Target Category (return visits only)', fontsize=14, y=1.02)
plt.tight_layout()
save('attention_on_target.png')

# Max attention distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, std_results, feat_results) in zip(axes, [
    ('Gowalla', attn_std_gw, attn_feat_gw),
    ('Foursquare', attn_std_fs, attn_feat_fs)
]):
    ax.hist([max(r['attn_values']) for r in std_results], bins=30, alpha=0.6, label='Standard', color='#2196F3')
    ax.hist([max(r['attn_values']) for r in feat_results], bins=30, alpha=0.6, label='Features', color='#FF5722')
    ax.set_xlabel('Max Attention Weight'); ax.set_ylabel('Count')
    ax.set_title(f'{name} — Max Attention Distribution'); ax.legend()
plt.tight_layout()
save('attention_max_distribution.png')

# Attention heatmap by target category (from visualize.py)
print('  attention heatmap by target...')
visualize.plot_attention_heatmap(
    attn_std_gw, data_gw['cat_to_group'],
    title='Gowalla — Standard: Attention by Target Category',
    save_path=os.path.join(RESULTS_DIR, 'attention_heatmap_gowalla_std.png')
)
visualize.plot_attention_heatmap(
    attn_feat_gw, data_gw['cat_to_group'],
    title='Gowalla — Feature: Attention by Target Category',
    save_path=os.path.join(RESULTS_DIR, 'attention_heatmap_gowalla_feat.png')
)
print('  saved: attention_heatmap_gowalla_std.png')
print('  saved: attention_heatmap_gowalla_feat.png')

# Attention lookback (from visualize.py)
print('  attention lookback...')
visualize.plot_attention_lookback(
    attn_std_gw,
    save_path=os.path.join(RESULTS_DIR, 'attention_lookback_gowalla_std.png')
)
visualize.plot_attention_lookback(
    attn_feat_gw,
    save_path=os.path.join(RESULTS_DIR, 'attention_lookback_gowalla_feat.png')
)
print('  saved: attention_lookback_gowalla_std.png')
print('  saved: attention_lookback_gowalla_feat.png')


# ═════════════════════════════════════════════════════════════════════════════
# 4. Simpson's Paradox
# ═════════════════════════════════════════════════════════════════════════════
print("[4] Simpson's Paradox figure...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plt.subplots_adjust(top=0.82)

models_x     = ['Gowalla\nStandard', 'Gowalla\nFeatures', 'Foursquare\nStandard', 'Foursquare\nFeatures']
high_overall = [29.7, 28.3, 19.4, 19.5]
low_overall  = [26.9, 29.9, 21.5, 21.4]
x = np.arange(len(models_x)); w = 0.35

bars1 = axes[0].bar(x - w/2, high_overall, w, label='Focused attention', color='#E53935', alpha=0.85)
bars2 = axes[0].bar(x + w/2, low_overall,  w, label='Spread attention',  color='#1E88E5', alpha=0.85)
axes[0].set_ylabel('Accuracy (%)', fontsize=13)
axes[0].set_xticks(x); axes[0].set_xticklabels(models_x, fontsize=10)
axes[0].legend(fontsize=10); axes[0].set_ylim(0, 40); axes[0].grid(axis='y', alpha=0.3)
axes[0].set_title('Overall Accuracy\nSpread attention appears better in 3/4 models', fontsize=13, fontweight='bold', linespacing=1.6)

x2 = np.arange(2)
bars3 = axes[1].bar(x2 - w/2, [48.0, 15.7], w, label='Focused attention', color='#E53935', alpha=0.85)
bars4 = axes[1].bar(x2 + w/2, [46.8,  6.2], w, label='Spread attention',  color='#1E88E5', alpha=0.85)
axes[1].set_ylabel('Accuracy (%)', fontsize=13)
axes[1].set_xticks(x2); axes[1].set_xticklabels(['Return Visits\n(easy)', 'True Transitions\n(hard)'], fontsize=11)
axes[1].legend(fontsize=10); axes[1].set_ylim(0, 58); axes[1].grid(axis='y', alpha=0.3)
axes[1].set_title('Controlling for Case Difficulty\nFocused attention wins in every subgroup', fontsize=13, fontweight='bold', linespacing=1.6)

for ax, bar_pairs in [(axes[0], [bars1, bars2]), (axes[1], [bars3, bars4])]:
    for bars in bar_pairs:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

fig.suptitle("Simpson's Paradox in Transformer Attention", fontsize=18, fontweight='bold')
plt.tight_layout()
save('simpsons_paradox.png')


# ═════════════════════════════════════════════════════════════════════════════
# 5. Attention path visualizations
# ═════════════════════════════════════════════════════════════════════════════
print('[5] Path visualization figures...')

CAT_COLORS = {
    'Food': '#4CAF50', 'Coffee/Dessert': '#795548', 'Bar/Nightlife': '#9C27B0',
    'Services': '#607D8B', 'Shopping': '#FF9800', 'Work': '#2196F3',
    'Entertainment': '#E91E63', 'Outdoors': '#8BC34A', 'Transit': '#00BCD4',
    'Home': '#FF5722', 'Education': '#3F51B5', 'Culture': '#009688',
    'Hotel': '#FFC107', 'Sports': '#F44336', 'Religious': '#673AB7',
    'Tech': '#03A9F4', 'Zoo/Nature': '#CDDC39', 'Other': '#9E9E9E',
}

def draw_path(ax, history, attn_vals, target, max_attn, spacing=1.15, box_w=1.0, fontsize=7.5):
    n = len(history)
    for i in range(n):
        cat   = history[i]
        attn  = attn_vals[i]
        color = CAT_COLORS.get(cat, '#9E9E9E')
        alpha = 0.3 + 0.7 * (attn / max_attn)
        box = mpatches.FancyBboxPatch(
            (i * spacing, 0.15), box_w, 0.7,
            boxstyle='round,pad=0.05', facecolor=color, alpha=alpha,
            edgecolor='black', linewidth=1.5 if attn > 0.1 else 0.5
        )
        ax.add_patch(box)
        label = cat.replace('Coffee/Dessert', 'Coffee').replace('Bar/Nightlife', 'Bar').replace('Entertainment', 'Entertain.')
        ax.text(i * spacing + box_w/2, 0.5, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if attn > 0.1 else 'normal',
                color='white' if attn > 0.2 else 'black')
        if attn > 0.01:
            ax.text(i * spacing + box_w/2, 0.0, f'{attn:.0%}', ha='center', va='top',
                    fontsize=9, fontweight='bold', color='#E53935')
        if i < n - 1:
            ax.annotate('', xy=((i+1)*spacing, 0.5), xytext=(i*spacing + box_w, 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax.annotate('', xy=(n*spacing + 0.15, 0.5), xytext=((n-1)*spacing + box_w, 0.5),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2.5))
    tbox = mpatches.FancyBboxPatch(
        (n*spacing + 0.15, 0.15), box_w, 0.7,
        boxstyle='round,pad=0.05', facecolor=CAT_COLORS.get(target, '#9E9E9E'), alpha=0.9,
        edgecolor='#4CAF50', linewidth=3
    )
    ax.add_patch(tbox)
    ax.text(n*spacing + 0.15 + box_w/2, 0.5, f'{target}\n(predicted)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    ax.set_xlim(-0.2, (n+1)*spacing + 0.3)
    ax.set_ylim(-0.15, 1.05)
    ax.axis('off')


# Return-visit path
good_feat = [r for r in attn_feat_gw if r['correct'] and 8 <= len(r['history']) <= 14]
good_std  = [r for r in attn_std_gw  if r['correct'] and 8 <= len(r['history']) <= 14]
ex_feat = good_feat[8]
ex_std  = next(
    (r for r in good_std if r['history'] == ex_feat['history'] and r['target'] == ex_feat['target']),
    good_std[8]
)
fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), gridspec_kw={'hspace': 0.6})
max_attn = max(max(ex_std['attn_values']), max(ex_feat['attn_values']))
for ax, (model_name, attn_vals) in zip(axes, [('Standard Model', ex_std['attn_values']), ('Feature Model', ex_feat['attn_values'])]):
    draw_path(ax, ex_feat['history'], attn_vals, ex_feat['target'], max_attn)
    ax.set_title(model_name, fontsize=14, fontweight='bold', loc='left')
fig.suptitle('Same Trajectory, Different Attention Strategies', fontsize=17, fontweight='bold', y=1.0)
fig.text(0.5, 0.93, 'Attention weights shown in red. Opacity = attention strength.', ha='center', fontsize=10.5, color='#666', style='italic')
save('attention_path.png')

# True-transition path
true_trans_feat = [r for r in attn_feat_gw if r['correct'] and r['target'] not in r['history'] and 6 <= len(r['history']) <= 12]
true_trans_std  = [r for r in attn_std_gw  if r['correct'] and r['target'] not in r['history'] and 6 <= len(r['history']) <= 12]
ex_feat_tt = true_trans_feat[9]
ex_std_tt  = next(
    (r for r in true_trans_std if r['history'] == ex_feat_tt['history'] and r['target'] == ex_feat_tt['target']),
    true_trans_std[9]
)
fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), gridspec_kw={'hspace': 0.6})
max_attn = max(max(ex_std_tt['attn_values']), max(ex_feat_tt['attn_values']))
for ax, (model_name, attn_vals) in zip(axes, [('Standard Model', ex_std_tt['attn_values']), ('Feature Model', ex_feat_tt['attn_values'])]):
    draw_path(ax, ex_feat_tt['history'], attn_vals, ex_feat_tt['target'], max_attn, spacing=1.3, box_w=1.1, fontsize=8.5)
    ax.set_title(model_name, fontsize=14, fontweight='bold', loc='left')
fig.suptitle('True Transition: Predicting a Category Never Seen Before', fontsize=17, fontweight='bold', y=1.0)
fig.text(0.5, 0.93, 'Both models correctly predict the next location — which never appeared in the trajectory.', ha='center', fontsize=10.5, color='#666', style='italic')
save('attention_path_transition.png')


# ═════════════════════════════════════════════════════════════════════════════
print(f'\nDone. All figures saved to: {RESULTS_DIR}/')
print('\nFiles:')
for f in sorted(os.listdir(RESULTS_DIR)):
    size = os.path.getsize(os.path.join(RESULTS_DIR, f))
    print(f'  {f}  ({size // 1024} KB)')
