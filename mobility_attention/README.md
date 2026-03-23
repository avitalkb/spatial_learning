# Mobility Attention

Predict next location category from movement history using a Transformer, then investigate what the attention mechanism actually learns.

For the full story — data, architecture, process, and findings — see [project_story.txt](project_story.txt).

---

## Data
Gowalla (Dallas/Austin) and Foursquare (Washington/Baltimore) check-in datasets.
Loaded automatically from URLs defined in `config.py`.

## Models
- `TrajectoryTransformer` — baseline: category sequence → next category
- `TrajectoryTransformerWithFeatures` — adds hour, day-of-week, time gap, distance

## How to Run

**Train and analyze (first time):**
```bash
# Open analysis_notebook_version.ipynb
# Set LOAD_SAVED = False in the first code cell
# Run all cells — saves results_clean.pkl at the end
```

**Generate all figures from saved results:**
```bash
cd mobility_attention/
python run_analysis.py   # requires results_clean.pkl
# figures saved to results/
```

## Key Findings
- Adding temporal/spatial features changes *how* the model uses attention, not just accuracy
- High-attention samples appear less accurate overall — this is Simpson's Paradox (the model focuses attention on harder cases)
- Standard model uses attention as a lookup; feature model uses it as context reading

## Files
```
config.py                        — URLs, category keywords
data.py                          — data loading, trajectory preparation
model.py                         — Transformer architectures
train.py                         — training loop, split logic
attention.py                     — attention extraction and analysis
visualize.py                     — reusable figure functions
analysis_notebook_version.ipynb  — full analysis notebook (training + exploration)
run_analysis.py                  — generate all figures from saved results
results/                         — generated figures
project_story.txt                — full writeup: architecture, process, findings
```
