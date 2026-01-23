# ============================================================================
# FILE 2: data.py
# ============================================================================

"""Data loading and preprocessing for mobility attention project."""

import pandas as pd
import numpy as np
import ast
from collections import Counter
from torch.utils.data import Dataset
import torch

from config import (FOURSQUARE_URL, GOWALLA_URL, MIN_CATEGORY_COUNT,
                    MIN_TRAJECTORY_LENGTH, MAX_TRAJECTORY_LENGTH, 
                    MAX_SEQ_LEN, CATEGORY_KEYWORDS)


def load_foursquare_data(url=FOURSQUARE_URL):
    """Load Foursquare Washington/Baltimore dataset."""
    df = pd.read_csv(url)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.sort_values(["userid", "timestamp"])
    return df


def load_gowalla_data(url=GOWALLA_URL):
    """Load Gowalla Dallas/Austin dataset."""
    df = pd.read_csv(url)
    
    # Parse nested category JSON
    def extract_category(cat_str):
        try:
            cat_list = ast.literal_eval(cat_str)
            if cat_list and isinstance(cat_list, list):
                return cat_list[0].get("name", "Unknown")
        except:
            pass
        return "Unknown"
    
    df["category"] = df["spot_categ"].apply(extract_category)
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["userid", "timestamp"])
    
    return df


def filter_categories(df, cat_col="category", min_count=MIN_CATEGORY_COUNT):
    """Keep only categories with minimum number of occurrences."""
    cat_counts = df[cat_col].value_counts()
    valid_cats = cat_counts[cat_counts >= min_count].index.tolist()
    return df[df[cat_col].isin(valid_cats)].copy()



def build_trajectories_with_users(df, cat_col="category", min_length=11, max_length=20):
    """Build trajectories, keeping user info."""
    trajectories = []
    user_ids = []
    
    for user_id, user_df in df.groupby("userid"):
        user_df = user_df.sort_values("timestamp")
        categories = user_df[cat_col].tolist()
        
        for i in range(0, len(categories) - min_length + 1, max_length // 2):
            seq = categories[i:i + max_length]
            if len(seq) >= min_length:
                trajectories.append(seq)
                user_ids.append(user_id)
    
    return trajectories, user_ids

def collapse_repeats(trajectory):
    """Remove consecutive duplicate categories."""
    if not trajectory:
        return trajectory
    result = [trajectory[0]]
    for cat in trajectory[1:]:
        if cat != result[-1]:
            result.append(cat)
    return result


def collapse_all_trajectories_with_users(trajectories, user_ids, min_length=11):
    """Collapse repeats and keep user_ids in sync."""
    result_trajs = []
    result_users = []
    
    for traj, user in zip(trajectories, user_ids):
        collapsed = collapse_repeats(traj)
        if len(collapsed) >= min_length:
            result_trajs.append(collapsed)
            result_users.append(user)
    
    return result_trajs, result_users


def build_vocab(categories):
    """Build vocabulary mapping from category list."""
    vocab = {cat: idx for idx, cat in enumerate(categories)}
    vocab["[PAD]"] = len(vocab)
    idx_to_cat = {idx: cat for cat, idx in vocab.items()}
    return vocab, idx_to_cat


def auto_categorize(cat_name):
    """Map category name to high-level group using keywords."""
    cat_lower = cat_name.lower()
    
    for group, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in cat_lower:
                return group
    return "Other"


def build_category_groups(categories):
    """Build mapping from categories to groups."""
    return {cat: auto_categorize(cat) for cat in categories}


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory prediction."""
    
    def __init__(self, trajectories, vocab, seq_len=MAX_SEQ_LEN):
        self.samples = []
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_idx = vocab["[PAD]"]
        
        for traj in trajectories:
            if len(traj) < 2:
                continue
            indices = [vocab.get(cat, self.pad_idx) for cat in traj]
            
            for i in range(1, len(indices)):
                history = indices[:i]
                target = indices[i]
                
                # Pad or truncate history
                if len(history) > seq_len:
                    history = history[-seq_len:]
                else:
                    history = [self.pad_idx] * (seq_len - len(history)) + history
                
                self.samples.append((torch.tensor(history), target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_data(dataset="gowalla", collapse=True):
    """Full data preparation pipeline."""
    
    # Load
    if dataset == "gowalla":
        df = load_gowalla_data()
        cat_col = "category"
    else:
        df = load_foursquare_data()
        cat_col = "spot_categ"
    
    # Filter
    df = filter_categories(df, cat_col=cat_col)
    categories = df[cat_col].unique().tolist()
    
    # Build trajectories
    trajectories,user_ids = build_trajectories_with_users(df, cat_col=cat_col)
    
    if collapse:
        trajectories, user_ids = collapse_all_trajectories_with_users(trajectories, user_ids)
    
    # Vocab
    vocab, idx_to_cat = build_vocab(categories)
    
    # Groups
    cat_to_group = build_category_groups(categories)
    
    # Dataset
    dataset = TrajectoryDataset(trajectories, vocab)
    
    return {
        "df": df,
        "trajectories": trajectories,
        "user_ids": user_ids,
        "categories": categories,
        "vocab": vocab,
        "idx_to_cat": idx_to_cat,
        "cat_to_group": cat_to_group,
        "dataset": dataset
    }
