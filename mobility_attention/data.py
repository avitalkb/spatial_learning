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

import math

from config import (FOURSQUARE_URL, GOWALLA_URL, MIN_CATEGORY_COUNT,
                    MIN_TRAJECTORY_LENGTH, MAX_TRAJECTORY_LENGTH,
                    MAX_SEQ_LEN, CATEGORY_KEYWORDS, HOUR_EMBED_DIM, DOW_EMBED_DIM)


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



def build_trajectories_with_users(df, cat_col="category", min_length=11, max_length=20,
                                   include_features=False):
    """Build trajectories, keeping user info.

    Args:
        include_features: If True, also return timestamps and coordinates
                         alongside categories for temporal/spatial features.
    """
    trajectories = []
    user_ids = []
    feature_trajs = [] if include_features else None

    for user_id, user_df in df.groupby("userid"):
        user_df = user_df.sort_values("timestamp")
        categories = user_df[cat_col].tolist()

        if include_features:
            timestamps = user_df["timestamp"].tolist()
            lats = user_df["lat"].tolist()
            lngs = user_df["lng"].tolist()

        for i in range(0, len(categories) - min_length + 1, max_length // 2):
            seq = categories[i:i + max_length]
            if len(seq) >= min_length:
                trajectories.append(seq)
                user_ids.append(user_id)
                if include_features:
                    feature_trajs.append({
                        "timestamps": timestamps[i:i + max_length],
                        "lats": lats[i:i + max_length],
                        "lngs": lngs[i:i + max_length],
                    })

    return trajectories, user_ids, feature_trajs

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


def collapse_with_features(trajectory, feature_traj):
    """Full collapse keeping features from the first occurrence of each run.

    [x, x, x, y, y, z] -> [x, y, z] with timestamps/coords of the first x, first y, z.
    """
    if not trajectory:
        return trajectory, feature_traj

    result = []
    feat_result = {"timestamps": [], "lats": [], "lngs": []}

    result.append(trajectory[0])
    feat_result["timestamps"].append(feature_traj["timestamps"][0])
    feat_result["lats"].append(feature_traj["lats"][0])
    feat_result["lngs"].append(feature_traj["lngs"][0])

    for i in range(1, len(trajectory)):
        if trajectory[i] != trajectory[i - 1]:
            result.append(trajectory[i])
            feat_result["timestamps"].append(feature_traj["timestamps"][i])
            feat_result["lats"].append(feature_traj["lats"][i])
            feat_result["lngs"].append(feature_traj["lngs"][i])

    return result, feat_result


def collapse_all_with_features(trajectories, user_ids, feature_trajs, min_length=3):
    """Full collapse all trajectories, keeping features from first occurrence."""
    result_trajs = []
    result_users = []
    result_feats = []

    for idx, (traj, user) in enumerate(zip(trajectories, user_ids)):
        collapsed, collapsed_ft = collapse_with_features(traj, feature_trajs[idx])
        if len(collapsed) >= min_length:
            result_trajs.append(collapsed)
            result_users.append(user)
            result_feats.append(collapsed_ft)

    return result_trajs, result_users, result_feats


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


def haversine_km(lat1, lng1, lat2, lng2):
    """Calculate distance in km between two lat/lng points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_temporal_spatial_features(feature_trajs):
    """Convert raw timestamps/coords into per-position features.

    For each trajectory, produces parallel lists:
        hours:    int 0-23
        dows:     int 0-6  (Monday=0)
        tgaps:    float in [0, 1], time gap since previous check-in (capped at 168h)
        dists:    float in [0, 1], distance from previous check-in (capped at 100km)

    First position gets 0.0 for tgap and dist.
    """
    MAX_HOURS = 168.0   # 1 week
    MAX_KM = 100.0

    processed = []
    for ft in feature_trajs:
        ts_list = ft["timestamps"]
        lats = ft["lats"]
        lngs = ft["lngs"]
        n = len(ts_list)

        hours = [t.hour for t in ts_list]
        dows = [t.weekday() for t in ts_list]

        tgaps = [0.0]
        dists = [0.0]
        for j in range(1, n):
            # Time gap in hours, capped and normalized
            delta_h = (ts_list[j] - ts_list[j - 1]).total_seconds() / 3600.0
            tgaps.append(min(delta_h, MAX_HOURS) / MAX_HOURS)

            # Distance in km, capped and normalized
            d = haversine_km(lats[j - 1], lngs[j - 1], lats[j], lngs[j])
            dists.append(min(d, MAX_KM) / MAX_KM)

        processed.append({
            "hours": hours,
            "dows": dows,
            "tgaps": tgaps,
            "dists": dists,
        })
    return processed


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


class TrajectoryDatasetWithFeatures(Dataset):
    """PyTorch Dataset for trajectory prediction with temporal/spatial features.

    Each sample is a 6-tuple:
        (cat_idx, hour, dow, tgap, dist, target)
    """

    def __init__(self, trajectories, feature_data, vocab, seq_len=MAX_SEQ_LEN):
        self.samples = []
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_idx = vocab["[PAD]"]

        for traj, feats in zip(trajectories, feature_data):
            if len(traj) < 2:
                continue
            indices = [vocab.get(cat, self.pad_idx) for cat in traj]
            hours = feats["hours"]
            dows = feats["dows"]
            tgaps = feats["tgaps"]
            dists = feats["dists"]

            for i in range(1, len(indices)):
                history = indices[:i]
                h_hours = hours[:i]
                h_dows = dows[:i]
                h_tgaps = tgaps[:i]
                h_dists = dists[:i]
                target = indices[i]

                # Pad or truncate history
                if len(history) > seq_len:
                    history = history[-seq_len:]
                    h_hours = h_hours[-seq_len:]
                    h_dows = h_dows[-seq_len:]
                    h_tgaps = h_tgaps[-seq_len:]
                    h_dists = h_dists[-seq_len:]
                else:
                    pad_len = seq_len - len(history)
                    history = [self.pad_idx] * pad_len + history
                    h_hours = [0] * pad_len + h_hours
                    h_dows = [0] * pad_len + h_dows
                    h_tgaps = [0.0] * pad_len + h_tgaps
                    h_dists = [0.0] * pad_len + h_dists

                self.samples.append((
                    torch.tensor(history),
                    torch.tensor(h_hours),
                    torch.tensor(h_dows),
                    torch.tensor(h_tgaps, dtype=torch.float32),
                    torch.tensor(h_dists, dtype=torch.float32),
                    target,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_data(dataset="gowalla", collapse=True, group_level=False,
                  include_features=False):
    """Full data preparation pipeline.

    Args:
        group_level: If True, convert trajectories to group-level categories
                     (e.g. "Chipotle" → "Food") before building vocab.
        include_features: If True, extract temporal/spatial features.
        collapse: If True, remove consecutive duplicates. Applied to both
                  standard and feature pipelines (feature pipeline keeps
                  timestamps/coords from the first occurrence of each run).
    """

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

    # Build category-to-group mapping (always needed)
    cat_to_group = build_category_groups(categories)

    # Build trajectories
    trajectories, user_ids, feature_trajs = build_trajectories_with_users(
        df, cat_col=cat_col, include_features=include_features
    )

    if collapse:
        if feature_trajs is not None:
            trajectories, user_ids, feature_trajs = collapse_all_with_features(
                trajectories, user_ids, feature_trajs, min_length=3
            )
        else:
            trajectories, user_ids = collapse_all_trajectories_with_users(
                trajectories, user_ids, min_length=3
            )

    if group_level:
        # Convert trajectories from categories to groups
        trajectories = [
            [cat_to_group.get(cat, "Other") for cat in traj]
            for traj in trajectories
        ]
        # Re-collapse since group mapping creates new consecutive duplicates
        if collapse:
            if feature_trajs is not None:
                trajectories, user_ids, feature_trajs = collapse_all_with_features(
                    trajectories, user_ids, feature_trajs, min_length=3
                )
            else:
                trajectories, user_ids = collapse_all_trajectories_with_users(
                    trajectories, user_ids, min_length=3
                )
        # Build vocab from groups
        groups = sorted(set(cat_to_group.values()))
        vocab, idx_to_cat = build_vocab(groups)
        categories = groups
    else:
        vocab, idx_to_cat = build_vocab(categories)

    # Extract temporal/spatial features
    processed_features = None
    if include_features and feature_trajs is not None:
        processed_features = extract_temporal_spatial_features(feature_trajs)

    # Dataset
    if include_features and processed_features is not None:
        ds = TrajectoryDatasetWithFeatures(trajectories, processed_features, vocab)
    else:
        ds = TrajectoryDataset(trajectories, vocab)

    result = {
        "df": df,
        "trajectories": trajectories,
        "user_ids": user_ids,
        "categories": categories,
        "vocab": vocab,
        "idx_to_cat": idx_to_cat,
        "cat_to_group": cat_to_group,
        "dataset": ds,
    }
    if include_features:
        result["feature_trajs"] = processed_features
    return result
