# Update train.py to add soft loss option

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import SoftCrossEntropyLoss, build_similarity_matrix, TrajectoryTransformerWithFeatures


def _forward_batch(model, batch, pad_idx, device):
    """Dispatch forward pass based on batch format.

    2-tuple batch: (cat_idx, target) — standard TrajectoryTransformer
    6-tuple batch: (cat_idx, hour, dow, tgap, dist, target) — WithFeatures
    """
    if len(batch) == 2:
        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x, pad_idx=pad_idx)
    else:
        cat_idx, hour, dow, tgap, dist, batch_y = batch
        cat_idx = cat_idx.to(device)
        hour = hour.to(device)
        dow = dow.to(device)
        tgap = tgap.to(device)
        dist = dist.to(device)
        batch_y = batch_y.to(device)
        logits = model(cat_idx, hour, dow, tgap, dist, pad_idx=pad_idx)
    return logits, batch_y


def train_model(model, train_loader, vocab, epochs=30, lr=0.001, print_every=5,
                loss_type="hard", cat_to_group=None, temperature=0.3,
                test_loader=None):
    """Train a trajectory prediction model.

    Args:
        loss_type: "hard" for standard CE, "soft" for similarity-weighted CE
        cat_to_group: Required if loss_type="soft"
        temperature: Soft loss temperature (lower = sharper)
        test_loader: If provided, evaluate test accuracy each epoch
    """

    # Device selection - CPU is faster for small models!
    # MPS/CUDA only beneficial for large batch sizes (512+) and large models
    device = torch.device("cpu")  # CPU is faster for this small model
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pad_idx = vocab["[PAD]"]

    # Setup loss
    if loss_type == "soft":
        idx_to_cat = {v: k for k, v in vocab.items() if k != "[PAD]"}
        categories = [idx_to_cat[i] for i in range(len(idx_to_cat))]
        similarity_matrix = build_similarity_matrix(categories, cat_to_group)
        criterion = SoftCrossEntropyLoss(similarity_matrix, temperature=temperature).to(device)
        print(f"Using soft loss (temp={temperature})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using hard loss")

    history = {"loss": [], "accuracy": [], "test_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            logits, batch_y = _forward_batch(model, batch, pad_idx, device)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        # Test accuracy
        test_acc = None
        if test_loader is not None:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch in test_loader:
                    logits, batch_y = _forward_batch(model, batch, pad_idx, device)
                    test_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    test_total += len(batch_y)
            test_acc = test_correct / test_total
            history["test_accuracy"].append(test_acc)

        if (epoch + 1) % print_every == 0:
            msg = f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train: {epoch_acc:.2%}"
            if test_acc is not None:
                msg += f" - Test: {test_acc:.2%}"
            print(msg)

    return history


def evaluate_model(model, test_loader, vocab, idx_to_cat, cat_to_group=None):
    """Evaluate model on test set."""

    model.eval()
    pad_idx = vocab["[PAD]"]
    device = next(model.parameters()).device

    correct = 0
    group_correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            logits, batch_y = _forward_batch(model, batch, pad_idx, device)
            preds = logits.argmax(dim=1)

            for pred, target in zip(preds, batch_y):
                pred_cat = idx_to_cat[pred.item()]
                target_cat = idx_to_cat[target.item()]

                is_correct = pred.item() == target.item()
                correct += is_correct

                if cat_to_group:
                    pred_group = cat_to_group.get(pred_cat, "Other")
                    target_group = cat_to_group.get(target_cat, "Other")
                    group_correct += (pred_group == target_group)

                predictions.append({
                    "predicted": pred_cat,
                    "target": target_cat,
                    "correct": is_correct
                })

                total += 1

    results = {
        "exact_accuracy": correct / total,
        "total": total,
        "predictions": predictions
    }

    if cat_to_group:
        results["group_accuracy"] = group_correct / total

    return results


def split_data(trajectories, user_ids, vocab, train_ratio=0.8, batch_size=64,
               stratified=True, feature_trajs=None):
    """Split by user with stratification, then create dataloaders.

    Args:
        stratified: If True, stratify by user activity level to ensure
                   train/test have similar user distributions.
        feature_trajs: If provided, split features alongside trajectories
                      and create TrajectoryDatasetWithFeatures instances.
    """
    from data import TrajectoryDataset, TrajectoryDatasetWithFeatures
    from collections import Counter
    import numpy as np
    import random

    random.seed(42)
    np.random.seed(42)

    # Count trajectories per user
    user_traj_counts = Counter(user_ids)
    unique_users = list(user_traj_counts.keys())

    if stratified:
        # Stratify by activity level (number of trajectories per user)
        # Bin users into quartiles
        counts = np.array([user_traj_counts[u] for u in unique_users])
        quartiles = np.percentile(counts, [25, 50, 75])

        def get_bin(count):
            if count <= quartiles[0]:
                return 0  # Low activity
            elif count <= quartiles[1]:
                return 1  # Medium-low
            elif count <= quartiles[2]:
                return 2  # Medium-high
            else:
                return 3  # High activity

        # Group users by activity bin
        bins = {0: [], 1: [], 2: [], 3: []}
        for user in unique_users:
            bins[get_bin(user_traj_counts[user])].append(user)

        # Sample proportionally from each bin
        train_users = []
        for bin_users in bins.values():
            random.shuffle(bin_users)
            n_train = int(len(bin_users) * train_ratio)
            train_users.extend(bin_users[:n_train])

        train_users = set(train_users)

        # Report stratification
        train_counts = [user_traj_counts[u] for u in train_users]
        test_counts = [user_traj_counts[u] for u in unique_users if u not in train_users]
        print(f"Stratified split by user activity:")
        print(f"  Train users: mean={np.mean(train_counts):.1f} trajs/user, median={np.median(train_counts):.1f}")
        print(f"  Test users:  mean={np.mean(test_counts):.1f} trajs/user, median={np.median(test_counts):.1f}")
    else:
        # Simple random split (original behavior)
        random.shuffle(unique_users)
        n_train = int(len(unique_users) * train_ratio)
        train_users = set(unique_users[:n_train])

    # Split trajectories (and features) by user
    train_trajs = []
    test_trajs = []
    train_feats = [] if feature_trajs is not None else None
    test_feats = [] if feature_trajs is not None else None

    for i, (t, u) in enumerate(zip(trajectories, user_ids)):
        if u in train_users:
            train_trajs.append(t)
            if feature_trajs is not None:
                train_feats.append(feature_trajs[i])
        else:
            test_trajs.append(t)
            if feature_trajs is not None:
                test_feats.append(feature_trajs[i])

    print(f"Users: {len(train_users)} train, {len(unique_users) - len(train_users)} test")
    print(f"Trajectories: {len(train_trajs)} train, {len(test_trajs)} test")

    # Create datasets and loaders
    if feature_trajs is not None:
        train_dataset = TrajectoryDatasetWithFeatures(train_trajs, train_feats, vocab)
        test_dataset = TrajectoryDatasetWithFeatures(test_trajs, test_feats, vocab)
    else:
        train_dataset = TrajectoryDataset(train_trajs, vocab)
        test_dataset = TrajectoryDataset(test_trajs, vocab)

    # DataLoader - keep simple for small datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, train_dataset, test_dataset
