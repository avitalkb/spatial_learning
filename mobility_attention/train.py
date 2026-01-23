# Update train.py to add soft loss option

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import SoftCrossEntropyLoss, build_similarity_matrix


def train_model(model, train_loader, vocab, epochs=30, lr=0.001, print_every=5, 
                loss_type="hard", cat_to_group=None, temperature=0.3):
    """Train a trajectory prediction model.
    
    Args:
        loss_type: "hard" for standard CE, "soft" for similarity-weighted CE
        cat_to_group: Required if loss_type="soft"
        temperature: Soft loss temperature (lower = sharper)
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pad_idx = vocab["[PAD]"]
    
    # Setup loss
    if loss_type == "soft":
        idx_to_cat = {v: k for k, v in vocab.items() if k != "[PAD]"}
        categories = [idx_to_cat[i] for i in range(len(idx_to_cat))]
        similarity_matrix = build_similarity_matrix(categories, cat_to_group)
        criterion = SoftCrossEntropyLoss(similarity_matrix, temperature=temperature)
        print(f"Using soft loss (temp={temperature})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using hard loss")
    
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x, pad_idx=pad_idx)
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
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2%}")
    
    return history


def evaluate_model(model, test_loader, vocab, idx_to_cat, cat_to_group=None):
    """Evaluate model on test set."""
    
    model.eval()
    pad_idx = vocab["[PAD]"]
    
    correct = 0
    group_correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x, pad_idx=pad_idx)
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

# New split_data function for train.py

def split_data(trajectories, user_ids, vocab, train_ratio=0.8, batch_size=64):
    """Split by user, then create dataloaders."""
    from data import TrajectoryDataset
    
    # Get unique users and split them
    unique_users = list(set(user_ids))
    n_train = int(len(unique_users) * train_ratio)
    
    import random
    random.seed(42)  # Reproducible
    random.shuffle(unique_users)
    train_users = set(unique_users[:n_train])
    
    # Split trajectories by user
    train_trajs = [t for t, u in zip(trajectories, user_ids) if u in train_users]
    test_trajs = [t for t, u in zip(trajectories, user_ids) if u not in train_users]
    
    print(f"Users: {len(train_users)} train, {len(unique_users) - len(train_users)} test")
    print(f"Trajectories: {len(train_trajs)} train, {len(test_trajs)} test")
    
    # Create datasets and loaders
    train_dataset = TrajectoryDataset(train_trajs, vocab)
    test_dataset = TrajectoryDataset(test_trajs, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, train_dataset, test_dataset
