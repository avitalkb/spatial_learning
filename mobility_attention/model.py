# ============================================================================
# FILE 3: model.py
# ============================================================================

"""Model architectures for mobility prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBED_DIM, N_HEADS, N_LAYERS


class TrajectoryTransformer(nn.Module):
    """Transformer for next-location prediction."""
    
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.pos_embedding = nn.Embedding(20, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, vocab_size - 1)  # Exclude PAD
        
    def forward(self, x, pad_idx):
        # Embeddings
        token_emb = self.embedding(x)
        positions = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
        pos_emb = self.pos_embedding(positions)
        x_emb = token_emb + pos_emb
        
        # Transformer with padding mask
        pad_mask = (x == pad_idx)
        out = self.transformer(x_emb, src_key_padding_mask=pad_mask)
        
        # Predict from last position
        return self.classifier(out[:, -1, :])


class HierarchicalTransformer(nn.Module):
    """Two-stage transformer: predict group, then category within group."""
    
    def __init__(self, vocab_size, n_groups, group_sizes, 
                 embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.pos_embedding = nn.Embedding(20, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Stage 1: Group classifier
        self.group_classifier = nn.Linear(embed_dim, n_groups)
        
        # Stage 2: Category classifier per group
        self.category_classifiers = nn.ModuleDict({
            str(i): nn.Linear(embed_dim, size) 
            for i, size in enumerate(group_sizes)
        })
        
    def forward(self, x, pad_idx):
        token_emb = self.embedding(x)
        positions = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
        pos_emb = self.pos_embedding(positions)
        x_emb = token_emb + pos_emb
        
        pad_mask = (x == pad_idx)
        out = self.transformer(x_emb, src_key_padding_mask=pad_mask)
        
        last_hidden = out[:, -1, :]
        group_logits = self.group_classifier(last_hidden)
        
        return group_logits, last_hidden
    
    def predict_category(self, hidden, group_idx):
        return self.category_classifiers[str(group_idx)](hidden)


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy with soft labels based on category similarity."""
    
    def __init__(self, similarity_matrix, temperature=0.3):
        super().__init__()
        self.similarity_matrix = similarity_matrix
        self.temperature = temperature
        
    def forward(self, logits, targets):
        batch_size = targets.shape[0]
        n_classes = logits.shape[1]
        
        soft_targets = torch.zeros(batch_size, n_classes)
        
        for i, target in enumerate(targets):
            if target.item() < self.similarity_matrix.shape[0]:
                sim = self.similarity_matrix[target.item()]
                probs = torch.exp(-sim / self.temperature)
                probs = probs / probs.sum()
                soft_targets[i, :len(probs)] = probs
            else:
                soft_targets[i, target] = 1.0
        
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        
        return loss


def build_similarity_matrix(categories, cat_to_group):
    """Build category similarity matrix based on groups."""
    n_cats = len(categories)
    similarity = torch.ones(n_cats, n_cats)
    
    for i, cat_i in enumerate(categories):
        for j, cat_j in enumerate(categories):
            group_i = cat_to_group.get(cat_i, "Other")
            group_j = cat_to_group.get(cat_j, "Other")
            
            if cat_i == cat_j:
                similarity[i, j] = 0.0
            elif group_i == group_j and group_i != "Other":
                similarity[i, j] = 0.3
            else:
                similarity[i, j] = 1.0
    
    return similarity
