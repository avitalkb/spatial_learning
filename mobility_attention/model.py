# ============================================================================
# FILE 3: model.py
# ============================================================================

"""Model architectures for mobility prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBED_DIM, N_HEADS, N_LAYERS, HOUR_EMBED_DIM, DOW_EMBED_DIM


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


class TrajectoryTransformerWithFeatures(nn.Module):
    """Transformer for next-location prediction with temporal/spatial features.

    Concatenates category embedding (embed_dim) + hour embedding (8) +
    day-of-week embedding (4) + time_gap (1) + distance (1), then projects
    back to embed_dim before feeding into the same transformer encoder.
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
                 hour_embed_dim=HOUR_EMBED_DIM, dow_embed_dim=DOW_EMBED_DIM):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
        self.pos_embedding = nn.Embedding(20, embed_dim)

        self.hour_embedding = nn.Embedding(24, hour_embed_dim)
        self.dow_embedding = nn.Embedding(7, dow_embed_dim)

        concat_dim = embed_dim + hour_embed_dim + dow_embed_dim + 2  # +2 for tgap, dist
        self.input_proj = nn.Linear(concat_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, vocab_size - 1)  # Exclude PAD

    def forward(self, cat_idx, hour, dow, tgap, dist, pad_idx):
        # Embeddings
        token_emb = self.embedding(cat_idx)                       # (B, T, embed_dim)
        hour_emb = self.hour_embedding(hour)                      # (B, T, 8)
        dow_emb = self.dow_embedding(dow)                         # (B, T, 4)
        tgap_feat = tgap.unsqueeze(-1)                            # (B, T, 1)
        dist_feat = dist.unsqueeze(-1)                            # (B, T, 1)

        combined = torch.cat([token_emb, hour_emb, dow_emb, tgap_feat, dist_feat], dim=-1)
        x_proj = self.input_proj(combined)                        # (B, T, embed_dim)

        positions = torch.arange(cat_idx.shape[1], device=cat_idx.device)
        positions = positions.unsqueeze(0).expand(cat_idx.shape[0], -1)
        pos_emb = self.pos_embedding(positions)
        x_emb = x_proj + pos_emb

        # Transformer with padding mask
        pad_mask = (cat_idx == pad_idx)
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
        # Register as buffer so it moves to GPU with model
        self.register_buffer('similarity_matrix', similarity_matrix)
        self.temperature = temperature

    def forward(self, logits, targets):
        n_classes = self.similarity_matrix.shape[0]

        # Clamp targets to valid range
        valid_targets = targets.clamp(max=n_classes - 1)

        # VECTORIZED: Index all targets at once (no loop!)
        sim_rows = self.similarity_matrix[valid_targets]  # (batch_size, n_classes)

        # Compute soft targets in one go
        soft_targets = torch.exp(-sim_rows / self.temperature)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

        # Pad to match logits size if needed
        if soft_targets.shape[1] < logits.shape[1]:
            padding = torch.zeros(soft_targets.shape[0], logits.shape[1] - soft_targets.shape[1],
                                  device=soft_targets.device)
            soft_targets = torch.cat([soft_targets, padding], dim=1)

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
