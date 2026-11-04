import torch
from torch import nn

class ViewSetPooling(nn.Module):
    """Drop-in replacement for ViewSetAggregator — same forward signature,
    (B, N, D) x (B, N) view_type_ids x (B, N) valid_mask -> (logit, weights).
    Replaces the transformer + gated-attention MIL with simple pooling
    over the per-view feature vectors.
    """
    def __init__(self, feat_dim, n_view_types=6, dropout=0.3,
                 pool_type='gem', gem_p=3.0, learnable_p=False):
        super().__init__()
        self.pool_type = pool_type  # 'max' | 'mean' | 'gem'
        self.view_type_embed = nn.Embedding(n_view_types, feat_dim)
        if pool_type == 'gem':
            init_p = torch.tensor([gem_p], dtype=torch.float32)
            self.gem_p = nn.Parameter(init_p) if learnable_p else init_p

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x, view_type_ids, valid_mask):
        x = x + self.view_type_embed(view_type_ids)          # (B, N, D)
        mask = valid_mask.unsqueeze(-1)                        # (B, N, 1)
        counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).float()

        if self.pool_type == 'max':
            pooled = x.masked_fill(~mask, float('-inf')).max(dim=1).values
            pooled = torch.nan_to_num(pooled, neginf=0.0)      # guards the rare all-invalid row

        elif self.pool_type == 'mean':
            pooled = x.masked_fill(~mask, 0.0).sum(dim=1) / counts

        elif self.pool_type == 'gem':
            p = self.gem_p.to(x.device).clamp(min=1.0, max=8.0)  # keep p in a stable range
            x_pow = x.clamp(min=1e-6).pow(p)                    # GeM needs non-negative input
            pooled = (x_pow.masked_fill(~mask, 0.0).sum(dim=1) / counts).clamp(min=1e-6).pow(1.0 / p)

        else:
            raise ValueError(f"unknown pool_type {self.pool_type}")

        logit = self.head(pooled).squeeze(-1)
        weights = None  
        return logit, weights
        