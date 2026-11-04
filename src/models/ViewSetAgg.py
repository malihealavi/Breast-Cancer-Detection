import torch
from torch import nn

class ViewSetAggregator(nn.Module):
    """Variable-N set of per-view encodings -> single logit."""
    def __init__(self, feat_dim, n_view_types=6, attn_dim=128,
                 use_self_attn=True, n_heads=4, n_layers=1, dropout=0.3):
        super().__init__()
        self.view_type_embed = nn.Embedding(n_view_types, feat_dim)  # 0=CC, 1=MLO
        self.use_self_attn = use_self_attn
        if use_self_attn:
            layer = nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=n_heads, dim_feedforward=feat_dim * 2,
                dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
            )
            self.self_attn = nn.TransformerEncoder(layer, num_layers=n_layers)

        # gated attention MIL pooling
        self.attn_V = nn.Linear(feat_dim, attn_dim)
        self.attn_U = nn.Linear(feat_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x, view_type_ids, valid_mask):
        # x: (B, MAX_VIEWS, D); view_type_ids: (B, MAX_VIEWS) long
        # valid_mask: (B, MAX_VIEWS) bool, True = real image
        x = x + self.view_type_embed(view_type_ids)

        if self.use_self_attn:
            x = self.self_attn(x, src_key_padding_mask=~valid_mask)  # True = ignore

        A_V = torch.tanh(self.attn_V(x))
        A_U = torch.sigmoid(self.attn_U(x))
        scores = self.attn_w(A_V * A_U).squeeze(-1)            # (B, MAX_VIEWS)
        scores = scores.masked_fill(~valid_mask, -1e4)         # -1e4, not -inf (NaN risk on TPU)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        pooled = (weights * x).sum(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return logit, weights.squeeze(-1)
