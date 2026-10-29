# src/ml/gesture_transformer/linformer_attention.py – Linformer Attention v1.0
# Linear-space self-attention via low-rank approximation of attention matrix
# O(N) time & space instead of O(N²), valence-modulated projection scaling
# PyTorch 2.3+, CUDA-ready, ONNX export compatible
# MIT License – Autonomicity Games Inc. 2026

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinformerAttention(nn.Module):
    """
    Linformer-style linear attention using low-rank projection of keys & values
    - Reduces attention complexity from O(N²) to O(N·k) where k = projection dim
    - Valence modulates projection rank & kernel sharpness
    - Supports causal masking for autoregressive use
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        seq_len: int,
        proj_dim: int = 64,           # low-rank projection dimension k << N
        dropout: float = 0.1,
        causal: bool = False,
        valence_mod: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.seq_len = seq_len
        self.proj_dim = proj_dim
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.valence_mod = valence_mod

        # Low-rank projection matrices E & F (shared across heads or per-head)
        self.E = nn.Parameter(torch.randn(nhead, seq_len, proj_dim) * 0.02)
        self.F = nn.Parameter(torch.randn(nhead, seq_len, proj_dim) * 0.02)

        # Learnable valence scaling (sharper projection when valence high)
        self.valence_scale = nn.Parameter(torch.ones(1)) if valence_mod else None

        self.out_proj = nn.Linear(d_model, d_model)

    def _project_kv(self, k: torch.Tensor, v: torch.Tensor, valence: torch.Tensor = None):
        """
        k, v: (B, H, N, d_head)
        Returns projected: (B, H, k, d_head)
        """
        B, H, N, d = k.shape

        # Project keys & values to low-rank space
        k_proj = torch.einsum('b h n d, h n k -> b h k d', k, self.E[:H, :N, :self.proj_dim])
        v_proj = torch.einsum('b h n d, h n k -> b h k d', v, self.F[:H, :N, :self.proj_dim])

        # Optional valence modulation (sharper projection when valence high)
        if self.valence_scale is not None and valence is not None:
            scale = 1.0 + self.valence_scale * valence.mean()
            k_proj = k_proj * scale
            v_proj = v_proj * scale

        return k_proj, v_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valence: torch.Tensor = None,
        attn_mask: torch.Tensor = None
    ):
        """
        query, key, value: (B, N, D)
        valence: (B,) or scalar
        Returns: (B, N, D), approximate attention weights
        """
        B, N, D = query.shape
        H = self.nhead
        d = self.head_dim

        # Split heads
        Q = query.view(B, N, H, d).transpose(1, 2)   # (B,H,N,d)
        K = key.view(B, N, H, d).transpose(1, 2)
        V = value.view(B, N, H, d).transpose(1, 2)

        # Project K & V to low-rank space
        K_proj, V_proj = self._project_kv(K, V, valence)

        # Attention scores: Q · (K_proj)^T
        scores = torch.einsum('b h n d, b h m d -> b h n m', Q, K_proj)  # (B,H,N,k)

        # Optional causal masking
        if self.causal:
            mask = torch.triu(torch.ones(N, N, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over projected dimension
        attn_weights = F.softmax(scores, dim=-1)  # (B,H,N,k)

        # Output: attn_weights · V_proj
        out = torch.einsum('b h n m, b h m d -> b h n d', attn_weights, V_proj)  # (B,H,N,d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # Final projection
        out = self.dropout(out)
        out = self.out_proj(out)

        # Approximate full attention weights (for debugging)
        approx_weights = torch.einsum('b h n m, b h m n -> b h n n', attn_weights, K_proj @ K_proj.transpose(-1, -2))
        approx_weights = F.softmax(approx_weights, dim=-1)

        return out, approx_weights.mean(dim=1)  # average over heads

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, nhead={self.nhead}, proj_dim={self.proj_dim}, causal={self.causal}'
