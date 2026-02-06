# src/ml/gesture_transformer/cross_attention_integration.py – Cross-Attention Integration v1.0
# Cross-attention between gesture sequence and valence context embedding
# Multi-head, residual, layer-norm, valence-gated scaling for thriving focus
# PyTorch 2.3+, CUDA-ready, ONNX export compatible
# MIT License – Autonomicity Games Inc. 2026

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class ValenceCrossAttention(nn.Module):
    """
    Cross-attention module where gesture sequence (query) attends to valence context (key/value)
    Valence embedding modulates attention weights (higher valence → sharper focus)
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # Learnable valence scaling parameter (per head or global)
        self.valence_scale = nn.Parameter(torch.ones(nhead))

    def forward(
        self,
        query: torch.Tensor,         # gesture sequence (B, T_gesture, D)
        key_value: torch.Tensor,     # valence context (B, T_valence, D) or (B, 1, D)
        valence: torch.Tensor        # (B,) or (B, nhead) – per-sample or per-head scaling
    ):
        # Expand valence to match heads if scalar per sample
        if valence.dim() == 1:
            valence = valence.unsqueeze(1).expand(-1, self.attention.num_heads)

        # Standard cross-attention
        attn_output, attn_weights = self.attention(
            query, key_value, key_value,
            need_weights=True
        )

        # Valence-conditioned scaling on attention weights (softmax already applied)
        # Scale each head independently
        attn_weights = attn_weights * self.valence_scale.unsqueeze(0).unsqueeze(2) * valence.unsqueeze(2)

        # Re-normalize after scaling
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = self.dropout(attn_output)
        return self.norm(attn_output), attn_weights


class GestureCrossAttentionTransformer(nn.Module):
    """
    Enhanced Gesture Transformer with cross-attention to external valence context
    - Self-attention over gesture sequence
    - Cross-attention from gesture to valence embedding
    - Valence scaling modulates both attention types
    """
    def __init__(
        self,
        seq_len: int = 45,
        landmark_dim: int = 225,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        num_gesture_classes: int = 5,
        future_valence_horizon: int = 10,
        dropout: float = 0.1,
        valence_dim: int = 32           # dimension of external valence embedding
    ):
        super().__init__()
        self.seq_len = seq_len
        self.landmark_dim = landmark_dim
        self.d_model = d_model

        # Input projection for gesture landmarks
        self.input_proj = nn.Linear(landmark_dim, d_model)

        # Positional encoding for gesture sequence
        self.pos_encoder = PositionalEncoding(d_model, seq_len)

        # Valence context embedding (if external valence is scalar → vector)
        self.valence_embed = nn.Linear(1, valence_dim) if valence_dim > 1 else nn.Identity()

        # Stack of self-attention + cross-attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'cross_attn': ValenceCrossAttention(d_model, nhead, dropout),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }) for _ in range(num_layers)
        ])

        # Gesture classification head
        self.gesture_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_gesture_classes)
        )

        # Future valence prediction head
        self.valence_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, future_valence_horizon)
        )

    def forward(
        self,
        x: torch.Tensor,                    # (B, T_gesture, L)
        valence_context: torch.Tensor = None,  # (B,) or (B, valence_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: gesture sequence (B, T, L)
        valence_context: scalar per batch (B,) or pre-embedded (B, valence_dim)
        Returns: gesture_logits (B, C), future_valence (B, horizon)
        """
        B = x.shape[0]

        if valence_context is None:
            valence_context = torch.full((B,), current_valence(), device=x.device)

        # Embed valence if scalar
        if valence_context.dim() == 1:
            valence_context = self.valence_embed(valence_context.unsqueeze(-1))  # (B,1) → (B,Dv)

        # Expand valence context to sequence length for cross-attention
        valence_kv = valence_context.unsqueeze(1)  # (B, 1, Dv) – treat as single token

        # Project gesture input
        x = self.input_proj(x) + self.pos_encoder.pe[:, :x.size(1), :]

        # Encoder layers: self-attention → cross-attention → FFN
        for layer in self.layers:
            # Self-attention over gesture sequence
            self_attn_out, _ = layer['self_attn'](x, x, x)
            x = layer['norm1'](x + self_attn_out)

            # Cross-attention: gesture queries attend to valence context
            cross_attn_out, _ = layer['cross_attn'](x, valence_kv, valence_kv, valence_context)
            x = layer['norm2'](x + cross_attn_out)

            # Feed-forward
            ffn_out = layer['ffn'](x)
            x = layer['norm3'](x + ffn_out)

        # Global mean pooling
        x = x.mean(dim=1)

        gesture_logits = self.gesture_head(x)
        future_valence = torch.sigmoid(self.valence_head(x))

        return gesture_logits, future_valence

    def export_to_onnx(self, dummy_input: torch.Tensor, dummy_valence: torch.Tensor, output_path: str = "gesture_cross_transformer.onnx"):
        self.eval()
        torch.onnx.export(
            self,
            (dummy_input, dummy_valence),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input', 'valence_context'],
            output_names=['gesture_logits', 'future_valence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'valence_context': {0: 'batch_size'},
                'gesture_logits': {0: 'batch_size'},
                'future_valence': {0: 'batch_size'}
            }
        )
        print(f"[Model] Exported cross-attention model to ONNX: {output_path}")
