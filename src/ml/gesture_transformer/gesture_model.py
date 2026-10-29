# src/ml/gesture_transformer/gesture_model.py – Spatiotemporal Transformer v1.2
# Multi-head self-attention over time + landmarks, positional encoding,
# valence-conditioned attention scaling, residual connections, layer norm
# PyTorch 2.3+, CUDA-ready, ONNX exportable
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


class ValenceConditionedAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.valence_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valence: torch.Tensor = None
    ):
        attn_output, attn_weights = self.attention(query, key, value)

        if valence is not None:
            scale_factor = 0.5 + 1.5 * valence.mean()
            attn_weights = attn_weights * scale_factor

        attn_output = self.dropout(attn_output)
        return self.norm(attn_output), attn_weights


class GestureTransformer(nn.Module):
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
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.landmark_dim = landmark_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(landmark_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.gesture_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_gesture_classes)
        )

        self.valence_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, future_valence_horizon)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        x = self.input_proj(x) + self.pos_encoder.pe[:, :x.size(1), :]

        x = self.transformer_encoder(x)

        x = x.mean(dim=1)

        gesture_logits = self.gesture_head(x)
        future_valence = torch.sigmoid(self.valence_head(x))

        return gesture_logits, future_valence

    def export_to_onnx(self, dummy_input: torch.Tensor, output_path: str = "gesture_transformer.onnx"):
        self.eval()
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['gesture_logits', 'future_valence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'gesture_logits': {0: 'batch_size'},
                'future_valence': {0: 'batch_size'}
            }
        )
        print(f"Model exported to {output_path}")
