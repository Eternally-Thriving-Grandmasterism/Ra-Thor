# src/ml/gesture_transformer/gesture_model.py – Spatiotemporal Transformer v1.0
# Gesture recognition + future valence prediction
# PyTorch 2.3+, CUDA-ready, ONNX exportable
# MIT License – Autonomicity Games Inc. 2026

import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 45,
        landmark_dim: int = 225,           # 33 pose * 3 + 21 left hand * 3 + 21 right hand * 3
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_gesture_classes: int = 5,      # none, pinch, spiral, figure8, etc.
        future_valence_horizon: int = 10   # predict next 10 steps valence
    ):
        super().__init__()
        self.seq_len = seq_len
        self.landmark_dim = landmark_dim
        self.d_model = d_model

        # Input embedding
        self.input_proj = nn.Linear(landmark_dim, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Gesture classification head
        self.gesture_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_gesture_classes)
        )

        # Future valence prediction head (sequence of scalars)
        self.valence_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, future_valence_horizon)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, landmark_dim)
        Returns:
            gesture_logits: (batch, num_gesture_classes)
            future_valence: (batch, future_horizon)
        """
        batch_size = x.shape[0]

        # Project landmarks → d_model
        x = self.input_proj(x)                  # (B, S, D)
        x = x + self.pos_encoder                # add positional encoding

        # Transformer encoder
        x = self.transformer_encoder(x)         # (B, S, D)

        # Global average pooling over sequence
        x = x.mean(dim=1)                       # (B, D)

        # Heads
        gesture_logits = self.gesture_head(x)   # (B, C)
        future_valence = torch.sigmoid(self.valence_head(x))  # (B, horizon) ∈ [0,1]

        return gesture_logits, future_valence

    def export_to_onnx(self, dummy_input: torch.Tensor, output_path: str = "gesture_transformer.onnx"):
        """Export model to ONNX for WebNN / TensorRT inference"""
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
        print(f"[Model] Exported to ONNX: {output_path}")
