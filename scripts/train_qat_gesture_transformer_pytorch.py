# scripts/train_qat_gesture_transformer_pytorch.py – QAT + Valence-Weighted KD Training v1.1
# Full gradient computation + backprop flow for spatiotemporal gesture transformer
# MIT License – Autonomicity Games Inc. 2026

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from typing import Tuple, Dict

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────────────────────
# Model – Placeholder (replace with your real transformer)
# ──────────────────────────────────────────────────────────────

class GestureTransformer(nn.Module):
    """Simplified placeholder – real model has attention + temporal layers"""
    def __init__(self, seq_len=45, landmark_dim=225, num_classes=4):
        super().__init__()
        self.quant = QuantStub()
        self.encoder = nn.Sequential(
            nn.Linear(landmark_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Linear(128 * seq_len, num_classes)
        self.valence_head = nn.Linear(128 * seq_len, 1)  # future valence predictor
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        logits = self.decoder(x_flat)
        future_valence = torch.sigmoid(self.valence_head(x_flat))  # [0,1]
        logits = self.dequant(logits)
        return logits, future_valence

    def fuse_modules(self):
        # Fuse if you have Conv+BN+ReLU etc.
        pass


# ──────────────────────────────────────────────────────────────
# Dummy high-valence dataset with valence labels
# ──────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=45, landmark_dim=225):
        self.data = torch.randn(num_samples, seq_len, landmark_dim)
        self.labels = torch.randint(0, 4, (num_samples,))
        # Simulated valence: higher for "good" sequences (dummy)
        self.valences = torch.rand(num_samples) * 0.5 + 0.5  # 0.5–1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.valences[idx]


# ──────────────────────────────────────────────────────────────
# Valence-weighted sampler
# ──────────────────────────────────────────────────────────────

def get_valence_sampler(valences: torch.Tensor, replacement=True):
    weights = torch.exp(6.0 * (valences - valences.mean()))
    weights /= weights.sum()
    return WeightedRandomSampler(weights, len(valences), replacement=replacement)


# ──────────────────────────────────────────────────────────────
# Gradient computation & training loop
# ──────────────────────────────────────────────────────────────

def train_qat(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    epochs: int = 60,
    lr: float = 1e-4,
    kd_alpha: float = 0.8,
    valence_alpha: float = 0.6,
    temperature: float = 5.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    student.to(device)
    teacher.to(device)
    teacher.eval()

    optimizer = optim.AdamW(student.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    # Progressive QAT activation
    qat_start_epoch = 10

    for epoch in range(epochs):
        if epoch == qat_start_epoch:
            print("[QAT] Activating quantization-aware training")
            student = prepare_qat(student, inplace=False)
            student.to(device)
            optimizer = optim.AdamW(student.parameters(), lr=lr * 0.1)

        student.train()
        running_loss = 0.0

        for inputs, labels, batch_valences in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # ─── Forward passes ──────────────────────────────────────
            student_logits, student_future_v = student(inputs)
            with torch.no_grad():
                teacher_logits, teacher_future_v = teacher(inputs)

            # ─── Losses ──────────────────────────────────────────────
            # Cross-entropy (hard labels)
            loss_ce = ce_loss_fn(student_logits, labels)

            # Soft KD
            loss_kd = kl_loss_fn(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature ** 2)

            # Future valence alignment
            loss_valence = F.mse_loss(student_future_v.squeeze(), teacher_future_v.squeeze())

            # Valence-weighted combination
            w = torch.exp(6.0 * (batch_valences - batch_valences.mean().item()))
            w = w / w.sum()   # normalize

            # Weighted mean
            loss = (w * loss_ce).mean() + \
                   kd_alpha * loss_kd + \
                   valence_alpha * loss_valence

            # ─── Gradient computation & step ─────────────────────────
            loss.backward()

            # Optional mercy gate (simplified projection check)
            projected_valence = student_future_v.mean().item()
            if projected_valence < 0.90 * teacher_future_v.mean().item():
                optimizer.zero_grad()
                print(f"[Mercy Gate] Epoch {epoch+1} – projected valence drop detected – skipping update")
                continue

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(student.state_dict(), f"qat_checkpoint_epoch_{epoch+1}.pth")

    # Final conversion to quantized model
    student.eval()
    student = convert(student, inplace=False)
    print("[QAT] Training complete – model converted to quantized state")

    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAT + Valence-Weighted KD training")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kd_alpha", type=float, default=0.8)
    parser.add_argument("--valence_alpha", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="models/qat")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dummy teacher (replace with real FP32 teacher)
    teacher = GestureTransformer()
    teacher.load_state_dict(torch.load("models/fp32_teacher.pth", map_location="cpu"))
    teacher.eval()

    # Dataset
    dataset = GestureDataset(num_samples=10000)
    sampler = get_valence_sampler(dataset.valences)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Student
    student = GestureTransformer()

    # Train
    quantized_model = train_qat(
        student=student,
        teacher=teacher,
        train_loader=loader,
        epochs=args.epochs,
        lr=args.lr,
        kd_alpha=args.kd_alpha,
        valence_alpha=args.valence_alpha,
        temperature=args.temperature
    )

    # Export to ONNX
    dummy_input = torch.randn(1, 45, 225)
    torch.onnx.export(
        quantized_model,
        dummy_input,
        os.path.join(args.output_dir, "gesture-transformer-qat.onnx"),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'future_valence'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'future_valence': {0: 'batch_size'}
        }
    )

    print(f"QAT training complete. Quantized ONNX saved to {args.output_dir}")
