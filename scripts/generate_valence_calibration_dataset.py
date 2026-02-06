# scripts/generate_valence_calibration_dataset.py – Valence-Aware Calibration Dataset Generator v1
# Creates synthetic high-valence gesture sequences for PTQ calibration (Olive/AIMET/ONNX Runtime)
# MIT License – Autonomicity Games Inc. 2026

import numpy as np
import os
import argparse
from typing import Tuple

def generate_single_sequence(
    sequence_length: int = 45,
    landmark_dim: int = 33 * 3 + 21 * 3 * 2,  # 99 pose + 126 hands = 225 total
    smoothness: float = 0.15,                 # temporal coherence factor
    noise_level: float = 0.08                 # base noise std
) -> np.ndarray:
    """
    Generate one synthetic gesture sequence with temporal smoothness.
    Returns shape (sequence_length, landmark_dim) float32 array.
    """
    # Base random walk
    seq = np.random.randn(sequence_length, landmark_dim).astype(np.float32) * noise_level
    
    # Apply cumulative smoothness (mimic real motion continuity)
    seq = np.cumsum(seq, axis=0)
    
    # Normalize per landmark (zero mean, unit std)
    seq -= np.mean(seq, axis=0, keepdims=True)
    seq /= (np.std(seq, axis=0, keepdims=True) + 1e-8)
    
    # Slight non-stationary drift (real gestures evolve)
    drift = np.linspace(-1.0, 1.0, sequence_length)[:, np.newaxis] * 0.02
    seq += drift
    
    return seq


def generate_valence_calibration_dataset(
    num_samples: int = 300,
    sequence_length: int = 45,
    landmark_dim: int = 225,
    output_dir: str = "calibration_data/high_valence_gestures",
    min_valence: float = 0.92,
    max_valence: float = 0.999,
    smoothness: float = 0.15,
    noise_level: float = 0.08
) -> str:
    """
    Generate dataset of high-valence synthetic gesture sequences.
    Each sample saved as individual .npy file.
    Returns output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} high-valence calibration samples...")
    print(f"Output directory: {output_dir}")
    print(f"Sequence shape per sample: ({sequence_length}, {landmark_dim})")
    
    for i in range(num_samples):
        seq = generate_single_sequence(
            sequence_length=sequence_length,
            landmark_dim=landmark_dim,
            smoothness=smoothness,
            noise_level=noise_level
        )
        
        # Optional: save valence metadata alongside (for future use)
        valence = np.random.uniform(min_valence, max_valence)
        
        file_path = os.path.join(output_dir, f"gesture_seq_{i:04d}.npy")
        np.save(file_path, seq)
        
        # Optional: save metadata json if needed for custom reader
        # meta_path = file_path.replace('.npy', '.json')
        # with open(meta_path, 'w') as f:
        #     json.dump({'valence': float(valence)}, f)
    
    print(f"Dataset generation complete: {num_samples} samples saved.")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate high-valence calibration dataset for PTQ")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of sequences to generate")
    parser.add_argument("--sequence_length", type=int, default=45,
                        help="Number of frames per sequence")
    parser.add_argument("--landmark_dim", type=int, default=225,
                        help="Number of landmark coordinates (pose + hands)")
    parser.add_argument("--output_dir", type=str, default="calibration_data/high_valence_gestures",
                        help="Output directory for .npy files")
    parser.add_argument("--min_valence", type=float, default=0.92,
                        help="Minimum simulated valence value")
    parser.add_argument("--max_valence", type=float, default=0.999,
                        help="Maximum simulated valence value")
    
    args = parser.parse_args()
    
    generate_valence_calibration_dataset(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        landmark_dim=args.landmark_dim,
        output_dir=args.output_dir,
        min_valence=args.min_valence,
        max_valence=args.max_valence
    )
