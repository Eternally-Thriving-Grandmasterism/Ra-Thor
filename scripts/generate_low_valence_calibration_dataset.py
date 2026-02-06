# scripts/generate_low_valence_calibration_dataset.py – Low-Valence Calibration Dataset Generator v1
# Creates synthetic low-valence (survival-mode) gesture sequences for PTQ calibration / boundary testing
# MIT License – Autonomicity Games Inc. 2026

import numpy as np
import os
import argparse
from typing import Tuple

def generate_low_valence_sequence(
    sequence_length: int = 45,
    landmark_dim: int = 33 * 3 + 21 * 3 * 2,  # 99 pose + 126 hands = 225 total
    smoothness: float = 0.05,                 # very low temporal coherence (erratic motion)
    noise_level: float = 0.35,                # high noise for survival-mode jitter
    drift_intensity: float = 0.08             # stronger non-stationary drift
) -> np.ndarray:
    """
    Generate one synthetic low-valence gesture sequence.
    Returns shape (sequence_length, landmark_dim) float32 array.
    """
    # High base noise + low smoothness = survival-mode jitter
    seq = np.random.randn(sequence_length, landmark_dim).astype(np.float32) * noise_level
    
    # Minimal cumulative smoothness
    seq = np.cumsum(seq * 0.1, axis=0)  # heavily damped
    
    # Stronger non-stationary drift
    drift = np.linspace(-1.5, 1.5, sequence_length)[:, np.newaxis] * drift_intensity
    seq += drift
    
    # Add random bursts (mimic stress / low-energy states)
    burst_mask = np.random.rand(sequence_length) < 0.08
    seq[burst_mask] += np.random.randn(np.sum(burst_mask), landmark_dim) * 0.6
    
    # Normalize per landmark (zero mean, unit std)
    seq -= np.mean(seq, axis=0, keepdims=True)
    seq /= (np.std(seq, axis=0, keepdims=True) + 1e-8)
    
    return seq


def generate_low_valence_calibration_dataset(
    num_samples: int = 300,
    sequence_length: int = 45,
    landmark_dim: int = 225,
    output_dir: str = "calibration_data/low_valence_gestures",
    min_valence: float = 0.0,
    max_valence: float = 0.75,
    smoothness: float = 0.05,
    noise_level: float = 0.35,
    drift_intensity: float = 0.08
) -> str:
    """
    Generate dataset of low-valence (survival-mode) gesture sequences.
    Each sample saved as individual .npy file.
    Returns output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} low-valence calibration samples...")
    print(f"Output directory: {output_dir}")
    print(f"Sequence shape per sample: ({sequence_length}, {landmark_dim})")
    
    for i in range(num_samples):
        seq = generate_low_valence_sequence(
            sequence_length=sequence_length,
            landmark_dim=landmark_dim,
            smoothness=smoothness,
            noise_level=noise_level,
            drift_intensity=drift_intensity
        )
        
        # Optional: save valence metadata (for future use)
        valence = np.random.uniform(min_valence, max_valence)
        
        file_path = os.path.join(output_dir, f"low_valence_seq_{i:04d}.npy")
        np.save(file_path, seq)
        
        # Optional: save metadata json if needed
        # meta_path = file_path.replace('.npy', '.json')
        # with open(meta_path, 'w') as f:
        #     json.dump({'valence': float(valence)}, f)
    
    print(f"Low-valence dataset generation complete: {num_samples} samples saved.")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate low-valence calibration dataset for PTQ / boundary testing")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of sequences to generate")
    parser.add_argument("--sequence_length", type=int, default=45,
                        help="Number of frames per sequence")
    parser.add_argument("--landmark_dim", type=int, default=225,
                        help="Number of landmark coordinates (pose + hands)")
    parser.add_argument("--output_dir", type=str, default="calibration_data/low_valence_gestures",
                        help="Output directory for .npy files")
    parser.add_argument("--min_valence", type=float, default=0.0,
                        help="Minimum simulated valence value")
    parser.add_argument("--max_valence", type=float, default=0.75,
                        help="Maximum simulated valence value")
    
    args = parser.parse_args()
    
    generate_low_valence_calibration_dataset(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        landmark_dim=args.landmark_dim,
        output_dir=args.output_dir,
        min_valence=args.min_valence,
        max_valence=args.max_valence
    )
