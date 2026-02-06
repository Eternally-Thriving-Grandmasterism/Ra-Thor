# scripts/calibration_reader.py – Valence-Aware CalibrationDataReader v1
# Iterator for high-valence gesture sequence .npy files – used in Olive/AIMET/ONNX PTQ
# MIT License – Autonomicity Games Inc. 2026

import os
import numpy as np
from typing import Dict, Optional, Iterator
from onnxruntime.quantization import CalibrationDataReader


class ValenceCalibrationDataReader(CalibrationDataReader):
    """
    Custom CalibrationDataReader for high-valence gesture sequences.
    Yields {'input': np.ndarray} shaped [1, sequence_length, landmark_dim].
    Iterates over .npy files in the provided directory.
    """
    def __init__(
        self,
        calibration_dir: str,
        input_name: str = "input",
        expected_shape: Tuple[int, int] = (45, 225),  # sequence_length, landmark_dim
        normalize: bool = True,
        add_batch_dim: bool = True
    ):
        """
        Args:
            calibration_dir: Folder containing .npy files (one sequence per file)
            input_name: Name of the input tensor in the ONNX model (default: "input")
            expected_shape: Expected shape per sample (without batch dim)
            normalize: Whether to normalize each sequence (zero mean, unit std)
            add_batch_dim: Add leading batch dimension [1, ...] (required by most models)
        """
        self.calibration_dir = os.path.abspath(calibration_dir)
        self.input_name = input_name
        self.expected_shape = expected_shape
        self.normalize = normalize
        self.add_batch_dim = add_batch_dim

        # Collect all .npy files
        self.files = [
            os.path.join(self.calibration_dir, f)
            for f in sorted(os.listdir(self.calibration_dir))
            if f.lower().endswith('.npy')
        ]

        if not self.files:
            raise ValueError(f"No .npy files found in {self.calibration_dir}")

        print(f"[ValenceCalibrationDataReader] Found {len(self.files)} calibration samples")
        self.iterator: Iterator[str] = iter(self.files)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Returns next calibration input feed or None when exhausted.
        Format: {'input': np.float32 array [1, seq_len, dim]}
        """
        try:
            file_path = next(self.iterator)
        except StopIteration:
            return None

        try:
            # Load raw sequence
            seq = np.load(file_path).astype(np.float32)

            # Validate shape
            if seq.shape != self.expected_shape:
                raise ValueError(
                    f"Shape mismatch in {file_path}: got {seq.shape}, expected {self.expected_shape}"
                )

            # Optional normalization (zero mean, unit std per landmark)
            if self.normalize:
                seq -= np.mean(seq, axis=0, keepdims=True)
                seq /= (np.std(seq, axis=0, keepdims=True) + 1e-8)

            # Add batch dimension if required
            if self.add_batch_dim:
                seq = seq[np.newaxis, ...]  # → [1, seq_len, dim]

            return {self.input_name: seq}

        except Exception as e:
            print(f"[ValenceCalibrationDataReader] Error loading {file_path}: {e}")
            return self.get_next()  # skip bad file, continue

    def __iter__(self):
        """Allow use as iterator."""
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        """Iterator protocol support."""
        next_data = self.get_next()
        if next_data is None:
            raise StopIteration
        return next_data


# Example usage (for testing / manual run)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ValenceCalibrationDataReader")
    parser.add_argument("--calibration_dir", type=str, required=True,
                        help="Directory containing .npy calibration files")
    args = parser.parse_args()

    reader = ValenceCalibrationDataReader(
        calibration_dir=args.calibration_dir,
        input_name="input",
        expected_shape=(45, 225),
        normalize=True,
        add_batch_dim=True
    )

    count = 0
    for feed in reader:
        if feed is None:
            break
        print(f"Sample {count}: shape = {feed['input'].shape}")
        count += 1

    print(f"Total samples processed: {count}")
