# scripts/onnxruntime_python_inference.py – ONNX Runtime Python Inference Runner v1
# Multi-backend (CPU, CUDA, TensorRT, DirectML), valence-aware model selection, batch inference
# MIT License – Autonomicity Games Inc. 2026

import argparse
import time
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# Model variants & paths
# ──────────────────────────────────────────────────────────────

MODEL_VARIANTS = {
    'fp16': {
        'path': 'models/gesture-transformer-onnx/model.onnx',
        'desc': 'Full FP16 (highest accuracy)'
    },
    'int8': {
        'path': 'models/gesture-transformer-qat-int8/model.onnx',
        'desc': 'QAT INT8 (balanced)'
    },
    'int4': {
        'path': 'models/gesture-transformer-qat-int4/model.onnx',
        'desc': 'QAT INT4 (extreme edge)'
    }
}

# ──────────────────────────────────────────────────────────────
# Available execution providers (ordered by preference)
# ──────────────────────────────────────────────────────────────

PROVIDER_PREFERENCE = [
    'TensorrtExecutionProvider',      # NVIDIA GPU + TensorRT
    'CUDAExecutionProvider',          # NVIDIA GPU
    'DirectMLExecutionProvider',      # Windows DirectML (AMD/Intel GPU)
    'OpenVINOExecutionProvider',      # Intel OpenVINO (CPU/NPU)
    'CPUExecutionProvider'            # Universal fallback
]

# ──────────────────────────────────────────────────────────────
# Inference function
# ──────────────────────────────────────────────────────────────

def run_inference(
    model_path: str,
    input_data: np.ndarray,
    providers: List[str] = None,
    input_name: str = 'input',
    output_names: List[str] = ['gesture', 'future_valence']
) -> Dict[str, np.ndarray]:
    """
    Run inference on ONNX model with given providers.
    Returns dict of output_name → np.ndarray
    """
    if providers is None:
        providers = PROVIDER_PREFERENCE

    print(f"[ONNXRuntime] Loading model: {model_path}")
    print(f"[ONNXRuntime] Providers (attempt order): {providers}")

    session = None
    used_provider = None

    for provider in providers:
        try:
            session = ort.InferenceSession(
                model_path,
                providers=[provider],
                provider_options={} if provider != 'TensorrtExecutionProvider' else {'trt_fp16_enable': True}
            )
            used_provider = provider
            print(f"[ONNXRuntime] Success with provider: {provider}")
            break
        except Exception as e:
            print(f"[ONNXRuntime] Provider {provider} failed: {str(e)}")

    if session is None:
        raise RuntimeError("No suitable execution provider found")

    # Validate input shape
    input_meta = session.get_inputs()[0]
    expected_shape = input_meta.shape
    if list(input_data.shape) != list(expected_shape):
        raise ValueError(f"Input shape mismatch: got {input_data.shape}, expected {expected_shape}")

    # Warm-up run
    print("[ONNXRuntime] Warm-up inference...")
    _ = session.run(None, {input_name: input_data})

    # Timed inference
    print("[ONNXRuntime] Running timed inference...")
    start_time = time.perf_counter()
    outputs = session.run(output_names, {input_name: input_data})
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    print(f"[ONNXRuntime] Inference time: {inference_time_ms:.2f} ms (provider: {used_provider})")

    return {name: out for name, out in zip(output_names, outputs)}


# ──────────────────────────────────────────────────────────────
# Valence-aware model selector
# ──────────────────────────────────────────────────────────────

def select_model_by_valence(valence: float = 0.96) -> str:
    if valence > 0.94:
        return MODEL_VARIANTS['int4']['path']
    elif valence > 0.88:
        return MODEL_VARIANTS['int8']['path']
    else:
        return MODEL_VARIANTS['fp16']['path']


# ──────────────────────────────────────────────────────────────
# Main CLI runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Runtime Python Inference Demo")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to ONNX model (overrides valence selection)")
    parser.add_argument("--valence", type=float, default=0.96,
                        help="Simulated valence for model selection (0.0–1.0)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for input")
    parser.add_argument("--sequence_length", type=int, default=45,
                        help="Sequence length")
    parser.add_argument("--landmark_dim", type=int, default=225,
                        help="Landmark dimension (pose + hands)")
    parser.add_argument("--output_names", type=str, default="gesture,future_valence",
                        help="Comma-separated output tensor names")
    args = parser.parse_args()

    # Select model path
    model_path = args.model if args.model else select_model_by_valence(args.valence)
    print(f"[Main] Using model: {model_path} (valence {args.valence:.3f})")

    # Generate dummy input
    input_shape = (args.batch_size, args.sequence_length, args.landmark_dim)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Run inference
    output_names = args.output_names.split(',')
    results = run_inference(
        model_path=model_path,
        input_data=input_data,
        output_names=output_names
    )

    # Print sample results
    for name, arr in results.items():
        print(f"\nOutput '{name}': shape = {arr.shape}")
        print(f"Sample values: {arr.flat[:8]}")
