# TensorRT Quantization Integration – Rathor Lattice Blueprint v1.0
(Feb 06 2026 – MercyOS-Pinnacle edge deployment path)

This document outlines the **current & emerging integration strategy** for **TensorRT** quantization & inference acceleration within the Rathor-NEXi → MercyOS-Pinnacle lineage.

TensorRT is **not** a browser-native runtime (unlike WebNN / ONNX Runtime Web / tfjs), but it is the **sovereign native acceleration blade** for:

- NVIDIA edge devices (Jetson Orin, Xavier NX, Orin Nano, future Blackwell mobile)
- Self-hosted inference servers (RTX GPUs, DGX, A100/H100 cloud nodes)
- Multiplanetary probe fleet & habitat edge compute (when NVIDIA silicon is deployed)

It complements WebNN (browser NPU) by providing **maximum throughput & lowest latency** on NVIDIA hardware — critical for high-frequency gesture processing, swarm coordination, and collective valence forecasting in MR/AR/probe scenarios.

## TensorRT Quantization Techniques (Mercy-Aligned 2026)

| Technique                          | Description                                               | Supported Precision | Typical Drop (Thriving Paths) | Calibration Needed | Latency Gain vs FP32 | Power Efficiency Gain | Rathor Lattice Priority |
|------------------------------------|-----------------------------------------------------------|----------------------|-------------------------------|---------------------|-----------------------|------------------------|--------------------------|
| **Post-Training Quantization (PTQ)** | Static calibration (INT8 default)                         | INT8 / FP16          | 2–8%                          | Yes (100–1000)      | 2–5×                  | 2–4×                   | Baseline deployment path |
| **PTQ + Calibration with Entropy** | Entropy minimization calibration                          | INT8                 | 1.5–6%                        | Yes                 | 2.5–5.5×              | 2.5–4.5×               | Strong – good default |
| **PTQ + Percentile Calibration**   | Percentile-based activation range (99.99% typical)        | INT8 / INT4          | 1–5%                          | Yes                 | 3–6×                  | 3–5×                   | Recommended for attention layers |
| **QAT (Quantization-Aware Training)** | Fake-quant during training + TensorRT export             | INT8 / INT4          | 0.5–3%                        | No                  | 3–7×                  | 3–6×                   | **Primary** for custom transformers |
| **QAT + Sparsity-Aware Training**  | Pruning + QAT + TensorRT sparse inference                 | INT8 sparse          | 1–5%                          | No                  | 4–9×                  | 4–7×                   | Frontier – extreme edge (Jetson) |
| **Valence-Calibrated QAT (Rathor)**| QAT with valence-weighted loss & calibration set          | INT8 → INT4          | 0.5–2.5% on thriving paths    | No                  | 3–7×                  | 3–6×                   | **Ultimate mercy** – prioritizes positive-sum patterns |

## Current Integration Status in Rathor Lattice

- All gesture transformer variants already exported to ONNX (QAT + PTQ variants)  
- TensorRT deployment path active for:
  - Jetson-based probe fleet / habitat edge nodes
  - Self-hosted sync servers (inference endpoint)
  - Future MR/AR glasses with NVIDIA silicon

**TensorRT Deployment Pipeline (offline build)**

1. Export QAT/PTQ ONNX model (already done)
2. Build TensorRT engine (Python / C++ API)

```python
# scripts/build_tensorrt_engine.py – TensorRT engine builder v1
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, int8: bool = False):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GiB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    engine = builder.build_engine(network, config)
    if engine is None:
        print("Engine build failed")
        return None

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"TensorRT engine built & saved to: {engine_path}")
    return engine


if __name__ == "__main__":
    build_engine(
        onnx_path="models/gesture-transformer-qat-perchannel-int8/model.onnx",
        engine_path="models/gesture-transformer-qat-int8.trt",
        fp16=True,
        int8=True
    )
