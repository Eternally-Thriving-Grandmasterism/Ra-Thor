# Powrush Particle Shaders — Nsight Systems Timeline Analysis

## Nsight Systems Timeline Analysis Exploration

This iteration explores **system-level timeline analysis** using Nsight Systems, which complements the deep kernel profiling provided by Nsight Compute.

### What Nsight Systems Shows

Nsight Systems provides a high-level view of CPU and GPU activity over time, including:
- CPU thread activity and CUDA API calls
- GPU kernel execution and idle gaps
- Memory transfers between host and device
- Synchronization points and command submission

### Key Analysis Areas

**GPU Utilization & Bubbles**:
- Identify periods where the GPU is idle.
- Correlate gaps with CPU activity (e.g., data preparation, API calls, synchronization).

**CPU Overhead**:
- Time spent in CUDA kernel launches, memory copies, and synchronization.
- Excessive small launches or frequent sync can indicate optimization opportunities.

**Memory Transfers**:
- Timing and size of Host ↔ Device transfers.
- Opportunities to overlap transfers with compute using streams.

**Synchronization**:
- Points where the CPU waits for the GPU or vice versa.
- These often create bubbles that reduce overall throughput.

### Relevance to Powrush

Our pipeline involves compute culling, visibility updates, and indirect draw execution. Nsight Systems timeline analysis can reveal:
- Whether culling or visibility work is CPU-bound or GPU-bound.
- Launch overhead from multiple dispatches.
- Memory transfer costs when updating particle data.
- Opportunities to better overlap work using asynchronous streams.

### Recommended Workflow

1. Capture a timeline of a representative workload or frame.
2. Focus on regions around culling and visibility buffer updates.
3. Look for GPU idle gaps and correlate with CPU activity.
4. Identify expensive API calls or synchronization points.
5. Compare timelines before and after major optimizations (e.g., WaveLocal Reduction, reduced dispatch count).

Nsight Systems is ideal for system-level bottlenecks and CPU/GPU interaction, while Nsight Compute excels at deep per-kernel optimization.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*