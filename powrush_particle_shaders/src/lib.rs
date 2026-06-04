/*!
# Powrush Particle Shaders — Nsight Systems Timeline Analysis

Exploration of system-level timeline analysis using Nsight Systems.

## What Nsight Systems Provides

Nsight Systems offers a high-level, system-wide timeline view of CPU and GPU activity. It complements Nsight Compute (which focuses on deep kernel metrics) by showing the overall execution flow over time.

Key views include:
- CPU thread activity and API calls
- GPU kernel execution and gaps
- Memory transfers (Host ↔ Device)
- CUDA synchronization and command submission
- Multi-GPU and multi-context behavior

## Key Things to Analyze

### GPU Utilization & Bubbles
- Look for gaps between kernel executions (GPU idle time).
- Identify whether the CPU is keeping the GPU fed.
- Long gaps often indicate CPU-side work, synchronization, or launch overhead.

### CPU Overhead
- Time spent in CUDA API calls (e.g., kernel launches, memory copies, synchronization).
- Excessive API time can indicate too many small launches or frequent synchronization.

### Memory Transfers
- Host-to-Device and Device-to-Host transfers.
- Overlapping transfers with compute (using streams and async copies).
- Unnecessary or poorly timed transfers that stall the GPU.

### Synchronization Points
- cudaDeviceSynchronize, cudaStreamSynchronize, or event-based sync.
- These often cause GPU bubbles if not overlapped properly.

## Relevance to Powrush Pipeline

Our architecture involves:
- Compute culling kernels (potentially multiple dispatches)
- Visibility buffer updates
- Indirect draw command generation and execution
- Possible multi-frame or asynchronous compute patterns

Nsight Systems timeline analysis can reveal:
- Whether culling is a CPU or GPU bottleneck
- Launch overhead from many small dispatches
- Memory transfer costs when updating particle data or reading back results
- Opportunities to overlap compute and transfer using streams

## Recommended Usage

1. Capture a timeline of a representative frame or workload.
2. Zoom into regions around culling and visibility updates.
3. Look for GPU idle gaps and correlate with CPU activity.
4. Identify expensive API calls or synchronization points.
5. Compare before/after major optimizations (e.g., WaveLocal Reduction, reduced dispatches).

Nsight Systems is excellent for system-level bottlenecks, while Nsight Compute is better for deep kernel optimization.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on Nsight Systems timeline analysis.
    pub const NSIGHT_SYSTEMS_TIMELINE_NOTES: &str = r#"
        // Use Nsight Systems for system-level view:
        // - GPU utilization gaps
        // - CPU API overhead
        // - Memory transfer timing
        // - Synchronization points
        //
        // Complement with Nsight Compute for deep kernel analysis.
        // Focus on culling and visibility update regions.
    "#;
}
