/*!
# Powrush Particle Shaders — CUDA API Call Overhead Analysis

Analysis of CUDA API call overhead and its impact on performance.

## What Contributes to API Overhead

Every CUDA API call (kernel launch, memory copy, event record, synchronization, etc.) incurs CPU-side overhead from the driver. This includes:
- Command buffer preparation and submission
- Validation and context management
- Potential serialization if not using streams properly

Frequent or expensive calls can make the CPU the bottleneck, leaving the GPU idle.

## Common Expensive Patterns

- Many small kernel launches instead of fewer, larger ones
- Frequent synchronization (`cudaDeviceSynchronize`, `cudaStreamSynchronize`)
- Small synchronous memory copies instead of batched/async transfers
- Repeated creation/destruction of events or streams

## Impact

High API overhead often appears in Nsight Systems as:
- Significant CPU time in CUDA API rows
- GPU idle gaps that correlate with CPU API activity
- Reduced overall frame throughput

## Relevance to Powrush

Our architecture may involve:
- Multiple compute dispatches for culling passes
- WaveLocal Reduction phases
- Visibility buffer updates
- Indirect draw command generation and execution

If these involve many small launches or frequent synchronization, API overhead can become noticeable.

## Mitigation Strategies

- Batch work into fewer, larger kernel launches when possible
- Use CUDA streams to overlap compute and transfers
- Minimize synchronization points (prefer events and stream-ordered operations)
- Reuse events and streams instead of creating them repeatedly
- Consider persistent kernels or grid-stride loops for repeated small work
- Profile with Nsight Systems to identify hot API calls

## Analysis in Nsight Systems

Look at the CPU row and CUDA API trace. Sort by total time or call count to find hot spots. Correlate with GPU timeline gaps to see where the CPU is stalling the GPU.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on CUDA API overhead.
    pub const CUDA_API_OVERHEAD_NOTES: &str = r#"
        // Common issues: many small launches, frequent syncs
        // Mitigation: batching, streams, event-based sync, reuse
        // Profile with Nsight Systems to find hot API calls
        // Correlate CPU API time with GPU idle gaps
    "#;
}
