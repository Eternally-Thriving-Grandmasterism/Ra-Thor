/*!
# Powrush Particle Shaders — CUDA Stream Overlap Efficiency

Analysis of CUDA stream usage and overlap efficiency.

## What is Stream Overlap?

CUDA streams allow concurrent execution of kernels and memory copies that have no data dependencies. Good overlap hides memory transfer latency behind compute or runs independent kernels in parallel, improving overall throughput.

## Key Analysis Points in Nsight Systems

### Timeline View
- Look for multiple streams executing in parallel.
- Identify gaps where one stream is idle while another has work.
- Check if memory transfers (H2D/D2H) overlap with compute kernels.

### Common Issues
- Overuse of the default stream (Stream 0), which can serialize operations.
- Unnecessary synchronization points that break concurrency.
- Memory copies not using async versions with streams.
- Dependencies between streams that prevent overlap.

### Positive Patterns
- Memory transfers running concurrently with unrelated compute.
- Independent compute kernels running on different streams.
- Proper use of events for lightweight cross-stream synchronization.

## Relevance to Powrush

Our pipeline may benefit from stream overlap in several areas:
- Updating particle data (Host → Device) while culling the previous frame.
- Overlapping culling compute passes with visibility buffer updates.
- Asynchronous result processing or readbacks.
- Running multiple independent culling or compaction phases concurrently.

Good stream usage can hide transfer latency and improve overall frame throughput.

## Best Practices

- Use non-default streams for work that can run concurrently.
- Use `cudaMemcpyAsync` with streams instead of synchronous copies.
- Use CUDA events for lightweight synchronization instead of `cudaStreamSynchronize` or `cudaDeviceSynchronize`.
- Minimize cross-stream dependencies when possible.
- Profile with Nsight Systems timeline to visualize and validate overlap.

## Analysis Workflow

1. Capture a timeline with Nsight Systems.
2. Examine stream activity around culling and data update regions.
3. Look for idle gaps that could be filled by overlapping transfers or independent compute.
4. Identify unnecessary synchronization that breaks concurrency.
5. Compare before/after stream optimization changes.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on stream overlap.
    pub const STREAM_OVERLAP_NOTES: &str = r#"
        // Use multiple streams for concurrent work.
        // Prefer async memory copies with streams.
        // Use events instead of heavy synchronization.
        // Profile with Nsight Systems to validate overlap.
        // Minimize unnecessary cross-stream dependencies.
    "#;
}
