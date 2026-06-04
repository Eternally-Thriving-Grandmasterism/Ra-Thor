/*!
# Powrush Particle Shaders — cudaMemPrefetchAsync Optimization

Exploration of `cudaMemPrefetchAsync` as an optimization for Unified Memory workloads.

## What cudaMemPrefetchAsync Does

`cudaMemPrefetchAsync` proactively migrates pages of Unified Memory to a target device (GPU or CPU) asynchronously before they are accessed. This helps reduce or eliminate expensive on-demand page faults during kernel execution.

Signature (simplified):
```c
cudaError_t cudaMemPrefetchAsync(
    const void* devPtr, size_t count, int dstDevice, cudaStream_t stream);
```

## Benefits

- Reduces page fault latency during kernel execution
- Can be overlapped with other work using streams
- Improves predictability of Unified Memory performance
- Allows the programmer to express data movement intent explicitly

## When to Use It

Use `cudaMemPrefetchAsync` when:
- You are using Unified Memory and want to improve performance
- You know in advance where data will be accessed (GPU or CPU)
- You want to overlap migration with compute or other transfers

It is especially useful before launching kernels that will access large managed memory regions.

## Best Practices

- Prefetch data to the GPU before launching compute kernels that will read it.
- Prefetch results back to the CPU after GPU processing if the CPU will read them soon.
- Use appropriate streams to allow overlap with other operations.
- Avoid unnecessary prefetching of data that won't be used soon (wastes bandwidth).
- Combine with `cudaMemAdvise` (e.g., `cudaMemAdviseSetPreferredLocation`) for more control.

## Comparison to Explicit Memory Copies

`cudaMemPrefetchAsync` is generally easier to use than explicit `cudaMemcpyAsync`, but explicit copies often have lower overhead and better performance for large, predictable transfers. Use prefetching when the simplicity of Unified Memory is desired and migration cost is acceptable.

## Relevance to Powrush

If Unified Memory is used for particle data that needs occasional CPU access (e.g., editing, loading, or result inspection), `cudaMemPrefetchAsync` can help make GPU access performant by migrating data to the GPU before culling or visibility passes.

It can also be used to bring results back to the CPU asynchronously after processing.

## Profiling

Use Nsight Systems to visualize migration traffic and verify that prefetching is reducing on-demand page faults. Compare kernel execution time and memory stall reasons with and without prefetching.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on cudaMemPrefetchAsync.
    pub const PREFETCH_ASYNC_NOTES: &str = r#"
        // Use cudaMemPrefetchAsync to reduce page faults in Unified Memory.
        // Prefetch to GPU before compute, to CPU after.
        // Overlap with streams when possible.
        // Profile migration with Nsight Systems.
        // Combine with cudaMemAdvise for more control.
    "#;
}
