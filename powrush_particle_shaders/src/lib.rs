/*!
# Powrush Particle Shaders — cudaMallocManaged

Exploration of `cudaMallocManaged` and Unified Memory allocation.

## What is cudaMallocManaged?

`cudaMallocManaged` allocates memory that can be accessed from both the CPU and GPU using a single pointer. It is the primary allocation function for **Unified Memory** (also called Managed Memory).

```c
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = 0);
```

The allocated memory is "managed" by the CUDA driver, which automatically migrates pages between host and device memory as they are accessed.

## Key Characteristics

- **Single Pointer**: Same address works on both CPU and GPU.
- **Automatic Migration**: Pages are moved on-demand via page faults.
- **Oversubscription Support**: Can allocate more memory than physically available on the GPU.
- **Coherence**: The driver maintains coherence between CPU and GPU views.

## Performance Model

- **On-Demand Page Migration**: First access on the non-resident side triggers a page fault and migration. This adds latency.
- **Bandwidth**: Migration happens over PCIe (or NVLink on supported systems), which is slower than GPU HBM.
- **Access Pattern Sensitivity**: Sequential, coalesced access is much more efficient than random or fine-grained access.

## Tuning APIs

- `cudaMemPrefetchAsync`: Proactively migrate pages before access.
- `cudaMemAdvise`: Provide hints (preferred location, accessed by, etc.).
- `cudaMemRangeGetAttribute`: Query memory properties.

## Relevance to Powrush

For high-performance particle processing (culling, visibility, compaction), explicit device memory (`cudaMalloc`) combined with `cudaMemcpyAsync` is generally preferred for predictability and performance.

`cudaMallocManaged` can be useful for:
- Data that needs occasional CPU read/write access (e.g., editing, debugging, loading).
- Prototyping and simpler code paths.
- Situations where memory oversubscription is beneficial.

When using it, always combine with `cudaMemPrefetchAsync` for hot paths.

## Best Practices

- Use `cudaMemPrefetchAsync` before kernels that will access the memory on the GPU.
- Avoid fine-grained random access patterns from the GPU.
- Profile migration traffic with Nsight Systems.
- For performance-critical loops, consider switching to explicit memory.
- Use `cudaMemAdviseSetPreferredLocation` to hint where data should reside most of the time.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on cudaMallocManaged.
    pub const CUDAMALLOCMANAGED_NOTES: &str = r#"
        // cudaMallocManaged enables Unified Memory.
        // Combine with cudaMemPrefetchAsync for best performance.
        // Good for data with occasional CPU access.
        // For hot paths, explicit memory is usually better.
        // Profile migration overhead with Nsight Systems.
    "#;
}
