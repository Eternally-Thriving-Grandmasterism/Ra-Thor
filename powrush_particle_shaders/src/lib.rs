/*!
# Powrush Particle Shaders — cudaMalloc vs cudaMallocManaged Performance Comparison

Direct performance comparison between standard device memory and Unified Memory.

## Comparison Table

| Aspect                        | cudaMalloc (Explicit Device Memory)                  | cudaMallocManaged (Unified Memory)                        | Winner (Typical Case)      |
|-------------------------------|-------------------------------------------------------|-----------------------------------------------------------|----------------------------|
| **Allocation Target**         | GPU device memory only                               | Managed memory (CPU + GPU accessible)                    | cudaMalloc (GPU-only)     |
| **Data Movement**             | Explicit cudaMemcpy / cudaMemcpyAsync                | Automatic page migration on access                       | Depends on use case       |
| **Hot Path Performance**      | Highest and most predictable                         | Generally lower due to migration overhead                | cudaMalloc                |
| **Programming Simplicity**    | Higher (must manage copies manually)                 | Lower (single pointer for CPU and GPU)                   | cudaMallocManaged         |
| **Page Fault / Migration**    | None                                                 | On-demand page faults + migration cost                   | cudaMalloc                |
| **Memory Oversubscription**   | Not supported natively                               | Supported                                                | cudaMallocManaged         |
| **Peak Bandwidth**            | Full GPU HBM bandwidth                               | Limited by interconnect during migration                 | cudaMalloc                |
| **Access Pattern Sensitivity**| Coalescing important                                 | Coalescing + migration efficiency important              | cudaMalloc (easier)       |
| **Best Use Case**             | Performance-critical hot paths                       | Simpler code, occasional CPU access, prototyping         | Context dependent         |

## Key Takeaways

- For performance-critical paths (particle culling, visibility buffer, high-throughput compaction), `cudaMalloc` + explicit async copies is almost always faster and more predictable.
- `cudaMallocManaged` shines when you need convenient CPU access to the same data or when development speed matters more than peak performance.
- Even with `cudaMemPrefetchAsync`, Unified Memory rarely matches the raw performance of well-managed explicit memory for hot compute kernels.

## Recommendation for Powrush

- Use `cudaMalloc` + `cudaMemcpyAsync` (or pinned memory + streams) for all high-throughput particle processing.
- Consider `cudaMallocManaged` only for data that needs frequent or convenient CPU read/write access, and always combine it with `cudaMemPrefetchAsync`.
- Profile both approaches when in doubt — the performance gap can be significant on memory-bound workloads.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Comparison notes.
    pub const MALLOC_VS_MANAGED_NOTES: &str = r#"
        // cudaMalloc + explicit copies = best performance for hot paths
        // cudaMallocManaged + prefetch = simpler but usually slower
        // Choose based on access pattern and performance needs.
        // Profile to confirm.
    "#;
}
