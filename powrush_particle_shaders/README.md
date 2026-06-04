# Powrush Particle Shaders — cudaMalloc vs cudaMallocManaged Performance

## cudaMalloc vs cudaMallocManaged Performance Comparison

This iteration provides a direct performance comparison between standard device memory allocation and Unified Memory.

### Summary Comparison

| Aspect                        | cudaMalloc (Explicit)                              | cudaMallocManaged (Unified)                        | Winner (Typical)          |
|-------------------------------|----------------------------------------------------|----------------------------------------------------|---------------------------|
| Hot Path Performance          | Highest and most predictable                       | Generally lower (migration overhead)               | cudaMalloc                |
| Programming Complexity        | Higher (manual copies)                             | Lower (single pointer)                             | cudaMallocManaged         |
| Page Migration / Faults       | None                                               | On-demand migration cost                           | cudaMalloc                |
| Memory Oversubscription       | Not supported                                      | Supported                                          | cudaMallocManaged         |
| Peak Bandwidth                | Full GPU HBM                                       | Limited by interconnect during migration           | cudaMalloc                |
| Best For                      | Performance-critical work                          | Simpler code + occasional CPU access               | Context dependent         |

### Key Takeaways

- For the core hot paths in Powrush (culling, visibility buffer updates, high-throughput compaction), `cudaMalloc` combined with explicit `cudaMemcpyAsync` is almost always the faster and more predictable choice.
- `cudaMallocManaged` becomes attractive when you need convenient CPU access to the same data or when faster development time is more important than maximum performance.
- Even with `cudaMemPrefetchAsync`, Unified Memory rarely matches the performance of well-optimized explicit memory for memory-bound compute kernels.

### Recommendation

- Use `cudaMalloc` + async copies for all performance-critical particle processing.
- Consider `cudaMallocManaged` only for data that needs frequent or convenient CPU read/write, and always pair it with `cudaMemPrefetchAsync`.
- When in doubt, profile both approaches — the performance difference can be substantial on memory-intensive workloads.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*