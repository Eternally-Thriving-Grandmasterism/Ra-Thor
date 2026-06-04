# Powrush Particle Shaders — cudaMallocManaged

## cudaMallocManaged Exploration

This iteration provides a focused exploration of `cudaMallocManaged`, the primary allocation function for **Unified Memory**.

### What It Does

`cudaMallocManaged` allocates memory accessible from both CPU and GPU via a single pointer. The CUDA driver automatically migrates memory pages between host and device as they are accessed (on-demand page migration).

### Key Characteristics

- Single pointer works on both host and device.
- Automatic page migration via page faults.
- Supports memory oversubscription.
- Maintains coherence between CPU and GPU views.

### Performance Considerations

- **Page Migration Latency**: On-demand migration adds latency compared to explicit copies.
- **Access Pattern Sensitivity**: Coalesced, sequential access performs much better than random access.
- **Interconnect Bandwidth**: Migration is limited by PCIe or NVLink speed.

### Tuning with Related APIs

- `cudaMemPrefetchAsync`: Proactively migrate pages to reduce faults.
- `cudaMemAdvise`: Provide hints about preferred location and access patterns.

### Relevance to Powrush

For performance-critical paths (particle culling, visibility buffer updates, high-throughput compaction), explicit device memory with `cudaMemcpyAsync` is generally preferred for speed and predictability.

`cudaMallocManaged` is more suitable for:
- Data that occasionally needs CPU read/write access.
- Prototyping or less performance-sensitive code.
- Workloads that benefit from memory oversubscription.

When using it for GPU hot paths, always combine with `cudaMemPrefetchAsync`.

### Best Practices

- Prefetch data to the GPU before kernels access it.
- Avoid fine-grained random GPU access on managed memory.
- Profile migration traffic and kernel performance.
- Use `cudaMemAdvise` to guide the driver.
- Consider switching hot data structures to explicit memory for maximum performance.

This provides the foundational allocation mechanism behind Unified Memory and guidance on its effective use.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*