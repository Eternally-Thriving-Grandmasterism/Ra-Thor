# Powrush Particle Shaders — NVLink vs PCIe Latency and Bandwidth

## NVLink vs PCIe Analysis

This iteration analyzes the performance differences between **NVLink** and **PCIe** interconnects for CPU-GPU data movement, with implications for memory management in our particle system.

### Key Differences

**PCIe (Most Common)**:
- PCIe Gen4 x16: ~32 GB/s theoretical, ~25-28 GB/s realistic
- PCIe Gen5 x16: ~64 GB/s theoretical
- Higher latency than NVLink
- Widely available

**NVLink (High-End)**:
- Dramatically higher bandwidth (hundreds of GB/s aggregate)
- Significantly lower latency
- Excellent for multi-GPU and tight CPU-GPU coherence
- Available on high-end data center and some professional GPUs

### Performance Implications

**Latency**:
- NVLink wins clearly. Lower latency benefits frequent small transfers and Unified Memory page migration.

**Bandwidth**:
- NVLink offers much higher throughput. This helps large particle dataset movements and makes Unified Memory more competitive.

**Unified Memory**:
- On PCIe systems, migration overhead can be significant.
- On NVLink systems, migration is much faster, making `cudaMallocManaged` + `cudaMemPrefetchAsync` more viable for performance-sensitive code.

### Relevance to Powrush

Most systems will be PCIe-based. On these platforms:
- Prefer explicit `cudaMalloc` + `cudaMemcpyAsync` for hot paths (culling, visibility, compaction).
- Use Unified Memory + prefetching judiciously and always profile.

On high-end NVLink hardware, Unified Memory becomes a much stronger option because the migration cost is substantially lower.

### Recommendation

- Profile on your target hardware. The interconnect makes a real difference.
- On PCIe: Be conservative with Unified Memory in performance-critical sections.
- On NVLink: Unified Memory + `cudaMemPrefetchAsync` can be a competitive choice.
- Always measure real-world impact rather than relying on theoretical numbers.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*