/*!
# Powrush Particle Shaders — NVLink vs PCIe Latency and Bandwidth

Analysis of NVLink vs PCIe interconnect performance for CPU-GPU data movement.

## Overview

The CPU-GPU interconnect is critical for memory copies, Unified Memory migration, and `cudaMemPrefetchAsync` performance. The two main options are PCIe and NVLink.

## PCIe (Most Common)

- **PCIe Gen4 x16**: ~32 GB/s theoretical bidirectional (~25-28 GB/s realistic)
- **PCIe Gen5 x16**: ~64 GB/s theoretical
- Higher latency compared to NVLink
- Widely available on consumer and many server GPUs
- Good performance for most workloads, but can become a bottleneck for very high bandwidth or latency-sensitive data movement

## NVLink (High-End)

- Much higher aggregate bandwidth (hundreds of GB/s on modern generations)
- Significantly lower latency than PCIe
- Excellent for multi-GPU communication and tight CPU-GPU coherence
- Available on high-end data center GPUs (A100, H100) and some professional workstation cards
- More expensive and less common on consumer hardware

## Performance Implications

### Latency
- NVLink has noticeably lower latency than PCIe.
- This benefits workloads with frequent small transfers or fine-grained access (including Unified Memory page migration).

### Bandwidth
- NVLink offers dramatically higher bandwidth.
- This helps large data movements (big particle datasets, frequent updates) and makes Unified Memory more competitive with explicit copies.

### Unified Memory Impact
- On PCIe systems, Unified Memory migration can be a noticeable bottleneck.
- On NVLink systems, migration is much faster and more efficient, making `cudaMallocManaged` + `cudaMemPrefetchAsync` more attractive for performance-sensitive code.

## Relevance to Powrush

Most development and deployment will likely be on PCIe systems. For these:
- Prefer explicit memory management (`cudaMalloc` + `cudaMemcpyAsync`) for hot paths.
- Use `cudaMemPrefetchAsync` carefully and profile its benefit.

On high-end NVLink systems (e.g., H100-class), Unified Memory becomes significantly more viable even for performance-critical work because migration overhead is much lower.

## Recommendation

- Profile on target hardware. The interconnect makes a big difference in memory movement cost.
- On PCIe: Be conservative with Unified Memory in hot paths.
- On NVLink: Unified Memory + prefetching can be a strong option.
- Always measure — theoretical differences don't always translate 1:1 to application speedup.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on NVLink vs PCIe.
    pub const NVLINK_VS_PCIE_NOTES: &str = r#"
        // NVLink: lower latency, much higher bandwidth
        // PCIe: good for most cases, but higher latency
        // Unified Memory benefits more from NVLink
        // Profile on target hardware
        // Prefer explicit memory on PCIe for hot paths
    "#;
}
