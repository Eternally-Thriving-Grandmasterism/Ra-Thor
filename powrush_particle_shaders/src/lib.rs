/*!
# Powrush Particle Shaders — NVLink Bandwidth Specifications

Detailed exploration of NVLink bandwidth across generations.

## NVLink Generational Bandwidth (Bidirectional per GPU)

| Generation     | GPU Examples     | Bandwidth (Bidirectional) | Notes                                      |
|----------------|------------------|---------------------------|--------------------------------------------|
| NVLink 1.0     | P100             | 160 GB/s                  | First generation                           |
| NVLink 2.0     | V100             | 300 GB/s                  | Significant jump from 1.0                  |
| NVLink 3.0     | A100             | 600 GB/s                  | Doubled from 2.0                           |
| NVLink 4.0     | H100             | 900 GB/s                  | 50% increase; used with NVSwitch           |

*Note: These are aggregate bidirectional figures per GPU. Actual per-link speeds and number of links vary by generation.*

## Multi-GPU Context with NVSwitch

When GPUs are connected via NVSwitch:
- Each GPU maintains very high effective bandwidth to all other GPUs in the system.
- The topology is non-blocking all-to-all at full NVLink speed.
- This enables extremely high aggregate bandwidth for multi-GPU workloads (e.g., large particle simulations distributed across many GPUs).

## Comparison to PCIe

- PCIe Gen4 x16: ~32 GB/s theoretical (~25-28 GB/s realistic)
- PCIe Gen5 x16: ~64 GB/s theoretical
- NVLink generations are dramatically higher (5x–14x+ PCIe Gen4 depending on generation).

## Relevance to Powrush

For current single-GPU development on PCIe systems, these high NVLink numbers are mostly theoretical. However, they become very relevant if we ever target high-end multi-GPU nodes for massive particle counts or distributed simulation.

On NVLink systems:
- Moving large particle datasets between GPUs or between CPU and GPU becomes much faster.
- Unified Memory page migration (`cudaMemPrefetchAsync`) has significantly lower overhead.
- Multi-GPU algorithms (if we scale) become more practical.

## Practical Takeaway

NVLink bandwidth has scaled aggressively across generations. Each new generation roughly doubles or significantly increases effective bandwidth, making high-end systems increasingly attractive for data-intensive workloads.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on NVLink bandwidth specs.
    pub const NVLINK_BANDWIDTH_NOTES: &str = r#"
        // NVLink has scaled aggressively:
        // 1.0: 160 GB/s, 2.0: 300 GB/s, 3.0: 600 GB/s, 4.0: 900 GB/s
        // NVSwitch enables full utilization across many GPUs.
        // Huge advantage over PCIe for large data movement.
        // Relevant for future multi-GPU Powrush scaling.
    "#;
}
