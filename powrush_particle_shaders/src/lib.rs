/*!
# Powrush Particle Shaders — PCIe Gen5 Bandwidth Investigation

Detailed investigation of PCIe Gen5 bandwidth specifications and implications.

## PCIe Gen5 Key Specifications

- **Signaling Rate**: 32 GT/s per lane
- **Encoding**: 128b/130b (very low overhead ~1.5%)
- **Effective Bandwidth per Lane** (one direction): ~3.938 GB/s
- **x16 Link (most common for GPUs)**:
  - Theoretical: **~64 GB/s bidirectional** (~32 GB/s each direction)
  - Realistic achievable: Typically 50–58+ GB/s bidirectional depending on platform and implementation

## Generational Comparison

| Generation     | x16 Theoretical Bidirectional | Realistic (approx.) | Notes                              |
|----------------|-------------------------------|---------------------|------------------------------------|
| PCIe Gen3      | ~16 GB/s                      | ~14 GB/s            | Older baseline                     |
| PCIe Gen4      | ~32 GB/s                      | ~25-28 GB/s         | Common on current systems          |
| PCIe Gen5      | ~64 GB/s                      | ~50-58+ GB/s        | Current high-end standard          |

## Comparison to NVLink

- PCIe Gen5 x16: ~64 GB/s theoretical
- NVLink 4.0: 900 GB/s bidirectional
- NVLink offers roughly **14x** the theoretical bandwidth of PCIe Gen5, with significantly lower latency.

This massive difference is why high-end multi-GPU systems feel dramatically faster for large data movement.

## Relevance to Powrush

Most current and near-future Powrush development and deployment will be on PCIe Gen4 or Gen5 systems. This is the baseline interconnect for:
- Explicit memory copies (`cudaMemcpyAsync`)
- Unified Memory page migration
- CPU-side data preparation and result readback

Understanding PCIe Gen5 limits helps explain:
- Why large particle dataset updates from CPU can become a bottleneck
- Why `cudaMemPrefetchAsync` on Unified Memory still has noticeable cost on typical hardware
- Why high-end NVLink systems offer such a large advantage for massive data movement

## Practical Implications

- On PCIe Gen5 systems, explicit async memory copies remain the most efficient way to move large amounts of particle data.
- Unified Memory migration will still feel the interconnect bottleneck compared to NVLink systems.
- For most workloads, PCIe Gen5 provides good headroom, but very large or frequent CPU-GPU transfers can still benefit from optimization (batching, compression, asynchronous overlap).
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on PCIe Gen5 bandwidth.
    pub const PCIE_GEN5_NOTES: &str = r#"
        // PCIe Gen5 x16: ~64 GB/s theoretical, ~50-58 GB/s realistic
        // Still the baseline for most systems
        // Much lower than NVLink (900 GB/s)
        // Explicit async copies remain best for hot paths
        // Unified Memory migration feels the PCIe limit
    "#;
}
