/*!
# Powrush Particle Shaders — NVLink vs Infinity Fabric Comparison

Comparison of NVIDIA NVLink and AMD Infinity Fabric interconnect technologies.

## Overview

Both NVLink and Infinity Fabric are high-speed interconnects designed to move large amounts of data between processors (primarily GPUs, and in some cases CPUs). They serve similar roles in their respective ecosystems but have different strengths.

## Comparison Table

| Aspect                        | NVLink (NVIDIA)                                      | Infinity Fabric (AMD)                                   | Notes / Winner                          |
|-------------------------------|------------------------------------------------------|---------------------------------------------------------|-----------------------------------------|
| **Primary Use**               | GPU-to-GPU (strong), some CPU-GPU                    | CPU-to-CPU + GPU-to-GPU / CPU-GPU                       | Depends on workload                     |
| **Bandwidth (per GPU)**       | Very high (up to 900 GB/s bidirectional on 4.0)     | High (varies by generation; competitive in MI300 era)  | NVLink generally leads in raw GPU bandwidth |
| **Latency**                   | Very low                                             | Low (competitive)                                       | Both excellent; NVLink often edges out  |
| **Multi-GPU Scaling**         | Excellent with NVSwitch (non-blocking all-to-all)   | Very good in Instinct + EPYC systems                    | NVSwitch gives NVIDIA an edge at scale  |
| **CPU-GPU Coherence**         | Limited / improving in some platforms                | Stronger in some configurations (e.g., MI300 + EPYC)   | Infinity Fabric often better for shared memory |
| **Ecosystem**                 | NVIDIA GPUs + some server CPUs                       | AMD EPYC CPUs + Instinct GPUs                           | Ecosystem dependent                     |
| **Collective Performance**    | Excellent (especially with NVSwitch)                 | Very good                                               | NVLink + NVSwitch typically leads       |
| **Availability**              | High-end data center and some professional GPUs     | High-end Instinct GPUs + EPYC CPUs                      | Both high-end only                      |

## Key Takeaways

**NVLink Strengths**:
- Higher per-GPU bandwidth in recent generations.
- Superior multi-GPU scaling when paired with NVSwitch.
- Excellent for tight GPU-to-GPU coupling and collectives.

**Infinity Fabric Strengths**:
- Better integrated CPU-GPU coherence in some platforms (useful for unified memory-like models).
- Strong performance in AMD's full-stack (EPYC + Instinct) systems.
- Competitive bandwidth and latency.

## Relevance to Powrush

For current single-GPU development, the differences are mostly academic. However, if we ever target high-end multi-GPU or CPU+GPU shared memory workloads:
- NVLink + NVSwitch is generally stronger for pure GPU-to-GPU scaling and large particle dataset distribution.
- Infinity Fabric may be preferable if tight CPU-GPU shared memory / coherence is important.

Most Powrush development will likely remain on NVIDIA hardware for the foreseeable future, making NVLink the more immediately relevant technology.

## Recommendation

- Continue prioritizing single-GPU performance on NVIDIA hardware (PCIe + NVLink where available).
- Monitor both technologies for future multi-GPU or heterogeneous computing decisions.
- Profile on target hardware — real-world results matter more than theoretical comparisons.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on NVLink vs Infinity Fabric.
    pub const NVLINK_VS_INFINITY_NOTES: &str = r#"
        // NVLink: higher GPU bandwidth, excellent multi-GPU with NVSwitch
        // Infinity Fabric: strong CPU-GPU coherence, competitive performance
        // NVLink currently more relevant for Powrush (NVIDIA focus)
        // Both excellent for high-end multi-GPU work
    "#;
}
