/*!
# Powrush Particle Shaders — NVSwitch Topology Benefits

Analysis of NVSwitch topology advantages compared to traditional multi-GPU interconnects.

## What is NVSwitch?

NVSwitch is NVIDIA's high-bandwidth, low-latency switch fabric designed to connect multiple GPUs within a node (e.g., DGX/HGX systems). It enables full NVLink bandwidth communication between any pair of GPUs, creating a non-blocking all-to-all topology.

## Key Benefits of NVSwitch

### 1. Full Bandwidth All-to-All Communication
- Every GPU can communicate with every other GPU at full NVLink speed simultaneously.
- No bandwidth sharing or contention between pairs (non-blocking).

### 2. Excellent Scalability
- Performs very well as the number of GPUs increases (8, 16, 32+ GPUs per node).
- Avoids the bottlenecks of mesh or tree topologies without a switch.

### 3. Superior Collective Performance
- Much faster AllReduce, AllGather, Broadcast, ReduceScatter, etc.
- Critical for distributed training and large-scale multi-GPU simulations.

### 4. Reduced Contention and Predictable Performance
- More uniform latency and bandwidth between any GPU pair.
- Easier to reason about and optimize multi-GPU algorithms.

### 5. Better CPU-GPU Coherence (on some systems)
- Some NVSwitch configurations also improve CPU-GPU communication characteristics.

## Comparison to Other Topologies

**Traditional PCIe + NVLink (no switch)**:
- Good for small numbers of GPUs (2-4).
- Bandwidth sharing and contention increase with more GPUs.
- Collectives can become bottlenecks.

**Direct NVLink Mesh (without switch)**:
- Works well for small GPU counts.
- Does not scale as cleanly to large numbers of GPUs.
- More complex wiring and potential hot spots.

**NVSwitch**:
- Best scaling and collective performance for larger GPU counts.
- Higher cost and only available in high-end server systems.

## Relevance to Powrush

Current Powrush development targets single-GPU performance (particle culling, visibility, rendering on one GPU). In this context, NVSwitch benefits are limited.

However, if we ever scale to multi-GPU nodes (e.g., massive particle simulations, distributed rendering, or multi-GPU AI components), NVSwitch becomes highly relevant because:
- It enables efficient distribution of large particle datasets across GPUs.
- It provides fast collectives if we need to synchronize state between GPUs.
- It makes multi-GPU programming more predictable and scalable.

For most current workloads, PCIe + single-GPU optimization remains the priority.

## Recommendation

- Focus on single-GPU performance for now (PCIe systems).
- Keep multi-GPU scaling considerations in mind for future architecture decisions.
- If targeting high-end multi-GPU nodes, NVSwitch systems offer significant advantages for large-scale work.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on NVSwitch.
    pub const NVSWITCH_NOTES: &str = r#"
        // NVSwitch excels at large-scale multi-GPU communication.
        // Full all-to-all bandwidth and fast collectives.
        // Limited relevance for current single-GPU Powrush focus.
        // Valuable if we scale to massive multi-GPU simulations.
        // Prioritize single-GPU optimization on PCIe for now.
    "#;
}
