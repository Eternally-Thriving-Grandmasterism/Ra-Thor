/*!
# Powrush Particle Shaders — Hardware Atomic Latency

Exploration of hardware-level latency characteristics of atomic operations on modern GPUs.

## Sources of Atomic Latency

Atomic operations have higher latency than regular loads/stores due to several factors:

1. **Memory Round-Trips**: Atomics typically require communication with L2 cache or device memory for coherence.
2. **Serialization under Contention**: When multiple threads/waves target the same address, the hardware must serialize the operations, increasing effective latency.
3. **Coherence Protocol Overhead**: Maintaining cache coherence across threads adds cost.
4. **Instruction Complexity**: Atomic instructions are more complex than simple loads/stores.

## Impact of Scope on Latency

- **Subgroup Scope**: Lowest latency. Can sometimes be optimized within the wave (registers or L1). Minimal coherence traffic.
- **Workgroup Scope**: Moderate to high latency. Usually involves L2 cache and workgroup-wide coherence.
- **Device Scope**: Highest latency. Full device-wide coherence traffic.

## Contention Amplifies Latency

High contention (many threads hitting the same atomic) is often the dominant source of observed latency in practice. Each contending thread may wait for previous operations to complete.

## Relevance to Powrush

We previously addressed atomic contention through **WaveLocal Reduction** (ballot + shuffle to perform one atomic per wave instead of per thread). This technique significantly reduces both the number of atomics issued *and* the effective latency caused by contention.

Staying at **Subgroup scope** further helps keep hardware atomic latency low.

## Key Takeaway

Hardware atomic latency is manageable when:
- Contention is minimized (via wave-local aggregation)
- Scope is kept as small as possible (Subgroup preferred)
- Atomics are used judiciously rather than as the primary synchronization mechanism
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on hardware atomic latency.
    pub const HARDWARE_ATOMIC_LATENCY_NOTES: &str = r#"
        // Subgroup scope atomics: lowest latency
        // Workgroup/Device: significantly higher
        // Contention is often the biggest real-world amplifier of latency
        //
        // WaveLocal Reduction helps on both count and effective latency.
    "#;
}
