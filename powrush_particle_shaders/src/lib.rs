/*!
# Powrush Particle Shaders — Memory Fence vs Atomic Costs

Comparison of performance costs: memory fences vs atomic operations.

## Key Differences

**Atomic Operations**:
- Main cost: Contention when many threads hit the same atomic.
- We mitigated this heavily with WaveLocal Reduction (ballot + shuffle → 1 atomic per wave).
- Scope sensitivity: High, but Subgroup scope helps a lot.

**Memory Fences**:
- Main cost: Stalls + prevention of instruction reordering (hurts latency hiding).
- Cache effects and memory traffic from visibility requirements.
- Scope sensitivity: Very high (Subgroup cheap, Workgroup/Device expensive).

## Comparison Table

| Aspect                    | Atomic Operations             | Memory Fences                    | Notes for Powrush                     |
|---------------------------|-------------------------------|----------------------------------|---------------------------------------|
| Main Cost Driver          | Contention                    | Stalls + reordering limits       | WaveLocal Reduction helps atomics     |
| Scope Sensitivity         | High                          | Very High                        | Subgroup scope best for both          |
| Latency Hiding Impact     | Moderate                      | Higher                           | Fences can hurt more broadly          |
| Frequency in our code     | High (counters)               | Lower                            | Atomics used more often               |
| Optimization Opportunity  | Wave-local aggregation        | Minimize scope + frequency       | We already do this well               |

## Conclusion

Our architecture (strong Subgroup scope preference + WaveLocal Reduction) keeps both atomic and fence costs low. Continue this approach.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    pub const COMPARISON_NOTES: &str = r#"
        // Atomic costs mainly from contention (mitigated by wave-local work)
        // Fence costs from stalls and reordering (mitigated by low frequency + Subgroup scope)
        // Both benefit strongly from staying at Subgroup scope.
    "#;
}
