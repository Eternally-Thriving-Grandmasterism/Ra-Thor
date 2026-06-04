# Powrush Particle Shaders — Memory Fence vs Atomic Operation Costs

## Memory Fence vs Atomic Operation Costs Comparison

This iteration compares the performance costs of memory fences and atomic operations.

### Summary

- **Atomic Operations**: Main cost is contention. We significantly reduced this using WaveLocal Reduction (turning many per-thread atomics into one per wave).
- **Memory Fences**: Main costs are execution stalls and prevention of instruction reordering (hurting latency hiding). Costs rise sharply with larger scopes.

### Comparison

| Aspect                    | Atomic Operations                  | Memory Fences                       | Notes                                      |
|---------------------------|------------------------------------|-------------------------------------|--------------------------------------------|
| Main Cost Driver          | Contention                         | Stalls + reordering restrictions    | WaveLocal Reduction helps atomics a lot    |
| Scope Sensitivity         | High                               | Very High                           | Subgroup scope preferred for both          |
| Latency Hiding Impact     | Moderate                           | Higher                              | Fences can hurt more broadly               |
| Frequency                 | High (counters, compaction)        | Lower                               | Atomics used more often                    |
| Best Optimization         | Wave-local aggregation             | Minimize scope + frequency          | We already apply both                      |

### Recommendation

Continue prioritizing Subgroup scope and wave-local techniques. This keeps both atomic and fence costs manageable while maintaining high scalability.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*