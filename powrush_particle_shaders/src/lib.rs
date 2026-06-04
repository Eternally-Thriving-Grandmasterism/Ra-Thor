/*!
# Powrush Particle Shaders — Memory Fence Performance Costs

Exploration of the performance costs associated with memory fences.

## What Contributes to Fence Cost?

Memory fences have performance costs because they can:

1. **Stall execution** — Threads wait for prior memory operations to complete and become visible.
2. **Flush or invalidate caches** — Depending on semantics and scope, fences may cause cache operations.
3. **Prevent instruction reordering** — Both compiler and hardware reordering is restricted, reducing latency hiding opportunities.
4. **Increase memory traffic** — Release/Acquire semantics require making data visible across threads/scopes.

## Cost by Scope

### Subgroup Scope
- Relatively low cost.
- Often just a lightweight wave synchronization.
- Minimal cache impact on most modern GPUs.
- Good for performance-critical paths.

### Workgroup Scope
- Moderate to high cost.
- Requires synchronization across the entire workgroup.
- Stronger memory visibility guarantees increase traffic and stall time.

### Device / QueueFamily Scope
- Highest cost.
- Very expensive due to broad visibility requirements across the device or queue family.
- Should be avoided in performance-critical rendering/compute paths.

## Impact on Our Architecture

Because we strongly prefer **Subgroup-scoped** operations and wave-local techniques (ballot, shuffle, wave-local reduction), memory fence costs in our particle culling and visibility pipelines remain low.

Using stronger scopes or unnecessary fences would introduce avoidable performance penalties.

## Best Practices

- Prefer Subgroup scope whenever possible.
- Only use stronger fences when functionally required (e.g., Workgroup-scoped cooperative matrices).
- Combine fences with wave-local work to minimize their frequency and strength.
- Profile and validate — unnecessary fences are a common source of hidden performance loss.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on memory fence performance costs.
    pub const MEMORY_FENCE_COST_NOTES: &str = r#"
        // Subgroup fences: low cost
        // Workgroup fences: moderate-high cost
        // Device fences: very high cost
        //
        // Minimize scope and frequency of fences.
        // Prefer wave-local techniques to reduce need for strong fences.
    "#;
}
