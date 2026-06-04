/*!
# Powrush Particle Shaders — Profiling Memory Transactions

Guidance on profiling memory transactions using Nsight Compute.

## Key Nsight Compute Sections

### Memory Workload Analysis
This is the primary section for memory transaction analysis:
- `l1tex__t_requests` / `l1tex__t_sectors` (L1)
- `lts__t_requests` / `lts__t_sectors` (L2)
- `dram__sectors` (device memory)
- `memory_replay_overhead`

### Memory Efficiency Metrics
- **Global Load/Store Efficiency** (% of bytes transferred that were useful)
- **L1/L2 Hit Rates**
- **Transactions per Request** (lower is better for coalescing)

### Warp Memory Metrics
- How many memory transactions a warp generates
- Coalescing efficiency indicators

### Stall Reasons
- "Long Scoreboard"
- "Memory Dependency"
These often correlate with memory transaction issues.

## What Good vs Bad Coalescing Looks Like

**Good Coalescing**:
- Low number of sectors/transactions per request
- High global load/store efficiency (> 80-90% ideal)
- Lower `memory_replay_overhead`

**Poor Coalescing**:
- High transaction count relative to requests
- Low efficiency percentages
- Elevated replay overhead and memory stalls

## Recommended Workflow for Powrush

1. Profile baseline culling kernel (focus on memory sections).
2. Profile after changes (WaveLocal Reduction, SoA layout, access pattern fixes).
3. Compare:
   - Transaction counts and efficiency percentages
   - `memory_replay_overhead`
   - Memory-related stall reasons
   - Overall kernel throughput
4. Look for improvements in coalescing efficiency and reductions in unnecessary traffic.

## Expected Benefits from Optimizations

- WaveLocal Reduction: Reduces write traffic to visible index buffers.
- SoA Layout + Linear Access: Improves read coalescing for particle data.
- Vectorized loads/stores: Increases effective transaction size and efficiency.

These metrics provide quantitative evidence of memory optimization effectiveness.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on profiling memory transactions.
    pub const PROFILE_MEMORY_TRANSACTIONS_NOTES: &str = r#"
        // Focus on Memory Workload Analysis section in Nsight Compute.
        // Compare transaction efficiency and replay overhead before/after changes.
        // Good coalescing = lower transactions per request + higher efficiency.
        // WaveLocal Reduction and SoA layouts should show measurable improvements.
    "#;
}
