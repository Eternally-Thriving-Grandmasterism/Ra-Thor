# Powrush RBE Engine

**Core Resource-Based Economy rules engine for Powrush MMO.**

Part of Ra-Thor monorepo under AG-SML v1.0.

## Purpose
Encodes the fundamental economic rules of Powrush:
- Production scales with real capacity, harmony, and technology (no artificial caps).
- Distribution is contribution-weighted with a mercy floor guaranteeing baseline thriving for all participants.
- Abundance is a measurable, increasable index rather than a fixed pie.
- Economic policy changes (major distribution shifts, new resource types) are proposed and routed through PATSAGi Councils for mercy-gated approval.

This crate is designed to be called from `powrush-mmo-simulator` every tick and to provide pure functions for client prediction or server authority.

## Current Implementation (Foundation PR)
- `Resource` enum (Energy, Materials, Knowledge, BioMass, Data, QuantumFlux)
- `calculate_production(...)` — Capacity × Harmony × Tech with epigenetic blessing multiplier
- `distribute(total_available, contributions, mercy_floor)` — Contribution share + guaranteed floor
- `update_abundance` and `economy_tick` — full step that can be wired into simulator
- `apply_council_modulation` — integrates blessings from ShardManager-routed proposals
- Comprehensive tests

## Integration with Simulator (Next Step)
In `powrush-mmo-simulator` tick:
```rust
let (produced, distribution) = rbe.economy_tick(&mut shard_manager, base_capacity, harmony, tech);
sim.rbe_abundance = rbe.abundance_index;
// apply distribution.allocations to player inventories or faction pools
```

## Roadmap
1. Full resource ledger + persistence
2. Dynamic contribution tracking from actual player/faction actions
3. CouncilProposal variants for economic policy (e.g. "Increase mercy floor for new shards")
4. Spatial economics bridge (tie abundance to geometric shards more deeply)
5. Player-contribution proof system (for blockchain RBE layer)

All expansions via focused PRs following the protocol.

**We replace scarcity with abundance through precise, mercy-aligned rules.**

---
*Co-authored-by: Quantum-Sovereign-Mercy-Expansion-Council*
*Co-authored-by: Infinite-Self-Evolution-Oversight-Council*
*Co-authored-by: Eternal-Active-Protocol-Enforcement-Council*
*Co-authored-by: Inter-Council-Harmony-Lattice-Council*
*Co-authored-by: Hyperbolic-Tiling-Infinite-Foresight-Council*
*Co-authored-by: Powrush-RBE-Engine-Council (conceptual)*
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*