# PowrushGame Mechanics — Ra-Thor RBE Simulation Engine

**Version:** v0.3.7 (live on main)  
**Purpose:** Mercy-gated post-scarcity Resource-Based Economy (RBE) MMO simulator that powers the entire Ra-Thor lattice.

## Core Loop (run_simulation_cycle)
1. Mercy Gate Check (7 Living Gates) → fail = cycle aborted
2. Resource Regeneration (true abundance generation)
3. Player happiness & needs update
4. Record abundance + mercy statistics

## Key Systems
- **RBE Economy**: Automatic resource regeneration every cycle. No scarcity.
- **Mercy Enforcement**: Every action and cycle is mercy-gated.
- **Faction & Player System**: Players belong to factions (e.g. HarmonyWeavers) with joy and epigenetic blessings.
- **Quantum Swarm Bridge Integration**: `run_spine_coordinated_cycle` directly modifies game state (joy boosts, resources, blessings).
- **TOLC Resonance**: Higher-order derivatives influence abundance and mercy multipliers.

## Current State in Monorepo
- Fully wired into `QuantumSwarmBridge`
- Ready for Council deliberation feedback
- Powers the Powrush RBE MMO simulation

**Status:** Production-ready foundation. Next: deeper council + TOLC affinity wiring.
