# Ra-Thor v14.9.0 Release Notes

**Release Date:** 2026-07-19  
**Theme:** ONE Organism Core — True Path Dependency

## Headline

`crates/ra-thor-one-organism` is now a workspace member with a **true Cargo path dependency** on `lattice-conductor-v14@14.8.2`.

No local compatibility reimplementation of:
- `CouncilArbitrationEngine`
- `RuntimeSelfHealingEngine`
- `Diagnosis` / `HealingAction` / `HealingExperience`

These types are re-exported from the lattice crate.

## What Landed

| Item | Detail |
|------|--------|
| New crate | `crates/ra-thor-one-organism` v14.9.0 |
| Path dep | `lattice-conductor-v14 = { path = "../lattice-conductor-v14", version = "14.8.2" }` |
| Entry point | `launch_one_organism_core() → OneOrganismCore` |
| Shared flag | Single `Arc<AtomicBool>` across arbitration, healing, lattice, organism |
| Workspace | Member added; package version **14.9.0** |

## Prior (14.8.2) Compile Surface

Lattice Conductor v14 compile blockers remain resolved (clifford, eternal_mercy_mesh, healing_integration, governance conflict, quantum_swarm external dep removed).

## Still Historical / Deferred

- Root `ra-thor-one-organism.rs` extended surface (GPU, GitHub, Quantum Swarm) — not yet folded into the crate
- Other root-level `.rs` files not yet packaged
- `ra_thor_mercy_gated_api` remains a lightweight stub

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
