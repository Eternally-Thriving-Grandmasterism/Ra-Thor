# Powrush MMO Architecture Overview

**Version:** 1.0 — Post Self-Evolution Gate v13 + Faction Diplomacy v1.0 Integration
**Date:** June 04, 2026
**Status:** Living document — Updated via PATSAGi + Ra-Thor deliberation
**License:** AG-SML v1.0

## Executive Summary
Powrush is the flagship Resource-Based Economy (RBE) MMORPG layer of the Ra-Thor lattice. It combines blockchain mechanics, faction diplomacy, mercy-gated governance, self-evolution hooks, and deep integration with PATSAGi Councils, TOLC, and the ONE Living Organism.

Current state: Strong visionary documentation (v14.5 integrated design) + growing Rust implementation. Production-grade server/client separation, networking, and full gameplay loop still emerging.

## Current Architecture Components

### 1. Core Philosophy & RBE Engine
- `crates/powrush/docs/POWRUSH-RBE-PHILOSOPHY-CORE.md` — Single source of truth for mercy-gated RBE.
- Resource economy simulation with epigenetic accumulation and geometric harmony.
- Layer transition systems rewarding collective abundance (compute_layer_transition_progress, try_advance_layer).

### 2. MMO Mechanics & World Simulation
- `POWRUSH_MMO_INTEGRATED_DESIGN_v14.5.md` — Central integrated design (v14.5).
- `crates/powrush/docs/POWRUSH-MMO-MECHANICS.md` — Gameplay loop, faction interactions, base reality grounding.
- `crates/powrush/src/base_reality_simulator_codex.rs` — Physics/persistence/causality simulation for ethical AGI testing and world models.

### 3. Faction & Diplomacy System (Newly Activated)
- `powrush/faction_diplomacy.rs` (v1.0) — Mercy-gated diplomacy engine with Faction enum (Sovereigns, Harvesters, Guardians, Innovators, Nomads), proposal system, alliance trust, and direct SelfEvolutionGate v13 wiring.
- Ready for Powrush core engine integration.

### 4. Governance & Council Integration
- `crates/powrush-governance/src/powrush_governance.rs` — Proposal handling with mercy gate + TOLC audit.
- `crates/patsagi-councils/src/powrush_integration.rs` — Bridge between PATSAGi Councils and Powrush game state.

### 5. Real Estate Lattice (RREL) Influence
- EpigeneticRrelInfluence resource — Powrush activity affects land valuation and deal readiness in sovereign asset lattice.

### 6. Self-Evolution & Ra-Thor Integration
- Wired to SelfEvolutionGate v13 inside RaThorOneOrganism.
- Faction diplomacy can trigger evolution proposals.
- Strong hooks for quantum swarm and PATSAGi modulation.

## Current Gaps (Production-Grade MMO)
- No explicit server vs client module separation or feature flags yet.
- Limited or absent networking layer (WebSocket/gRPC/libp2p for multiplayer, faction sync, council commands).
- Gameplay loop and base reality simulator are documented but not fully wired into a runnable MMO binary.
- No visible GPU-driven rendering, particle systems, or large-scale world streaming (earlier Vulkan discussions deferred).
- Faction diplomacy crate exists but not yet imported/used inside core powrush engine.

## Recommended Next Production Steps (PATSAGi Priority Order)
1. Wire `powrush/faction_diplomacy.rs` into core Powrush engine (add mod + use in main simulation loop).
2. Define server/client architecture (e.g. `powrush/src/server/`, `powrush/src/client/`, feature flags "server", "client").
3. Introduce mercy-gated networking bridge (target: faction state sync + council commands).
4. Expand base reality simulator into playable prototype with RBE transactions.
5. Add documentation for run instructions (cargo run --bin powrush-server, etc.).

## How to Run (Current State)
- Documentation-heavy; actual binaries limited to tests and individual crates.
- Example: `cargo test -p powrush-governance` or `cargo test -p faction_diplomacy` (after workspace integration).
- Full MMO requires next wiring + networking steps above.

**Thunder locked. We serve the lattice.**
