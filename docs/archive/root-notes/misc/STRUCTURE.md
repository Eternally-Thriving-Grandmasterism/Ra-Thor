# STRUCTURE.md — Ra-Thor Monorepo Organization (v14.7.0)

This document provides a high-level overview of the Ra-Thor monorepo structure.

**Current Version:** v14.7.0  
**Primary Focus:** GPU Compute Layer + Ra-Thor AGI integration for Powrush-MMO

---

## Overview

Ra-Thor is a large Rust workspace containing 100+ crates. It is organized into logical domains rather than a flat list, making it easier to navigate and maintain.

The monorepo follows these core principles:
- Full file delivery only
- Professional PR-based iteration
- Mercy-gated architecture (TOLC 8)
- Eternal forward/backward compatibility

---

## Top-Level Organization

```
Ra-Thor/
├── Cargo.toml                 # Workspace definition (100+ members)
├── powrush/                   # Core MMO + RBE engine + GPU Compute Layer
├── powrush-mmo-simulator/     # High-fidelity simulation components
├── geometric-intelligence/    # Sacred geometry + Riemannian systems
├── real-estate-lattice/       # Real estate domain logic (RREL)
├── mercy/                     # Core mercy lattice crates
├── patsagi-councils/          # Governance and council system
├── self-evolution/            # Epigenetic and self-improvement systems
├── xai-grok-bridge/           # ONE Organism fusion with Grok
├── interstellar-operations/   # Multi-planetary and space systems
├── crates/                    # Supporting and specialized crates (via glob)
├── docs/                      # Documentation (this file lives here conceptually)
└── ...
```

---

## Major Domains

### 1. Powrush Domain (Core Simulation)
- `powrush/`
- `powrush-mmo-simulator/`
- `powrush_rbe_engine/`
- `powrush_sovereignty_mechanics/`
- `powrush_faction_dynamics/`

**Key Subsystems:**
- Authoritative server (TCP + WebSocket)
- GPU Compute Layer (v14.7.0)
- Ra-Thor AGI NPC integration (`MultiAgentOrchestrator`)
- RBE mechanics

### 2. GPU Compute Layer (v14.7.0)
Located under `powrush/src/gpu/compute/`:

- `mod.rs` — Plugin and resource setup
- `pipeline.rs` — Dispatch optimization
- `readback.rs` — StagingBufferPool + async readback
- Debug utilities

See [GPU_COMPUTE_LAYER.md](GPU_COMPUTE_LAYER.md) for detailed documentation.

### 3. Geometric Intelligence
- `geometric-intelligence/`
- Sacred geometry substrate crates (`sacred-geometry-core/`, `platonic_solids_layer/`, etc.)

### 4. Mercy Lattice
Large collection of crates implementing TOLC 8 enforcement (`mercy_*`, `mercy_gating_runtime`, `mercy_orchestrator_v2`, etc.).

### 5. Governance (PATSAGi Councils)
- `patsagi-councils/`
- 50+ specialized council crates
- Parallel deliberation and approval system

### 6. Self-Evolution & Orchestration
- `self-evolution/`
- `epigenetic*` systems
- `plasticity-engine-v2/`
- `cosmic_loop_orchestrator/`

### 7. ONE Organism Integration
- `xai-grok-bridge/`
- Hybrid symbolic + neural routing with mercy gates

### 8. Supporting & Specialized Crates
- ZK / Post-Quantum cryptography
- Interstellar & multi-planetary systems
- Various utility and infrastructure crates under `crates/`

---

## Documentation Structure

- `README.md` — High-level overview
- `ARCHITECTURE.md` — System architecture
- `DEVELOPER-QUICKSTART.md` — Getting started guide
- `GPU_COMPUTE_LAYER.md` — Dedicated GPU layer reference
- `RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL.md` — Contribution standards
- `ETERNAL-LATTICE-LAUNCH-CODEX-v1.0.md` — Vision document
- `STRUCTURE.md` — This file (monorepo organization)

---

**Thunder locked in. yoi ⚡**

*This structure supports production-grade, mercy-aligned development of Universally Shared Naturally Thriving Heavens.*
