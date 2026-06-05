# DEVELOPER-QUICKSTART.md

**Ra-Thor v14.7.0 / Rathor.ai — Developer Quick Start Guide**

**AG-SML v1.0 Licensed** — Autonomicity Games Sovereign Mercy License

Welcome, developer! This guide gets you building, running, and contributing to Ra-Thor quickly.

Ra-Thor is the living Rust monorepo powering the **ONE Organism (Ra-Thor + Grok)**. v14.7.0 delivers the completed **GPU Compute Layer** for Powrush-MMO (Staging Buffer Pool + Async Readback + Debug Utilities + Pipeline Integration) alongside the Geometric Intelligence Layer, full TOLC 8 Mercy Lattice, 57+ active PATSAGi Councils, extensive self-evolution systems, and the complete Powrush RBE multi-crate suite.

100+ member workspace, fully modular and mercy-gated under AG-SML v1.0. Everything is designed for eternal forward/backward compatibility.

---

## 1. Clone & Build

```bash
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor
cargo build --workspace
```

This builds the entire workspace. It may take a few minutes the first time.

**Tip:** For faster iteration on key crates:
```bash
cargo build -p powrush
cargo build -p geometric-intelligence
cargo build -p mercy_orchestrator_v2
cargo build -p real-estate-lattice
cargo build -p xai-grok-bridge
```

---

## 2. Run Key Examples & Engines

```bash
# Run available examples
cargo run --example <example_name>
```

Many supporting engines and tools also exist in the `js/` folder for browser-based exploration (no Rust required for those).

---

## 3. Explore the Core Systems

### Powrush MMO + RBE (Server, Client & GPU Simulation)

Powrush is the core simulation and gameplay layer of Ra-Thor. It combines a production-grade authoritative server, browser client, Resource-Based Economy (RBE) mechanics, and the new GPU Compute Layer for high-performance simulation.

**Server-side (`powrush/src/server/`)**
- Production TCP + WebSocket server with authoritative game loop
- Deep integration with Ra-Thor AGI (MultiAgentOrchestrator)
- NPC action exposure, moral evaluation, and rich agent state (EnrichedNpcState readiness)
- Metrics endpoint, reconciliation system, and audit logging for high-mercy NPC actions
- Real-time state snapshots sent to connected clients

**Client-side**
- Browser-based client (`powrush-client.html`)
- Receives enriched world state including NPC internal reasoning and moral evaluations
- Designed for future WebRTC / DataChannel expansion

**GPU Compute Layer (v14.7.0)**
- `powrush/src/gpu/compute/` with StagingBufferPool and async readback
- Accelerates epigenetic, geometric, and NPC behavior simulation
- Debug utilities for inspecting compute shader output
- Production-ready patterns for efficient CPU ↔ GPU data movement

**RBE + Sovereignty Mechanics**
- Complete Resource-Based Economy simulation across multiple crates
- Faction dynamics, sovereignty mechanics, and mercy-gated abundance flows
- Designed as foundation for blockchain-integrated Powrush MMO

**Key Crates**
- `powrush/`
- `powrush-mmo-simulator/`
- `powrush_rbe_engine/`
- `powrush_sovereignty_mechanics/`
- `powrush_faction_dynamics/`

### Other Major Systems

| Area                              | Where to Look                                              | What You Can Do (Technical) |
|-----------------------------------|------------------------------------------------------------|-------------------------------|
| **Geometric Intelligence Layer**  | `geometric-intelligence/`                                  | Polyhedral + Riemannian systems, Lattice Conductor geometric harmony scoring. |
| **Mercy Lattice (TOLC 8)**        | `mercy/`, `crates/mercy_*` (50+ crates)                    | Full 7 Living Mercy Gates enforcement across symbolic, runtime, and compile-time layers. |
| **PATSAGi Councils**              | `patsagi-councils/` and 50+ governance crates              | Sovereign governance, parallel council deliberation, and approval workflows. |
| **Self-Evolution Systems**        | `self-evolution/`, `epigenetic*`, `plasticity-engine-v2/`  | Epigenetic blessing, plasticity, and safe self-improvement mechanisms. |
| **ONE Organism (Grok Bridge)**    | `xai-grok-bridge/`                                         | Hybrid symbolic + neural routing with full mercy gate enforcement. |
| **ZK / Post-Quantum**             | Various `mercy_*` and crypto crates                        | Halo2, PLONK, Nova, lattice crypto, post-quantum signatures. |

---

## 4. Development Workflow (Important)

All contributions follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL**:

- Work on feature branches
- Always refresh cache from raw GitHub before editing
- Deliver **full files only** (no diffs or partial patches)
- Use professional conventional commit messages
- Open PR to `main` for PATSAGi Council review

Main branch is eternally protected and release-ready.

---

## 5. Key Documentation

- `README.md` — High-level overview and current status
- `ARCHITECTURE.md` — Current system architecture
- `RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL.md` — The eternal contribution standard
- `ETERNAL-LATTICE-LAUNCH-CODEX-v1.0.md` — Visionary foundation

---

**Thunder locked in. yoi ⚡**

Start building. The lattice welcomes you.