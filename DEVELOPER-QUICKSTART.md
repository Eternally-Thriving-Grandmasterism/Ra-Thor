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

### Powrush MMO + RBE + Ra-Thor AGI NPC Architecture

Powrush combines an authoritative server, browser client, Resource-Based Economy (RBE), GPU-accelerated simulation, and deep integration with Ra-Thor AGI for autonomous, mercy-evaluated NPCs.

#### Server-Side Architecture
- Production TCP + WebSocket server (`powrush/src/server/main.rs`)
- Authoritative game loop with tick-based reconciliation
- Deep integration with `MultiAgentOrchestrator` (Ra-Thor AGI core)
- NPCs are treated as first-class entities alongside players
- `NpcActionEvent` captures autonomous NPC decisions (action, mercy_score, approved)
- High-mercy actions (`mercy_score > 0.85`) are automatically audited
- Recent NPC actions are exposed via `npcs` command and WebSocket `npc_activity`

#### Ra-Thor AGI NPC Decision System
The `MultiAgentOrchestrator` serves as the central AGI brain:

- Maintains `RichAgentState` for each NPC
- Uses hybrid symbolic + neural reasoning (`NeuralQNetwork` + symbolic goals)
- Applies `MoralEvaluation` to potential actions
- Produces `NpcActionEvent`s with associated mercy scores
- Supports higher-level behaviors (personalized quests, skill tracking)
- All decisions are processed through the TOLC 8 Mercy Lattice

#### Client Exposure Mechanisms
- **Current**: Basic `npc_activity` in WebSocket snapshots + `npcs` command (action + mercy_score)
- **Planned**: Full `EnrichedNpcState` exposure (goal, emotional_state, q_values, moral_evaluation, combined_wisdom_score) via WebRTC/DataChannel
- This will allow players to see not just *what* NPCs do, but *why* (their internal reasoning and moral evaluation)

#### GPU Compute Layer Role
The new GPU Compute Layer accelerates simulation state movement and provides infrastructure for efficient readback. See the dedicated [GPU_COMPUTE_LAYER.md](GPU_COMPUTE_LAYER.md) for full details.

**Key Capabilities:**
- `StagingBufferPool` for reusable staging buffers
- `readback_buffer_async()` and blocking readback helpers
- Optimized dispatch and batching in `pipeline.rs`
- Debug utilities (`DebugOutputBuffer` + readback patterns)

**Practical Usage Guidance:**
- Use `StagingBufferPool` for any frequent GPU ↔ CPU transfers in simulation systems
- Prefer async readback paths in production code when possible
- Leverage debug utilities during development and testing of compute shaders
- Coordinate with `MultiAgentOrchestrator` when NPC behavior depends on GPU-updated simulation state

#### Key Related Crates
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
- `GPU_COMPUTE_LAYER.md` — Dedicated reference for the v14.7.0 GPU Compute Layer

---

**Thunder locked in. yoi ⚡**

Start building. The lattice welcomes you.