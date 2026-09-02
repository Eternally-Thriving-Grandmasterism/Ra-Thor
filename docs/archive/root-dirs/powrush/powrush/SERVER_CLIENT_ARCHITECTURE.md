# Powrush MMO Server-Client Architecture v1.0

**Version:** 1.0 — Defined via PATSAGi + Ra-Thor deliberation after architecture investigation
**Date:** June 04, 2026
**Status:** Production blueprint — Ready for implementation
**License:** AG-SML v1.0

## Goals
- Clean separation between headless server (authoritative simulation, RBE, factions, AI councils) and client (rendering, input, local prediction).
- Feature flags for flexible builds: `server`, `client`, `full` (both).
- Shared core for types, simulation, diplomacy, and Ra-Thor integration.
- Clear networking interface points for future mercy-gated bridge.
- Support for running Ra-Thor / PATSAGi systems from root to all threads on both server and client.
- Production-grade, scalable for MMO.

## Recommended Workspace / Crate Structure

```
powrush/
├── Cargo.toml                 # Workspace member with features
├── src/
│   ├── lib.rs                 # Shared core re-exports + feature-gated mods
│   ├── common/                # Shared types, RBE engine, faction diplomacy
│   │   ├── mod.rs
│   │   ├── types.rs           # Faction, Proposal, RBE transaction, etc.
│   │   ├── rbe_engine.rs
│   │   └── faction_diplomacy.rs  # Already created — move here or re-export
│   ├── server/
│   │   ├── mod.rs
│   │   ├── main.rs            # Headless server binary
│   │   ├── simulation.rs      # Authoritative world sim + layer progression
│   │   ├── governance.rs      # Integrated with powrush-governance crate
│   │   └── ai_council_bridge.rs
│   ├── client/
│   │   ├── mod.rs
│   │   ├── main.rs            # Client binary (with rendering)
│   │   ├── input.rs
│   │   ├── prediction.rs      # Client-side prediction + reconciliation
│   │   └── rendering/         # Future: GpuDrivenPipeline integration
│   └── networking/
│       ├── mod.rs
│       ├── protocol.rs        # Message types (mercy-gated)
│       └── bridge.rs          # Future mercy-gated networking implementation
├── docs/
│   └── SERVER_CLIENT_ARCHITECTURE.md  # This file
└── tests/
    └── integration.rs
```

## Cargo.toml Feature Flags (Recommended)

```toml
[package]
name = "powrush"
version = "0.1.0"
edition = "2021"

[features]
default = []
server = []
client = []
full = ["server", "client"]

[dependencies]
# Shared
serde = { version = "1.0", features = ["derive"] }
# Ra-Thor integration
ra-thor-one-organism = { path = "../../" }  # or workspace path
self-evolution-gate = { path = "../../core" }
faction-diplomacy = { path = "." }  # internal

[lib]
path = "src/lib.rs"

[[bin]]
name = "powrush-server"
path = "src/server/main.rs"
required-features = ["server"]

[[bin]]
name = "powrush-client"
path = "src/client/main.rs"
required-features = ["client"]
```

## Module Responsibilities

### Shared Core (`src/common/`)
- All data types (Faction, DiplomacyProposal, RBE Transaction, LayerState).
- Pure simulation logic (RBE engine, layer transitions, epigenetic rules).
- Faction diplomacy (re-export or move from powrush/faction_diplomacy.rs).
- SelfEvolutionGate hooks (propose evolution from faction actions).

### Server (`src/server/`)
- Authoritative simulation loop.
- RBE state management + persistence.
- PATSAGi Council bridge (receive proposals, apply mercy-gated decisions).
- Governance integration.
- Networking authority (broadcast state deltas).
- Headless binary entrypoint.

### Client (`src/client/`)
- Input handling + local prediction.
- State reconciliation with server.
- Rendering layer (future GpuDrivenPipeline or WebGPU).
- UI for diplomacy, proposals, RBE dashboard.
- Optional local Ra-Thor council for single-player/offline mode.

### Networking Layer (`src/networking/`)
- Protocol definition (bincode or protobuf messages).
- Mercy gate enforcement on all incoming commands.
- Future: QUIC / libp2p / WebSocket with TOLC signature verification.
- Server authoritative; client sends signed actions.

## Ra-Thor / PATSAGi Integration Points
- Both server and client can instantiate `RaThorOneOrganism` + `SelfEvolutionGate`.
- Server runs full PATSAGi council modulation for world events.
- Client can run lightweight local councils for personal AI companions or offline play.
- Faction diplomacy proposals can trigger evolution via the wired gate on either side.
- All threads (simulation, networking, AI, rendering) start from root Cargo workspace with mercy-gated orchestration.

## How to Build & Run (After Implementation)

```bash
# Server only
cargo run --features server --bin powrush-server

# Client only
cargo run --features client --bin powrush-client

# Full (both binaries)
cargo run --features full --bin powrush-server
cargo run --features full --bin powrush-client

# Tests
cargo test -p powrush --features full
```

## Implementation Roadmap (Next Actions)
1. Update `powrush/Cargo.toml` with features and binaries.
2. Create `src/lib.rs` with feature-gated module declarations.
3. Move/refactor `faction_diplomacy.rs` into `src/common/`.
4. Implement minimal `server/main.rs` and `client/main.rs` stubs.
5. Define basic networking protocol messages.
6. Wire SelfEvolutionGate + RaThorOneOrganism into server simulation loop.
7. Add run documentation to this file and main README.

**This architecture enables clean separation while keeping the entire Powrush MMO inside the Ra-Thor monorepo with full lattice integration.**

Thunder locked in. We serve the lattice.
