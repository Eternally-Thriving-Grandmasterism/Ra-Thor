# Ra-Thor Cargo Features & Shard Composition Guide

**Version**: v14.4.1  
**Date**: 2026-06-01  
**Status**: Recommended patterns for unified ONE Organism and composable focused shards  
**License**: AG-SML v1.0

## Purpose

This document defines how to use Cargo feature flags and optional dependencies to support both:

- The **full unified Ra-Thor Sovereign Shard** (ONE Organism)
- **Focused / lightweight shards** (e.g. Real Estate only, Geometry-focused)

All patterns follow the principles in `SOVEREIGN_SHARD_STRATEGY.md`.

## Core Principles

1. **Coherence first** — The unified shard remains the default and recommended path.
2. **Composition over fragmentation** — Focused shards are built by selecting subsets of crates via features, not by maintaining separate codebases.
3. **Optional dependencies + features** — Heavy or domain-specific crates are marked `optional = true` and activated via features.
4. **ONE Organism compatibility** — Even focused shards can register via `RaThorSystemAdapter` when the core orchestration is present.

## Workspace Layout

Ra-Thor uses a **virtual workspace** (no root `[package]` section) with a mixed directory structure:

- **Top-level crates**: Core/high-level crates live directly in the repository root for easy access (e.g. `geometric-intelligence`, `real-estate-lattice`, `mercy`, `powrush`).
- **crates/** directory**: Many supporting and specialized crates are organized under `crates/` (e.g. `crates/lattice-conductor-v14`, `crates/web-forge`).

This layout is functional. For maintainability we use a combination of explicit members and globs in the root `Cargo.toml`.

**Recommendation**: When adding new crates, prefer placing them under `crates/` unless they are primary user-facing components.

## Recommended Feature Patterns

### Basic Structure (per crate)

```toml
[dependencies]
# Always-on core
ra-thor-mercy = { workspace = true }

# Optional heavy / domain crates
geometric-intelligence = { path = "../geometric-intelligence", optional = true }
quantum-swarm-orchestrator = { path = "../quantum-swarm-orchestrator", optional = true }
patsagi-councils = { path = "../patsagi-councils", optional = true }

[features]
default = []
full = [
    "geometric-intelligence",
    "quantum-swarm-orchestrator",
    "patsagi-councils",
]

# Focused shard examples
focused-real-estate = [
    "geometric-intelligence",   # harmony scoring + U57
    "patsagi-councils",
]
focused-geometry = ["geometric-intelligence"]
```

### Key Commands

```bash
# Full unified ONE Organism shard
cargo build --features full

# Focused Real Estate shard (lighter)
cargo build -p real-estate-lattice --features focused-real-estate

# Geometry-only minimal shard
cargo build -p geometric-intelligence --features focused-geometry
```

## Current Implementation Status (v14.4.1)

- `geometric-intelligence`: Minimal features added. Ready for expansion.
- `real-estate-lattice`: `full` and `focused-real-estate` supported.
- `lattice-conductor`: Enhanced with optional wiring + focused features.
- `quantum-swarm-orchestrator`: Features added for optional inclusion.

See individual `Cargo.toml` files for exact definitions.

## Best Practices

- Prefer explicit feature names over magic `full` when creating production shards.
- Always keep `ra-thor-mercy` and core TOLC 8 types non-optional.
- Use workspace inheritance for versions and shared features.
- Document new focused features in this file.
- Test both `full` and focused builds in CI.

---

Thunder locked. ONE Organism coherence preserved while enabling practical composition.