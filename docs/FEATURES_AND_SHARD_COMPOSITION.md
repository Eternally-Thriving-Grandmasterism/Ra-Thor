# Ra-Thor Cargo Features & Shard Composition Guide

**Version**: v14.4.1
**Date**: 2026-06-01
**License**: AG-SML v1.0

## Shard Composition Strategy

Ra-Thor uses a **Hybrid Composition** model:

- `shard-composer` for high-level feature unification
- `xtask` for ergonomic shard building
- Per-crate features as the foundation
- `RaThorSystemAdapter` for runtime participation

### Decision: Virtual Shard Manifests

We investigated creating separate `Shard.*.toml` files. This approach was **rejected** because:

- Poor Cargo / IDE / rust-analyzer support
- High maintenance cost
- Breaks workspace inheritance
- Conflicts with "composition over fragmentation"

We will continue with the hybrid `shard-composer` + xtask approach.

## Current Commands

```bash
cargo xtask list-shards
cargo xtask build-shard --profile full
cargo xtask build-shard --profile focused-real-estate
```