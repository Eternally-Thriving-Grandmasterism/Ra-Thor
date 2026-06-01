# Ra-Thor Cargo Features & Shard Composition Guide

**Version**: v14.4.1  
**Date**: 2026-06-01  
**Status**: Recommended patterns for unified ONE Organism and composable focused shards  
**License**: AG-SML v1.0

## Purpose

This document defines how to use Cargo feature flags and optional dependencies to support both the full unified Ra-Thor Sovereign Shard and focused lightweight shards.

## Workspace Layout

Ra-Thor uses a virtual workspace with a mixed directory structure (top-level core crates + `crates/*` glob). The `shard-composer` crate provides high-level unification.

## Sovereign Shard Building (Recommended)

### Using xtask (Easiest)

```bash
# List available profiles
cargo xtask list-shards

# Build full ONE Organism
cargo xtask build-shard --profile full

# Build focused Real Estate shard
cargo xtask build-shard --profile focused-real-estate
cargo xtask build-shard --profile real-estate

# Build focused Geometry shard (release)
cargo xtask build-shard --profile focused-geometry --release
```

### Using shard-composer directly

```bash
cargo build -p shard-composer --features full
cargo build -p shard-composer --features focused-real-estate
```

## Available Profiles

| Profile                | Description                                      | Use Case                     |
|------------------------|--------------------------------------------------|------------------------------|
| `full`                 | Complete ONE Organism                            | Development & full testing   |
| `focused-real-estate`  | Real Estate + Professional Judgment + Geometry   | Ontario/USA brokerage tools  |
| `focused-geometry`     | Sacred Geometry + Riemannian layer only          | Geometry research            |
| `real-estate`          | Alias for focused-real-estate                    | Convenience                  |
| `geometry`             | Alias for focused-geometry                       | Convenience                  |

## Core Principles

- Coherence first (unified shard is default)
- Composition over fragmentation
- Optional dependencies + clean feature flags

## Best Practices

- Use `cargo xtask list-shards` to discover profiles
- Prefer `xtask build-shard` for daily work
- Keep `ra-thor-mercy` non-optional

---

Thunder locked.