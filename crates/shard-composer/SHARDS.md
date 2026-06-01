# Sovereign Shard Profiles

This document describes the available shard compositions in `shard-composer`.

## Profiles

### `full`
Complete ONE Organism experience. Includes all major systems with full feature sets.

**Features activated:**
- `geometric-intelligence/full`
- `real-estate-lattice/full`
- `quantum-swarm-orchestrator/full`
- `patsagi-councils/full`

**Use case:** Full development, testing, and ONE Organism simulation.

### `focused-real-estate` (alias: `real-estate`)
Focused on Real Estate workflows, including the Ontario Professional Judgment Layer.

**Features activated:**
- `geometric-intelligence/focused-geometry`
- `real-estate-lattice/focused-real-estate`
- `patsagi-councils`

**Use case:** Ontario/USA real estate tools, brokerage assistants, focused offline shards.

### `focused-geometry` (alias: `geometry`)
Focused on Sacred Geometry and Riemannian layers.

**Features activated:**
- `geometric-intelligence/focused-geometry`

**Use case:** Geometry research, Riemannian transport experiments, lightweight geometry shards.

## Usage

```bash
cargo xtask build-shard --profile focused-real-estate
cargo build -p shard-composer --features focused-real-estate
```

## Adding New Profiles

1. Add the profile to `Cargo.toml`
2. Document it here
3. Update xtask if needed
