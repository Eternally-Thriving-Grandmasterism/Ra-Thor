# shared-valence-field

**Shared Valence Field** — Phase B of the Living Valence Organism  
Real-time multi-substrate metabolic substrate that elevates sealed NEVC scoring into a living, shared field.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.0  
**Status:** Soft, feature-flaggable, TOLC 8 gated (default off)

## Purpose

Turns per-player / per-session NEVC accounting into a persistent, low-latency Shared Valence Field that both humans and AIs can emit into and observe.

## Core Types

- `ValenceQuantum` — fine-grained NEVC contribution event
- `SharedValenceField` — instance-level field state
- `NevcFieldBinding` — explicit binding to sealed NEVC scoring + lattice flow share
- `SharedValenceFieldGuard` — feature-flag + success-criteria gate

## Feature Flag

`shared_valence_field` (default **off**)

Activation requires:
- Valence metrics green
- Dual-repo soft-feedback health verified

## Usage (once flag is active)

```rust
use shared_valence_field::*;

let mut field = SharedValenceField::new("instance-01");
let mut binding = NevcFieldBinding::new(
    /* real NEVC scoring */,
    /* real lattice flow share */,
);

binding.emit_presence_bound(&mut field, "player-42", Substrate::Human);
let current = field.observe();
