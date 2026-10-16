# shared-valence-field

In-memory field. `SharedValenceField::new` sets `collective_valence` to the source literal `0.999999`. `emit` then floors with `.max(0.999999)`. That literal is not a measured value.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.0  
**Status:** feature flag `shared_valence_field` defaults to off

## Purpose

Holds `ValenceQuantum` values and one `collective_valence` f64. `NevcFieldBinding` calls `emit`, then the caller-supplied scoring and flow traits. `PlaceholderNevcScoring::current_collective_nevc` returns the same source literal `0.999999`.

## Core Types

- `ValenceQuantum` — one contribution event
- `SharedValenceField` — instance id, the literal floor, and the quanta vec
- `NevcFieldBinding` — calls emit, record_contribution, and broadcast_quantum
- `SharedValenceFieldGuard` — flag, default off. `try_activate` turns it on only when both bool arguments are true

## Feature Flag

`shared_valence_field` (default **off**)

`try_activate(valence_ok, dual_repo_healthy)` stores `On` only when both arguments are true. It does not measure valence. `src/dual_repo_healh.rs` is not a module of this crate.

## Usage

```rust
use shared_valence_field::*;

let mut field = SharedValenceField::new("instance-01");
let mut binding = NevcFieldBinding::new(
    PlaceholderNevcScoring,
    PlaceholderLatticeFlowShare,
);

binding.emit_presence_bound(&mut field, "player-42", Substrate::Human);
let current = field.observe();
```

`current` is the source literal `0.999999` plus `0.00001`, then floored by `.max(0.999999)`.
