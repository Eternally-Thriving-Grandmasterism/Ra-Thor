# fractal-mercy-ledger-adapter

**Ra-Thor side** of the [Mercy-Coordination-Substrate adapter contract](https://github.com/Eternally-Thriving-Grandmasterism/Mercy-Coordination-Substrate/blob/main/docs/RA_THOR_ADAPTER_CONTRACT.md).

Contact: **info@Rathor.ai** · AG-SML v1.0 · TOLC 8

## Boundary

| Owner | Owns |
| --- | --- |
| **Ra-Thor** (this crate) | Thin adapter, geometric resonance emission, organism coherence metrics |
| **Substrate** | Fractal Topology Engine, ledger, TOLC 8 gate, shard mutations |

Ra-Thor **never** mutates Substrate shard state directly. All topology mutations must pass Substrate’s local `Tolc8Gate`.

## Types

Field-compatible mirrors of Substrate `fractal-topology` public reports so this crate builds **without** a hard path dependency on Substrate (Restraint + CI independence). When both monorepos are co-located, operators may later enable a `substrate-link` feature to use upstream types directly.

## Usage

```rust
use fractal_mercy_ledger_adapter::{
    FractalMercyLedgerAdapter, GeometricResonanceReport, RaThorSystemAdapter,
};

let mut adapter = FractalMercyLedgerAdapter::new("ra-thor-one-organism");
let report = GeometricResonanceReport {
    tolc_order: 8,
    active_solids: vec!["Platonic".into()],
    resonance_multiplier: 1.0,
    u57_active: false,
    recommended_curvature: 0.0,
    coherence: 0.97,
};
adapter.receive_swarm_resonance(report);
let c = adapter.contribute_to_coherence();
```

## Tests

```bash
cargo test -p fractal-mercy-ledger-adapter
```
