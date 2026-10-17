# abundance-breath

`BreathCycle::trigger` appends two `ValenceQuantum` values to a `SharedValenceField`.

Inhale amount is the source literal `0.00003`. Exhale amount is the source literal `0.00004`. Those amounts are not measured values.

The field floor `0.999999` lives in `shared-valence-field`. `emit` applies `.max(0.999999)`. That literal is not a measured value.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.2  
**Status:** `src/feature_flag.rs` is not a module. The flag is not compiled.

## What the function does

`trigger` copies the bool argument into `offered_to_air_foundation`. The exhale context is `abundance-exhale-air-foundation` when that bool is true, and `abundance-exhale` when it is false. No transfer runs in this crate.

## Usage

```rust
use abundance_breath::BreathCycle;
use shared_valence_field::{SharedValenceField, Substrate};

let mut field = SharedValenceField::new("instance-01");
let cycle = BreathCycle::trigger("player-1", Substrate::Human, &mut field, false);
```

`cycle.inhale_valence` is `0.00003`. `cycle.exhale_valence` is `0.00004`. `field.quanta.len()` is 2.
