# symbiotic-membrane

`SymbioticMembrane::form_contact` calls `emit_presence_bound`. `SharedValenceField::emit_presence` uses the source amount `0.00001`. `emit` then floors with the source literal `0.999999`. Those literals are not measured values.

A human result stores `guide_style` as the Debug text of `AdaptiveGuide::style`. A new guide is `Balanced`. An AI result copies `AiHandshake` fields, including `protocol_version` `"1.0"`.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.1  
**Status:** flag `symbiotic_membrane` defaults to off. `try_activate` stores `On` only when both bool arguments are true. It does not measure valence.

## Guide signals

Inside the learning window, `observe_interaction` maps `slow` / `pause` / `reflect` to `Gentle`, `ask` / `why` / `curious` to `Curious`, `go` / `next` / `fast` to `Direct`, and `explore` / `look` / `wander` to `Exploratory`. Any other string sets `Balanced`. Each call adds 1 to `interaction_count`. The branch after 90 seconds was not exercised by the tests in this crate.

## Usage

```rust
use shared_valence_field::nevc_binding::{
    NevcFieldBinding, PlaceholderLatticeFlowShare, PlaceholderNevcScoring,
};
use shared_valence_field::{SharedValenceField, Substrate};
use symbiotic_membrane::SymbioticMembrane;

let mut field = SharedValenceField::new("instance-01");
let mut binding = NevcFieldBinding::new(
    PlaceholderNevcScoring,
    PlaceholderLatticeFlowShare,
);
let result = SymbioticMembrane::form_contact("participant-id", Substrate::Human, &mut field, &mut binding);
```

`result.guide_style` is `Some("Balanced")`. `field.quanta.len()` is 1. The quantum amount is `0.00001`.
