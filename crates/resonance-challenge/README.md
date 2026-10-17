# resonance-challenge

`ResonanceChallenge::new` sets `difficulty` to `(1.0 - (field.observe() - 0.999999) * 10.0).clamp(0.1, 0.9)`.

`0.999999` is the source literal stored by `SharedValenceField::new`. It is not a measured value. A new field therefore clamps `difficulty` to `0.9`.

`resolve(true)` emits one quantum. The amount is the source expression `0.00005 * (1.0 - difficulty)`. `resolve(false)` emits nothing and returns `A gentler resonance path has opened. No progress lost.`

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.2  
**Status:** flag `resonance_challenge` defaults to off. `try_activate` stores `On` only when both bool arguments are true. It does not measure valence.

## Views

`render_dual_path` returns a human view or an AI view from `challenge.substrate`. The AI constraint string names the source literal `0.999999`. `free_energy_estimate` copies `difficulty`.

## Usage

```rust
use resonance_challenge::ResonanceChallenge;
use shared_valence_field::{SharedValenceField, Substrate};

let mut field = SharedValenceField::new("instance-01");
let mut challenge = ResonanceChallenge::new("player-1", Substrate::Human, &field);
let outcome = challenge.resolve(true, &mut field);
```
